import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from video_dataset_mm import  VideoFrameDataset, ImglistToTensor
from comet_ml import Experiment
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from torchvision.models.video import r3d_18
from torchvision import models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from validate import validate_feat_concat
from models.models import VIS_PHY_MODEL_CONCAT   
import statistics


def train(train_annotation,test_annotation,v_m_path,phy_m_path,weight_name):


    """
    Training settings

    """
    num_epochs = 100
    best_epoch = 0
    check_every = 1
    b_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_acc=0

    l_rate = 0.00001



    experiment = Experiment(
        api_key="U0t7sSZhwEHvDLko0tJ4kbPH0",
        project_name="biovid",
        disabled=True,)

    parameters = {'batch_size': b_size,
                'learning_rate': l_rate,
                'epochs':num_epochs            
                }
    experiment.log_parameters(parameters)


    num_frames = 5  # Number of frames in each video clip
    num_channels = 3  # Number of channels (e.g., RGB)
    video_length = 112  # Length of the video in each dimension
    num_classes = 5  # Number of classes



    criterion = nn.CrossEntropyLoss()

    vis_phy_model=VIS_PHY_MODEL_CONCAT(v_m_path,phy_m_path,num_classes).to(device=device)

    vis_phy_optimizer = optim.Adam(list(vis_phy_model.out_layer1.parameters()) + list(vis_phy_model.out_layer2.parameters()) + list(vis_phy_model.out_layer3.parameters()) + list(vis_phy_model.out_layer4.parameters()), lr=0.00001)
    vis_optimizer = optim.Adam(vis_phy_model.vis_model.parameters(), lr=0)
    phy_optimizer = optim.Adam(vis_phy_model.phy_model.parameters(), lr=0.00001)

    scheduler = ReduceLROnPlateau(vis_phy_optimizer, mode='max', factor=0.1, patience=10, verbose=True)

    

    videos_root = '/projets2/AS84330/Datasets/Biovid/PartA/subject_images/subject_images_organised'
    # videos_root = '/home/livia/work/Biovid/PartB/Video-Dataset-Loading-Pytorch-main/demo_dataset'
    five_fold_annotations_path = '/projets2/AS84330/Datasets/Biovid/PartA/5folds_annotations_five/'
    train_annotation_file = os.path.join(five_fold_annotations_path, train_annotation)
    val_annotation_file = os.path.join(five_fold_annotations_path, test_annotation)

    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(112),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(112),  # image batch, center crop to square 299x299
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=train_annotation_file,
        num_segments=1,
        frames_per_segment=5,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        test_mode=False)
    
    val_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=val_annotation_file,
        num_segments=1,
        frames_per_segment=5,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        test_mode=True)


    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=b_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False)
    

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False)
    


    with experiment.train():
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            
            vis_phy_model.vis_model.eval()
            vis_phy_model.phy_model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            for i,(spec_2d,video_batch, labels) in enumerate(train_dataloader,0):
                vis_phy_optimizer.zero_grad()
                vis_optimizer.zero_grad()
                phy_optimizer.zero_grad()

                video_batch=video_batch.permute(0, 2, 1, 3, 4)
                out = vis_phy_model.feat_concat(video_batch,spec_2d)
                labels = labels.to(device)
                t_loss = criterion(out, labels)

                t_loss.backward()
                # vis_phy_model.vis_optim.step()
                # vis_phy_model.phy_optim.step()
                vis_phy_optimizer.step()
                vis_optimizer.step()
                phy_optimizer.step()

                running_loss += t_loss.item()
                _, predicted = torch.max(out.data, 1)   
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0
            
            train_accuracy= 100 * correct / total
            # print("*********************************************\n")
            print(f"Accuracy after epoch {epoch + 1}: {train_accuracy}%")
            train_loss= running_loss / 100
            experiment.log_metric('Loss', train_loss,epoch= epoch)
            experiment.log_metric('Accuracy', train_accuracy ,epoch= epoch)
            # last_lr=scheduler.get_last_lr()
            # experiment.log_metric('Learning Rate', last_lr,epoch= epoch)
            if epoch % check_every == 0:
                val_acc, val_loss = validate_feat_concat(vis_phy_model, val_dataloader, criterion, device)
                scheduler.step(val_acc)
                # print( "Validation accuracy: ", val_acc)
                experiment.log_metric('Val Accuracy', val_acc,epoch= epoch)
                experiment.log_metric('Val Loss', val_loss,epoch= epoch)
                current_lr = vis_phy_optimizer.param_groups[0]['lr']
                experiment.log_metric('Learning Rate', current_lr,epoch= epoch)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    weight_path = '/projets2/AS84330/Projets/MM_transformer/biovid_codes/temp_weights/' + f'{weight_name}{round(best_val_acc,2)}.pth'
                    model_save_path = os.path.join(weight_path)
                    torch.save(vis_phy_model.state_dict(), model_save_path)
                    print('Best model saved at epoch: ', epoch+1)
                    best_epoch = epoch+1


        print("Finished Training")

        train_accuracy = 100 * correct / total
        avg_train_loss = running_loss / len(train_dataloader)
        print(f'Training accuracy: {train_accuracy}%')
        print(f'Training loss: {avg_train_loss}')

        print("Best model saved at epoch: ", best_epoch)
        print("Best validation accuracy: ", best_val_acc)

        return best_val_acc
    
def test(test_annotation,v_m_path,phy_m_path, test_weights):

    num_epochs = 100
    b_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    vis_phy_model=VIS_PHY_MODEL_CONCAT(v_m_path,phy_m_path).to(device=device)
    vis_phy_model_checkpoint_path = f'/projets2/AS84330/Projets/MM_transformer/biovid_codes/all_weights/best_weights_feat_concat2/{test_weights}'
    vis_phy_model.load_state_dict(torch.load(vis_phy_model_checkpoint_path))

    videos_root = '/projets2/AS84330/Datasets/Biovid/PartA/subject_images/subject_images_organised'
    val_annotation_file = os.path.join(videos_root,'../../5folds_annotations', test_annotation)

    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(112),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(112),  # image batch, center crop to square 299x299
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=val_annotation_file,
        num_segments=10,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        test_mode=True)
    

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False)
    
    
    val_acc, _ = validate_feat_concat(vis_phy_model, val_dataloader, criterion, device)
    
    return val_acc


##### Train + Evaluate #####
if __name__ == '__main__':
    print('Training...')
    kfold_accuracy = []
    for i in range (1,6):
        train_annotation = f'train_fold{i}.txt'
        test_annotation = f'test_fold{i}.txt'
        v_m_path = f'/projets2/AS84330/Projets/MM_transformer/biovid_codes/all_weights/weights_physio_viso_five/model_best_vis_only_fold{i}.pth'
        phy_m_path = f'/projets2/AS84330/Projets/MM_transformer/biovid_codes/all_weights/weights_physio_viso_five/model_best_gsr_fold{i}.pth'

        weight_name = f'model_best_feature_concat_fold{i}_'
        best_accuracy = train(train_annotation,test_annotation,v_m_path,phy_m_path, weight_name)
        kfold_accuracy.append(round(best_accuracy,1))

    print('accuracy on fold 1', kfold_accuracy[0])
    print('accuracy on fold 2', kfold_accuracy[1])
    print('accuracy on fold 3', kfold_accuracy[2])
    print('accuracy on fold 4', kfold_accuracy[3])
    print('accuracy on fold 5', kfold_accuracy[4])
    print('mean: ', statistics.mean(kfold_accuracy))


##### Test #####
if __name__ == '__main__':
    print('Testing...')
    kfold_accuracy = []
    for i in range (1,6):
        test_annotation = f'test_fold{i}.txt'
        v_m_path = f'/projets2/AS84330/Projets/MM_transformer/biovid_codes/all_weights/weights_physio_viso5_fps5/model_best_vis_only_fold{i}.pth'
        phy_m_path = f'/projets2/AS84330/Projets/MM_transformer/biovid_codes/all_weights/weights_physio_viso5_fps5/model_best_gsr_fold{i}.pth'

        weight_name = f'model_best_feature_concat_fold{i}.pth'
        best_accuracy = test(test_annotation,v_m_path,phy_m_path, weight_name)
        kfold_accuracy.append(round(best_accuracy,1))

    print('accuracy on fold 1', kfold_accuracy[0])
    print('accuracy on fold 2', kfold_accuracy[1])
    print('accuracy on fold 3', kfold_accuracy[2])
    print('accuracy on fold 4', kfold_accuracy[3])
    print('accuracy on fold 5', kfold_accuracy[4])
    print('mean: ', statistics.mean(kfold_accuracy))
