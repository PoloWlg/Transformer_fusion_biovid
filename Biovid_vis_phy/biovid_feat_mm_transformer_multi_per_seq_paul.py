import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from video_dataset_mm import  VideoFrameDataset, ImglistToTensor
from comet_ml import Experiment
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import statistics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from torchvision.models.video import r3d_18
from torchvision import models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from validate import validate_mmtransformer_per_seq
from models.models_seq import  VIS_PHY_MODEL, Multi_cross_attention  

# seed = 42
# torch.manual_seed(seed)
# random.seed(seed)



def train(train_annotation,test_annotation,concat_m_path, weight_name):
    
    videos_root = '/projets2/AS84330/Datasets/Biovid/PartA/subject_images/subject_images_organised'
    # videos_root = '/home/livia/work/Biovid/PartB/Video-Dataset-Loading-Pytorch-main/demo_dataset'
    train_annotation_file = os.path.join(videos_root,'../../5folds_annotations2', train_annotation)
    val_annotation_file = os.path.join(videos_root,'../../5folds_annotations2', test_annotation)
    
    
    """
    Training settings
    """
    num_epochs = 100
    best_epoch = 0
    check_every = 1
    b_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    lr_vis_phy = 0
    lr_mmtransformer = 0.000005
    
    # best learning rates 1:
    # lr_vis_phy = 0
    # lr_mmtransformer = 0.000005
    
    # Best learning rates_old:
    # lr_vis_phy = 0
    # lr_mmtransformer = 0.000005

    experiment = Experiment(
        api_key="17GX9zixZo9vF8cTq9qhF2nch",
        project_name="biovid",
        disabled=True)

    parameters = {'batch_size': b_size,
                'learning_rate bb': lr_vis_phy,
                'learning_rate mmtransformer': lr_mmtransformer,
                'epochs':num_epochs            
                }
    experiment.log_parameters(parameters)


    num_frames = 5  # Number of frames in each video clip
    num_channels = 3  # Number of channels (e.g., RGB)
    video_length = 112  # Length of the video in each dimension
    num_classes = 2  # Number of classes


    criterion = nn.CrossEntropyLoss()

    vis_phy_model=VIS_PHY_MODEL(concat_m_path).to(device=device)
    mm_transformer = Multi_cross_attention(visual_dim=512, physiological_dim=512, num_heads=1, hidden_dim=512, num_layers=1, num_classes=2)

    mm_transformer = mm_transformer.to(device=device)
    # classifier_m = classifier_m.to(device=device)


    # params = [{'params': vis_phy_model.parameters(), 'lr': lr_vis_phy},{'params': mm_transformer.parameters(), 'lr': lr_mmtransformer}]

    vis_phy_optimizer = optim.Adam(vis_phy_model.parameters(), lr=lr_vis_phy)
    mmtransformer_optimizer = optim.Adam(mm_transformer.parameters(), lr=lr_mmtransformer)


    # vis_phy_mmtransformer_optimizer = optim.Adam(params)
    # scheduler = optim.lr_scheduler.StepLR(vis_phy_optimizer, step_size=5, gamma=0.01)
    # scheduler = ReduceLROnPlateau(mmtransformer_optimizer, mode='max', factor=0.1, patience=10,verbose=True)

    

    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(112),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(112),  # image batch, center crop to square 299x299
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=train_annotation_file,
        num_segments=4,
        frames_per_segment=5,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        test_mode=False)

    val_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=val_annotation_file,
        num_segments=4,
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


    
    best_val_acc=0
    with experiment.train():
        # for epoch in tqdm(range(num_epochs), desc='Epochs'):
        for epoch in range(num_epochs):
            print("*********************************************\n")
            # vis_phy_model.vis_model.train()
            # vis_phy_model.phy_model.train()
            vis_phy_model.train()
            mm_transformer.train()
            
            # classifier_m.train()
            
            running_loss = 0.0
            correct = 0
            total = 0
            for i,(spec_2d,video_batch, labels, record_images) in tqdm(enumerate(train_dataloader,0),total=len(train_dataloader), desc=f'Training'):
                mmtransformer_optimizer.zero_grad()
                vis_phy_optimizer.zero_grad()
                

                video_batch=video_batch.permute(0, 2, 1, 3, 4)
                video_batch = video_batch.to(device)
                
                # Sequence of 20 frames unflattened to 4x5
                unflatten = torch.nn.Unflatten(2, (4,5))
                video_batch = unflatten(video_batch)
                video_batch=video_batch.permute(2, 0, 1, 3, 4, 5)
                
                phy_feats_seqs = []                   
                vis_feats_seqs = []
                

                
                for count,video_batch_sequence in enumerate(video_batch):
                    vis_feats, phy_feats = vis_phy_model.model_out_feats(video_batch_sequence,spec_2d)
                    phy_feats_seqs.append(phy_feats.to(device))
                    vis_feats_seqs.append(vis_feats.to(device))
                
                phy_feats_seqs = torch.stack(phy_feats_seqs)
                vis_feats_seqs = torch.stack(vis_feats_seqs)
                phy_feats_seqs.to(device)
                vis_feats_seqs.to(device)
                
                phy_feats_seqs = phy_feats_seqs.permute(1,0,2)
                vis_feats_seqs = vis_feats_seqs.permute(1,0,2)
                
                outs = mm_transformer(vis_feats_seqs, phy_feats_seqs)

                labels = labels.to(device)
                t_loss = criterion(outs, labels)

                t_loss.backward()
                mmtransformer_optimizer.step()
                vis_phy_optimizer.step()

                running_loss += t_loss.item()
                _, predicted = torch.max(outs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0
            
            train_accuracy= 100 * correct / total
            print(f"Accuracy after epoch {epoch + 1}: {train_accuracy}%")
            train_loss= running_loss / 100
            experiment.log_metric('Loss', train_loss,epoch= epoch)
            experiment.log_metric('Accuracy', train_accuracy ,epoch= epoch)
            # last_lr=scheduler.get_last_lr()
            # experiment.log_metric('Learning Rate', last_lr,epoch= epoch)
            if epoch % check_every == 0:
                val_acc, val_loss = validate_mmtransformer_per_seq(vis_phy_model,mm_transformer, val_dataloader, criterion, device)
                # print( "Validation accuracy: ", val_acc)
                experiment.log_metric('Val Accuracy', val_acc,epoch= epoch)
                experiment.log_metric('Val Loss', val_loss,epoch= epoch)
                # scheduler.step(val_acc)
                current_lr = mmtransformer_optimizer.param_groups[0]['lr']
                experiment.log_metric('Learning Rate', current_lr,epoch= epoch)
                # print('Current learning rate: ', current_lr)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model_save_path_bb = os.path.join('/projets2/AS84330/Projets/MM_transformer/biovid_codes/temp_weights',f'{weight_name}{round(best_val_acc,2)}_visphy.pth')
                    model_save_path_mmtransformer = os.path.join('/projets2/AS84330/Projets/MM_transformer/biovid_codes/temp_weights',f'{weight_name}{round(best_val_acc,2)}.pth')

                    torch.save(vis_phy_model.state_dict(), model_save_path_bb)
                    torch.save(mm_transformer.state_dict(), model_save_path_mmtransformer)  
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
    

"""
Testing
"""



if __name__ == '__main__':
    kfold_accuracy = []
    for i in range (1,6):
        train_annotation = f'train_fold{i}.txt'
        test_annotation = f'test_fold{i}.txt'
        concat_m_path = f'/projets2/AS84330/Projets/MM_transformer/biovid_codes/all_weights/best_weights_feat_concat5_fps5_folddiff/model_best_feature_concat_fold{i}.pth'

        weight_name = f'model_best_feat_concat_fusion_multi_mmtransformer_fold{i}_'
        best_accuracy = train(train_annotation,test_annotation,concat_m_path, weight_name)
        kfold_accuracy.append(round(best_accuracy,1))

    print('accuracy on fold 1', kfold_accuracy[0])
    print('accuracy on fold 2', kfold_accuracy[1])
    print('accuracy on fold 3', kfold_accuracy[2])
    print('accuracy on fold 4', kfold_accuracy[3])
    print('accuracy on fold 5', kfold_accuracy[4])
    print(statistics.mean(kfold_accuracy))

    # for i in range (1,6):
    #     test_annotation = f'test_fold{i}.txt'
    #     test_weights = f'model_best_vis_only_fold{i}.pth'
    #     test(test_annotation, test_weights)