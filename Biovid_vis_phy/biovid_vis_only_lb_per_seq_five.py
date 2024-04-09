import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from video_dataset_mm import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms
from comet_ml import Experiment
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
from validate import validate_vis_only
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
from models.models import Visual_model 
warnings.filterwarnings("ignore")

# Set a seed for PyTorch
# seed = 42
# torch.manual_seed(seed)
# random.seed(seed)



preprocess_train = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(112),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(112),  # image batch, center crop to square 299x299 Remove potentially later 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

preprocess_val = transforms.Compose([
    ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    transforms.Resize(112),  # image batch, resize smaller edge to 299
    transforms.CenterCrop(112),  # image batch, center crop to square 299x299 Remove potentially later 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def train(train_annotation,test_annotation,weight_name):

    """
    Training settings

    """
    num_epochs = 50
    best_epoch = 0
    check_every = 1
    b_size = 64
    lr=0.001
    
    experiment = Experiment(
    api_key="17GX9zixZo9vF8cTq9qhF2nch",
    project_name="biovid",
    disabled=True)

    parameters = {'batch_size': b_size,
                'learning_rate': lr,
                'epochs':num_epochs            
                }
    experiment.log_parameters(parameters)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_acc=0


    visual_model = Visual_model(num_classes=5)
    visual_model = visual_model.to(device)


    criterion = nn.CrossEntropyLoss()
    vis_optimizer = optim.SGD(visual_model.parameters(), lr=lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(vis_optimizer, mode='max', factor=0.1, patience=10, verbose=True)


    videos_root = '/projets2/AS84330/Datasets/Biovid/PartA/subject_images/subject_images_organised'
    
    five_fold_annotations_path = '/projets2/AS84330/Datasets/Biovid/PartA/5folds_annotations_five/'
    train_annotation_file = os.path.join(five_fold_annotations_path, train_annotation)
    val_annotation_file = os.path.join(five_fold_annotations_path, test_annotation)

    train_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=train_annotation_file,
        num_segments=1,
        frames_per_segment=5,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess_train,
        test_mode=False)
    
    val_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=val_annotation_file,
        num_segments=1,
        frames_per_segment=5,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess_val,
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
    

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        with experiment.train():
            visual_model.train()

            running_loss = 0.0
            correct = 0
            total = 0
            for i,(spec_2d,video_batch, labels) in enumerate(train_dataloader,0):
                
                vis_optimizer.zero_grad()

                
                video_batch=video_batch.permute(0, 2, 1, 3, 4)
                
                video_batch = video_batch.to(device)
                labels = labels.to(device)
                # spec_2d = spec_2d.to(device= device, dtype=torch.float)
                outputs = visual_model(video_batch)
                vis_loss = criterion(outputs, labels)
                
                vis_loss.backward()
                vis_optimizer.step()

                running_loss += vis_loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            
            train_acc = 100 * correct / total
            print("*********************************************\n")
            print(f"Accuracy after epoch {epoch + 1}: {train_acc}%")
            if epoch % check_every == 0:
                val_acc, val_loss = validate_vis_only(visual_model, val_dataloader, criterion, device)
                scheduler.step(val_acc)
                experiment.log_metric('Val loss', val_loss,epoch= epoch)
                experiment.log_metric('Val accuracy', val_acc ,epoch= epoch)
                
                experiment.log_metric('Train accuracy', train_acc ,epoch= epoch)
                
                current_lr = vis_optimizer.param_groups[0]['lr']
                experiment.log_metric('Learning Rate', current_lr,epoch= epoch)
                print('Current learning rate: ', current_lr)
                
                # print( "Validation accuracy: ", val_acc)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model_save_path = os.path.join('/projets2/AS84330/Projets/MM_transformer/biovid_codes/Biovid_vis_phy',f'{weight_name}{round(best_val_acc,2)}.pth')
                    torch.save(visual_model.state_dict(), model_save_path)
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

def test(test_annotation, test_weights):

    b_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    visual_model = Visual_model(num_classes=2)
    visual_model_checkpoint_path = f'/projets2/AS84330/Projets/MM_transformer/biovid_codes/all_weights/weights_physio_viso2/{test_weights}'
    visual_model.load_state_dict(torch.load(visual_model_checkpoint_path))

    visual_model = visual_model.to(device)
    visual_model.eval()

    videos_root = '/projets2/AS84330/Datasets/Biovid/PartA/subject_images/subject_images_organised'
    five_fold_annotations_path = '/projets2/AS84330/Projets/MM_transformer/biovid_codes/5fold_annotations'
    
    val_annotation_file = os.path.join(five_fold_annotations_path, test_annotation)

    
    val_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=val_annotation_file,
        num_segments=10,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess_val,
        test_mode=True)

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False)
    
    val_acc, val_loss = validate_vis_only(visual_model, val_dataloader, criterion, device)
    
    return val_acc

    
##### Train + Evaluate #####
if __name__ == '__main__':
    kfold_accuracy = []
    for i in range (2,6):
        train_annotation = f'train_fold{i}.txt'
        test_annotation = f'test_fold{i}.txt'
        weight_name = f'model_best_vis_only_fold{i}_'
        best_accuracy = train(train_annotation,test_annotation,weight_name)
        kfold_accuracy.append(round(best_accuracy,1))

    print('accuracy on fold 1', kfold_accuracy[0])
    print('accuracy on fold 2', kfold_accuracy[1])
    print('accuracy on fold 3', kfold_accuracy[2])
    print('accuracy on fold 4', kfold_accuracy[3])
    print('accuracy on fold 5', kfold_accuracy[4])
    print(statistics.mean(kfold_accuracy))



##### Test #####
# if __name__ == '__main__':
#     kfold_accuracy = []
#     for i in range (1,6):
#         test_annotation = f'test_fold{i}.txt'
#         weight_name = f'model_best_vis_only_fold{i}.pth'
#         best_accuracy = test(test_annotation,weight_name)
#         kfold_accuracy.append(round(best_accuracy,1))

#     print('accuracy on fold 1', kfold_accuracy[0])
#     print('accuracy on fold 2', kfold_accuracy[1])
#     print('accuracy on fold 3', kfold_accuracy[2])
#     print('accuracy on fold 4', kfold_accuracy[3])
#     print('accuracy on fold 5', kfold_accuracy[4])
#     print(statistics.mean(kfold_accuracy))



