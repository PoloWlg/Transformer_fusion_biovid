import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from video_dataset_mm import  VideoFrameDataset, ImglistToTensor
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
from models.models2 import VIS_PHY_MODEL  
from torch.optim.lr_scheduler import ReduceLROnPlateau







def validate(vis_phy_model, val_dataloader, criterion, device):
    # Validation phase
    vis_phy_model.eval() 
    val_correct = 0
    val_total = 0
    val_vis_loss = 0.0
    val_physio_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)

            val_physio_outputs, val_vis_outputs = vis_phy_model.model_out_feats(val_inputs, spec_2d)

            val_physio_loss = criterion(val_physio_outputs, val_labels)
            val_vis_loss += criterion(val_vis_outputs, val_labels).item()

            val_both_outputs = val_physio_outputs + val_vis_outputs

            # _,val_predicted = torch.max(val_physio_outputs.data, 1)

            _, val_both_predicted = torch.max(val_both_outputs.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_both_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_physio_loss+val_vis_loss)/2) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy



# Define a custom collate function
def custom_collate_fn(batch):
    # Assuming batch is a list of (image, label) tuples
    images, labels = zip(*batch)
    # Convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])
    images = [transform(image) for image in images]
    return torch.stack(images), torch.tensor(labels)








# batch_size = 2  # Adjust as needed
num_frames = 5  # Number of frames in each video clip
num_channels = 3  # Number of channels (e.g., RGB)
video_length = 112  # Length of the video in each dimension
num_classes = 2  # Number of classes

# dummy_data = torch.randn(batch_size, num_frames, num_channels, video_length, video_length)  # Example shape


"""
Model definition 
Visual model: R3D-18
Physiological model: Resnet 18 layer MLP

"""





def train(train_annotation,test_annotation,v_m_path,phy_m_path, weight_name):

    num_epochs = 100
    best_epoch = 0
    check_every = 1
    b_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_acc=0


    
    
    vis_phy_model=VIS_PHY_MODEL(v_m_path,phy_m_path).to(device=device)
    vis_phy_model.vis_model.fc = nn.Linear(512, 2).to(device=device)
    vis_phy_model.phy_model.fc2 = nn.Linear(512, 2).to(device=device)

    criterion = nn.CrossEntropyLoss()
    vis_optimizer = optim.SGD(vis_phy_model.vis_model.parameters(), lr=0.00005, momentum=0.9)
    physio_optimizer = optim.SGD(vis_phy_model.phy_model.parameters(), lr=0.0005, momentum=0.9)

    scheduler_vis = ReduceLROnPlateau(vis_optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    scheduler_phy = ReduceLROnPlateau(physio_optimizer, mode='max', factor=0.1, patience=10, verbose=True)

    videos_root = '/projets2/AS84330/Datasets/Biovid/PartA/subject_images/subject_images_organised'
    # videos_root = '/home/livia/work/Biovid/PartB/Video-Dataset-Loading-Pytorch-main/demo_dataset'
    train_annotation_file = os.path.join(videos_root,'../../5folds_annotations', train_annotation)
    val_annotation_file = os.path.join(videos_root,'../../5folds_annotations', test_annotation)




    """ DEMO 3 WITH TRANSFORMS """
    # As of torchvision 0.8.0, torchvision transforms support batches of images
    # of size (BATCH x CHANNELS x HEIGHT x WIDTH) and apply deterministic or random
    # transformations on the batch identically on all images of the batch. Any torchvision
    # transform for image augmentation can thus also be used  for video augmentation.
    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(112),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(112),  # image batch, center crop to square 299x299
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=train_annotation_file,
        num_segments=10,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        test_mode=False)
    
    val_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=val_annotation_file,
        num_segments=10,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        test_mode=True)


    def denormalize(video_tensor):
        """
        Undoes mean/standard deviation normalization, zero to one scaling,
        and channel rearrangement for a batch of images.
        args:
            video_tensor: a (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        """
        inverse_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        return (inverse_normalize(video_tensor) * 255.).type(torch.uint8).permute(0, 2, 3, 1).numpy()


    # frame_tensor = denormalize(frame_tensor)
    # plot_video(rows=1, cols=5, frame_list=frame_tensor, plot_width=15., plot_height=3.,
    #            title='Evenly Sampled Frames, + Video Transform')



    """ DEMO 3 CONTINUED: DATALOADER """
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
        vis_phy_model.vis_model.train()
        vis_phy_model.phy_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i,(spec_2d,video_batch, labels) in enumerate(train_dataloader,0):
            
            vis_optimizer.zero_grad()
            physio_optimizer.zero_grad()
            
            video_batch=video_batch.permute(0, 2, 1, 3, 4)
            
            video_batch = video_batch.to(device)
            labels = labels.to(device)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)


            vis_outputs, physio_outputs = vis_phy_model.model_out_feats(video_batch, spec_2d)
            vis_loss = criterion(vis_outputs, labels)
            
            physio_loss = criterion(physio_outputs, labels)

            vis_loss.backward()
            vis_optimizer.step()

            physio_loss.backward()
            physio_optimizer.step()

            
            running_loss += vis_loss.item()
            running_loss += physio_loss.item()
            
            both_outputs = vis_outputs + physio_outputs
            _, both_predicted = torch.max(both_outputs.data, 1)
            # _, physio_predicted = torch.max(physio_outputs.data, 1)
            total += labels.size(0)
            correct += (both_predicted == labels).sum().item()


            # _, predicted = torch.max(vis_outputs.data, 1)
            
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

            
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
        print("*********************************************\n")
        print(f"Accuracy after epoch {epoch + 1}: {100 * correct / total}%")
        if epoch % check_every == 0:
            val_acc = validate(vis_phy_model, val_dataloader, criterion, device)
            scheduler_vis.step(val_acc)
            scheduler_phy.step(val_acc)
            # print( "Validation accuracy: ", val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_save_path = os.path.join('/projets2/AS84330/Projets/MM_transformer/biovid_codes/Biovid_vis_phy',f'{weight_name}{round(best_val_acc,2)}.pth')
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



if __name__ == '__main__':
    kfold_accuracy = []
    for i in range (1,6):
        train_annotation = f'train_fold{i}.txt'
        test_annotation = f'test_fold{i}.txt'
        v_m_path = f'/projets2/AS84330/Projets/MM_transformer/biovid_codes/weights/model_best_vis_only_fold{i}.pth'
        phy_m_path = f'/projets2/AS84330/Projets/MM_transformer/biovid_codes/weights/model_best_gsr_fold{i}.pth'

        weight_name = f'weights_fusion/model_best_decision_fusion_fold{i}_'
        best_accuracy = train(train_annotation,test_annotation,v_m_path,phy_m_path, weight_name)
        kfold_accuracy.append(round(best_accuracy,1))

    print('accuracy on fold 1', kfold_accuracy[0])
    print('accuracy on fold 2', kfold_accuracy[1])
    print('accuracy on fold 3', kfold_accuracy[2])
    print('accuracy on fold 4', kfold_accuracy[3])
    print('accuracy on fold 5', kfold_accuracy[4])

    # for i in range (1,6):
    #     test_annotation = f'test_fold{i}.txt'
    #     test_weights = f'model_best_vis_only_fold{i}.pth'
    #     test(test_annotation, test_weights)
    


