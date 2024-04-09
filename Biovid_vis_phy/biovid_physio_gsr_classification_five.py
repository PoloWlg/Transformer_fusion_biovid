import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Union, Tuple, Any
import statistics
from validate import validate_physio_gsr_only
from models.models import Conv1D_model
from video_dataset_mm import GSRDataset


def train(train_annotation,test_annotation,weight_name):
    batch_size = 1024
    num_epochs = 100 
    lr = 0.0001
    num_classes = 5
    check_every = 1
    best_val_acc = 0

    biosignals_path = '/projets2/AS84330/Datasets/Biovid/PartA/physio/physio_organised'
    five_fold_annotations_path = '/projets2/AS84330/Datasets/Biovid/PartA/5folds_annotations_five/'
    train_annotation_file = os.path.join(five_fold_annotations_path, train_annotation)
    val_annotation_file = os.path.join(five_fold_annotations_path, test_annotation)


    train_dataset = GSRDataset(train_annotation_file, biosignals_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = GSRDataset(val_annotation_file, biosignals_path)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physio_model = Conv1D_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    physio_optimizer = optim.SGD(physio_model.parameters(), lr=lr, momentum=0.9)
    #scheduler = ReduceLROnPlateau(physio_optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        physio_model.train()
        
        
        running_loss = 0
        correct = 0
        total = 0
        
        for i,(physio_batch, labels) in enumerate(train_dataloader):
            physio_optimizer.zero_grad()
            physio_batch = physio_batch.reshape(physio_batch.shape[0],1,physio_batch.shape[1])
            physio_batch = physio_batch.to(device, dtype=torch.float)
            labels = labels.to(device)
            
            physio_outputs = physio_model(physio_batch)
            
            physio_loss = criterion(physio_outputs, labels)
            
            physio_loss.backward()
            physio_optimizer.step()
            # print(physio_loss.data)
            
            running_loss += physio_loss.item()
            
            _, physio_predicted = torch.max(physio_outputs.data, 1)
            total += labels.size(0)
            # print('output: ', physio_outputs)
            # print('predicted: ', physio_predicted)
            # print('labels: ', labels)
            # print('**************************')
            correct += (physio_predicted == labels).sum().item()
            #print(physio_loss.item())

        print(f"Accuracy after epoch {epoch + 1}: {100 * correct / total}%")

        if epoch % check_every == 0:
                val_acc, val_loss = validate_physio_gsr_only(physio_model, val_dataloader, criterion, device)
                # scheduler.step(val_loss)
                # print( "Validation accuracy: ", val_acc)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    
                    # # delete previous best model 
                    # for file in os.listdir('/projets2/AS84330/Projets/MM_transformer/biovid_codes/weights/'):
                    #     if weight_name in file:
                    #         os.remove(os.path.join('/projets2/AS84330/Projets/MM_transformer/biovid_codes/weights/', file))
                            
                    model_save_path = os.path.join('/projets2/AS84330/Projets/MM_transformer/biovid_codes/Biovid_vis_phy',f'{weight_name}{round(best_val_acc,2)}.pth')
                    torch.save(physio_model.state_dict(), model_save_path)
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

    batch_size = 1024
    num_classes = 2
    biosignals_path = '/projets2/AS84330/Datasets/Biovid/PartA/physio/physio_organised'
   
    videos_root = '/projets2/AS84330/Datasets/Biovid/PartA/subject_images/subject_images_organised'
    val_annotation_file = os.path.join(videos_root,'../../5folds_annotations', test_annotation)


    val_dataset = GSRDataset(val_annotation_file, biosignals_path)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physio_model = Conv1D_model(num_classes).to(device)
    
    physio_model.load_state_dict(torch.load('/projets2/AS84330/Projets/MM_transformer/biovid_codes/all_weights/weights_physio_viso2/' + test_weights))

    criterion = nn.CrossEntropyLoss()

    
    val_acc, _ = validate_physio_gsr_only(physio_model, val_dataloader, criterion, device)
    
    return val_acc

##### Train + Evaluate #####
if __name__ == '__main__':
    kfold_accuracy = []
    for i in range (2,6):
        train_annotation = f'train_fold{i}.txt'
        test_annotation = f'test_fold{i}.txt'
        weight_name = f'model_best_gsr_fold{i}_'
        best_accuracy = train(train_annotation,test_annotation,weight_name)
        kfold_accuracy.append(round(best_accuracy,1))

    print('accuracy on fold 1', kfold_accuracy[0])
    print('accuracy on fold 2', kfold_accuracy[1])
    print('accuracy on fold 3', kfold_accuracy[2])
    print('accuracy on fold 4', kfold_accuracy[3])
    print('accuracy on fold 5', kfold_accuracy[4])
    print('mean: ', statistics.mean(kfold_accuracy))
    

##### Test#####
# if __name__ == '__main__':
#     kfold_accuracy = []
#     for i in range (1,6):
#         test_annotation = f'test_fold{i}.txt'
#         weight_name = f'model_best_gsr_fold{i}.pth'
#         best_accuracy = test(test_annotation,weight_name)
#         kfold_accuracy.append(round(best_accuracy,1))

#     print('accuracy on fold 1', kfold_accuracy[0])
#     print('accuracy on fold 2', kfold_accuracy[1])
#     print('accuracy on fold 3', kfold_accuracy[2])
#     print('accuracy on fold 4', kfold_accuracy[3])
#     print('accuracy on fold 5', kfold_accuracy[4])
#     print('mean: ', statistics.mean(kfold_accuracy))
    
    