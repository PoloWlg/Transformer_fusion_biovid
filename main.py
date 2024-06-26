import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from Biovid_vis_phy.video_dataset_mm import  VideoFrameDataset, ImglistToTensor
from comet_ml import Experiment
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import statistics

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
from torchvision import models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Biovid_vis_phy.validate import validate_mmtransformer_per_seq, validate_feat_concat
from Biovid_vis_phy.models.models_seq import  VIS_PHY_MODEL, Multi_cross_attention,Two_cross_attention, Concat_plus_fc
from parseit import get_args
from Biovid_vis_phy.models.models import VIS_PHY_MODEL_CONCAT   


from tools import fmsg


from dllogger import ArbJSONStreamBackend
from dllogger import Verbosity
from dllogger import ArbStdOutBackend
from dllogger import ArbTextStreamBackend
import dllogger as DLLogger

import datetime as dt


def train(train_annotation,test_annotation,concat_m_path, weight_name_visphy, weight_name_fusion):
    args = get_args()

    # Init logging 
    log_backends = [
            ArbJSONStreamBackend(Verbosity.VERBOSE,
                                 os.path.join(args['outd'], "log.json")),
            ArbTextStreamBackend(Verbosity.VERBOSE,
                                 os.path.join(args['outd'], "log.txt")),
        ]

    # if args['verbose']:
    #     log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))

    DLLogger.init_arb(backends=log_backends, master_pid=os.getpid())
    args['t0'] = dt.datetime.now()
    DLLogger.log(fmsg("Start time: {}".format(args['t0'])))
    DLLogger.flush()

    # '/projets2/AS84330/Datasets/Biovid/PartA/subject_images/subject_images_organised'
    videos_root = '/projets2/AS84330/Datasets/Biovid/PartA'
    train_annotation_file = os.path.join(videos_root, '5folds_annotations2', train_annotation)
    val_annotation_file = os.path.join(videos_root, '5folds_annotations2', test_annotation)
    

    if (args['SEED'] == True):
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
    
    """
        Training settings
    """
    num_epochs = args['num_epochs']
    b_size = args['batch_size']
    lr_vis_phy = args['lr_backbones']
    lr_mmtransformer = args['lr_fusion']

    best_epoch = 0
    check_every = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

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


    # num_frames = 5  # Number of frames in each video clip
    # num_channels = 3  # Number of channels (e.g., RGB)
    # video_length = 112  # Length of the video in each dimension
    # num_classes = 2  # Number of classes


    criterion = nn.CrossEntropyLoss()

    vis_phy_model=VIS_PHY_MODEL(concat_m_path).to(device=device)

    fusion_model = None
    if (args['fusion_technique'] == 'Multi_cross_attention'):
        fusion_model = Multi_cross_attention(visual_dim=512, physiological_dim=512, num_heads=1, hidden_dim=512, num_layers=1, num_classes=2)
    elif (args['fusion_technique'] == 'Two_cross_attention'):
        fusion_model = Two_cross_attention(visual_dim=512, physiological_dim=512, num_heads=1, hidden_dim=512, num_layers=1, num_classes=2)
    elif (args['fusion_technique'] == 'Feature_concat'):
        fusion_model=Concat_plus_fc()
    else:
        raise NotImplementedError(args['fusion_technique'])

    fusion_model = fusion_model.to(device=device)


    vis_phy_optimizer = optim.Adam(vis_phy_model.parameters(), lr=lr_vis_phy)
    fusion_model_optimizer = optim.Adam(fusion_model.parameters(), lr=lr_mmtransformer)
    # vis_phy_model=VIS_PHY_MODEL_CONCAT(v_m_path,phy_m_path,2).to(device=device)

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
        num_segments=1,
        frames_per_segment=4,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        test_mode=False)

    val_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=val_annotation_file,
        num_segments=1,
        frames_per_segment=4,
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


    DLLogger.log(fmsg("Start Trainging"))
    best_val_acc=0
    with experiment.train():
        for epoch in range(num_epochs):
            DLLogger.log(f'Epoch: {epoch}/{num_epochs-1}')
            vis_phy_model.train()
            fusion_model.train()

            running_loss = 0.0
            correct = 0
            total = 0
            for i,(spec_2d,video_batch, labels, record_images) in tqdm(enumerate(train_dataloader,0),total=len(train_dataloader), desc=f'Training'):                
                
                fusion_model_optimizer.zero_grad()
                vis_phy_optimizer.zero_grad()
                
                video_batch = video_batch.to(device)
                labels = labels.to(device)

                video_batch=video_batch.permute(0, 2, 1, 3, 4)
                if (args['fusion_technique'] == 'Feature_concat'):
                    vis_feats, phy_feats = vis_phy_model.model_out_feats(video_batch,spec_2d)
                    outs = fusion_model(vis_feats, phy_feats)
                else :
                    # Other fusion techniques
                    # Sequence of 20 frames unflattened to 4x5
                    unflatten = torch.nn.Unflatten(2, (4,5))
                    video_batch = unflatten(video_batch)
                    video_batch=video_batch.permute(2, 0, 1, 3, 4, 5)
                    
                    phy_feats_seqs = []                   
                    vis_feats_seqs = []
                    

                    
                    for video_batch_sequence in video_batch:
                        vis_feats, phy_feats = vis_phy_model.model_out_feats(video_batch_sequence,spec_2d)
                        phy_feats_seqs.append(phy_feats.to(device))
                        vis_feats_seqs.append(vis_feats.to(device))
                    
                    phy_feats_seqs = torch.stack(phy_feats_seqs)
                    vis_feats_seqs = torch.stack(vis_feats_seqs)
                    phy_feats_seqs.to(device)
                    vis_feats_seqs.to(device)
                    
                    phy_feats_seqs = phy_feats_seqs.permute(1,0,2)
                    vis_feats_seqs = vis_feats_seqs.permute(1,0,2)
                    
                    outs = fusion_model(vis_feats_seqs, phy_feats_seqs)

                t_loss = criterion(outs, labels)

                t_loss.backward()
                fusion_model_optimizer.step()
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
            DLLogger.log(f'Training accuracy: {train_accuracy}%')
            train_loss= running_loss / 100
            experiment.log_metric('Loss', train_loss,epoch= epoch)
            experiment.log_metric('Accuracy', train_accuracy ,epoch= epoch)
            if epoch % check_every == 0:
                val_acc, val_loss = validate_feat_concat(vis_phy_model,fusion_model, val_dataloader, criterion, device)
                print(f'Validation accuracy: {val_acc}%')
                print(f'Validation loss: {val_loss}')
                
                
                experiment.log_metric('Val Accuracy', val_acc,epoch= epoch)
                experiment.log_metric('Val Loss', val_loss,epoch= epoch)
                current_lr = fusion_model_optimizer.param_groups[0]['lr']
                experiment.log_metric('Learning Rate', current_lr,epoch= epoch)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model_save_path_visphy = os.path.join(args['outd'],'weights', f'{weight_name_visphy}.pth')
                    model_save_path_fusion = os.path.join(args['outd'],'weights', f'{weight_name_fusion}.pth')

                    torch.save(vis_phy_model.state_dict(), model_save_path_visphy)
                    torch.save(fusion_model.state_dict(), model_save_path_fusion)  
                    print('Best model saved at epoch: ', epoch+1)
                    best_epoch = epoch
                DLLogger.log(f'Validation @EPOCH {epoch}: '
                             f'accuracy: {round(val_acc,2)}% | ' 
                             f'[BEST: {round(best_val_acc,2)}% '
                             f'@EPOCH: {best_epoch}] \n ')  
                DLLogger.flush()

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
        visphy_path = f'/home/ens/AS84330/transformer_biovid_code/pretrained_weights/best_weights_feat_concat/model_best_feature_concat_fold{i}.pth'

        weight_name_visphy = f'visphy_model_fold{i}_'
        weight_name_fusion = f'fusion_model_fold{i}_'
        best_accuracy = train(train_annotation,test_annotation,visphy_path, weight_name_visphy,weight_name_fusion)
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