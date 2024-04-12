from video_dataset_mm import  VideoFrameDataset, ImglistToTensor
from comet_ml import Experiment
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from torchvision.models.video import r3d_18
from torchvision import models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

"""
Training settings

"""



def validate_vis_only(vis_model, val_dataloader, criterion, device):
    # Validation phase
    vis_model.eval() 
    val_correct = 0
    val_total = 0
    val_vis_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)

            val_vis_outputs = vis_model(val_inputs)

            val_vis_loss += criterion(val_vis_outputs, val_labels).item()

            val_outputs = val_vis_outputs

            _,val_predicted = torch.max(val_outputs.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_vis_loss)/2) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss

def validate_physio_gsr_only(physio_model, val_dataloader, criterion, device):
    # Validation phase
    physio_model.eval() 
    val_correct = 0
    val_total = 0
    val_vis_loss = 0.0
    val_physio_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            val_inputs, val_labels = val_data

            val_inputs = val_inputs.reshape(val_inputs.shape[0],1,val_inputs.shape[1])
            
            val_inputs = val_inputs.to(device, dtype=torch.float)
            val_labels = val_labels.to(device)
        


            val_physio_outputs = physio_model(val_inputs)
            # val_vis_outputs = vis_model(val_inputs)

            val_physio_loss += criterion(val_physio_outputs, val_labels)
            # val_vis_loss += criterion(val_vis_outputs, val_labels).item()

            # val_both_outputs = val_physio_outputs + val_vis_outputs

            _,val_predicted = torch.max(val_physio_outputs.data, 1)

            # _, val_both_predicted = torch.max(val_both_outputs.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_physio_loss)) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss

def validate(vis_phy_mod, val_dataloader, criterion, device):
    # Validation phase
    vis_phy_mod.vis_model.eval() 
    vis_phy_mod.phy_model.eval()
    val_correct = 0
    val_total = 0
    val_t_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)

            val_out = vis_phy_mod.feat_concat_att(val_inputs,spec_2d)


            # val_physio_loss = criterion(val_out, val_labels)
            val_t_loss += criterion(val_out, val_labels).item()


            _, val_predicted = torch.max(val_out.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_t_loss)/2) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss


def validate_feat_concat(vis_phy_mod, val_dataloader, criterion, device):
    # Validation phase
    vis_phy_mod.vis_model.eval() 
    vis_phy_mod.phy_model.eval()
    val_correct = 0
    val_total = 0
    val_t_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)

            val_out = vis_phy_mod.feat_concat(val_inputs,spec_2d)


            # val_physio_loss = criterion(val_out, val_labels)
            val_t_loss += criterion(val_out, val_labels).item()


            _, val_predicted = torch.max(val_out.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_t_loss)/2) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss


def validate_cam(vis_phy_mod,cam, val_dataloader, criterion, device):
    # Validation phase
    vis_phy_mod.vis_model.eval() 
    vis_phy_mod.phy_model.eval()
    cam.eval()
    val_correct = 0
    val_total = 0
    val_t_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)

            visfeats, phyfeats = vis_phy_mod.model_out_feats(val_inputs,spec_2d)
            visual_feats=visfeats.unsqueeze(1)
            physio_feats=phyfeats.unsqueeze(1)
            physiovisual_outs = cam(visual_feats, physio_feats)
            val_out=physiovisual_outs.squeeze(1)


            # val_physio_loss = criterion(val_out, val_labels)
            val_t_loss += criterion(val_out, val_labels).item()


            _, val_predicted = torch.max(val_out.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_t_loss)/2) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss


def validate_attn(vis_phy_mod,attention_m, val_dataloader, criterion, device):
    # Validation phase
    vis_phy_mod.vis_model.eval() 
    vis_phy_mod.phy_model.eval()
    attention_m.eval()
    # classifier_m.eval()
    val_correct = 0
    val_total = 0
    val_t_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)

            vis_feats, phy_feats = vis_phy_mod.model_out_feats(val_inputs,spec_2d)
            concatenated_features = torch.cat((vis_feats, phy_feats), dim=1)
            val_out = attention_m(concatenated_features)
            # val_out = classifier_m(attended_features)
            # val_physio_loss = criterion(val_out, val_labels)
            val_t_loss += criterion(val_out, val_labels).item()


            _, val_predicted = torch.max(val_out.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_t_loss)/2) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss


def validate_mmtransformer(vis_phy_mod,mm_transformer, val_dataloader, criterion, device):
    # Validation phase
    vis_phy_mod.eval() 
    mm_transformer.eval()
    # classifier_m.eval()
    val_correct = 0
    val_total = 0
    val_t_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)

            vis_feats, phy_feats = vis_phy_mod.model_out_feats(val_inputs,spec_2d)
            vis_feats = vis_feats.unsqueeze(1)
            phy_feats = phy_feats.unsqueeze(1)
            val_out = mm_transformer(vis_feats, phy_feats)
            # val_out = classifier_m(attended_features)
            # val_physio_loss = criterion(val_out, val_labels)
            val_t_loss += criterion(val_out, val_labels).item()


            _, val_predicted = torch.max(val_out.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_t_loss)/2) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss

def validate_mmtransformer_per_seq(vis_phy_mod,mm_transformer, val_dataloader, criterion, device):
    # Validation phase
    vis_phy_mod.eval() 
    mm_transformer.eval()
    # classifier_m.eval()
    val_correct = 0
    val_total = 0
    val_t_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,video_batch, val_labels, record_id = val_data
            video_batch = video_batch.to(device)
            val_labels = val_labels.to(device)
            video_batch = video_batch.permute(0, 2, 1, 3, 4)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)

          # Sequence of 20 frames unflattened to 4x5
            unflatten = torch.nn.Unflatten(2, (4,5))
            video_batch = unflatten(video_batch)
            video_batch=video_batch.permute(2, 0, 1, 3, 4, 5)
            
            phy_feats_seqs = []                   
            vis_feats_seqs = []
            

            
            for count,video_batch_sequence in enumerate(video_batch):
                vis_feats, phy_feats = vis_phy_mod.model_out_feats(video_batch_sequence,spec_2d)
                phy_feats_seqs.append(phy_feats.to(device))
                vis_feats_seqs.append(vis_feats.to(device))
            
            phy_feats_seqs = torch.stack(phy_feats_seqs)
            vis_feats_seqs = torch.stack(vis_feats_seqs)
            phy_feats_seqs.to(device)
            vis_feats_seqs.to(device)
            
            phy_feats_seqs = phy_feats_seqs.permute(1,0,2)
            vis_feats_seqs = vis_feats_seqs.permute(1,0,2)    
            
            val_out = mm_transformer(vis_feats_seqs, phy_feats_seqs)
            # val_out = classifier_m(attended_features)
            # val_physio_loss = criterion(val_out, val_labels)
            val_t_loss += criterion(val_out, val_labels).item()


            _, val_predicted = torch.max(val_out.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_t_loss)/2) / len(val_dataloader)
    return val_accuracy, avg_val_loss


from collections import OrderedDict
def load_model(concat_m_path, name):
    torch_load = torch.load(concat_m_path)
    model_keys = [key for key in torch_load.keys() if key.startswith(f'{name}.')]
    model_checkpoint = OrderedDict((key.replace(f'{name}.', ''), torch_load[key]) for key in model_keys)
    return model_checkpoint

def validate_mmtransformer_with_weights(vis_phy_mod,mm_transformer, val_dataloader, criterion, device,concat_m_path, transformer_path):
    # Validation phase
    
    vis_model_checkpoint = load_model(concat_m_path, 'vis_model')
    vis_phy_mod.vis_model.load_state_dict(vis_model_checkpoint)
    vis_phy_mod.vis_model.eval()

    phy_model_checkpoint = load_model(concat_m_path, 'phy_model')
    vis_phy_mod.phy_model.load_state_dict(phy_model_checkpoint)
    vis_phy_mod.phy_model.eval()
    
    transformer_load = torch.load(transformer_path)
    mm_transformer.load_state_dict(transformer_load)
    mm_transformer.eval()
    # classifier_m.eval()
    val_correct = 0
    val_total = 0
    val_t_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)

            vis_feats, phy_feats = vis_phy_mod.model_out_feats(val_inputs,spec_2d)
            vis_feats = vis_feats.unsqueeze(1)
            phy_feats = phy_feats.unsqueeze(1)
            val_out = mm_transformer(vis_feats, phy_feats)
            # val_out = classifier_m(attended_features)
            # val_physio_loss = criterion(val_out, val_labels)
            val_t_loss += criterion(val_out, val_labels).item()


            _, val_predicted = torch.max(val_out.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_t_loss)/2) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss