# from video_dataset_mm import  VideoFrameDataset, ImglistToTensor
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
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # Number of classes

class Visual_model(nn.Module):
    def __init__(self, num_classes=2):
        super(Visual_model, self).__init__()
        self.visual_model = r3d_18(pretrained=True, progress=True)
        self.visual_model.stem[0] = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.visual_model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.visual_model(x)
        return x
    
class Visual_model2(nn.Module):
    def __init__(self, num_classes=2):
        super(Visual_model2, self).__init__()
        self.visual_model = r3d_18(pretrained=True, progress=True)
        self.visual_model.stem[0] = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.visual_model.fc = nn.Linear(512, 512)
        self.out_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = self.visual_model(x)
        x = self.out_layers(x)
        return x

class Conv1D_model(nn.Module):
    def __init__(self, num_classes=2):
        super(Conv1D_model, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(22336, 512)  
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor to 1D
        x = self.fc1(x)
        
        x = self.fc2(x)
        #x = self.sigmoid(x)
        return x
    
def load_model(concat_m_path, name):
    torch_load = torch.load(concat_m_path)
    model_keys = [key for key in torch_load.keys() if key.startswith(f'{name}.')]
    model_checkpoint = OrderedDict((key.replace(f'{name}.', ''), torch_load[key]) for key in model_keys)
    return model_checkpoint


class VIS_PHY_MODEL_CONCAT(nn.Module):
    
    def __init__(self,v_m_path, phy_m_path, num_classes):
        super(VIS_PHY_MODEL_CONCAT,self).__init__()
        
        visual_model = Visual_model2(num_classes)
        visual_model.out_layers= nn.Sequential()
        
        v_m=torch.load(v_m_path)
        del v_m['out_layers.0.weight']
        del v_m['out_layers.0.bias']
        del v_m['out_layers.2.weight']
        del v_m['out_layers.2.bias']
        del v_m['out_layers.4.weight']
        del v_m['out_layers.4.bias']
        # del v_m['out_layers']

        visual_model.load_state_dict(v_m)
        visual_model.eval()
        
        physio_model=Conv1D_model(num_classes)
        physio_model.load_state_dict(torch.load(phy_m_path))
        physio_model.eval()

        physio_model.fc2 = nn.Sequential()
        
        #Freeze the visual_model
        for param in visual_model.parameters():
            param.requires_grad = False
            
        for param in physio_model.parameters():
            param.requires_grad = True


        self.out_layer1=nn.Linear(1024,512)
        self.out_layer2=nn.Linear(512,256)
        self.out_layer3=nn.Linear(256,64)  
        self.out_layer4=nn.Linear(64,num_classes)  
        
        self.relu = nn.ReLU() 
        self.layer_norm = nn.LayerNorm(512)

        self.vis_model=visual_model
        self.phy_model=physio_model
        
        physio_model = physio_model.to(device)
        visual_model = visual_model.to(device)

    def feat_concat(self,video_batch,specs_2d):
        output=[]
        video_batch = video_batch.to(device)

        specs_2d = specs_2d.reshape(specs_2d.shape[0],1,specs_2d.shape[1])
        specs_2d = specs_2d.to(device= device, dtype=torch.float)

        vis_out=self.vis_model(video_batch)
        phy_out=self.phy_model(specs_2d)
        # normalise the data before concatenating
        vis_out = vis_out / vis_out.norm(dim=1, keepdim=True)
        phy_out = phy_out / phy_out.norm(dim=1, keepdim=True)
        
        # make layer normalisation layer
        # vis_out = self.layer_norm(vis_out)
        # phy_out = self.layer_norm(phy_out)

        #contatenate vis_out and phy_out
        combined_out=torch.cat((vis_out,phy_out),1)
        
        output = self.out_layer1(combined_out)
        output = self.relu(output)
        output = self.out_layer2(output)
        output = self.relu(output)
        output = self.out_layer3(output)
        output = self.relu(output)
        output = self.out_layer4(output)
        return output
    
    def model_out_feats(self,video_batch,specs_2d):

        video_batch = video_batch.to(device)

        specs_2d = specs_2d.reshape(specs_2d.shape[0],1,specs_2d.shape[1])
        specs_2d = specs_2d.to(device= device, dtype=torch.float)

        vis_out=self.vis_model(video_batch)
        phy_out=self.phy_model(specs_2d)

        vis_out = vis_out / vis_out.norm(dim=1, keepdim=True)
        phy_out = phy_out / phy_out.norm(dim=1, keepdim=True)

        return vis_out,phy_out


class Concat_plus_fc(nn.Module):
    
    def __init__(self):
        super(VIS_PHY_MODEL_CONCAT,self).__init__()
        self.out_layer1=nn.Linear(1024,512)
        self.out_layer2=nn.Linear(512,256)
        self.out_layer3=nn.Linear(256,64)  
        self.out_layer4=nn.Linear(64,num_classes)  
        
        self.relu = nn.ReLU() 
        
    def forward(self, vis_feats, phy_feats):
        output = torch.cat((vis_feats,phy_feats),1)
        
        output = self.out_layer1(output)
        output = self.relu(output)
        output = self.out_layer2(output)
        output = self.relu(output)
        output = self.out_layer3(output)
        output = self.relu(output)
        output = self.out_layer4(output)
        
        return output
        



class VIS_PHY_MODEL(nn.Module):
    def __init__(self,concat_m_path):
        super(VIS_PHY_MODEL,self).__init__()
        
        visual_model = Visual_model(num_classes=2)
        visual_model.visual_model.fc= nn.Sequential()
        
        physio_model=Conv1D_model(num_classes=2)
        physio_model.fc2 = nn.Sequential()
        
        #Freeze the visual_model
        for param in visual_model.parameters():
            param.requires_grad = False
            
        for param in physio_model.parameters():
            param.requires_grad = True

        self.out_layer1=nn.Linear(1024,512)
        self.out_layer2=nn.Linear(512,256)
        self.out_layer3=nn.Linear(256,64)  
        self.out_layer4=nn.Linear(64,num_classes)   
        
        # Loading weights 
        vis_model_checkpoint = load_model(concat_m_path, 'vis_model')
        visual_model.load_state_dict(vis_model_checkpoint)
        visual_model.eval()

        phy_model_checkpoint = load_model(concat_m_path, 'phy_model')
        physio_model.load_state_dict(phy_model_checkpoint)
        physio_model.eval()

        self.vis_model=visual_model
        self.phy_model=physio_model
        
        physio_model = physio_model.to(device)
        visual_model = visual_model.to(device)


        self.W = nn.Linear(1024, 1024)
        self.V = nn.Linear(1024, 1024, bias=False)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU() 
        self.layer_norm = nn.LayerNorm(1024)
        



    def feat_concat_att(self,video_batch,specs_2d):
        output=[]
        # self.vis_model.zero_grad()
        # self.vis_optim.zero_grad()
        # self.phy_optim.zero_grad()
        video_batch = video_batch.to(device)

        specs_2d = specs_2d.reshape(specs_2d.shape[0],1,specs_2d.shape[1])
        specs_2d = specs_2d.to(device= device, dtype=torch.float)

        vis_out=self.vis_model(video_batch)
        phy_out=self.phy_model(specs_2d)

        # normalise the data before concatenating
        vis_out = vis_out / vis_out.norm(dim=1, keepdim=True)
        phy_out = phy_out / phy_out.norm(dim=1, keepdim=True)

        #contatenate vis_out and phy_out
        combined_out=torch.cat((vis_out,phy_out),1)
        
        # q = self.W(combined_out)
        attn_weights = torch.softmax(self.V(self.tanh(combined_out)), dim=1)
        attended_x = attn_weights * combined_out
        out = self.layer_norm(attended_x + combined_out)
        out = self.out_layer1(attended_x)
        out = self.relu(out)
        out = self.out_layer2(out)
        out = self.relu(out)
        out = self.out_layer3(out)
        out = self.relu(out)
        out = self.out_layer4(out)
        return out
    
    def feat_concat(self,video_batch,specs_2d):
        output=[]
        # self.vis_model.zero_grad()
        # self.vis_optim.zero_grad()
        # self.phy_optim.zero_grad()
        video_batch = video_batch.to(device)

        specs_2d = specs_2d.reshape(specs_2d.shape[0],1,specs_2d.shape[1])
        specs_2d = specs_2d.to(device= device, dtype=torch.float)

        vis_out=self.vis_model(video_batch)
        phy_out=self.phy_model(specs_2d)

        # normalise the data before concatenating
        vis_out = vis_out / vis_out.norm(dim=1, keepdim=True)
        phy_out = phy_out / phy_out.norm(dim=1, keepdim=True)
        #contatenate vis_out and phy_out
        combined_out=torch.cat((vis_out,phy_out),1)
        
        out = self.out_layer1(combined_out)
        out = self.out_layer2(out)
        out = self.out_layer3(out)
        return out
    
    def model_out_feats(self,video_batch,specs_2d):
        output=[]
        # self.vis_model.zero_grad()
        # self.vis_optim.zero_grad()
        # self.phy_optim.zero_grad()
        video_batch = video_batch.to(device)

        specs_2d = specs_2d.reshape(specs_2d.shape[0],1,specs_2d.shape[1])
        specs_2d = specs_2d.to(device= device, dtype=torch.float)

        vis_out=self.vis_model(video_batch)
        phy_out=self.phy_model(specs_2d)

        return vis_out,phy_out

    
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.V = nn.Linear(input_dim, input_dim, bias=False)
        self.tanh = nn.Tanh()
        self.fc=nn.Linear(input_dim,2)

        self.out_layer1=nn.Linear(512,256)
        self.out_layer2=nn.Linear(256,64)  
        self.out_layer3=nn.Linear(64,2)   

    def forward(self, x):
        q = self.W(x)
        attn_weights = torch.softmax(self.V(self.tanh(q)), dim=1)
        attended_x = attn_weights * x
        out = self.out_layer1(attended_x)
        out = self.out_layer2(out)
        out = self.out_layer3(out)
        return out
    

class SequentialEncoder(nn.Sequential):
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoderBlock, self).__init__()
        self.layers = SequentialEncoder(*[TransformerEncoderLayer(input_dim, num_heads, hidden_dim)
                                      for _ in range(num_layers)])
    
    def forward(self, x):
        x = self.layers(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)    
        x = x + attn_output
        x = self.layer_norm1(x)
        
        # Apply feed forward network
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        return x

class MultimodalTransformer(nn.Module):
    def __init__(self, visual_dim, physiological_dim, num_heads, hidden_dim, num_layers, num_classes):
        super(MultimodalTransformer, self).__init__()
        self.visual_encoder = TransformerEncoderBlock(visual_dim, num_heads, hidden_dim, num_layers)
        self.physiological_encoder = TransformerEncoderBlock(physiological_dim, num_heads, hidden_dim, num_layers)
        self.cross_attention_v = nn.MultiheadAttention(visual_dim, num_heads)
        self.cross_attention_p = nn.MultiheadAttention(physiological_dim, num_heads)
        self.gated_attention = nn.Linear(visual_dim + physiological_dim, 1)
        self.fc = nn.Linear(visual_dim + physiological_dim, num_classes)
        
        self.out_layer1=nn.Linear(1024,512)
        self.out_layer2=nn.Linear(512,256)
        self.out_layer3=nn.Linear(256,64)  
        self.out_layer4=nn.Linear(64,num_classes)  
        self.relu = nn.ReLU() 
        
        self.layer_norm = nn.LayerNorm(1024)
        
        
    
    def forward(self, visual_features, physiological_features):
        
        # concat_features = torch.cat((visual_features, physiological_features), dim=2)
        # concat_features=concat_features.squeeze(1)
        # concat_features = self.layer_norm(concat_features)
        
        # Pass the visual and physiological features through their respective encoders
        visual_encoded = self.visual_encoder(visual_features)
        physiological_encoded = self.physiological_encoder(physiological_features)
        
        # Do all the cross-attention for visual features
        cross_attention_output_v, _ = self.cross_attention_v(visual_encoded.permute(1, 0, 2), physiological_encoded.permute(1, 0, 2), physiological_encoded.permute(1, 0, 2))
        cross_attention_output_v = cross_attention_output_v.permute(1, 0, 2)
        
        # Do all the cross-attention for physio features
        cross_attention_output_p, _ = self.cross_attention_p(physiological_encoded.permute(1, 0, 2), visual_encoded.permute(1, 0, 2), visual_encoded.permute(1, 0, 2))
        cross_attention_output_p = cross_attention_output_p.permute(1, 0, 2)

        # Concatenate the cross-attention outputs
        concat_attention = torch.cat((cross_attention_output_v, cross_attention_output_p), dim=2) 
        
        # Normalisation of the concatenated attention
        # concat_attention = torch.norm(concat_attention, dim=1, keepdim=True)
        concat_attention=concat_attention.squeeze(1)
        # concat_attention = self.layer_norm(concat_attention)
        # concat_attention = concat_attention + concat_features

        
        # Pass the concatenated attention through the final layers
        out = self.out_layer1(concat_attention)
        out = self.relu(out)
        out = self.out_layer2(out)
        out = self.relu(out)
        out = self.out_layer3(out)
        out = self.relu(out)
        output = self.out_layer4(out)
        
        return output
    

class MultimodalTransformer_multi(nn.Module):
    def __init__(self, visual_dim, physiological_dim, num_heads, hidden_dim, num_layers, num_classes):
        super(MultimodalTransformer_multi, self).__init__()
        
        # Encoder blocks
        self.visual_encoder = TransformerEncoderBlock(visual_dim, num_heads, hidden_dim, num_layers)
        self.physiological_encoder = TransformerEncoderBlock(physiological_dim, num_heads, hidden_dim, num_layers)
        self.joint_representation_encoder = TransformerEncoderBlock(physiological_dim, num_heads, hidden_dim, num_layers)

        # Cross attention
        self.cross_attention_v = nn.MultiheadAttention(visual_dim, num_heads)
        self.cross_attention_p = nn.MultiheadAttention(physiological_dim, num_heads)
        self.cross_attention_pv = nn.MultiheadAttention(512, num_heads)
        
        # Fully connected layer for joint representation
        self.out_layer_pv = nn.Linear(1024, 512)
        
        # Fully connected layers for cross-attention outputs
        self.out_layer_ca=nn.Linear(512,256)
        
        # Fully connected layers
        self.out_layerAll1=nn.Linear(3072,1024)
        self.out_layerAll2=nn.Linear(1024,num_classes)
        
        self.out_layer1=nn.Linear(3072,1024)
        self.out_layer2=nn.Linear(1024,512)
        self.out_layer3=nn.Linear(512,256)
        self.out_layer4=nn.Linear(256,64)  
        self.out_layer5=nn.Linear(64,num_classes)  
        self.relu = nn.ReLU() 
        
        
    
    def forward(self, visual_features, physiological_features):
        
        # Concatenate the visual and physiological features
        joint_representation = torch.cat((visual_features, physiological_features), dim=2)
        
        # Decrease the dimensionality of the joint representation
        joint_representation = torch.norm(joint_representation, dim=1, keepdim=True)
        joint_representation=joint_representation.squeeze(1)
        joint_representation = self.out_layer_pv(joint_representation)
        joint_representation = torch.unsqueeze(joint_representation, 1)
        
        # Pass the visual, physiological and joint representation features through their respective encoders
        visual_encoded = self.visual_encoder(visual_features)
        physiological_encoded = self.physiological_encoder(physiological_features)
        joint_representation_encoded = self.joint_representation_encoder(joint_representation)
        
        # Do all the cross-attention between the visual encoded and physio encoded features
        cross_attention_output_v_p, _ = self.cross_attention_v(visual_encoded.permute(1, 0, 2), physiological_encoded.permute(1, 0, 2), physiological_encoded.permute(1, 0, 2))
        cross_attention_output_v_p = cross_attention_output_v_p.permute(1, 0, 2)
        # cross_attention_output_v_p = self.out_layer_ca(cross_attention_output_v_p)
        
        # Do all the cross-attention between the physio encoded and visio encoded features
        cross_attention_output_p_v, _ = self.cross_attention_p(physiological_encoded.permute(1, 0, 2), visual_encoded.permute(1, 0, 2), visual_encoded.permute(1, 0, 2))
        cross_attention_output_p_v = cross_attention_output_p_v.permute(1, 0, 2)
        # cross_attention_output_p_v = self.out_layer_ca(cross_attention_output_p_v)

        # Do all the cross-attention between the joint representation encoded and visio encoded features
        cross_attention_output_pv_v, _ = self.cross_attention_pv(joint_representation_encoded.permute(1, 0, 2), visual_encoded.permute(1, 0, 2), visual_encoded.permute(1, 0, 2))
        cross_attention_output_pv_v = cross_attention_output_pv_v.permute(1, 0, 2)
        # cross_attention_output_pv_v = self.out_layer_ca(cross_attention_output_pv_v)
        
        # Do all the cross-attention between the visio encoded and joint representation encoded features
        cross_attention_output_v_pv, _ = self.cross_attention_v(visual_encoded.permute(1, 0, 2), joint_representation_encoded.permute(1, 0, 2), joint_representation_encoded.permute(1, 0, 2))
        cross_attention_output_v_pv = cross_attention_output_v_pv.permute(1, 0, 2)
        # cross_attention_output_v_pv = self.out_layer_ca(cross_attention_output_v_pv)
        
        # Do all the cross-attention between the joint representation encoded and physio encoded features
        cross_attention_output_pv_p, _ = self.cross_attention_pv(joint_representation_encoded.permute(1, 0, 2), physiological_encoded.permute(1, 0, 2), physiological_encoded.permute(1, 0, 2))
        cross_attention_output_pv_p = cross_attention_output_pv_p.permute(1, 0, 2)
        # cross_attention_output_pv_p = self.out_layer_ca(cross_attention_output_pv_p)
        
        # Do all the cross-attention between the physio encoded and joint representation encoded features
        cross_attention_output_p_pv, _ = self.cross_attention_p(physiological_encoded.permute(1, 0, 2), joint_representation_encoded.permute(1, 0, 2), joint_representation_encoded.permute(1, 0, 2))
        cross_attention_output_p_pv = cross_attention_output_p_pv.permute(1, 0, 2)
        # cross_attention_output_p_pv = self.out_layer_ca(cross_attention_output_p_pv)
        
        # Addition the cross-attention outputs
        # added_attention = (cross_attention_output_v_p + cross_attention_output_p_v + cross_attention_output_pv_v + cross_attention_output_v_pv + cross_attention_output_pv_p + cross_attention_output_p_pv)
        
        # Concatenate Cross-attention outputs
        concat_attention = torch.cat((cross_attention_output_v_p, cross_attention_output_p_v, cross_attention_output_pv_v, cross_attention_output_v_pv, cross_attention_output_pv_p, cross_attention_output_p_pv), dim=2)
        
        # Normalisation of the concatenated attention
        # concat_attention = torch.norm(concat_attention, dim=1, keepdim=True)
        concat_attention=concat_attention.squeeze(1)
        
        # Pass the concatenated attention through the final layers
        # out = self.out_layer1(concat_attention)
        # out = self.relu(out)
        # out = self.out_layer2(out)
        # out = self.relu(out)
        # out = self.out_layer3(out)
        # out = self.relu(out)
        # out = self.out_layer4(out)
        # out = self.relu(out)
        # output = self.out_layer5(out)
        output = self.out_layerAll1(concat_attention)
        output = self.out_layerAll2(output)
        
        return output