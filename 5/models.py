import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *

class AdvancedMotionLoss(nn.Module):
    def __init__(self, w_pos=1.0, w_vel=1.0, w_acc=0.5, w_bone=0.5, w_foot=0.5):
        super().__init__()
        self.weights = {
            'pos': w_pos, 
            'vel': w_vel, 
            'acc': w_acc, 
            'bone': w_bone, 
            'foot': w_foot
        }
        self.bones = JOINT_CONNECTIONS
        self.feet_idx = [Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE]

    def forward(self, pred, target):
        l_pos = F.mse_loss(pred, target)
        
        v_pred = pred[:, 1:] - pred[:, :-1]
        v_target = target[:, 1:] - target[:, :-1]
        l_vel = F.mse_loss(v_pred, v_target)
        
        a_pred = v_pred[:, 1:] - v_pred[:, :-1]
        a_target = v_target[:, 1:] - v_target[:, :-1]
        l_acc = F.mse_loss(a_pred, a_target)
        
        l_bone = 0.0
        for j1, j2 in self.bones:
            b_pred = torch.norm(pred[:, :, j1, :] - pred[:, :, j2, :], dim=-1)
            b_target = torch.norm(target[:, :, j1, :] - target[:, :, j2, :], dim=-1)
            l_bone += F.mse_loss(b_pred, b_target)
        l_bone = l_bone / len(self.bones)
        
        l_foot = 0.0
        contact_threshold = 0.05 
        
        for f_idx in self.feet_idx:
            v_targ_foot = target[:, 1:, f_idx, :] - target[:, :-1, f_idx, :]
            v_pred_foot = pred[:, 1:, f_idx, :] - pred[:, :-1, f_idx, :]
            
            contact_mask = (torch.norm(v_targ_foot, dim=-1) < contact_threshold).float()
            skating_penalty = torch.norm(v_pred_foot, dim=-1) * contact_mask
            l_foot += torch.mean(skating_penalty)
            
        total_loss = (self.weights['pos'] * l_pos + 
                      self.weights['vel'] * l_vel + 
                      self.weights['acc'] * l_acc + 
                      self.weights['bone'] * l_bone + 
                      self.weights['foot'] * l_foot)
                      
        return total_loss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MotionTransformerDiffusion(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.d_model = d_model
        self.joint_dim = NUM_JOINTS * DIM 
        
        self.input_proj = nn.Linear(self.joint_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.cond_emb = nn.Embedding(2, d_model) 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            batch_first=True, 
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, self.joint_dim)

    def forward(self, x, t, cond):
        B, F_len, J, D = x.shape
        x = x.view(B, F_len, J * D) 
        
        x = self.input_proj(x)      
        x = self.pos_encoder(x)
        
        t_emb = self.time_mlp(t.unsqueeze(-1).float()) 
        c_emb = self.cond_emb(cond)                    
        
        emb = (t_emb + c_emb).unsqueeze(1)             
        x = x + emb 
        
        x = self.transformer(x)
        out = self.output_proj(x)                      
        
        return out.view(B, F_len, J, D)