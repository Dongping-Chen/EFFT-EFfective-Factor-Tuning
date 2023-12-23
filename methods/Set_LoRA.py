import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import random
import timm

from typing import Optional

def ViT_LoRA_qv_forward_attn(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x)
    new_q = self.dp(self.LoRAq1) @ self.dp(self.LoRAq2)
    new_v = self.dp(self.LoRAv1) @ self.dp(self.LoRAv2)
    qkv += torch.cat([(x @ new_q), (torch.zeros_like(x @ new_q)), (x @ new_v)], dim=2)
    qkv = qkv.reshape(B, N, 3,
                      self.num_heads,
                      C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    proj = self.proj(x)
    x = self.proj_drop(proj)
    return x

def ViT_LoRA_all_forward_attn(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x)
    new_q = self.dp(self.LoRAq1) @ self.dp(self.LoRAq2)
    new_v = self.dp(self.LoRAv1) @ self.dp(self.LoRAv2)
    new_k = self.dp(self.LoRAk1) @ self.dp(self.LoRAk2)
    new_o = self.dp(self.LoRAo1) @ self.dp(self.LoRAo2)
    qkv += torch.cat([(x @ new_q), (x @ new_k), (x @ new_v)], dim=2)
    qkv = qkv.reshape(B, N, 3,
                      self.num_heads,
                      C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    proj = self.proj(x)
    new_proj = x @ new_o
    x = self.proj_drop(proj) + new_proj
    return x

def ViT_LoRA_all_forward_mlp(self, x):
    B, N, C = x.shape
    new_ffn1 = self.dp(self.LoRA_ffn1_1) @ self.dp(self.LoRA_ffn1_2)
    new_ffn2 = self.dp(self.LoRA_ffn2_1) @ self.dp(self.LoRA_ffn2_2)
    h = self.fc1(x)
    h += (x @ new_ffn1)
        
    x = self.act(h)
    x = self.drop1(x)
    h = self.fc2(x)
    h += (x @ new_ffn2)
    x = self.drop2(h)
    return x

def set_ViT_LoRA_qv(model,dim=8):
        
    for layer in model.children():
        if type(layer) == timm.models.vision_transformer.Attention:
            layer.dp = nn.Dropout(0.1)
            layer.dim = dim
            hidden_dim = layer.proj.shape[0]
            layer.LoRAq1 = nn.Parameter(torch.zeros([hidden_dim,dim], dtype=torch.float), requires_grad=True)
            layer.LoRAq2 = nn.Parameter(torch.zeros([dim,hidden_dim], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRAq1)
            nn.init.zeros_(layer.LoRAq2)
            layer.LoRAv1 = nn.Parameter(torch.zeros([hidden_dim,dim], dtype=torch.float), requires_grad=True)
            layer.LoRAv2 = nn.Parameter(torch.zeros([dim,hidden_dim], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRAv1)
            nn.init.zeros_(layer.LoRAv2)
            bound_method = ViT_LoRA_qv_forward_attn.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_ViT_LoRA_qv(layer, dim)

def set_ViT_LoRA_all(model,dim=4):
              
    for layer in model.children():
        if type(layer) == timm.models.vision_transformer.Attention:
            layer.dp = nn.Dropout(0.1)
            layer.dim = dim
            
            layer.LoRAq1 = nn.Parameter(torch.zeros([768,dim], dtype=torch.float), requires_grad=True)
            layer.LoRAq2 = nn.Parameter(torch.zeros([dim,768], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRAq1)
            nn.init.zeros_(layer.LoRAq2)
            layer.LoRAv1 = nn.Parameter(torch.zeros([768,dim], dtype=torch.float), requires_grad=True)
            layer.LoRAv2 = nn.Parameter(torch.zeros([dim,768], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRAv1)
            nn.init.zeros_(layer.LoRAv2)
            layer.LoRAk1 = nn.Parameter(torch.zeros([768,dim], dtype=torch.float), requires_grad=True)
            layer.LoRAk2 = nn.Parameter(torch.zeros([dim,768], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRAk1)
            nn.init.zeros_(layer.LoRAk2)
            layer.LoRAo1 = nn.Parameter(torch.zeros([768,dim], dtype=torch.float), requires_grad=True)
            layer.LoRAo2 = nn.Parameter(torch.zeros([dim,768], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRAo1)
            nn.init.zeros_(layer.LoRAo2)
            
            bound_method = ViT_LoRA_all_forward_attn.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif type(layer) == timm.models.layers.mlp.Mlp:
            layer.dim = dim
            layer.dp = nn.Dropout(0.1)
            layer.LoRA_ffn1_1 = nn.Parameter(torch.zeros([768,dim], dtype=torch.float), requires_grad=True)
            layer.LoRA_ffn1_2 = nn.Parameter(torch.zeros([dim,4*768], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRA_ffn1_1)
            nn.init.zeros_(layer.LoRA_ffn1_2)
            layer.LoRA_ffn2_1 = nn.Parameter(torch.zeros([4*768,dim], dtype=torch.float), requires_grad=True)
            layer.LoRA_ffn2_2 = nn.Parameter(torch.zeros([dim,768], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRA_ffn2_1)
            nn.init.zeros_(layer.LoRA_ffn2_2)
            bound_method = ViT_LoRA_all_forward_mlp.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_ViT_LoRA_all(layer, dim)

def set_ViT_LoRA_ffn(model,dim=4):
        
    for layer in model.children():
        if type(layer) == timm.models.layers.mlp.Mlp:
            layer.dim = dim
            layer.dp = nn.Dropout(0.1)
            layer.LoRA_ffn1_1 = nn.Parameter(torch.zeros([768,dim], dtype=torch.float), requires_grad=True)
            layer.LoRA_ffn1_2 = nn.Parameter(torch.zeros([dim,4*768], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRA_ffn1_1)
            nn.init.zeros_(layer.LoRA_ffn1_2)
            layer.LoRA_ffn2_1 = nn.Parameter(torch.zeros([4*768,dim], dtype=torch.float), requires_grad=True)
            layer.LoRA_ffn2_2 = nn.Parameter(torch.zeros([dim,768], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRA_ffn2_1)
            nn.init.zeros_(layer.LoRA_ffn2_2)
            bound_method = ViT_LoRA_all_forward_mlp.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_ViT_LoRA_ffn(layer, dim)

    
def Swin_LoRA_attn_forward(self, x, mask: Optional[torch.Tensor] = None):
    B_, N, C = x.shape
    qkv = self.qkv(x)
    new_q = (self.dp(self.LoRA_q1) @ self.dp(self.LoRA_q2))
    new_v = (self.dp(self.LoRA_v1) @ self.dp(self.LoRA_v2))
    qkv += torch.cat([(x @ new_q), (torch.zeros_like(x @ new_q)), (x @ new_v)], dim=2)
    qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

    
def set_Swin_LoRA_qv(model,dim=4):
    
    for layer in model.layers:
        for block in layer.blocks:
            attn = block.attn
            new_dim = attn.dim // 128 * dim
            attn.LoRA_q1 = nn.Parameter(torch.zeros([attn.dim,new_dim], dtype=torch.float), requires_grad=True)
            attn.LoRA_q2 = nn.Parameter(torch.zeros([new_dim,attn.dim], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(attn.LoRA_q1)
            nn.init.zeros_(attn.LoRA_q2)
            attn.LoRA_v1 = nn.Parameter(torch.zeros([attn.dim,new_dim], dtype=torch.float), requires_grad=True)
            attn.LoRA_v2 = nn.Parameter(torch.zeros([new_dim,attn.dim], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(attn.LoRA_v1)
            nn.init.zeros_(attn.LoRA_v2)
            attn.dp = nn.Dropout(0.1)
            bound_method = Swin_LoRA_attn_forward.__get__(attn, attn.__class__)
            setattr(attn, 'forward', bound_method)
        
