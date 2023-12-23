import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import functional as F
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
import numpy as np
import random
import timm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from torch.cuda.amp import autocast, GradScaler

def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')

def ViT_SSF_LoRA_all_forward_attn(self, x):
    B, N, C = x.shape
    x = ssf_ada(x, self.SSF_attn_x_scale, self.SSF_attn_x_shift)
    qkv = self.qkv(x)
    qkv = ssf_ada(qkv, self.SSF_attn_qkv_scale, self.SSF_attn_qkv_shift)
    new_q = x @ self.dp(self.LoRA_q1)
    new_q = ssf_ada(new_q, self.SSF_attn_lora_hidden_qkv_scale[:self.dim], self.SSF_attn_lora_hidden_qkv_shift[:self.dim])
    new_q = new_q @ self.dp(self.LoRA_q2)
    new_q = ssf_ada(new_q, self.SSF_attn_lora_qkv_scale[ :768], self.SSF_attn_lora_qkv_shift[ :768])
    new_k = x @ self.dp(self.LoRA_k1)
    new_k = ssf_ada(new_k, self.SSF_attn_lora_hidden_qkv_scale[self.dim:2*self.dim], self.SSF_attn_lora_hidden_qkv_shift[self.dim:2*self.dim])
    new_k = new_k @ self.dp(self.LoRA_k2)
    new_k = ssf_ada(new_k, self.SSF_attn_lora_qkv_scale[ 768:2*768], self.SSF_attn_lora_qkv_shift[ 768:2*768])
    new_v = x @ self.dp(self.LoRA_v1)
    new_v = ssf_ada(new_v, self.SSF_attn_lora_hidden_qkv_scale[2*self.dim:3*self.dim], self.SSF_attn_lora_hidden_qkv_shift[2*self.dim:3*self.dim])
    new_v = new_v @ self.dp(self.LoRA_v2)
    new_v = ssf_ada(new_v, self.SSF_attn_lora_qkv_scale[ 2*768:3*768], self.SSF_attn_lora_qkv_shift[ 2*768:3*768])
    qkv += torch.cat([new_q, new_k, new_v], dim=2)
    qkv = qkv.reshape(B, N, 3,
                      self.num_heads,
                      C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    new_o = x @ self.dp(self.LoRA_o1)
    new_o = ssf_ada(new_o, self.SSF_attn_lora_hidden_o_scale, self.SSF_attn_lora_hidden_o_shift)
    new_o = new_o @ self.dp(self.LoRA_o2)
    new_o = ssf_ada(new_o, self.SSF_attn_lora_o_scale, self.SSF_attn_lora_o_shift)
    proj = self.proj(x)
    proj = ssf_ada(proj, self.SSF_attn_o_scale, self.SSF_attn_o_shift)
    proj = proj + new_o
    x = self.proj_drop(proj)
    return x

def ViT_SSF_LoRA_qv_forward_attn(self, x):
    B, N, C = x.shape
    x = ssf_ada(x, self.SSF_attn_x_scale, self.SSF_attn_x_shift)
    qkv = self.qkv(x)
    qkv = ssf_ada(qkv, self.SSF_attn_qkv_scale, self.SSF_attn_qkv_shift)
    new_q = x @ self.dp(self.LoRA_q1)
    new_q = ssf_ada(new_q, self.SSF_attn_lora_hidden_qkv_scale[:self.dim], self.SSF_attn_lora_hidden_qkv_shift[:self.dim])
    new_q = new_q @ self.dp(self.LoRA_q2)
    new_q = ssf_ada(new_q, self.SSF_attn_lora_qkv_scale[ :768], self.SSF_attn_lora_qkv_shift[ :768])
    new_v = x @ self.dp(self.LoRA_v1)
    new_v = ssf_ada(new_v, self.SSF_attn_lora_hidden_qkv_scale[2*self.dim:3*self.dim], self.SSF_attn_lora_hidden_qkv_shift[2*self.dim:3*self.dim])
    new_v = new_v @ self.dp(self.LoRA_v2)
    new_v = ssf_ada(new_v, self.SSF_attn_lora_qkv_scale[ 2*768:3*768], self.SSF_attn_lora_qkv_shift[ 2*768:3*768])
    qkv += torch.cat([new_q, torch.zeros_like(new_q), new_v], dim=2)
    qkv = qkv.reshape(B, N, 3,
                      self.num_heads,
                      C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    proj = self.proj(x)
    proj = ssf_ada(proj, self.SSF_attn_o_scale, self.SSF_attn_o_shift)
    proj = proj
    x = self.proj_drop(proj)
    return x

def ViT_SSF_LoRA_forward_mlp(self, x):
    B, N, C = x.shape
    x = ssf_ada(x, self.SSF_ffnx_scale, self.SSF_ffnx_shift)
    h = self.fc1(x)
    new_h = x @ self.dp(self.LoRA_ffn1_1)
    new_h = ssf_ada(new_h, self.SSF_lora_hidden_ffn1_scale, self.SSF_lora_hidden_ffn1_shift)
    new_h = new_h @ self.dp(self.LoRA_ffn1_2)
    new_h = ssf_ada(new_h, self.SSF_lora_ffn1_scale, self.SSF_lora_ffn1_shift)
    h = ssf_ada(h, self.SSF_ffn1_scale, self.SSF_ffn1_shift)
    h = h + new_h
    x = self.act(h)
    x = self.drop1(x)
    h = self.fc2(x)
    new_h = x @ self.dp(self.LoRA_ffn2_1)
    new_h = ssf_ada(new_h, self.SSF_lora_hidden_ffn2_scale, self.SSF_lora_hidden_ffn2_shift)
    new_h = new_h @ self.dp(self.LoRA_ffn2_2)
    new_h = ssf_ada(new_h, self.SSF_lora_ffn2_scale, self.SSF_lora_ffn2_shift)
    h = ssf_ada(h, self.SSF_ffn2_scale, self.SSF_ffn2_shift)
    h = h + new_h
    x = self.drop2(h)
    return x

def ViT_SSF_LoRA_forward_PatchEmbed(self, x):
    B, C, H, W = x.shape
    assert H == self.img_size[0] and W == self.img_size[1], \
        f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
    x = self.proj(x)
    if self.flatten:
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
    x = ssf_ada(x, self.SSF_embed_scale, self.SSF_embed_shift)
    x = self.norm(x)
    x = ssf_ada(x, self.SSF_embed_norm_scale, self.SSF_embed_norm_shift)
    return x
    
def ViT_SSF_LoRA_forward_features(self, x):
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    if self.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = self.pos_drop(x + self.pos_embed)
    x = self.blocks(x)
    x = self.norm(x)
    x = ssf_ada(x, self.SSF_ViT_scale, self.SSF_ViT_shift)
    if self.dist_token is None:
        return self.pre_logits(x[:, 0])
    else:
        return x[:, 0], x[:, 1]
    
def set_ViT_SSF_LoRA_all(model,dim=8):
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.SSF_ViT_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
        model.SSF_ViT_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
        nn.init.normal_(model.SSF_ViT_scale, mean=1, std=.02)
        nn.init.normal_(model.SSF_ViT_shift, std=.02)
        
        bound_method = ViT_SSF_LoRA_forward_features.__get__(model, model.__class__)
        setattr(model, 'forward_features', bound_method)
        
        
    for layer in model.children():
        if type(layer) == timm.models.vision_transformer.Attention:
            layer.dim = dim
            layer.dp = nn.Dropout(0.1)
            
            layer.LoRA_q1 = nn.Parameter(torch.zeros([768,dim], dtype=torch.float), requires_grad=True)
            layer.LoRA_q2 = nn.Parameter(torch.zeros([dim,768], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRA_q1)
            nn.init.zeros_(layer.LoRA_q2)
            layer.LoRA_v1 = nn.Parameter(torch.zeros([768,dim], dtype=torch.float), requires_grad=True)
            layer.LoRA_v2 = nn.Parameter(torch.zeros([dim,768], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRA_v1)
            nn.init.zeros_(layer.LoRA_v2)
            layer.LoRA_k1 = nn.Parameter(torch.zeros([768,dim], dtype=torch.float), requires_grad=True)
            layer.LoRA_k2 = nn.Parameter(torch.zeros([dim,768], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRA_k1)
            nn.init.zeros_(layer.LoRA_k2)
            layer.LoRA_o1 = nn.Parameter(torch.zeros([768,dim], dtype=torch.float), requires_grad=True)
            layer.LoRA_o2 = nn.Parameter(torch.zeros([dim,768], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRA_o1)
            nn.init.zeros_(layer.LoRA_o2)
        
            layer.SSF_attn_x_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_x_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_qkv_scale = nn.Parameter(torch.zeros([3*768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_qkv_shift = nn.Parameter(torch.zeros([3*768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_o_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_o_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_lora_qkv_scale = nn.Parameter(torch.zeros([3*768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_lora_qkv_shift = nn.Parameter(torch.zeros([3*768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_lora_hidden_qkv_scale = nn.Parameter(torch.zeros([3*dim], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_lora_hidden_qkv_shift = nn.Parameter(torch.zeros([3*dim], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_lora_o_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_lora_o_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_lora_hidden_o_scale = nn.Parameter(torch.zeros([dim], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_lora_hidden_o_shift = nn.Parameter(torch.zeros([dim], dtype=torch.float), requires_grad=True)
            nn.init.normal_(layer.SSF_attn_x_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_x_shift, std=.02)
            nn.init.normal_(layer.SSF_attn_qkv_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_qkv_shift, std=.02)
            nn.init.normal_(layer.SSF_attn_o_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_o_shift, std=.02)
            nn.init.normal_(layer.SSF_attn_lora_qkv_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_lora_qkv_shift, std=.02)
            nn.init.normal_(layer.SSF_attn_lora_hidden_qkv_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_lora_hidden_qkv_shift, std=.02)
            nn.init.normal_(layer.SSF_attn_lora_o_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_lora_o_shift, std=.02)
            nn.init.normal_(layer.SSF_attn_lora_hidden_o_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_lora_hidden_o_shift, std=.02)
            
            bound_method = ViT_SSF_LoRA_all_forward_attn.__get__(layer, layer.__class__)
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

            layer.SSF_ffnx_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_ffnx_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_ffn1_scale = nn.Parameter(torch.zeros([4*768], dtype=torch.float), requires_grad=True)
            layer.SSF_ffn1_shift = nn.Parameter(torch.zeros([4*768], dtype=torch.float), requires_grad=True)
            layer.SSF_ffn2_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_ffn2_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_ffn1_scale = nn.Parameter(torch.zeros([4*768], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_ffn1_shift = nn.Parameter(torch.zeros([4*768], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_hidden_ffn1_scale = nn.Parameter(torch.zeros([dim], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_hidden_ffn1_shift = nn.Parameter(torch.zeros([dim], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_ffn2_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_ffn2_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_hidden_ffn2_scale = nn.Parameter(torch.zeros([dim], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_hidden_ffn2_shift = nn.Parameter(torch.zeros([dim], dtype=torch.float), requires_grad=True)
            nn.init.normal_(layer.SSF_ffnx_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_ffnx_shift, std=.02)
            nn.init.normal_(layer.SSF_ffn1_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_ffn1_shift, std=.02)
            nn.init.normal_(layer.SSF_ffn2_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_ffn2_shift, std=.02)
            nn.init.normal_(layer.SSF_lora_ffn1_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_lora_ffn1_shift, std=.02)
            nn.init.normal_(layer.SSF_lora_hidden_ffn1_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_lora_hidden_ffn1_shift, std=.02)
            nn.init.normal_(layer.SSF_lora_ffn2_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_lora_ffn2_shift, std=.02)
            nn.init.normal_(layer.SSF_lora_hidden_ffn2_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_lora_hidden_ffn2_shift, std=.02)
            
            bound_method = ViT_SSF_LoRA_forward_mlp.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif type(layer) == timm.models.layers.PatchEmbed:
            layer.SSF_embed_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_embed_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_embed_norm_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_embed_norm_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            nn.init.normal_(layer.SSF_embed_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_embed_shift, std=.02)
            nn.init.normal_(layer.SSF_embed_norm_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_embed_norm_shift, std=.02)
            bound_method = ViT_SSF_LoRA_forward_PatchEmbed.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_ViT_SSF_LoRA_all(layer,dim)

def set_ViT_SSF_LoRA_qv(model,dim=8):
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.SSF_ViT_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
        model.SSF_ViT_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
        nn.init.normal_(model.SSF_ViT_scale, mean=1, std=.02)
        nn.init.normal_(model.SSF_ViT_shift, std=.02)
        
        bound_method = ViT_SSF_LoRA_forward_features.__get__(model, model.__class__)
        setattr(model, 'forward_features', bound_method)
        
        
    for layer in model.children():
        if type(layer) == timm.models.vision_transformer.Attention:
            layer.dim = dim
            layer.dp = nn.Dropout(0.1)
            
            layer.LoRA_q1 = nn.Parameter(torch.zeros([768,dim], dtype=torch.float), requires_grad=True)
            layer.LoRA_q2 = nn.Parameter(torch.zeros([dim,768], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRA_q1)
            nn.init.zeros_(layer.LoRA_q2)
            layer.LoRA_v1 = nn.Parameter(torch.zeros([768,dim], dtype=torch.float), requires_grad=True)
            layer.LoRA_v2 = nn.Parameter(torch.zeros([dim,768], dtype=torch.float), requires_grad=True)
            nn.init.xavier_uniform_(layer.LoRA_v1)
            nn.init.zeros_(layer.LoRA_v2)
        
            layer.SSF_attn_x_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_x_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_qkv_scale = nn.Parameter(torch.zeros([3*768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_qkv_shift = nn.Parameter(torch.zeros([3*768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_o_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_o_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_lora_qkv_scale = nn.Parameter(torch.zeros([3*768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_lora_qkv_shift = nn.Parameter(torch.zeros([3*768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_lora_hidden_qkv_scale = nn.Parameter(torch.zeros([3*dim], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_lora_hidden_qkv_shift = nn.Parameter(torch.zeros([3*dim], dtype=torch.float), requires_grad=True)
            nn.init.normal_(layer.SSF_attn_x_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_x_shift, std=.02)
            nn.init.normal_(layer.SSF_attn_qkv_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_qkv_shift, std=.02)
            nn.init.normal_(layer.SSF_attn_o_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_o_shift, std=.02)
            nn.init.normal_(layer.SSF_attn_lora_qkv_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_lora_qkv_shift, std=.02)
            nn.init.normal_(layer.SSF_attn_lora_hidden_qkv_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_lora_hidden_qkv_shift, std=.02)
            
            bound_method = ViT_SSF_LoRA_qv_forward_attn.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif type(layer) == timm.models.layers.PatchEmbed:
            layer.SSF_embed_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_embed_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_embed_norm_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_embed_norm_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            nn.init.normal_(layer.SSF_embed_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_embed_shift, std=.02)
            nn.init.normal_(layer.SSF_embed_norm_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_embed_norm_shift, std=.02)
            bound_method = ViT_SSF_LoRA_forward_PatchEmbed.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_ViT_SSF_LoRA_qv(layer,dim)
            
def set_ViT_SSF_LoRA_ffn(model,dim=8):
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.SSF_ViT_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
        model.SSF_ViT_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
        nn.init.normal_(model.SSF_ViT_scale, mean=1, std=.02)
        nn.init.normal_(model.SSF_ViT_shift, std=.02)
        
        bound_method = ViT_SSF_LoRA_forward_features.__get__(model, model.__class__)
        setattr(model, 'forward_features', bound_method)
        
        
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

            layer.SSF_ffnx_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_ffnx_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_ffn1_scale = nn.Parameter(torch.zeros([4*768], dtype=torch.float), requires_grad=True)
            layer.SSF_ffn1_shift = nn.Parameter(torch.zeros([4*768], dtype=torch.float), requires_grad=True)
            layer.SSF_ffn2_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_ffn2_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_ffn1_scale = nn.Parameter(torch.zeros([4*768], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_ffn1_shift = nn.Parameter(torch.zeros([4*768], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_hidden_ffn1_scale = nn.Parameter(torch.zeros([dim], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_hidden_ffn1_shift = nn.Parameter(torch.zeros([dim], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_ffn2_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_ffn2_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_hidden_ffn2_scale = nn.Parameter(torch.zeros([dim], dtype=torch.float), requires_grad=True)
            layer.SSF_lora_hidden_ffn2_shift = nn.Parameter(torch.zeros([dim], dtype=torch.float), requires_grad=True)
            nn.init.normal_(layer.SSF_ffnx_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_ffnx_shift, std=.02)
            nn.init.normal_(layer.SSF_ffn1_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_ffn1_shift, std=.02)
            nn.init.normal_(layer.SSF_ffn2_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_ffn2_shift, std=.02)
            nn.init.normal_(layer.SSF_lora_ffn1_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_lora_ffn1_shift, std=.02)
            nn.init.normal_(layer.SSF_lora_hidden_ffn1_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_lora_hidden_ffn1_shift, std=.02)
            nn.init.normal_(layer.SSF_lora_ffn2_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_lora_ffn2_shift, std=.02)
            nn.init.normal_(layer.SSF_lora_hidden_ffn2_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_lora_hidden_ffn2_shift, std=.02)
            
            bound_method = ViT_SSF_LoRA_forward_mlp.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif type(layer) == timm.models.layers.PatchEmbed:
            layer.SSF_embed_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_embed_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_embed_norm_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_embed_norm_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            nn.init.normal_(layer.SSF_embed_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_embed_shift, std=.02)
            nn.init.normal_(layer.SSF_embed_norm_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_embed_norm_shift, std=.02)
            bound_method = ViT_SSF_LoRA_forward_PatchEmbed.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_ViT_SSF_LoRA_ffn(layer,dim)
