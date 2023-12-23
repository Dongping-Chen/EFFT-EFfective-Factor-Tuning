import torch
from torch import nn
import timm
from typing import Optional

def ViT_EFFT_forward_attn(self, x):
    B, N, C = x.shape
    dropped_c = self.dp(self.EFFTc[:,:,:4])
    dropped_u = self.dp(self.EFFTu)
    dropped_v = self.dp(self.EFFTv)
    q,k,v,o = dropped_u @ dropped_c[:,:,0] @ dropped_v, dropped_u @ dropped_c[:,:,1] @ dropped_v, dropped_u @ dropped_c[:,:,2] @ dropped_v,dropped_u @ dropped_c[:,:,3] @ dropped_v
    qkv = self.qkv(x)
    qkv += torch.cat([(x @ q), (x @ k), (x @ v)], dim=2) * self.s

    qkv = qkv.reshape(B, N, 3,
                      self.num_heads,
                      C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    proj = self.proj(x)
    proj += (x @ o) * self.s
    x = self.proj_drop(proj)
    return x

def ViT_EFFT_forward_mlp(self, x):
    B, N, C = x.shape
    EFFT_fc1 = self.dp(self.EFFTu) @ self.dp(self.EFFTc[:,:,0]) @ self.dp(self.EFFTv)
    EFFT_fc2 = self.dp(self.EFFTu) @ self.dp(self.EFFTc[:,:,1]) @ self.dp(self.EFFTv)
    EFFT_fc2 = EFFT_fc2.transpose(0,1)
    h = self.fc1(x)
    h += (x @ EFFT_fc1) * self.s
        
    x = self.act(h)
    x = self.drop1(x)
    h = self.fc2(x)
    h += (x @ EFFT_fc2) * self.s
    x = self.drop2(h)
    return x
        
def set_ViT_EFFT(model,dim=16,s=1,init = 'vv', root_model=None):
    if root_model is None:
        root_model = model
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.EFFTu1 = nn.Parameter(torch.zeros([768, dim], dtype=torch.float), requires_grad=True)
        model.EFFTv1 = nn.Parameter(torch.zeros([dim, 768], dtype=torch.float), requires_grad=True)
        model.EFFTc1 = nn.Parameter(torch.zeros([dim, dim, 4], dtype=torch.float), requires_grad=True)
        if init[0] == 'v':
            nn.init.zeros_(model.EFFTv1)
            nn.init.xavier_uniform_(model.EFFTu1)
        else:
            nn.init.zeros_(model.EFFTu1)
            nn.init.xavier_uniform_(model.EFFTv1)
        nn.init.xavier_uniform_(model.EFFTc1)
        
        model.EFFTu2 = nn.Parameter(torch.zeros([768, dim], dtype=torch.float), requires_grad=True)
        model.EFFTv2 = nn.Parameter(torch.zeros([dim, 4*768], dtype=torch.float), requires_grad=True)
        model.EFFTc2 = nn.Parameter(torch.zeros([dim, dim, 2], dtype=torch.float), requires_grad=True)
        if init[0] == 'v':
            nn.init.zeros_(model.EFFTv2)
            nn.init.xavier_uniform_(model.EFFTu2)
        else:
            nn.init.zeros_(model.EFFTu2)
            nn.init.xavier_uniform_(model.EFFTv2)
        nn.init.xavier_uniform_(model.EFFTc2)
        
    for layer in model.children():
        if type(layer) == timm.models.vision_transformer.Attention:
            layer.dp = nn.Dropout(0.1)
            layer.s = s
            layer.dim = dim
            layer.EFFTu = root_model.EFFTu1
            layer.EFFTv = root_model.EFFTv1
            layer.EFFTc = root_model.EFFTc1
            bound_method = ViT_EFFT_forward_attn.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif type(layer) == timm.models.layers.mlp.Mlp:
            layer.dim = dim
            layer.s = s
            layer.dp = nn.Dropout(0.1)
            layer.EFFTu = root_model.EFFTu2
            layer.EFFTv = root_model.EFFTv2
            layer.EFFTc = root_model.EFFTc2
            bound_method = ViT_EFFT_forward_mlp.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_ViT_EFFT(layer, dim, s, root_model)

def set_ViT_L_EFFT(model,dim=24,s=1,root_model=None):
    if root_model is None:
        root_model = model
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.EFFTu1 = nn.Parameter(torch.zeros([1024, dim], dtype=torch.float), requires_grad=True)
        model.EFFTv1 = nn.Parameter(torch.zeros([dim, 1024], dtype=torch.float), requires_grad=True)
        model.EFFTc1 = nn.Parameter(torch.zeros([dim, dim, 4], dtype=torch.float), requires_grad=True)
        nn.init.zeros_(model.EFFTv1)
        nn.init.xavier_uniform_(model.EFFTu1)
        nn.init.xavier_uniform_(model.EFFTc1)
        
        model.EFFTu2 = nn.Parameter(torch.zeros([1024, dim], dtype=torch.float), requires_grad=True)
        model.EFFTv2 = nn.Parameter(torch.zeros([dim, 4*1024], dtype=torch.float), requires_grad=True)
        model.EFFTc2 = nn.Parameter(torch.zeros([dim, dim, 2], dtype=torch.float), requires_grad=True)
        nn.init.zeros_(model.EFFTv2)
        nn.init.xavier_uniform_(model.EFFTu2)
        nn.init.xavier_uniform_(model.EFFTc2)
        
    for layer in model.children():
        if type(layer) == timm.models.vision_transformer.Attention:
            layer.dp = nn.Dropout(0.1)
            layer.s = s
            layer.dim = dim
            layer.EFFTu = root_model.EFFTu1
            layer.EFFTv = root_model.EFFTv1
            layer.EFFTc = root_model.EFFTc1
            bound_method = ViT_EFFT_forward_attn.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif type(layer) == timm.models.layers.mlp.Mlp:
            layer.dim = dim
            layer.s = s
            layer.dp = nn.Dropout(0.1)
            layer.EFFTu = root_model.EFFTu2
            layer.EFFTv = root_model.EFFTv2
            layer.EFFTc = root_model.EFFTc2
            bound_method = ViT_EFFT_forward_mlp.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_ViT_EFFT(layer, dim, s, root_model)
            
def set_ViT_H_EFFT(model,dim=32,s=1,root_model=None):
    if root_model is None:
        root_model = model
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.EFFTu1 = nn.Parameter(torch.zeros([1280, dim], dtype=torch.float), requires_grad=True)
        model.EFFTv1 = nn.Parameter(torch.zeros([dim, 1280], dtype=torch.float), requires_grad=True)
        model.EFFTc1 = nn.Parameter(torch.zeros([dim, dim, 4], dtype=torch.float), requires_grad=True)
        nn.init.zeros_(model.EFFTv1)
        nn.init.xavier_uniform_(model.EFFTu1)
        nn.init.xavier_uniform_(model.EFFTc1)
        
        model.EFFTu2 = nn.Parameter(torch.zeros([1280, dim], dtype=torch.float), requires_grad=True)
        model.EFFTv2 = nn.Parameter(torch.zeros([dim, 4*1280], dtype=torch.float), requires_grad=True)
        model.EFFTc2 = nn.Parameter(torch.zeros([dim, dim, 2], dtype=torch.float), requires_grad=True)
        nn.init.zeros_(model.EFFTv2)
        nn.init.xavier_uniform_(model.EFFTu2)
        nn.init.xavier_uniform_(model.EFFTc2)
        
    for layer in model.children():
        if type(layer) == timm.models.vision_transformer.Attention:
            layer.dp = nn.Dropout(0.1)
            layer.s = s
            layer.dim = dim
            layer.EFFTu = root_model.EFFTu1
            layer.EFFTv = root_model.EFFTv1
            layer.EFFTc = root_model.EFFTc1
            bound_method = ViT_EFFT_forward_attn.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif type(layer) == timm.models.layers.mlp.Mlp:
            layer.dim = dim
            layer.s = s
            layer.dp = nn.Dropout(0.1)
            layer.EFFTu = root_model.EFFTu2
            layer.EFFTv = root_model.EFFTv2
            layer.EFFTc = root_model.EFFTc2
            bound_method = ViT_EFFT_forward_mlp.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_ViT_EFFT(layer, dim, s, root_model)
            
            
def Swin_EFFT_attn_forward(self, x, mask: Optional[torch.Tensor] = None):
    B_, N, C = x.shape
    dropped_c = self.dp(self.EFFT_c[:4,:,:])
    dropped_u = self.dp(self.EFFT_u)
    dropped_v = self.dp(self.EFFT_v)
    
    q,k,v,o = dropped_u @ dropped_c[0] @ dropped_v, dropped_u @ dropped_c[1] @ dropped_v, dropped_u @ dropped_c[2] @ dropped_v,dropped_u @ dropped_c[3] @ dropped_v
    qkv = self.qkv(x)
    qkv += torch.cat([(x @ q), (x @ k), (x @ v)], dim=2) * self.scale_s
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
    x = self.proj(x) + (x @ o) * self.scale_s
    x = self.proj_drop(x)
    return x

def Swin_EFFT_mlp_forward(self,x):
    B, N, C = x.shape     
    EFFT_fc1 = self.dp(self.EFFT_u) @ self.dp(self.EFFT_c[0]) @ self.dp(self.EFFT_v)
    EFFT_fc2 = self.dp(self.EFFT_u) @ self.dp(self.EFFT_c[1]) @ self.dp(self.EFFT_v)
       
    EFFT_fc2 = EFFT_fc2.transpose(0,1)
    h = self.fc1(x)
    h += x @ EFFT_fc1 * self.scale_s
        
    x = self.act(h)
    x = self.drop1(x)
    h = self.fc2(x)
    h += x @ EFFT_fc2 * self.scale_s
    x = self.drop2(h)
    return x

def set_Swin_EFFT(model, dim=2, s=1):
    print(s)
    if type(model) == timm.models.swin_transformer.SwinTransformer:
        model.EFFT_1_attn_u = nn.Parameter(torch.zeros([128,2*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_1_attn_c = nn.Parameter(torch.zeros([4,2*dim,2*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_1_attn_v = nn.Parameter(torch.zeros([2*dim,128],dtype=torch.float),requires_grad=True)
        model.EFFT_2_attn_u = nn.Parameter(torch.zeros([256,4*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_2_attn_c = nn.Parameter(torch.zeros([4,4*dim,4*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_2_attn_v = nn.Parameter(torch.zeros([4*dim,256],dtype=torch.float),requires_grad=True)
        model.EFFT_3_attn_u = nn.Parameter(torch.zeros([512,8*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_3_attn_c = nn.Parameter(torch.zeros([4,8*dim,8*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_3_attn_v = nn.Parameter(torch.zeros([8*dim,512],dtype=torch.float),requires_grad=True)
        model.EFFT_4_attn_u = nn.Parameter(torch.zeros([1024,16*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_4_attn_c = nn.Parameter(torch.zeros([4,16*dim,16*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_4_attn_v = nn.Parameter(torch.zeros([16*dim,1024],dtype=torch.float),requires_grad=True)
        nn.init.zeros_(model.EFFT_1_attn_v)
        nn.init.xavier_uniform_(model.EFFT_1_attn_u)
        nn.init.xavier_uniform_(model.EFFT_1_attn_c)
        nn.init.zeros_(model.EFFT_2_attn_v)
        nn.init.xavier_uniform_(model.EFFT_2_attn_u)
        nn.init.xavier_uniform_(model.EFFT_2_attn_c)
        nn.init.zeros_(model.EFFT_3_attn_v)
        nn.init.xavier_uniform_(model.EFFT_3_attn_u)
        nn.init.xavier_uniform_(model.EFFT_3_attn_c)
        nn.init.zeros_(model.EFFT_4_attn_v)
        nn.init.xavier_uniform_(model.EFFT_4_attn_u)
        nn.init.xavier_uniform_(model.EFFT_4_attn_c)
        
        model.EFFT_1_mlp_u = nn.Parameter(torch.zeros([128,dim],dtype=torch.float),requires_grad=True)
        model.EFFT_1_mlp_c = nn.Parameter(torch.zeros([2,dim,2 * dim],dtype=torch.float),requires_grad=True)
        model.EFFT_1_mlp_v = nn.Parameter(torch.zeros([2 * dim,512],dtype=torch.float),requires_grad=True)
        model.EFFT_2_mlp_u = nn.Parameter(torch.zeros([256,2*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_2_mlp_c = nn.Parameter(torch.zeros([2,2*dim,4*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_2_mlp_v = nn.Parameter(torch.zeros([4*dim,1024],dtype=torch.float),requires_grad=True)
        model.EFFT_3_mlp_u = nn.Parameter(torch.zeros([512,4*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_3_mlp_c = nn.Parameter(torch.zeros([18,4*dim,8*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_3_mlp_v = nn.Parameter(torch.zeros([8*dim,2048],dtype=torch.float),requires_grad=True)
        model.EFFT_4_mlp_u = nn.Parameter(torch.zeros([1024,8*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_4_mlp_c = nn.Parameter(torch.zeros([2,8*dim,16*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_4_mlp_v = nn.Parameter(torch.zeros([16*dim,4096],dtype=torch.float),requires_grad=True)
        nn.init.zeros_(model.EFFT_1_mlp_v)
        nn.init.xavier_uniform_(model.EFFT_1_mlp_u)
        nn.init.xavier_uniform_(model.EFFT_1_mlp_c)
        nn.init.zeros_(model.EFFT_2_mlp_v)
        nn.init.xavier_uniform_(model.EFFT_2_mlp_u)
        nn.init.xavier_uniform_(model.EFFT_2_mlp_c)
        nn.init.zeros_(model.EFFT_3_mlp_v)
        nn.init.xavier_uniform_(model.EFFT_3_mlp_u)
        nn.init.xavier_uniform_(model.EFFT_3_mlp_c)
        nn.init.zeros_(model.EFFT_4_mlp_v)
        nn.init.xavier_uniform_(model.EFFT_4_mlp_u)
        nn.init.xavier_uniform_(model.EFFT_4_mlp_c)

        model.layer_idx = 0
    
    for layer in model.layers:
        for block in layer.blocks:
            attn = block.attn
            attn.layer_idx = model.layer_idx
            attn.scale_s = s
            attn.dp = nn.Dropout(0.1)
            bound_method = Swin_EFFT_attn_forward.__get__(attn, attn.__class__)
            setattr(attn, 'forward', bound_method)
            
            mlp = block.mlp
            mlp.layer_idx = model.layer_idx
            mlp.scale_s = s
            mlp.dp = nn.Dropout(0.1)
            bound_method = Swin_EFFT_mlp_forward.__get__(mlp, mlp.__class__)
            setattr(mlp, 'forward', bound_method)
            
            if model.layer_idx == 0:
                attn.EFFT_u = model.EFFT_1_attn_u
                attn.EFFT_v = model.EFFT_1_attn_v
                attn.EFFT_c = model.EFFT_1_attn_c
                mlp.EFFT_u = model.EFFT_1_mlp_u
                mlp.EFFT_v = model.EFFT_1_mlp_v
                mlp.EFFT_c = model.EFFT_1_mlp_c
            elif model.layer_idx == 1:
                attn.EFFT_u = model.EFFT_2_attn_u
                attn.EFFT_v = model.EFFT_2_attn_v
                attn.EFFT_c = model.EFFT_2_attn_c
                mlp.EFFT_u = model.EFFT_2_mlp_u
                mlp.EFFT_v = model.EFFT_2_mlp_v
                mlp.EFFT_c = model.EFFT_2_mlp_c
            elif model.layer_idx == 2:
                attn.EFFT_u = model.EFFT_3_attn_u
                attn.EFFT_v = model.EFFT_3_attn_v
                attn.EFFT_c = model.EFFT_3_attn_c
                mlp.EFFT_u = model.EFFT_3_mlp_u
                mlp.EFFT_v = model.EFFT_3_mlp_v
                mlp.EFFT_c = model.EFFT_3_mlp_c
            elif model.layer_idx == 3:
                attn.EFFT_u = model.EFFT_4_attn_u
                attn.EFFT_v = model.EFFT_4_attn_v
                attn.EFFT_c = model.EFFT_4_attn_c
                mlp.EFFT_u = model.EFFT_4_mlp_u
                mlp.EFFT_v = model.EFFT_4_mlp_v
                mlp.EFFT_c = model.EFFT_4_mlp_c
        model.layer_idx += 1
        
        
def set_Swin_S_EFFT(model, dim=4, s=1):
    print(s)
    if type(model) == timm.models.swin_transformer.SwinTransformer:
        model.EFFT_1_attn_u = nn.Parameter(torch.zeros([96,dim],dtype=torch.float),requires_grad=True)
        model.EFFT_1_attn_c = nn.Parameter(torch.zeros([4,dim,dim],dtype=torch.float),requires_grad=True)
        model.EFFT_1_attn_v = nn.Parameter(torch.zeros([dim,96],dtype=torch.float),requires_grad=True)
        model.EFFT_2_attn_u = nn.Parameter(torch.zeros([192,2*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_2_attn_c = nn.Parameter(torch.zeros([4,2*dim,2*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_2_attn_v = nn.Parameter(torch.zeros([2*dim,192],dtype=torch.float),requires_grad=True)
        model.EFFT_3_attn_u = nn.Parameter(torch.zeros([384,4*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_3_attn_c = nn.Parameter(torch.zeros([4,4*dim,4*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_3_attn_v = nn.Parameter(torch.zeros([4*dim,384],dtype=torch.float),requires_grad=True)
        model.EFFT_4_attn_u = nn.Parameter(torch.zeros([768,8*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_4_attn_c = nn.Parameter(torch.zeros([4,8*dim,8*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_4_attn_v = nn.Parameter(torch.zeros([8*dim,768],dtype=torch.float),requires_grad=True)
        nn.init.zeros_(model.EFFT_1_attn_v)
        nn.init.xavier_uniform_(model.EFFT_1_attn_u)
        nn.init.xavier_uniform_(model.EFFT_1_attn_c)
        nn.init.zeros_(model.EFFT_2_attn_v)
        nn.init.xavier_uniform_(model.EFFT_2_attn_u)
        nn.init.xavier_uniform_(model.EFFT_2_attn_c)
        nn.init.zeros_(model.EFFT_3_attn_v)
        nn.init.xavier_uniform_(model.EFFT_3_attn_u)
        nn.init.xavier_uniform_(model.EFFT_3_attn_c)
        nn.init.zeros_(model.EFFT_4_attn_v)
        nn.init.xavier_uniform_(model.EFFT_4_attn_u)
        nn.init.xavier_uniform_(model.EFFT_4_attn_c)
        
        model.EFFT_1_mlp_u = nn.Parameter(torch.zeros([96,dim],dtype=torch.float),requires_grad=True)
        model.EFFT_1_mlp_c = nn.Parameter(torch.zeros([2,dim,2 * dim],dtype=torch.float),requires_grad=True)
        model.EFFT_1_mlp_v = nn.Parameter(torch.zeros([2 * dim,384],dtype=torch.float),requires_grad=True)
        model.EFFT_2_mlp_u = nn.Parameter(torch.zeros([192,2*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_2_mlp_c = nn.Parameter(torch.zeros([2,2*dim,4*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_2_mlp_v = nn.Parameter(torch.zeros([4*dim,768],dtype=torch.float),requires_grad=True)
        model.EFFT_3_mlp_u = nn.Parameter(torch.zeros([384,4*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_3_mlp_c = nn.Parameter(torch.zeros([18,4*dim,8*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_3_mlp_v = nn.Parameter(torch.zeros([8*dim,1536],dtype=torch.float),requires_grad=True)
        model.EFFT_4_mlp_u = nn.Parameter(torch.zeros([768,8*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_4_mlp_c = nn.Parameter(torch.zeros([2,8*dim,16*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_4_mlp_v = nn.Parameter(torch.zeros([16*dim,3072],dtype=torch.float),requires_grad=True)
        nn.init.zeros_(model.EFFT_1_mlp_v)
        nn.init.xavier_uniform_(model.EFFT_1_mlp_u)
        nn.init.xavier_uniform_(model.EFFT_1_mlp_c)
        nn.init.zeros_(model.EFFT_2_mlp_v)
        nn.init.xavier_uniform_(model.EFFT_2_mlp_u)
        nn.init.xavier_uniform_(model.EFFT_2_mlp_c)
        nn.init.zeros_(model.EFFT_3_mlp_v)
        nn.init.xavier_uniform_(model.EFFT_3_mlp_u)
        nn.init.xavier_uniform_(model.EFFT_3_mlp_c)
        nn.init.zeros_(model.EFFT_4_mlp_v)
        nn.init.xavier_uniform_(model.EFFT_4_mlp_u)
        nn.init.xavier_uniform_(model.EFFT_4_mlp_c)

        model.layer_idx = 0
    
    for layer in model.layers:
        for block in layer.blocks:
            attn = block.attn
            attn.layer_idx = model.layer_idx
            attn.scale_s = s
            attn.dp = nn.Dropout(0.1)
            bound_method = Swin_EFFT_attn_forward.__get__(attn, attn.__class__)
            setattr(attn, 'forward', bound_method)
            
            mlp = block.mlp
            mlp.layer_idx = model.layer_idx
            mlp.scale_s = s
            mlp.dp = nn.Dropout(0.1)
            bound_method = Swin_EFFT_mlp_forward.__get__(mlp, mlp.__class__)
            setattr(mlp, 'forward', bound_method)
            
            if model.layer_idx == 0:
                attn.EFFT_u = model.EFFT_1_attn_u
                attn.EFFT_v = model.EFFT_1_attn_v
                attn.EFFT_c = model.EFFT_1_attn_c
                mlp.EFFT_u = model.EFFT_1_mlp_u
                mlp.EFFT_v = model.EFFT_1_mlp_v
                mlp.EFFT_c = model.EFFT_1_mlp_c
            elif model.layer_idx == 1:
                attn.EFFT_u = model.EFFT_2_attn_u
                attn.EFFT_v = model.EFFT_2_attn_v
                attn.EFFT_c = model.EFFT_2_attn_c
                mlp.EFFT_u = model.EFFT_2_mlp_u
                mlp.EFFT_v = model.EFFT_2_mlp_v
                mlp.EFFT_c = model.EFFT_2_mlp_c
            elif model.layer_idx == 2:
                attn.EFFT_u = model.EFFT_3_attn_u
                attn.EFFT_v = model.EFFT_3_attn_v
                attn.EFFT_c = model.EFFT_3_attn_c
                mlp.EFFT_u = model.EFFT_3_mlp_u
                mlp.EFFT_v = model.EFFT_3_mlp_v
                mlp.EFFT_c = model.EFFT_3_mlp_c
            elif model.layer_idx == 3:
                attn.EFFT_u = model.EFFT_4_attn_u
                attn.EFFT_v = model.EFFT_4_attn_v
                attn.EFFT_c = model.EFFT_4_attn_c
                mlp.EFFT_u = model.EFFT_4_mlp_u
                mlp.EFFT_v = model.EFFT_4_mlp_v
                mlp.EFFT_c = model.EFFT_4_mlp_c
        model.layer_idx += 1
        
def set_Swin_L_EFFT(model, dim=8, s=1):
    print(s)
    if type(model) == timm.models.swin_transformer.SwinTransformer:
        model.EFFT_1_attn_u = nn.Parameter(torch.zeros([192,dim],dtype=torch.float),requires_grad=True)
        model.EFFT_1_attn_c = nn.Parameter(torch.zeros([4,dim,dim],dtype=torch.float),requires_grad=True)
        model.EFFT_1_attn_v = nn.Parameter(torch.zeros([dim,192],dtype=torch.float),requires_grad=True)
        model.EFFT_2_attn_u = nn.Parameter(torch.zeros([384,2*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_2_attn_c = nn.Parameter(torch.zeros([4,2*dim,2*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_2_attn_v = nn.Parameter(torch.zeros([2*dim,384],dtype=torch.float),requires_grad=True)
        model.EFFT_3_attn_u = nn.Parameter(torch.zeros([768,4*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_3_attn_c = nn.Parameter(torch.zeros([4,4*dim,4*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_3_attn_v = nn.Parameter(torch.zeros([4*dim,768],dtype=torch.float),requires_grad=True)
        model.EFFT_4_attn_u = nn.Parameter(torch.zeros([1536,8*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_4_attn_c = nn.Parameter(torch.zeros([4,8*dim,8*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_4_attn_v = nn.Parameter(torch.zeros([8*dim,1536],dtype=torch.float),requires_grad=True)
        nn.init.zeros_(model.EFFT_1_attn_v)
        nn.init.xavier_uniform_(model.EFFT_1_attn_u)
        nn.init.xavier_uniform_(model.EFFT_1_attn_c)
        nn.init.zeros_(model.EFFT_2_attn_v)
        nn.init.xavier_uniform_(model.EFFT_2_attn_u)
        nn.init.xavier_uniform_(model.EFFT_2_attn_c)
        nn.init.zeros_(model.EFFT_3_attn_v)
        nn.init.xavier_uniform_(model.EFFT_3_attn_u)
        nn.init.xavier_uniform_(model.EFFT_3_attn_c)
        nn.init.zeros_(model.EFFT_4_attn_v)
        nn.init.xavier_uniform_(model.EFFT_4_attn_u)
        nn.init.xavier_uniform_(model.EFFT_4_attn_c)
        
        model.EFFT_1_mlp_u = nn.Parameter(torch.zeros([192,dim],dtype=torch.float),requires_grad=True)
        model.EFFT_1_mlp_c = nn.Parameter(torch.zeros([2,dim,2 * dim],dtype=torch.float),requires_grad=True)
        model.EFFT_1_mlp_v = nn.Parameter(torch.zeros([2 * dim,768],dtype=torch.float),requires_grad=True)
        model.EFFT_2_mlp_u = nn.Parameter(torch.zeros([384,2*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_2_mlp_c = nn.Parameter(torch.zeros([2,2*dim,4*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_2_mlp_v = nn.Parameter(torch.zeros([4*dim,1536],dtype=torch.float),requires_grad=True)
        model.EFFT_3_mlp_u = nn.Parameter(torch.zeros([768,4*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_3_mlp_c = nn.Parameter(torch.zeros([18,4*dim,8*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_3_mlp_v = nn.Parameter(torch.zeros([8*dim,3072],dtype=torch.float),requires_grad=True)
        model.EFFT_4_mlp_u = nn.Parameter(torch.zeros([1536,8*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_4_mlp_c = nn.Parameter(torch.zeros([2,8*dim,16*dim],dtype=torch.float),requires_grad=True)
        model.EFFT_4_mlp_v = nn.Parameter(torch.zeros([16*dim,6144],dtype=torch.float),requires_grad=True)
        nn.init.zeros_(model.EFFT_1_mlp_v)
        nn.init.xavier_uniform_(model.EFFT_1_mlp_u)
        nn.init.xavier_uniform_(model.EFFT_1_mlp_c)
        nn.init.zeros_(model.EFFT_2_mlp_v)
        nn.init.xavier_uniform_(model.EFFT_2_mlp_u)
        nn.init.xavier_uniform_(model.EFFT_2_mlp_c)
        nn.init.zeros_(model.EFFT_3_mlp_v)
        nn.init.xavier_uniform_(model.EFFT_3_mlp_u)
        nn.init.xavier_uniform_(model.EFFT_3_mlp_c)
        nn.init.zeros_(model.EFFT_4_mlp_v)
        nn.init.xavier_uniform_(model.EFFT_4_mlp_u)
        nn.init.xavier_uniform_(model.EFFT_4_mlp_c)

        model.layer_idx = 0
    
    for layer in model.layers:
        for block in layer.blocks:
            attn = block.attn
            attn.layer_idx = model.layer_idx
            attn.scale_s = s
            attn.dp = nn.Dropout(0.1)
            bound_method = Swin_EFFT_attn_forward.__get__(attn, attn.__class__)
            setattr(attn, 'forward', bound_method)
            
            mlp = block.mlp
            mlp.layer_idx = model.layer_idx
            mlp.scale_s = s
            mlp.dp = nn.Dropout(0.1)
            bound_method = Swin_EFFT_mlp_forward.__get__(mlp, mlp.__class__)
            setattr(mlp, 'forward', bound_method)
            
            if model.layer_idx == 0:
                attn.EFFT_u = model.EFFT_1_attn_u
                attn.EFFT_v = model.EFFT_1_attn_v
                attn.EFFT_c = model.EFFT_1_attn_c
                mlp.EFFT_u = model.EFFT_1_mlp_u
                mlp.EFFT_v = model.EFFT_1_mlp_v
                mlp.EFFT_c = model.EFFT_1_mlp_c
            elif model.layer_idx == 1:
                attn.EFFT_u = model.EFFT_2_attn_u
                attn.EFFT_v = model.EFFT_2_attn_v
                attn.EFFT_c = model.EFFT_2_attn_c
                mlp.EFFT_u = model.EFFT_2_mlp_u
                mlp.EFFT_v = model.EFFT_2_mlp_v
                mlp.EFFT_c = model.EFFT_2_mlp_c
            elif model.layer_idx == 2:
                attn.EFFT_u = model.EFFT_3_attn_u
                attn.EFFT_v = model.EFFT_3_attn_v
                attn.EFFT_c = model.EFFT_3_attn_c
                mlp.EFFT_u = model.EFFT_3_mlp_u
                mlp.EFFT_v = model.EFFT_3_mlp_v
                mlp.EFFT_c = model.EFFT_3_mlp_c
            elif model.layer_idx == 3:
                attn.EFFT_u = model.EFFT_4_attn_u
                attn.EFFT_v = model.EFFT_4_attn_v
                attn.EFFT_c = model.EFFT_4_attn_c
                mlp.EFFT_u = model.EFFT_4_mlp_u
                mlp.EFFT_v = model.EFFT_4_mlp_v
                mlp.EFFT_c = model.EFFT_4_mlp_c
        model.layer_idx += 1