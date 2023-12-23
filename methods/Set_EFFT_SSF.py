import torch
from torch import nn
from typing import Optional
import timm


def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')

def ViT_SSF_PoFT_forward_attn(self, x):
    B, N, C = x.shape
    dropped_c = self.dp(self.PoFTc[:,:,self.idx:self.idx+4])
    dropped_u = self.dp(self.PoFTu)
    dropped_v = self.dp(self.PoFTv)
    q,k,v,o = dropped_u @ dropped_c[:,:,0] @ dropped_v, dropped_u @ dropped_c[:,:,1] @ dropped_v, dropped_u @ dropped_c[:,:,2] @ dropped_v,dropped_u @ dropped_c[:,:,3] @ dropped_v
    x = ssf_ada(x, self.SSF_attn_x_scale, self.SSF_attn_x_shift)
    qkv = self.qkv(x)
    qkv = ssf_ada(qkv, self.SSF_attn_qkv_scale, self.SSF_attn_qkv_shift)
    new_qkv = torch.cat([(x @ q), (x @ k), (x @ v)], dim=2) 
    new_qkv = ssf_ada(new_qkv, self.SSF_attn_poft_qkv_scale, self.SSF_attn_poft_qkv_shift)
    qkv += new_qkv
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
    new_proj = x @ o
    new_proj = ssf_ada(new_proj, self.SSF_attn_poft_o_scale, self.SSF_attn_poft_o_shift)
    proj += new_proj
    x = self.proj_drop(proj)
    return x

def ViT_SSF_PoFT_forward_mlp(self, x):
    B, N, C = x.shape
    PoFT_fc1 = self.dp(self.PoFTu) @ self.dp(self.PoFTc[:,:,2*self.idx]) @ self.dp(self.PoFTv)
    PoFT_fc2 = self.dp(self.PoFTu) @ self.dp(self.PoFTc[:,:,2*self.idx + 1]) @ self.dp(self.PoFTv)
    PoFT_fc2 = PoFT_fc2.transpose(0,1)
    x = ssf_ada(x, self.SSF_ffnx_scale, self.SSF_ffnx_shift)
    h = self.fc1(x)
    h = ssf_ada(h, self.SSF_ffn1_scale, self.SSF_ffn1_shift)
    new_h = x @ PoFT_fc1
    new_h = ssf_ada(new_h, self.SSF_poft_ffn1_scale, self.SSF_poft_ffn1_shift)
    h = h + new_h
    x = self.act(h)
    x = self.drop1(x)
    h = self.fc2(x)
    h = ssf_ada(h, self.SSF_ffn2_scale, self.SSF_ffn2_shift)
    new_h = x @ PoFT_fc2
    new_h = ssf_ada(new_h, self.SSF_poft_ffn2_scale, self.SSF_poft_ffn2_shift)
    h = h + new_h
    x = self.drop2(h)
    return x

def ViT_SSF_forward_PatchEmbed(self, x):
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
    
def ViT_SSF_forward_features(self, x):
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
    
def set_ViT_PoFT_SSF(model, dim=16, root_model=None):
    if root_model is None:
        root_model = model
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.PoFTu1 = nn.Parameter(torch.zeros([768, dim], dtype=torch.float), requires_grad=True)
        model.PoFTv1 = nn.Parameter(torch.zeros([dim, 768], dtype=torch.float), requires_grad=True)
        model.PoFTc1 = nn.Parameter(torch.zeros([dim, dim, 12 * 4], dtype=torch.float), requires_grad=True)
        nn.init.zeros_(model.PoFTv1)
        nn.init.xavier_uniform_(model.PoFTu1)
        nn.init.xavier_uniform_(model.PoFTc1)
        
        model.PoFTu2 = nn.Parameter(torch.zeros([768, dim], dtype=torch.float), requires_grad=True)
        model.PoFTv2 = nn.Parameter(torch.zeros([dim, 4*768], dtype=torch.float), requires_grad=True)
        model.PoFTc2 = nn.Parameter(torch.zeros([dim, dim, 12 * 2], dtype=torch.float), requires_grad=True)
        nn.init.zeros_(model.PoFTv2)
        nn.init.xavier_uniform_(model.PoFTu2)
        nn.init.xavier_uniform_(model.PoFTc2)
        
        model.SSF_ViT_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
        model.SSF_ViT_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
        nn.init.normal_(model.SSF_ViT_scale, mean=1, std=.02)
        nn.init.normal_(model.SSF_ViT_shift, std=.02)
        
        bound_method = ViT_SSF_forward_features.__get__(model, model.__class__)
        setattr(model, 'forward_features', bound_method)
        
        model.idx = 0
        
    for layer in model.children():
        if type(layer) == timm.models.vision_transformer.Attention:
            layer.idx = root_model.idx
            layer.dim = dim
            layer.dp = nn.Dropout(0.1)
            
            layer.SSF_attn_x_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_x_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_qkv_scale = nn.Parameter(torch.zeros([3*768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_qkv_shift = nn.Parameter(torch.zeros([3*768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_o_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_o_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_poft_qkv_scale = nn.Parameter(torch.zeros([3*768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_poft_qkv_shift = nn.Parameter(torch.zeros([3*768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_poft_o_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_attn_poft_o_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            nn.init.normal_(layer.SSF_attn_x_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_x_shift, std=.02)
            nn.init.normal_(layer.SSF_attn_qkv_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_qkv_shift, std=.02)
            nn.init.normal_(layer.SSF_attn_o_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_o_shift, std=.02)
            nn.init.normal_(layer.SSF_attn_poft_qkv_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_poft_qkv_shift, std=.02)
            nn.init.normal_(layer.SSF_attn_poft_o_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_attn_poft_o_shift, std=.02)

            layer.PoFTu = root_model.PoFTu1
            layer.PoFTv = root_model.PoFTv1
            layer.PoFTc = root_model.PoFTc1
            bound_method = ViT_SSF_PoFT_forward_attn.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif type(layer) == timm.models.layers.mlp.Mlp:
            layer.idx = root_model.idx
            layer.dim = dim
            layer.dp = nn.Dropout(0.1)
            root_model.idx += 1

            layer.SSF_ffnx_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_ffnx_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_ffn1_scale = nn.Parameter(torch.zeros([4*768], dtype=torch.float), requires_grad=True)
            layer.SSF_ffn1_shift = nn.Parameter(torch.zeros([4*768], dtype=torch.float), requires_grad=True)
            layer.SSF_ffn2_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_ffn2_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_poft_ffn1_scale = nn.Parameter(torch.zeros([4*768], dtype=torch.float), requires_grad=True)
            layer.SSF_poft_ffn1_shift = nn.Parameter(torch.zeros([4*768], dtype=torch.float), requires_grad=True)
            layer.SSF_poft_ffn2_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            layer.SSF_poft_ffn2_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
            nn.init.normal_(layer.SSF_ffnx_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_ffnx_shift, std=.02)
            nn.init.normal_(layer.SSF_ffn1_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_ffn1_shift, std=.02)
            nn.init.normal_(layer.SSF_ffn2_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_ffn2_shift, std=.02)
            nn.init.normal_(layer.SSF_poft_ffn1_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_poft_ffn1_shift, std=.02)
            nn.init.normal_(layer.SSF_poft_ffn2_scale, mean=1, std=.02)
            nn.init.normal_(layer.SSF_poft_ffn2_shift, std=.02)
            
            layer.PoFTu = root_model.PoFTu2
            layer.PoFTv = root_model.PoFTv2
            layer.PoFTc = root_model.PoFTc2
            
            bound_method = ViT_SSF_PoFT_forward_mlp.__get__(layer, layer.__class__)
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
            bound_method = ViT_SSF_forward_PatchEmbed.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_ViT_PoFT_SSF(layer, dim, root_model)



def Swin_SSF_attn_forward(self, x, mask: Optional[torch.Tensor] = None):
    B_, N, C = x.shape
    x = ssf_ada(x, self.SSF_attn_x_scale, self.SSF_attn_x_shift)
    qkv = self.qkv(x)
    qkv = ssf_ada(qkv, self.SSF_attn_qkv_scale, self.SSF_attn_qkv_shift)
    dropped_c = self.dp(self.PoFTc[:,:,self.idx:self.idx+4])
    dropped_u = self.dp(self.PoFTu)
    dropped_v = self.dp(self.PoFTv)
    q,k,v,o = dropped_u @ dropped_c[:,:,0] @ dropped_v, dropped_u @ dropped_c[:,:,1] @ dropped_v, dropped_u @ dropped_c[:,:,2] @ dropped_v,dropped_u @ dropped_c[:,:,3] @ dropped_v
    new_qkv = torch.cat([(x @ q), (x @ k), (x @ v)], dim=2) 
    new_qkv = ssf_ada(new_qkv, self.SSF_attn_poft_qkv_scale, self.SSF_attn_poft_qkv_shift)
    qkv += new_qkv
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
    new_proj = x @ o
    new_proj = ssf_ada(new_proj, self.SSF_attn_poft_o_scale, self.SSF_attn_poft_o_shift)
    x = self.proj(x)
    x = ssf_ada(x, self.SSF_attn_o_scale, self.SSF_attn_o_shift)
    x = self.proj_drop(x+new_proj)
    return x

def Swin_SSF_mlp_forward(self,x):
    B, N, C = x.shape
    PoFT_fc1 = self.dp(self.PoFTu) @ self.dp(self.PoFTc[:,:,2*self.idx]) @ self.dp(self.PoFTv)
    PoFT_fc2 = self.dp(self.PoFTu) @ self.dp(self.PoFTc[:,:,2*self.idx + 1]) @ self.dp(self.PoFTv)
    PoFT_fc2 = PoFT_fc2.transpose(0,1)  
    x = ssf_ada(x, self.SSF_ffn_x_scale, self.SSF_ffn_x_shift)
    h = self.fc1(x)
    h = ssf_ada(h, self.SSF_ffn_1_scale, self.SSF_ffn_1_shift)
    new_h = x @ PoFT_fc1
    new_h = ssf_ada(new_h, self.SSF_poft_ffn1_scale, self.SSF_poft_ffn1_shift)
    h = h + new_h
    x = self.act(h)
    x = self.drop1(x)
    h = self.fc2(x)
    h = ssf_ada(h, self.SSF_ffn_2_scale, self.SSF_ffn_2_shift)
    new_h = x @ PoFT_fc2
    new_h = ssf_ada(new_h, self.SSF_poft_ffn2_scale, self.SSF_poft_ffn2_shift)
    h = h + new_h
    x = self.drop2(h)
    return x

def Swin_SSF_forward_PatchEmbed(self, x):
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

def Swin_SSF_forward_features(self, x):
    x = self.patch_embed(x)
    if self.absolute_pos_embed is not None:
        x = x + self.absolute_pos_embed
    x = self.pos_drop(x)
    x = self.layers(x)
    x = self.norm(x)  # B L C
    x = ssf_ada(x, self.SSF_scale, self.SSF_shift)
    x = self.avgpool(x.transpose(1, 2))  # B C 1
    x = torch.flatten(x, 1)
    return x
    
def set_Swin_SSF(model,dim=2):
    if type(model) == timm.models.swin_transformer.SwinTransformer: 
        model.PoFT_1_attn_u = nn.Parameter(torch.zeros([128,dim],dtype=torch.float),requires_grad=True)
        model.PoFT_1_attn_c = nn.Parameter(torch.zeros([2*4,dim,dim],dtype=torch.float),requires_grad=True)
        model.PoFT_1_attn_v = nn.Parameter(torch.zeros([dim,128],dtype=torch.float),requires_grad=True)
        model.PoFT_2_attn_u = nn.Parameter(torch.zeros([256,2*dim],dtype=torch.float),requires_grad=True)
        model.PoFT_2_attn_c = nn.Parameter(torch.zeros([2*4,2*dim,2*dim],dtype=torch.float),requires_grad=True)
        model.PoFT_2_attn_v = nn.Parameter(torch.zeros([2*dim,256],dtype=torch.float),requires_grad=True)
        model.PoFT_3_attn_u = nn.Parameter(torch.zeros([512,4*dim],dtype=torch.float),requires_grad=True)
        model.PoFT_3_attn_c = nn.Parameter(torch.zeros([18*4,4*dim,4*dim],dtype=torch.float),requires_grad=True)
        model.PoFT_3_attn_v = nn.Parameter(torch.zeros([4*dim,512],dtype=torch.float),requires_grad=True)
        model.PoFT_4_attn_u = nn.Parameter(torch.zeros([1024,8*dim],dtype=torch.float),requires_grad=True)
        model.PoFT_4_attn_c = nn.Parameter(torch.zeros([2*4,8*dim,8*dim],dtype=torch.float),requires_grad=True)
        model.PoFT_4_attn_v = nn.Parameter(torch.zeros([8*dim,1024],dtype=torch.float),requires_grad=True)
        nn.init.zeros_(model.PoFT_1_attn_v)
        nn.init.xavier_uniform_(model.PoFT_1_attn_u)
        nn.init.xavier_uniform_(model.PoFT_1_attn_c)
        nn.init.zeros_(model.PoFT_2_attn_v)
        nn.init.xavier_uniform_(model.PoFT_2_attn_u)
        nn.init.xavier_uniform_(model.PoFT_2_attn_c)
        nn.init.zeros_(model.PoFT_3_attn_v)
        nn.init.xavier_uniform_(model.PoFT_3_attn_u)
        nn.init.xavier_uniform_(model.PoFT_3_attn_c)
        nn.init.zeros_(model.PoFT_4_attn_v)
        nn.init.xavier_uniform_(model.PoFT_4_attn_u)
        nn.init.xavier_uniform_(model.PoFT_4_attn_c)
        
        model.PoFT_1_mlp_u = nn.Parameter(torch.zeros([128,dim],dtype=torch.float),requires_grad=True)
        model.PoFT_1_mlp_c = nn.Parameter(torch.zeros([2*2,dim,dim],dtype=torch.float),requires_grad=True)
        model.PoFT_1_mlp_v = nn.Parameter(torch.zeros([dim,512],dtype=torch.float),requires_grad=True)
        model.PoFT_2_mlp_u = nn.Parameter(torch.zeros([256,2*dim],dtype=torch.float),requires_grad=True)
        model.PoFT_2_mlp_c = nn.Parameter(torch.zeros([2*2,2*dim,2*dim],dtype=torch.float),requires_grad=True)
        model.PoFT_2_mlp_v = nn.Parameter(torch.zeros([2*dim,1024],dtype=torch.float),requires_grad=True)
        model.PoFT_3_mlp_u = nn.Parameter(torch.zeros([512,4*dim],dtype=torch.float),requires_grad=True)
        model.PoFT_3_mlp_c = nn.Parameter(torch.zeros([2*18,4*dim,4*dim],dtype=torch.float),requires_grad=True)
        model.PoFT_3_mlp_v = nn.Parameter(torch.zeros([4*dim,2048],dtype=torch.float),requires_grad=True)
        model.PoFT_4_mlp_u = nn.Parameter(torch.zeros([1024,8*dim],dtype=torch.float),requires_grad=True)
        model.PoFT_4_mlp_c = nn.Parameter(torch.zeros([2*2,8*dim,8*dim],dtype=torch.float),requires_grad=True)
        model.PoFT_4_mlp_v = nn.Parameter(torch.zeros([8*dim,4096],dtype=torch.float),requires_grad=True)
        nn.init.zeros_(model.PoFT_1_mlp_v)
        nn.init.xavier_uniform_(model.PoFT_1_mlp_u)
        nn.init.xavier_uniform_(model.PoFT_1_mlp_c)
        nn.init.zeros_(model.PoFT_2_mlp_v)
        nn.init.xavier_uniform_(model.PoFT_2_mlp_u)
        nn.init.xavier_uniform_(model.PoFT_2_mlp_c)
        nn.init.zeros_(model.PoFT_3_mlp_v)
        nn.init.xavier_uniform_(model.PoFT_3_mlp_u)
        nn.init.xavier_uniform_(model.PoFT_3_mlp_c)
        nn.init.zeros_(model.PoFT_4_mlp_v)
        nn.init.xavier_uniform_(model.PoFT_4_mlp_u)
        nn.init.xavier_uniform_(model.PoFT_4_mlp_c)
        model.layer_idx = 0
        model.block_idx = 0
        
        for layer in model.layers:
            for block in layer.blocks:
                attn = block.attn
                attn.layer_idx = model.layer_idx
                attn.block_idx = model.block_idx
                attn.SSF_attn_x_scale = nn.Parameter(torch.zeros([attn.dim],dtype=torch.float),requires_grad = True)
                attn.SSF_attn_x_shift = nn.Parameter(torch.zeros([attn.dim],dtype=torch.float),requires_grad = True)
                attn.SSF_attn_qkv_scale = nn.Parameter(torch.zeros([3*attn.dim],dtype=torch.float),requires_grad = True)
                attn.SSF_attn_qkv_shift = nn.Parameter(torch.zeros([3*attn.dim],dtype=torch.float),requires_grad = True)
                attn.SSF_attn_o_scale = nn.Parameter(torch.zeros([attn.dim],dtype=torch.float),requires_grad = True)
                attn.SSF_attn_o_shift = nn.Parameter(torch.zeros([attn.dim],dtype=torch.float),requires_grad = True)
                nn.init.normal_(attn.SSF_attn_x_scale, mean=1, std=.02)
                nn.init.normal_(attn.SSF_attn_x_shift, std=.02)
                nn.init.normal_(attn.SSF_attn_qkv_scale, mean=1, std=.02)
                nn.init.normal_(attn.SSF_attn_qkv_shift, std=.02)
                nn.init.normal_(attn.SSF_attn_o_scale, mean=1, std=.02)
                nn.init.normal_(attn.SSF_attn_o_shift, std=.02)
                
                attn.SSF_attn_poft_qkv_scale = nn.Parameter(torch.zeros([3*768], dtype=torch.float), requires_grad=True)
                attn.SSF_attn_poft_qkv_shift = nn.Parameter(torch.zeros([3*768], dtype=torch.float), requires_grad=True)
                attn.SSF_attn_poft_o_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
                attn.SSF_attn_poft_o_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
                nn.init.normal_(attn.SSF_attn_poft_qkv_scale, mean=1, std=.02)
                nn.init.normal_(attn.SSF_attn_poft_qkv_shift, std=.02)
                nn.init.normal_(attn.SSF_attn_poft_o_scale, mean=1, std=.02)
                nn.init.normal_(attn.SSF_attn_poft_o_shift, std=.02)
            
                attn.dp = nn.Dropout(0.1)
                bound_method = Swin_SSF_attn_forward.__get__(attn, attn.__class__)
                setattr(attn, 'forward', bound_method)
                
                mlp = block.mlp
                mlp.layer_idx = model.layer_idx
                mlp.block_idx = model.block_idx
                mlp.SSF_ffn_x_scale = nn.Parameter(torch.zeros([attn.dim],dtype=torch.float),requires_grad = True)
                mlp.SSF_ffn_x_shift = nn.Parameter(torch.zeros([attn.dim],dtype=torch.float),requires_grad = True)
                mlp.SSF_ffn_1_scale = nn.Parameter(torch.zeros([4*attn.dim],dtype=torch.float),requires_grad = True)
                mlp.SSF_ffn_1_shift = nn.Parameter(torch.zeros([4*attn.dim],dtype=torch.float),requires_grad = True)
                mlp.SSF_ffn_2_scale = nn.Parameter(torch.zeros([attn.dim],dtype=torch.float),requires_grad = True)
                mlp.SSF_ffn_2_shift = nn.Parameter(torch.zeros([attn.dim],dtype=torch.float),requires_grad = True)
                nn.init.normal_(mlp.SSF_ffn_x_scale, mean=1, std=.02)
                nn.init.normal_(mlp.SSF_ffn_x_shift, std=.02)
                nn.init.normal_(mlp.SSF_ffn_1_scale, mean=1, std=.02)
                nn.init.normal_(mlp.SSF_ffn_1_shift, std=.02)
                nn.init.normal_(mlp.SSF_ffn_2_scale, mean=1, std=.02)
                nn.init.normal_(mlp.SSF_ffn_2_shift, std=.02)
                mlp.SSF_poft_ffn1_scale = nn.Parameter(torch.zeros([4*768], dtype=torch.float), requires_grad=True)
                mlp.SSF_poft_ffn1_shift = nn.Parameter(torch.zeros([4*768], dtype=torch.float), requires_grad=True)
                mlp.SSF_poft_ffn2_scale = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
                mlp.SSF_poft_ffn2_shift = nn.Parameter(torch.zeros([768], dtype=torch.float), requires_grad=True)
                nn.init.normal_(mlp.SSF_poft_ffn1_scale, mean=1, std=.02)
                nn.init.normal_(mlp.SSF_poft_ffn1_shift, std=.02)
                nn.init.normal_(mlp.SSF_poft_ffn2_scale, mean=1, std=.02)
                nn.init.normal_(mlp.SSF_poft_ffn2_shift, std=.02)
                mlp.dp = nn.Dropout(0.1)
                bound_method = Swin_SSF_mlp_forward.__get__(mlp, mlp.__class__)
                setattr(mlp, 'forward', bound_method)
                if model.layer_idx == 0:
                    attn.PoFT_u = model.PoFT_1_attn_u
                    attn.PoFT_v = model.PoFT_1_attn_v
                    attn.PoFT_c = model.PoFT_1_attn_c
                    mlp.PoFT_u = model.PoFT_1_mlp_u
                    mlp.PoFT_v = model.PoFT_1_mlp_v
                    mlp.PoFT_c = model.PoFT_1_mlp_c
                elif model.layer_idx == 1:
                    attn.PoFT_u = model.PoFT_2_attn_u
                    attn.PoFT_v = model.PoFT_2_attn_v
                    attn.PoFT_c = model.PoFT_2_attn_c
                    mlp.PoFT_u = model.PoFT_2_mlp_u
                    mlp.PoFT_v = model.PoFT_2_mlp_v
                    mlp.PoFT_c = model.PoFT_2_mlp_c
                elif model.layer_idx == 2:
                    attn.PoFT_u = model.PoFT_3_attn_u
                    attn.PoFT_v = model.PoFT_3_attn_v
                    attn.PoFT_c = model.PoFT_3_attn_c
                    mlp.PoFT_u = model.PoFT_3_mlp_u
                    mlp.PoFT_v = model.PoFT_3_mlp_v
                    mlp.PoFT_c = model.PoFT_3_mlp_c
                elif model.layer_idx == 3:
                    attn.PoFT_u = model.PoFT_4_attn_u
                    attn.PoFT_v = model.PoFT_4_attn_v
                    attn.PoFT_c = model.PoFT_4_attn_c
                    mlp.PoFT_u = model.PoFT_4_mlp_u
                    mlp.PoFT_v = model.PoFT_4_mlp_v
                    mlp.PoFT_c = model.PoFT_4_mlp_c
                model.block_idx += 1
            model.block_idx = 0
            model.layer_idx += 1
        
        patch = model.patch_embed
        patch.SSF_embed_scale = nn.Parameter(torch.zeros([128],dtype=torch.float),requires_grad = True)
        patch.SSF_embed_shift = nn.Parameter(torch.zeros([128],dtype=torch.float),requires_grad = True)
        patch.SSF_embed_norm_scale = nn.Parameter(torch.zeros([128],dtype=torch.float),requires_grad = True)
        patch.SSF_embed_norm_shift = nn.Parameter(torch.zeros([128],dtype=torch.float),requires_grad = True)
        nn.init.normal_(patch.SSF_embed_scale, mean=1, std=.02)
        nn.init.normal_(patch.SSF_embed_shift, std=.02)
        nn.init.normal_(patch.SSF_embed_norm_scale, mean=1, std=.02)
        nn.init.normal_(patch.SSF_embed_norm_shift, std=.02)
        bound_method = Swin_SSF_forward_PatchEmbed.__get__(patch, patch.__class__)
        setattr(patch, 'forward', bound_method)
        
        model.SSF_scale = nn.Parameter(torch.zeros([1024],dtype=torch.float),requires_grad = True)
        model.SSF_shift = nn.Parameter(torch.zeros([1024],dtype=torch.float),requires_grad = True)
        nn.init.normal_(model.SSF_scale, mean=1, std=.02)
        nn.init.normal_(model.SSF_shift, std=.02)
        bound_method = Swin_SSF_forward_features.__get__(model, model.__class__)
        setattr(model, 'forward_features', bound_method)
            