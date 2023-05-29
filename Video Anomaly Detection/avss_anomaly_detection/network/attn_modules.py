import sys
sys.path.append('./network')

import torch
import math
from torch import nn, einsum
from einops import rearrange
from monai.networks.layers.utils import get_norm_layer
from dynunet_block import get_conv_layer, UnetResBlock
from normal_modules import ResBlock, DSConv

'''
==============================================================================================
Attention Fusion
==============================================================================================
'''
class AttentionFusion(nn.Module):
    def __init__(self,in_depths,hidden_size,is_three=True,norm_name="instance",depth=1):
        super().__init__()

        self.is_three=is_three

        if is_three:
            self.out_proj = nn.Linear(hidden_size, int(hidden_size // 3))
            self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 3))
            self.out_proj3 = nn.Linear(hidden_size, int(hidden_size // 3))
        
        else:
            self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
            self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))
        
        self.conv2d = ResBlock(spatial_dims=2, in_channels=hidden_size*in_depths, out_channels=hidden_size, norm_name=norm_name, depth=depth)

    def forward(self,x1,x2,x3=None):
        if self.is_three:
            x1 = rearrange(x1, "b c d h w -> b d h w c")        
            x2 = rearrange(x2, "b c d h w -> b d h w c")   
            x3 = rearrange(x3,"b c d h w -> b d h w c")

            x1 = self.out_proj(x1)
            x2 = self.out_proj2(x2)
            x3 = self.out_proj3(x3)

            x = torch.cat((x1,x2,x3),dim=-1)            

        else:
            x1 = rearrange(x1, "b c d h w -> b d h w c")        
            x2 = rearrange(x2, "b c d h w -> b d h w c")   

            x1 = self.out_proj(x1)
            x2 = self.out_proj2(x2)
            
            x = torch.cat((x1,x2),dim=-1)
        
        x = rearrange(x,"b d h w c -> b c d h w")
        x = rearrange(x,'b c d h w-> b (c d) h w') # [B,CD,H,W]
        x = self.conv2d(x)

        return x
    

'''
==============================================================================================
Attention Block
==============================================================================================
'''
class SpatialAttnBlock(nn.Module):
     def __init__(self, conv_hidden, input_size, dim, num_heads=4, qkv_bias=False, attn_drop=0.1, proj_drop=0.1,is_pos_embed=False):
          '''
          input_size: resolution (H*W)
          dim: channel * depth (C*D)
          '''
          super().__init__()

          self.norm = nn.LayerNorm(dim)
          self.is_pos_embed = is_pos_embed
          self.pos_embed = nn.Parameter(torch.zeros(1, input_size, dim))
          self.spatial_attn = SpatialAttn(input_size, dim, num_heads, qkv_bias, attn_drop, proj_drop)
          self.dsconv = DSConv(in_channels=conv_hidden, out_channels=conv_hidden)

     def forward(self,x):
          '''
          x: [B, C, D, H, W]
          '''
          B, C, D, H, W = x.shape
          save = x
          
          x = rearrange(x,'b c d h w-> b (h w) (c d)', b=B, c=C, d=D, h=H, w=W) # [B,HW,DC]
          if self.is_pos_embed:
               x = x + self.pos_embed

          # spatial attn -> norm
          x = self.norm(self.spatial_attn(x))
          x = rearrange(x,'b (h w) (c d)-> b c d h w', b=B, c=C, d=D, h=H, w=W) # [B,C,D,H,W]
          x = x + save

          # conv -> norm
          x = x + self.dsconv(x)

          return x
     

class TemporalAttnBlock(nn.Module):
     def __init__(self, conv_hidden, input_size, dim, proj_size=64, squeeze=8, num_heads=4, qkv_bias=False, attn_drop=0.1, proj_drop=0.1, is_pos_embed=False):
          '''
          input_size: depth (D)
          dim: resolution * channel (H*W*C)
          '''
          super().__init__()         

          self.norm = nn.LayerNorm(dim)
          self.is_pos_embed = is_pos_embed
          self.pos_embed = nn.Parameter(torch.zeros(1, input_size, dim))
          self.temporal_attn = TemporalAttn(input_size, dim, proj_size, squeeze, num_heads, qkv_bias, attn_drop, proj_drop)
          self.dsconv = DSConv(in_channels=conv_hidden, out_channels=conv_hidden)

     def forward(self,x):
          '''
          x: [B, C, D, H, W]
          '''
          B, C, D, H, W = x.shape
          save = x
          
          x = rearrange(x,'b c d h w-> b d (h w c)', b=B, c=C, d=D, h=H, w=W) # [B,D,HWC]
          if self.is_pos_embed:
            x = x + self.pos_embed

          # temporal attn -> norm
          x = self.norm(self.temporal_attn(x))
          x = rearrange(x,'b d (h w c)-> b c d h w', b=B, c=C, d=D, h=H, w=W) # [B,C,D,H,W]
          x = x + save

          # conv -> norm
          x = x + self.dsconv(x)
        
          return x
          

class SpatioTemporalAttnBlock(nn.Module):
     def __init__(self, conv_hidden, input_size, dim, proj_size=64, num_heads=4, qkv_bias=False, attn_drop=0.1, proj_drop=0.1,is_pos_embed=False):
          '''
          input_size: resolution * depth (H*W*D)
          dim: channel (C)
          '''
          super().__init__()

          self.norm = nn.LayerNorm(dim)
          self.is_pos_embed = is_pos_embed
          self.pos_embed = nn.Parameter(torch.zeros(1, input_size, dim))
          self.spatio_temporal_attn = SpatioTemporalAttn(input_size, dim, proj_size, num_heads, qkv_bias, attn_drop, proj_drop)
          self.dsconv = DSConv(in_channels=conv_hidden, out_channels=conv_hidden)

     def forward(self,x):
          '''
          x: [B, C, D, H, W]
          '''
          B, C, D, H, W = x.shape
          save = x
          
          x = rearrange(x,'b c d h w-> b (h w d) c', b=B, c=C, d=D, h=H, w=W) # [B,HWD,C]
          if self.is_pos_embed:
               x = x + self.pos_embed

          # spatio temporal attn -> norm
          x = self.norm(self.spatio_temporal_attn(x))
          x = rearrange(x,'b (h w d) c-> b c d h w', b=B, c=C, d=D, h=H, w=W) # [B,C,D,H,W]
          x = x + save

          # conv -> norm
          x = x + self.dsconv(x)

          return x
    

class ChannelAttnBlock(nn.Module):
     def __init__(self, conv_hidden, input_size, dim, proj_size=64, num_heads=4, qkv_bias=False, attn_drop=0.1, proj_drop=0.1,is_pos_embed=False):
          '''
          input_size: channel (C)
          dim: resolution * Depth (H*W*D)
          '''
          super().__init__()
        
          self.norm = nn.LayerNorm(dim)
          self.is_pos_embed = is_pos_embed
          self.pos_embed = nn.Parameter(torch.zeros(1, input_size, dim))
          self.channel_attn = ChannelAttn(input_size, dim, proj_size, num_heads, qkv_bias, attn_drop, proj_drop)
          self.dsconv = DSConv(in_channels=conv_hidden, out_channels=conv_hidden)

     def forward(self,x):
          '''
          x: [B, C, D, H, W]
          '''
          B, C, D, H, W = x.shape
          save = x

          x = rearrange(x,'b c d h w-> b c (h w d)', b=B, c=C, d=D, h=H, w=W) # [B,C,HWD]
          if self.is_pos_embed:
               x = x + self.pos_embed
          
          # channel attn -> norm
          x = self.norm(self.channel_attn(x))
          x = rearrange(x,'b c (h w d)-> b c d h w', b=B, c=C, d=D, h=H, w=W) # [B,C,D,H,W]
          x = x + save

          # conv -> norm
          x = x + self.dsconv(x)

          return x
     

class SpatialAttn2DBlock(nn.Module):
     def __init__(self, conv_hidden, input_size, dim, num_heads=4, qkv_bias=False, attn_drop=0.1, proj_drop=0.1,is_pos_embed=False):
          '''
          input_size: resolution (H*W)
          dim: channel (C)
          '''
          super().__init__()

          self.norm = nn.LayerNorm(dim)
          self.is_pos_embed = is_pos_embed
          self.pos_embed = nn.Parameter(torch.zeros(1, input_size, dim))
          self.spatial_attn_2d = SpatialAttn2D(input_size, dim, num_heads, qkv_bias, attn_drop, proj_drop)
          self.dsconv = DSConv(in_channels=conv_hidden, out_channels=conv_hidden, spatial_dims=2)

     def forward(self,x):
          '''
          x: [B, C, H, W]
          '''
          B, C, H, W = x.shape
          save = x
          
          x = rearrange(x,'b c h w-> b (h w) c', b=B, c=C, h=H, w=W) # [B,HW,C]
          if self.is_pos_embed:
               x = x + self.pos_embed

          # spatial attn -> norm
          x = self.norm(self.spatial_attn_2d(x))
          x = rearrange(x,'b (h w) c-> b c h w', b=B, c=C, h=H, w=W) # [B,C,H,W]
          x = x + save

          # conv -> norm
          x = x + self.dsconv(x)

          return x
    

'''
==============================================================================================
Attention Main Function 
==============================================================================================
'''
class SpatialAttn(nn.Module):
    def __init__(self, input_size, dim, num_heads=4, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        '''
        input_size: resolution (H*W)
        dim: channel * depth (C*D)
        '''
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkv are 3 linear layers (query, key, value)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop) 

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):        
        '''
        Spatial Attention
        : no projection 

        x: [B, HW, DC] 
        '''
        B, HW, DC = x.shape 

        qkv = self.qkv(x).reshape(B, HW, 3, self.num_heads, DC // self.num_heads) # B x HW x 3 x h x C/h
        qkv = qkv.permute(2, 0, 3, 1, 4) # 3 x B x h x HW x DC/h
        q, k, v = qkv[0], qkv[1], qkv[2] # B x h x HW x DC/h

        q = torch.nn.functional.normalize(q, dim=-2)
        k = torch.nn.functional.normalize(k, dim=-2)
        k_t = k.permute(0, 1, 3, 2) # K_T : B x h x DC/h x HW

        attn_SA = (q @ k_t) * self.temperature  # [Q x K_T] B x h x HW x HW
        
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop(attn_SA) # [Spatial Attn Map] B x h x HW x HW
        
        # [Spatial Attn Map x V] B x h x HW x DC/h -> B x HW x h x DC/h -> B x HW x DC
        x_SA = (attn_SA @ v).permute(0, 2, 1, 3).reshape(B, HW, DC) 
        
        # linear projection for msa
        x = self.proj(x_SA)
        x = self.proj_drop(x)

        return x
    

class TemporalAttn(nn.Module):
    def __init__(self, input_size, dim, proj_size, squeeze=8, num_heads=4, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        '''
        input_size: depth (D)
        dim: resolution * channel (H*W*C)
        '''
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkv are 3 linear layers (query, key, value)
        # we use bottlenect architecture for efficient calculation!!
        self.qkv = nn.Sequential(
            nn.Linear(dim, squeeze, bias=qkv_bias),
            nn.Linear(squeeze, proj_size, bias=qkv_bias),
            nn.Linear(proj_size,squeeze, bias=qkv_bias),
            nn.Linear(squeeze, 3*dim, bias=qkv_bias),
        )

        # projection matrices with shared weights used in attention module to project
        self.proj_size = proj_size
        self.proj_q = self.proj_k = nn.Linear(dim, proj_size)

        self.attn_drop = nn.Dropout(attn_drop) 

        self.proj = nn.Sequential(
            nn.Linear(dim, squeeze, bias=qkv_bias),
            nn.Linear(squeeze, proj_size, bias=qkv_bias),
            nn.Linear(proj_size,squeeze, bias=qkv_bias),
            nn.Linear(squeeze, dim, bias=qkv_bias),
        )
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):        
        '''
        Temporal Attention
        : Q -> Q(p), K -> K(p) [ Q(p) x K_T(p) ]

        x: [B, D, HWC] 
        '''
        B, D, HWC = x.shape 

        qkv = self.qkv(x).reshape(B, D, 3, HWC).permute(2,0,1,3) # B x D x 3 x HWC -> 3 x B x D x HWC
        q, k, v = qkv[0], qkv[1], qkv[2] # B x D x HWC

        q_projected = self.proj_q(q) # B x D x P
        k_projected = self.proj_k(k) # B x D x p

        q_projected = q_projected.reshape(B, D, self.num_heads, self.proj_size // self.num_heads).permute(0,2,1,3) # B x D x h x P/h -> B x h x D x P/h
        k_projected = k_projected.reshape(B, D, self.num_heads, self.proj_size // self.num_heads).permute(0,2,1,3) # B x D x h x P/h -> B x h x D x P/h
        v = v.reshape(B, D, self.num_heads, self.dim // self.num_heads).permute(0,2,1,3) # B x D x h x HWC/h -> B x h x D x HWC/h

        q_projected = torch.nn.functional.normalize(q_projected, dim=-2)
        k_projected = torch.nn.functional.normalize(k_projected, dim=-2)
        k_t_projected = k_projected.transpose(-2, -1) # K_T : B x h x P/h x D

        attn_TA = (q_projected @ k_t_projected) * self.temperature  # [Q(p) x K_T(p)] B x h x D x D
        
        attn_TA = attn_TA.softmax(dim=-1)
        attn_TA = self.attn_drop(attn_TA) # [Temporal Attn Map] B x h x D x D
        
        # [Temporal Attn Map x V(p)] B x h x D x HWC/h -> B x D x h x HWC/h -> B x D x HWC
        x_TA = (attn_TA @ v).permute(0, 2, 1, 3).reshape(B, D, HWC) 
        
        # linear projection for msa
        x = self.proj(x_TA)
        x = self.proj_drop(x)

        return x
    

class SpatioTemporalAttn(nn.Module):
    def __init__(self, input_size, dim, proj_size, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.1):
        super().__init__()
        '''
        input_size: resolution * depth (H*W*D)
        dim: channel (C)
        '''
        self.num_heads = num_heads
        self.dim = dim
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkv are 3 linear layers (query, key, value)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # projection matrices with shared weights used in attention module to project
        self.proj_k = self.proj_v = nn.Linear(input_size, proj_size)
        self.attn_drop = nn.Dropout(attn_drop) 

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, HWD, C = x.shape 
        
        '''
        Spatio-Temporal Attention
        : K -> K(p), V -> V(p) [ Q x K_T(p) ]
        '''
        qkv = self.qkv(x).reshape(B, HWD, 3, self.num_heads, C // self.num_heads) # B x HWD x 3 x h x C/h
        qkv = qkv.permute(2, 0, 3, 1, 4) # 3 x B x h x HWD x C/h
        q, k, v = qkv[0], qkv[1], qkv[2] # B x h x HWD x C/h

        q = torch.nn.functional.normalize(q, dim=-2)
        k = torch.nn.functional.normalize(k, dim=-2)

        k_t = k.transpose(-2, -1) # B x h x C/h x HWD
        v_t = v.transpose(-2, -1) # B x h x C/h x HWD

        k_t_projected = self.proj_k(k_t) # B x h x C/h x p
        v_t_projected = self.proj_v(v_t) # B x h x C/h x p

        attn_STA = (q @ k_t_projected) * self.temperature  # [Q x K_T(p)] B x h x HWD x p
        
        attn_STA = attn_STA.softmax(dim=-1)
        attn_STA = self.attn_drop(attn_STA) # [Spatial-Temporal Attn Map] B x h x HWD x p
        
        v_projected = v_t_projected.transpose(-2, -1) # V(p) : B x h x p x C/h

        # [Spatio-Temporal Attn Map x V] B x h x HWD x C/h -> B x HWD x h x C/h -> B x HWD x C
        x_STA = (attn_STA @ v_projected).permute(0, 2, 1, 3).reshape(B, HWD, C) 
        
        # linear projection for msa
        x = self.proj(x_STA)
        x = self.proj_drop(x)

        return x
    

class ChannelAttn(nn.Module):
    def __init__(self, input_size, dim, proj_size, num_heads=4, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        '''
        input_size: channel (C)
        dim: resolution * Depth (H*W*D)
        '''
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkv are 3 linear layers (query, key, value)
        self.qkv = nn.Sequential(
            nn.Linear(dim, proj_size, bias=qkv_bias),
            nn.Linear(proj_size, 3*dim, bias=qkv_bias),
        )

        # projection matrices with shared weights used in attention module to project
        self.proj_size = proj_size
        self.proj_q = self.proj_k = nn.Linear(dim, proj_size)

        self.attn_drop = nn.Dropout(attn_drop) 

        self.proj = nn.Sequential(
            nn.Linear(dim, proj_size, bias=qkv_bias),
            nn.Linear(proj_size, dim, bias=qkv_bias),
        )
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):        
        '''
        Channel Attention
        : Q -> Q(p), K -> K(p) [ Q(p) x K_T(p) ]

        x: [B, C, HWD] 
        '''
        B, C, HWD = x.shape 

        qkv = self.qkv(x).reshape(B, C, 3, HWD).permute(2,0,1,3) # B x C x 3 x HWD -> 3 x B x C x HWD
        q, k, v = qkv[0], qkv[1], qkv[2] # B x C x HWD

        q_projected = self.proj_q(q) # B x C x P
        k_projected = self.proj_k(k) # B x C x p

        q_projected = q_projected.reshape(B, C, self.num_heads, self.proj_size // self.num_heads).permute(0,2,1,3) # B x C x h x P/h -> B x h x C x P/h
        k_projected = k_projected.reshape(B, C, self.num_heads, self.proj_size // self.num_heads).permute(0,2,1,3) # B x C x h x P/h -> B x h x C x P/h
        v = v.reshape(B, C, self.num_heads, self.dim // self.num_heads).permute(0,2,1,3) # B x C x h x HWD/h -> B x h x C x HWD/h

        q_projected = torch.nn.functional.normalize(q_projected, dim=-2)
        k_projected = torch.nn.functional.normalize(k_projected, dim=-2)
        k_t_projected = k_projected.transpose(-2, -1) # K_T : B x h x P/h x C

        attn_CA = (q_projected @ k_t_projected)   # [Q(p) x K_T(p)] B x h x C x C
        attn_CA = attn_CA * self.temperature
        
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA) # [Channel Attn Map] B x h x C x C

        # [Channel Attn Map x V(p)] B x h x C x HWD/h -> B x C x h x HWD/h -> B x C x HWD
        x_CA = (attn_CA @ v).permute(0, 2, 1, 3).reshape(B, C, HWD) 
        
        # linear projection for msa
        x = self.proj(x_CA)
        x = self.proj_drop(x)

        return x
    

class SpatialAttn2D(nn.Module):
    def __init__(self, input_size, dim, num_heads=4, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        '''
        input_size: resolution (H*W)
        dim: channel (C)
        '''
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkv are 3 linear layers (query, key, value)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop) 

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):        
        '''
        Spatial Attention
        : no projection 

        x: [B, HW, C] 
        '''
        B, HW, C = x.shape 

        qkv = self.qkv(x).reshape(B, HW, 3, self.num_heads, C // self.num_heads) # B x HW x 3 x h x C/h
        qkv = qkv.permute(2, 0, 3, 1, 4) # 3 x B x h x HW x C/h
        q, k, v = qkv[0], qkv[1], qkv[2] # B x h x HW x C/h

        q = torch.nn.functional.normalize(q, dim=-2)
        k = torch.nn.functional.normalize(k, dim=-2)
        k_t = k.permute(0, 1, 3, 2) # K_T : B x h x C/h x HW

        attn_SA = (q @ k_t) * self.temperature  # [Q x K_T] B x h x HW x HW
        
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop(attn_SA) # [Spatial Attn Map] B x h x HW x HW
        
        # [Spatial Attn Map x V] B x h x HW x C/h -> B x HW x h x DC/h -> B x HW x C
        x_SA = (attn_SA @ v).permute(0, 2, 1, 3).reshape(B, HW, C) 
        
        # linear projection for msa
        x = self.proj(x_SA)
        x = self.proj_drop(x)

        return x