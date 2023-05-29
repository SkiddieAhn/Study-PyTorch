import sys
sys.path.extend(['./network'])

from typing import List
from timm.models.layers import trunc_normal_
import torch
import math
from torch import nn, einsum
from einops import rearrange
from pytorch_model_summary import summary
from monai.networks.layers.utils import get_norm_layer
from dynunet_block import get_conv_layer, UnetResBlock
from attn_modules import *
from normal_modules import *

'''
==============================================================================================
Proposed Model 2
(Encoder: Three Stream Attention)
(Decoder: Concat, Residual Block)
(Downsampling: Convolution)
==============================================================================================
'''

class AttnStream(nn.Module):
    def __init__(
            self, img_size=[4,256,256], hidden_sizes=[48,96,192,384], depths=[2,2,2,2], num_heads=4,
            kernel_size=(1,2,2), proj_size=None, squeeze=None, dropout=0.1, attn_drop=0.1, proj_drop=0.1, AttnType='Spatial', downsample='conv') -> None:
        super().__init__()

        '''
        If you config kernel_size(1,2,2), you should config below code too!
        '''
        D, H, W = img_size[0]//1, img_size[1]//4, img_size[2]//4 # 4, 64, 64

        if AttnType == 'Spatial':
            input_sizes = [H*W, (H//2)*(W//2), (H//4)*(W//4), (H//8)*(W//8)] # 64*64, 32*32, 16*16, 8*8
            dims = [D*hidden_sizes[0], D*hidden_sizes[1], D*hidden_sizes[2], D*hidden_sizes[3]] # 4*48, 4*96, 4*192, 4*384

        elif AttnType == 'Temporal':
            input_sizes = [D, D, D, D] # 4, 4, 4, 4
            dims = [H*W*hidden_sizes[0], (H//2)*(W//2)*hidden_sizes[1], (H//4)*(W//4)*hidden_sizes[2], (H//8)*(W//8)*hidden_sizes[3]] # 64*64*48, 32*32*96, 16*16*192, 8*8*384
            if proj_size == None:
                proj_size = 64
            if squeeze == None:
                squeeze = 8

        elif AttnType == 'Channel':
            input_sizes = [hidden_sizes[0], hidden_sizes[1], hidden_sizes[2], hidden_sizes[3]] # 48, 96, 192, 384
            dims = [H*W*D, (H//2)*(W//2)*D, (H//4)*(W//4)*D, (H//8)*(W//8)*D] # 64*64*4, 32*32*4, 16*16*4, 8*8*4
            if proj_size == None:
                proj_size = 64
        
        '''
        1. Downsampling
        '''
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            if downsample == 'conv':
                downsample_layer = nn.Sequential(
                    Downsample(in_channels=hidden_sizes[i], out_channels=hidden_sizes[i+1], kernel_size=kernel_size, stride=kernel_size, dropout=dropout),
                    get_norm_layer(name=("group", {"num_groups": hidden_sizes[i]}), channels=2 * hidden_sizes[i])
                )   
            else:
                downsample_layer = nn.Sequential(
                    VideoPatchMerging(in_channels=hidden_sizes[i]),
                    get_norm_layer(name=("group", {"num_groups": hidden_sizes[i]}), channels=2 * hidden_sizes[i])
                )
            self.downsample_layers.append(downsample_layer)

        '''
        2. Attention
        '''
        self.attn_stages = nn.ModuleList()
        for i in range(4):
            # 3 attn_layers per stage
            attn_layers = []
            for _ in range(depths[i]):
                if AttnType == 'Spatial':
                    attn_layers.append(SpatialAttnBlock(conv_hidden=hidden_sizes[i], input_size=input_sizes[i], dim=dims[i], qkv_bias=False, num_heads = num_heads,
                                                        attn_drop=attn_drop,proj_drop=proj_drop,is_pos_embed=True))
                elif AttnType == 'Temporal':
                    attn_layers.append(TemporalAttnBlock(conv_hidden=hidden_sizes[i], input_size=input_sizes[i], dim=dims[i], proj_size = proj_size,
                                                            squeeze = squeeze, qkv_bias=False, num_heads = num_heads, attn_drop=attn_drop, proj_drop=proj_drop, is_pos_embed=True))
                elif AttnType == 'Channel':
                    attn_layers.append(ChannelAttnBlock(conv_hidden=hidden_sizes[i], input_size=input_sizes[i], dim=dims[i], proj_size = proj_size, num_heads = num_heads, qkv_bias=False,
                                                        attn_drop=attn_drop, proj_drop=proj_drop, is_pos_embed=True))
            self.attn_stages.append(nn.Sequential(*attn_layers))

    def forward(self, x):
        stage_output = []

        for i in range(3):
            x = self.attn_stages[i](x)
            stage_output.append(x)
            x = self.downsample_layers[i](x)

        x = self.attn_stages[-1](x)
        stage_output.append(x)

        return stage_output


class Decoder(nn.Module):
    def __init__(
            self, hidden_sizes=[384,192,96,48], out_channels=3, depths=[2,2,2,2], kernel_size=(2,2), dropout=0.1, downsample='conv') -> None:
        super().__init__()
        
        '''
        1. Upsampling
        '''
        self.upsample_layers = nn.ModuleList()
        for i in range(4):
            if i == 3: # kernel_size = (4,4) for future frame prediction
                upsample_layers = nn.Sequential(
                    Upsample(in_channels=hidden_sizes[i], out_channels=hidden_sizes[i]//4, kernel_size=(kernel_size[0]*2, kernel_size[1]*2), stride=(kernel_size[0]*2, kernel_size[1]*2), dropout=dropout)
                ) 
            elif downsample == 'conv':
                upsample_layers = nn.Sequential(
                    Upsample(in_channels=hidden_sizes[i], out_channels=hidden_sizes[i+1], kernel_size=kernel_size, stride=kernel_size, dropout=dropout)
                )   
            else:
                upsample_layers = nn.Sequential(
                    PatchExpanding(in_channels=hidden_sizes[i])
                )
            self.upsample_layers.append(upsample_layers)

        '''
        2. Local-Global Fusion
        '''
        self.lgf_layers = nn.ModuleList()
        for i in range(3):
            self.lgf_layers.append(ConcatConv(in_channels=hidden_sizes[i+1]))

        '''
        3. Head
        '''
        self.out = Head(in_channels=hidden_sizes[-1]//4*2, out_channels=out_channels)

    def forward(self, fusions, x0):
        '''
        fusions = [8x8x384, 16x16x192, 32x32x96, 64x64x48]
        x0 = [256x256x12]
        '''
        x = fusions[0]
        for i in range(3):
            x = self.upsample_layers[i](x) # [192,16,16 -> 96,32,32 -> 48,64,64]
            x = self.lgf_layers[i](x, fusions[i+1]) 

        x = self.upsample_layers[-1](x) # [12, 256, 256]
        x = torch.cat([x,x0], dim=1) # [24, 256, 256]

        x = self.out(x) # [3,256,256]

        return torch.tanh(x)


class PM2(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            in_depths: int = 4,
            img_size: List[int] = [4, 256, 256],
            hidden_sizes: List[int] = [48, 96, 192, 384],
            depths: List[int] = [2,2,2,2],
            num_heads: int = 4,
            patch_size: List[int] = [1, 4, 4],
            kernel_size: List[int] = [1, 2, 2],
            proj_size: int = 64,
            squeeze: int = 8, 
            dropout: float = 0.1,
            attn_drop: float = 0.1,
            proj_drop: float = 0.1,
            downsample: str = 'conv',
            res_norm_name: str = 'instance',
            res_depth: int = 1,
    ) -> None:
        super().__init__()
        '''
        in_channels = input image channels
        in_depths = input image number
        img_size = input imgage resolution and channels
        hidden_size = feature channel size per UNet Stage
        depths = attention module depth per UNet Stage
        num_heads = attention head
        patch_size = patch size for Patch Partition (see. VisionTransformer)
        kernel_size = kerel size for Upsampling and Downsampling
        proj_size = attention projection size
        squeeze = squeezed dim for temporal attention
        dropout = dropout for Upsampling and Downsampling
        attn_drop = dropout for attention
        proj_drop = dropout for attention multi head attention
        downsample = downsample and upsample method (ex. conv, patchmerging)
        res_norm_name = residual block normalization
        res_depth = residual block depth
        '''

        # residual block
        self.residual_block = ResBlock(in_channels=in_channels*in_depths, out_channels=hidden_sizes[0]//4, depth=res_depth) # in_channels = 12, out_channels = 12

        # patch embedding
        self.patch_embedding = PatchEmbedding(spatial_dims=3, in_channels=in_channels, out_channels=hidden_sizes[0], kernel_size=patch_size,
                                              stride=patch_size)
        
        # attention stream for encoder
        self.spatial_attn_stream = AttnStream(img_size=img_size, hidden_sizes=hidden_sizes, depths=depths, num_heads= num_heads, kernel_size=kernel_size,
                                              dropout=dropout, attn_drop=attn_drop, proj_drop=proj_drop, AttnType='Spatial', downsample=downsample)
        self.temporal_attn_stream = AttnStream(img_size=img_size, hidden_sizes=hidden_sizes, depths=depths, num_heads= num_heads, kernel_size=kernel_size,
                                               proj_size = proj_size, squeeze = squeeze, dropout=dropout, attn_drop=attn_drop, proj_drop=proj_drop, AttnType='Temporal', downsample=downsample)
        self.channel_attn_stream = AttnStream(img_size=img_size, hidden_sizes=hidden_sizes, depths=depths, num_heads= num_heads, kernel_size=kernel_size, proj_size = proj_size,
                                              dropout=dropout, attn_drop=attn_drop, proj_drop=proj_drop, AttnType='Channel', downsample=downsample)

        # fusion modules
        self.attn_fusions = nn.ModuleList()
        for i in range(4):
            self.attn_fusions.append(AttentionFusion(in_depths=in_depths, hidden_size=hidden_sizes[i], is_three=True, depth=res_depth, norm_name=res_norm_name)) # 48 -> 96 -> 192 -> 384

        # spatial attention stream for decoder
        r_hidden_sizes = hidden_sizes[::-1].copy() # [384, 192, 96, 48]
        self.decoder = Decoder(hidden_sizes=r_hidden_sizes, out_channels=in_channels, depths=depths,
                               kernel_size=[kernel_size[1], kernel_size[2]], dropout=dropout, downsample=downsample)
        
    def forward(self, x):
        '''
        x: B,C,D,H,W [B,3,4,256,256]
        '''
        x0 = rearrange(x,'b c d h w-> b (c d) h w')
        x0 = self.residual_block(x0)

        x = self.patch_embedding(x)

        spatial_outputs = self.spatial_attn_stream(x)
        temporal_outputs = self.temporal_attn_stream(x)
        channel_outputs = self.channel_attn_stream(x)

        fusions = []
        for i in range(4):
            enc_output = self.attn_fusions[3-i](spatial_outputs[3-i], temporal_outputs[3-i], channel_outputs[3-i])
            fusions.append(enc_output)

        x = self.decoder(fusions, x0)

        return x        


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.zeros(4,3,4,256,256).cuda() # [B,C,D,H,W] 256 x 256 x 4 x 3
    print('Input Shape:',x.shape)
    model = PM2(downsample='conv').to(device)

    print(summary(model,x))
    print(model.parameters)
