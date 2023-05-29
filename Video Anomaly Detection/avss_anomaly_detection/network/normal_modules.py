import sys
sys.path.append('./network')

import torch
import math
from torch import nn, einsum
from einops import rearrange
from monai.networks.layers.utils import get_norm_layer
from dynunet_block import get_conv_layer, UnetResBlock


class PatchEmbedding(nn.Module):
    def __init__(self, spatial_dims=3, in_channels=3, out_channels=24, kernel_size=(1,4,4), stride=(1,4,4), dropout=0.0):
        super().__init__()
        self.conv=get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size, stride, dropout, conv_only=True)
        self.norm=get_norm_layer(name=("group", {"num_groups": in_channels}), channels=out_channels)
    
    def forward(self,x):
        x=self.conv(x)
        x=self.norm(x)
        return x
    

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dims=3, kernel_size=(1,2,2), stride=(1,2,2), dropout=0.0):
        super().__init__()
        self.conv=get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size, stride, dropout, conv_only=True)
    
    def forward(self,x):
        x=self.conv(x)
        return x
    

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dims=2, kernel_size=(2,2), stride=(2,2), dropout=0.0):
        super().__init__()
        self.deconv=get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size, stride, dropout, conv_only=True, is_transposed=True)
    
    def forward(self,x):
        x=self.deconv(x)
        return x
    

class VideoPatchMerging(nn.Module):
    def __init__(self, in_channels):
        '''
        we assume that h,w,d are even numbers.
        out_channels = 2 * in_channels.
        '''
        super().__init__()
        self.dim = in_channels
        self.reduction = nn.Linear(4 * in_channels, 2 * in_channels, bias=False)

    def forward(self, x):
        '''
        x: B,C,D,H,W
        '''
        x = x.permute(0,2,3,4,1) # [B, D, H, W, C]

        x0 = x[:, :, 0::2, 0::2, :]  # [B, D, H/2, W/2, C]
        x1 = x[:, :, 1::2, 0::2, :] 
        x2 = x[:, :, 0::2, 1::2, :]  
        x3 = x[:, :, 1::2, 1::2, :]  
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, D, H/2, W/2, 4C]

        x = self.reduction(x) # [B, D, H/2, W/2, 2C]
        x = x.permute(0, 4, 1, 2, 3) # [B, 2C, D, H/2, W/2]

        return x
    

class PatchExpanding(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.expand = nn.Linear(in_channels, 2 * in_channels, bias=False)

    def forward(self, y):
        """
        y: B,C,H,W
        """
        y=y.permute(0,2,3,1) # [B, H, W, C]
        B, H, W, C = y.size()

        y=self.expand(y) # B, H, W, 2*C
    
        y=rearrange(y,'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//2) # B, 2*H, 2*W, C//2

        y=y.permute(0,3,1,2) # B, C//2, 2*H, 2*W
        
        return y
    

class Globalpool(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.pool=nn.AdaptiveMaxPool3d((1, height, width))
    
    def forward(self,x):
        x=self.pool(x) # [B,C,1,H,W]
        x=x.squeeze(2) # [B,C,H,W]
        return x


class DSConv(nn.Module):
    """
    Depthwise seperable convolution. 
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, spatial_dims=3):
        super().__init__()

        if spatial_dims == 3:
            self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, 
                                padding, dilation, groups=in_channels, bias=False)
            self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                                    stride=1, padding=0, dilation=1, groups=1, bias=False)
            self.bn = nn.BatchNorm3d(out_channels, momentum=0.9997, eps=4e-5)

        else:
            self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                                padding, dilation, groups=in_channels, bias=False)
            self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                    stride=1, padding=0, dilation=1, groups=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)

        self.act = nn.ReLU()
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dims=2, kernel_size=3, stride=1, norm_name="instance",depth=1):
        super().__init__()

        self.depth = depth 
        self.resblock_set = nn.ModuleList()

        for i in range(depth):
            if i==0:
                self.resblock_set.append(UnetResBlock(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, 
                                        kernel_size=kernel_size, stride=stride, norm_name=norm_name))
            else:
                self.resblock_set.append(UnetResBlock(spatial_dims=spatial_dims, in_channels=out_channels, out_channels=out_channels, 
                         kernel_size=kernel_size, stride=stride, norm_name=norm_name))
    
    def forward(self,x):
        for i in range(self.depth):
            x = self.resblock_set[i](x)
        return x
    

class ConcatConv(nn.Module):
    def __init__(self, in_channels, depth=1):
        super().__init__()
        '''
        in_channels = C
        '''
        self.conv = ResBlock(in_channels=in_channels*2, out_channels=in_channels, depth=depth)
    
    def forward(self, x1, x2):
        '''
        x1, x2: [B, C, H, W]
        '''
        x = torch.cat((x1,x2),dim=1) # [B, 2C, H, W]
        x = self.conv(x)

        return x
    

class Head(nn.Module):
    def __init__(self, in_channels, out_channels=3, spatial_dims=2, dropout=0.0):
        super().__init__()
        self.conv = ResBlock(in_channels=in_channels, out_channels=in_channels)
        self.head=get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=1, dropout=dropout, bias=True, conv_only=True)
    
    def forward(self,x):
        x=self.conv(x)
        x=self.head(x)
        return x