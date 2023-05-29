import ltr.admin.settings as ws_settings
import ltr.models.tracking.transt as transt_models
import ltr.models.backbone as backbones
import torch
import torch.nn as nn
import math,copy
from torchvision_edit import models_e
from efficientnet_pytorch_edit import EfficientNet,utils

class MyEfficientNet(nn.Module):
    def __init__(self,model):
        super().__init__()

        def CBR2d(in_channels,out_channels,kernel_size,stride=1,padding=0,bias=True):
            layers=[]
            layers+=[nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
            kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)]
            layers+=[nn.BatchNorm2d(num_features=out_channels)]
            layers+=[nn.ReLU()]
            conv=nn.Sequential(*layers)
            return conv

        self.downsample=nn.Upsample(scale_factor=1/8, mode='bilinear', align_corners=False) 

        self.upsampleSin=nn.Upsample(scale_factor=(300/256), mode='bicubic', align_corners=False) 
        self.upsampleSout=nn.Upsample(scale_factor=(32/10), mode='bicubic', align_corners=False) 
        self.upsampleTin=nn.Upsample(scale_factor=(128/128), mode='bicubic', align_corners=False) 
        self.upsampleTout=nn.Upsample(scale_factor=(16/4), mode='bicubic', align_corners=False) 

        self.cnl1024=CBR2d(1280,1024,1)

        self.btn1=CBR2d(1024,3,1)
        self.btn2=CBR2d(2*3,6,3,padding=1)
        self.btn3=CBR2d(6,1024,1)

        self.relu=nn.ReLU()
        self.effconv=model

    def forward(self,x):
        # tensor x_width size 
        y=x.size(dim=3) 

        # downsampling (1/8)
        # ds=self.downsample(x)

        # Search Region
        if y==256:
            x=self.upsampleSin(x) 
            x = self.effconv(x)
            x=self.upsampleSout(x) 
        # Template
        elif y==128:
            x=self.upsampleTin(x) 
            x = self.effconv(x) 
            x=self.upsampleTout(x)

        # channel 1280 to 1024
        x=self.cnl1024(x)

        # skip connection with bottleneck
        # btn1=self.btn1(x)
        # cat=torch.cat((btn1,ds),dim=1)
        # btn2=self.btn2(cat)
        # btn3=self.btn3(btn2)
        # x=x+btn3
        # x=self.relu(x)

        return x

def effnet(pretrained=True):
    """Constructs a MyEfficientNet model.
    """
    model=models_e.efficientnet_v2_s(weights=models_e.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model=MyEfficientNet(model)

    return model
