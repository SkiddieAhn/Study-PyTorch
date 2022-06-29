import ltr.admin.settings as ws_settings
import ltr.models.tracking.transt as transt_models
import ltr.models.backbone as backbones
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_model_summary
import math,copy
from torchvision_edit import models_e
from torchvision_edit.ops.misc import Conv2dNormActivation
from efficientnet_pytorch_edit import EfficientNet,utils

class MyEfficientNet(nn.Module):
    def __init__(self,efficientnet,eff_in_size=600,eff_out_chnl=2560):
        super().__init__()

        # EfficientNet Input Size (default:600x600)
        self.EffInSize=eff_in_size
        # EfficientNet Output Size (default:12x12)
        self.EffOutSize=math.ceil(eff_in_size/32)

        # Search Region Size
        self.SearchImageSize=256
        self.SearchFeatureSize=32
        # Template Size
        self.TemplateImageSize=128
        self.TemplateFeatureSize=16
        
        # upsampling for identical resolution
        self.upsampleSin=nn.Upsample(scale_factor=(self.EffInSize/self.SearchImageSize), mode='bilinear', align_corners=False) 
        self.upsampleSout=nn.Upsample(scale_factor=(self.SearchFeatureSize/self.EffOutSize), mode='bilinear', align_corners=False) 
        self.upsampleTin=nn.Upsample(scale_factor=(self.EffInSize/self.TemplateImageSize), mode='bilinear', align_corners=False) 
        self.upsampleTout=nn.Upsample(scale_factor=(self.TemplateFeatureSize/self.EffOutSize), mode='bilinear', align_corners=False) 

        # EfficientNet Feature Extractor
        self.effConv = efficientnet

        # output channel number is identical with resnet50 
        self.stage1=nn.Sequential(
            nn.Conv2d(in_channels=eff_out_chnl,out_channels=1024,kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        # BottleNeck
        self.stage2=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=1),
            nn.BatchNorm2d(1024)
        )
        self.relu=nn.ReLU()

    def forward(self,x):
        # tensor x_width size 
        y=x.size(dim=3) 

        # Search Region
        if y == self.SearchImageSize:
            x=self.upsampleSin(x) # (256x256) -> (600x600)
            x = self.effConv(x) # (600x600) -> (76x76)
            x=self.upsampleSout(x) # (19x19) -> (32x32)
        # Template
        elif y==self.TemplateImageSize:
            x=self.upsampleTin(x) # (128x128) -> (600x600)
            x = self.effConv(x) # (600x600) -> (19x19)
            x=self.upsampleTout(x) # (19x19) -> (16x16)
            
        # output channel number is identical with resnet50 
        # (32x32x2560) -> (32x32x1024), (16x16x2560) -> (16x16x1024)
        x=self.stage1(x)
        
        # BottleNeck with Residual Conn256ection 
        fx=self.stage2(x) # F(x) 
        x=fx+x  # F(x)+x
        x=self.relu(x)
        
        return x


def effnet(pretrained=True):
    """Constructs a MyEfficientNet model.
    """

    # 1. load a modified EfficientNet B7
    if pretrained:
        model=EfficientNet.from_pretrained('efficientnet-b7')
    else:
        model=EfficientNet.fom_name('efficientnet-b7')

    # conv2d (same padding)
    Conv2dSame = utils.get_same_padding_conv2d()

    # config efficientnet b7
    for ct, child in enumerate(model.children()):
        if type(child) == nn.modules.container.ModuleList:
            for gct, gchild in enumerate(child.children()):
                if gct==4:
                    gchild._depthwise_conv=Conv2dSame(in_channels=192, out_channels=192, kernel_size=1, dilation=2, bias=False)
                elif gct==11:
                    gchild._depthwise_conv=Conv2dSame(in_channels=288, out_channels=288, kernel_size=1, dilation=2, bias=False)
    model._conv_head=Conv2dSame(in_channels=640, out_channels=1024, kernel_size=1, bias=False)
    model._bn1=nn.BatchNorm2d(1024, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)

    # 2. Construct My model with B7
    model=MyEfficientNet(model)

    return model
