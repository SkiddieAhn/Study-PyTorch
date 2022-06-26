from efficientnet_pytorch_edit import EfficientNet
import torch.nn as nn
import math

class MyEfficientNet(nn.Module):
    def __init__(self,efficientnet,eff_in_size=380,eff_out_chnl=1792):
        super().__init__()

        # EfficientNet Input Size (default:380x380)
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
            x=self.upsampleSin(x) # (256x256) -> (380x380)
            x = self.effConv(x) # (380x380) -> (12x12)
            x=self.upsampleSout(x) # (12x12) -> (32x32)
        # Template
        elif y==self.TemplateImageSize:
            x=self.upsampleTin(x) # (256x256) -> (380x380)
            x = self.effConv(x) # (380x380) -> (12x12)
            x=self.upsampleTout(x) # (12x12) -> (16x16)
            
        # output channel number is identical with resnet50 
        # (32x32x1792) -> (32x32x1024), (16x16x1792) -> (16x16x1024)
        x=self.stage1(x)
        
        # BottleNeck with Residual Connection 
        fx=self.stage2(x) # F(x) 
        x=fx+x  # F(x)+x
        x=self.relu(x)
        
        return x

class MyEfficientNet2(nn.Module):
    def __init__(self,efficientnet):
        super().__init__()
        # EfficientNet Feature Extractor
        self.effConv = efficientnet

        # output channel number is identical with resnet50 
        self.stage=nn.Sequential(
            nn.Conv2d(in_channels=80,out_channels=160,kernel_size=1),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Conv2d(in_channels=160,out_channels=320,kernel_size=3,padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Conv2d(in_channels=320,out_channels=640,kernel_size=1),
            nn.BatchNorm2d(640),
            nn.ReLU(),
            nn.Conv2d(in_channels=640,out_channels=1024,kernel_size=3,padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
    
    def forward(self,x):
        x=self.effConv(x)
        x=self.stage(x)
        return x

def freeze_model(model):
    for ct, child in enumerate(model.children()):
        if isinstance(child,nn.modules.batchnorm.BatchNorm2d):
            for param in child.parameters():
                param.requires_grad = False
    return model

def effnet(pretrained=True):
    """Constructs a MyEfficientNet model.
    """

    # 1. load a modified EfficientNet B3
    if pretrained:
        efficientnet=EfficientNet.from_pretrained('efficientnet-b7')
    else:
        efficientnet=EfficientNet.fom_name('efficientnet-b7')

    efficientnet._blocks=nn.Sequential(*list(efficientnet._blocks.children())[:-37])
    efficientnet._conv_head=nn.Identity()
    efficientnet._bn1=nn.Identity()
    
    # 2. freeze EfficientNet
    efficientnet=freeze_model(efficientnet)

    # 3. Construct My model with B3
    model=MyEfficientNet2(efficientnet)

    return model
