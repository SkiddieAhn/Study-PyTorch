from torchvision_edit import models_e
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self,resnet):
        super().__init__()

        self.resnet=resnet
        self.upsample=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) 
        # output channel number is identical with resnet50 
        self.stage1=nn.Sequential(
            nn.Conv2d(in_channels=2048,out_channels=1024,kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
    
    def forward(self,x):
        x=self.resnet(x)
        x=self.upsample(x)
        x=self.stage1(x)

        return x

def myresnet(pretrained=True):
    model=models_e.resnet50(pretrained=True)
    mymodel=MyModel(model)
    return mymodel 