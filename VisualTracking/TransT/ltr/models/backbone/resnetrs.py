from pytorch_resnet_rs.model import ResnetRS
import torch.nn as nn

class MyResNetRS(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model
        self.upsample=nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

    def forward(self,x):
        x=self.model(x)
        x=self.upsample(x)
        return x

def resnetrs(pretrained=True):
    model=ResnetRS.create_pretrained('resnetrs50', in_ch=3, num_classes=1024,
                           drop_rate=0.)

    # layer4~fc는 사용하지 않음
    model.layer4=nn.Identity()
    model.avg_pool=nn.Identity()
    model.fc=nn.Identity()

    # upsampling 적용
    mymodel=MyResNetRS(model)

    return mymodel 