from efficientnet_pytorch_edit import EfficientNet
import torch.nn as nn

class MyEfficientNet(nn.Module):
    def __init__(self,efficientnet):
        super().__init__()
        # 수정된 efficientnet
        self.effConv = efficientnet
        # 1/8로 축소돼야 하는 데 1/32로 축소되므로 scale_factor을 4로 줘서 width,height를 4배씩 증가
        self.upsample=nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self,x):
        x = self.effConv(x)
        x=self.upsample(x)
        return x


def effnet(pretrained=True):
    """Constructs a MyEfficientNet model.
    """

    # 1. load a modified EfficientNet B2
    if pretrained:
        efficientnet=EfficientNet.from_pretrained('efficientnet-b2')
    else:
        efficientnet=EfficientNet.fom_name('efficientnet-b2')
    # 2. Construct My model with B2
    model=MyEfficientNet(efficientnet)

    return model
