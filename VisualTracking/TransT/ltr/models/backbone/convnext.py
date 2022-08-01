import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torchvision_edit.models_e as models 

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        # pytorch version >= 0.9.0
        # x = x.permute(0, 2, 3, 1)
        # x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # x = x.permute(0, 3, 1, 2)

        # pytorch version < 0.9.0
        x = x.transpose(1,2)
        x = x.transpose(2,3)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.transpose(2,3)
        x = x.transpose(1,2)
        return x
    
class MyConvNeXt(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model
        self.layernorm=LayerNorm2d((384,), eps=1e-06, elementwise_affine=True)
        self.upsample=nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

    def forward(self,x):
        x=self.model(x)
        x=self.layernorm(x)
        x=self.upsample(x)
        return x
    
def convnext(pretrained=True):
    model=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    # 모델 일부만 사용 
    model.features=nn.Sequential(*list(model.features.children())[0:6])
    model.avgpool=nn.Identity()
    model.classifier=nn.Identity()

    # upsampling 적용
    mymodel=MyConvNeXt(model)

    return mymodel 
    
    