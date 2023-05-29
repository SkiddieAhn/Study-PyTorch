import torch
import torch.nn as nn
from .resnet_cbam import ResidualNet
import os
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls


# 딕셔너리 키 삭제 
def remove(d):
    remove_list=['layer4.0.conv1.weight', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.conv2.weight', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.conv3.weight', 'layer4.0.bn3.running_mean', 'layer4.0.bn3.running_var', 'layer4.0.bn3.weight', 'layer4.0.bn3.bias', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.1.conv1.weight', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.conv2.weight', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.conv3.weight', 'layer4.1.bn3.running_mean', 'layer4.1.bn3.running_var', 'layer4.1.bn3.weight', 'layer4.1.bn3.bias', 'layer4.2.conv1.weight', 'layer4.2.bn1.running_mean', 'layer4.2.bn1.running_var', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.conv2.weight', 'layer4.2.bn2.running_mean', 'layer4.2.bn2.running_var', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.conv3.weight', 'layer4.2.bn3.running_mean', 'layer4.2.bn3.running_var', 'layer4.2.bn3.weight', 'layer4.2.bn3.bias', 'fc.weight', 'fc.bias']
    for i in range(len(remove_list)):
        key=remove_list[i]
        del d[key]
    return d


# 딕셔너리 아이템 추가
def add(d, d2):
    add_list=["layer1.0.cbam.ChannelGate.mlp.1.weight", "layer1.0.cbam.ChannelGate.mlp.1.bias", "layer1.0.cbam.ChannelGate.mlp.3.weight", "layer1.0.cbam.ChannelGate.mlp.3.bias", "layer1.0.cbam.SpatialGate.spatial.conv.weight", "layer1.0.cbam.SpatialGate.spatial.bn.weight", "layer1.0.cbam.SpatialGate.spatial.bn.bias", "layer1.0.cbam.SpatialGate.spatial.bn.running_mean", "layer1.0.cbam.SpatialGate.spatial.bn.running_var", "layer1.1.cbam.ChannelGate.mlp.1.weight", "layer1.1.cbam.ChannelGate.mlp.1.bias", "layer1.1.cbam.ChannelGate.mlp.3.weight", "layer1.1.cbam.ChannelGate.mlp.3.bias", "layer1.1.cbam.SpatialGate.spatial.conv.weight", "layer1.1.cbam.SpatialGate.spatial.bn.weight", "layer1.1.cbam.SpatialGate.spatial.bn.bias", "layer1.1.cbam.SpatialGate.spatial.bn.running_mean", "layer1.1.cbam.SpatialGate.spatial.bn.running_var", "layer1.2.cbam.ChannelGate.mlp.1.weight", "layer1.2.cbam.ChannelGate.mlp.1.bias", "layer1.2.cbam.ChannelGate.mlp.3.weight", "layer1.2.cbam.ChannelGate.mlp.3.bias", "layer1.2.cbam.SpatialGate.spatial.conv.weight", "layer1.2.cbam.SpatialGate.spatial.bn.weight", "layer1.2.cbam.SpatialGate.spatial.bn.bias", "layer1.2.cbam.SpatialGate.spatial.bn.running_mean", "layer1.2.cbam.SpatialGate.spatial.bn.running_var", "layer2.0.cbam.ChannelGate.mlp.1.weight", "layer2.0.cbam.ChannelGate.mlp.1.bias", "layer2.0.cbam.ChannelGate.mlp.3.weight", "layer2.0.cbam.ChannelGate.mlp.3.bias", "layer2.0.cbam.SpatialGate.spatial.conv.weight", "layer2.0.cbam.SpatialGate.spatial.bn.weight", "layer2.0.cbam.SpatialGate.spatial.bn.bias", "layer2.0.cbam.SpatialGate.spatial.bn.running_mean", "layer2.0.cbam.SpatialGate.spatial.bn.running_var", "layer2.1.cbam.ChannelGate.mlp.1.weight", "layer2.1.cbam.ChannelGate.mlp.1.bias", "layer2.1.cbam.ChannelGate.mlp.3.weight", "layer2.1.cbam.ChannelGate.mlp.3.bias", "layer2.1.cbam.SpatialGate.spatial.conv.weight", "layer2.1.cbam.SpatialGate.spatial.bn.weight", "layer2.1.cbam.SpatialGate.spatial.bn.bias", "layer2.1.cbam.SpatialGate.spatial.bn.running_mean", "layer2.1.cbam.SpatialGate.spatial.bn.running_var", "layer2.2.cbam.ChannelGate.mlp.1.weight", "layer2.2.cbam.ChannelGate.mlp.1.bias", "layer2.2.cbam.ChannelGate.mlp.3.weight", "layer2.2.cbam.ChannelGate.mlp.3.bias", "layer2.2.cbam.SpatialGate.spatial.conv.weight", "layer2.2.cbam.SpatialGate.spatial.bn.weight", "layer2.2.cbam.SpatialGate.spatial.bn.bias", "layer2.2.cbam.SpatialGate.spatial.bn.running_mean", "layer2.2.cbam.SpatialGate.spatial.bn.running_var", "layer2.3.cbam.ChannelGate.mlp.1.weight", "layer2.3.cbam.ChannelGate.mlp.1.bias", "layer2.3.cbam.ChannelGate.mlp.3.weight", "layer2.3.cbam.ChannelGate.mlp.3.bias", "layer2.3.cbam.SpatialGate.spatial.conv.weight", "layer2.3.cbam.SpatialGate.spatial.bn.weight", "layer2.3.cbam.SpatialGate.spatial.bn.bias", "layer2.3.cbam.SpatialGate.spatial.bn.running_mean", "layer2.3.cbam.SpatialGate.spatial.bn.running_var", "layer3.0.cbam.ChannelGate.mlp.1.weight", "layer3.0.cbam.ChannelGate.mlp.1.bias", "layer3.0.cbam.ChannelGate.mlp.3.weight", "layer3.0.cbam.ChannelGate.mlp.3.bias", "layer3.0.cbam.SpatialGate.spatial.conv.weight", "layer3.0.cbam.SpatialGate.spatial.bn.weight", "layer3.0.cbam.SpatialGate.spatial.bn.bias", "layer3.0.cbam.SpatialGate.spatial.bn.running_mean", "layer3.0.cbam.SpatialGate.spatial.bn.running_var", "layer3.1.cbam.ChannelGate.mlp.1.weight", "layer3.1.cbam.ChannelGate.mlp.1.bias", "layer3.1.cbam.ChannelGate.mlp.3.weight", "layer3.1.cbam.ChannelGate.mlp.3.bias", "layer3.1.cbam.SpatialGate.spatial.conv.weight", "layer3.1.cbam.SpatialGate.spatial.bn.weight", "layer3.1.cbam.SpatialGate.spatial.bn.bias", "layer3.1.cbam.SpatialGate.spatial.bn.running_mean", "layer3.1.cbam.SpatialGate.spatial.bn.running_var", "layer3.2.cbam.ChannelGate.mlp.1.weight", "layer3.2.cbam.ChannelGate.mlp.1.bias", "layer3.2.cbam.ChannelGate.mlp.3.weight", "layer3.2.cbam.ChannelGate.mlp.3.bias", "layer3.2.cbam.SpatialGate.spatial.conv.weight", "layer3.2.cbam.SpatialGate.spatial.bn.weight", "layer3.2.cbam.SpatialGate.spatial.bn.bias", "layer3.2.cbam.SpatialGate.spatial.bn.running_mean", "layer3.2.cbam.SpatialGate.spatial.bn.running_var", "layer3.3.cbam.ChannelGate.mlp.1.weight", "layer3.3.cbam.ChannelGate.mlp.1.bias", "layer3.3.cbam.ChannelGate.mlp.3.weight", "layer3.3.cbam.ChannelGate.mlp.3.bias", "layer3.3.cbam.SpatialGate.spatial.conv.weight", "layer3.3.cbam.SpatialGate.spatial.bn.weight", "layer3.3.cbam.SpatialGate.spatial.bn.bias", "layer3.3.cbam.SpatialGate.spatial.bn.running_mean", "layer3.3.cbam.SpatialGate.spatial.bn.running_var", "layer3.4.cbam.ChannelGate.mlp.1.weight", "layer3.4.cbam.ChannelGate.mlp.1.bias", "layer3.4.cbam.ChannelGate.mlp.3.weight", "layer3.4.cbam.ChannelGate.mlp.3.bias", "layer3.4.cbam.SpatialGate.spatial.conv.weight", "layer3.4.cbam.SpatialGate.spatial.bn.weight", "layer3.4.cbam.SpatialGate.spatial.bn.bias", "layer3.4.cbam.SpatialGate.spatial.bn.running_mean", "layer3.4.cbam.SpatialGate.spatial.bn.running_var", "layer3.5.cbam.ChannelGate.mlp.1.weight", "layer3.5.cbam.ChannelGate.mlp.1.bias", "layer3.5.cbam.ChannelGate.mlp.3.weight", "layer3.5.cbam.ChannelGate.mlp.3.bias", "layer3.5.cbam.SpatialGate.spatial.conv.weight", "layer3.5.cbam.SpatialGate.spatial.bn.weight", "layer3.5.cbam.SpatialGate.spatial.bn.bias", "layer3.5.cbam.SpatialGate.spatial.bn.running_mean", "layer3.5.cbam.SpatialGate.spatial.bn.running_var"]
    for i in range(len(add_list)):
        key=add_list[i]
        key2="module."+add_list[i]
        d[key]=d2[key2]
    return d


# 모델 생성 및 반환
def myresnet(pretrained=True):
    # 모델 정의 (수정된 ResNet50)
    model = ResidualNet("ImageNet",50,1000,'CBAM')

    if pretrained:
        # weight 불러오기 (torchvision)
        d=model_zoo.load_url(model_urls['resnet50'])
        
        # weigtht 불러오기 (CBAM 논문)
        path='/home/seinkwon/ahnsunghyun/TransT2/ltr/models/backbone/cbam_weight/RESNET50_CBAM.pth'
        checkpoint = torch.load(path)
        d2=checkpoint['state_dict']
        
        # CBAM weight 추가
        d=add(d,d2)
        
        # 학습된 weight 일부 삭제
        d=remove(d)

        # 학습된 모델 로드
        model.load_state_dict(d)
        print('=========================================')
        print('pretrained resnet_cbam...ok!')
        print('=========================================')
    
    return model 