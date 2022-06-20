import torch 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np

# 나의 모델 
class MyEfficientNet(nn.Module):
    def __init__(self):
        super(MyEfficientNet,self).__init__()
        self.conv=nn.Conv2d(1,3,3).to(device)
        self.network=model.to(device)
        self.lin=nn.Linear(1000,10).to(device)
        
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv(x)),(2,2)).to(device)
        x=F.relu(self.network(x)).to(device)
        x=F.softmax(self.lin(x),dim=1).to(device)
        return x

# GPU 설정 
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(device," run!")

# 데이터 트랜스폼 
transform=transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.5,),(0.5))])

# trainset, testset 구축 
trainset=datasets.FashionMNIST(root='content',
train=True,download=True,
transform=transform)

testset=datasets.FashionMNIST(root='content',
train=False,download=True,
transform=transform)
print("data ok!")

# 데이터 로더 생성
train_loader=DataLoader(trainset,batch_size=128,shuffle=True,num_workers=6)
test_loader=DataLoader(testset,batch_size=128,shuffle=False,num_workers=6)
print("data_loader ok!")


# 모델 생성
model = models.efficientnet_b0(pretrained=True) # 학습된 모델
mynet=MyEfficientNet()
mynet.to(device)
print("model ok!")

# 목적 함수 및 옵티마이저
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(mynet.parameters(), lr=0.0001, momentum=0.9)
print("loss & opt ok! training start!!")

# 학습 (epoch 수:200)
for epoch in range(200):
    running_loss=0.0

    for i, data in enumerate(train_loader,0):
        # 데이터 입력
        inputs, labels=data
        inputs=inputs.to(device)
        labels=labels.to(device)
        # Gradient -> 0
        optimizer.zero_grad()
        # 순전파+역전파+최적화 
        outputs=mynet(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss +=loss.item()

        if i % 100 == 99:
            print('Epoch: {}, Iter: {}, Loss:{}'.format(epoch+1,i+1,running_loss/1000))
            running_loss=0.0

print("training end!!")

# 모델 테스트
correct=0
total=0
with torch.no_grad():
    for data in test_loader:
        images, labels=data
        images=images.to(device)
        labels=labels.to(device)
        outputs=mynet(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

print(correct/total*100)