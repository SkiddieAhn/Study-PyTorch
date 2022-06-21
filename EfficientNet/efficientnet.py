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
        self.batchN1=nn.BatchNorm2d(3)
        self.network=model.to(device)
        self.batchN2=nn.BatchNorm1d(1000)
        self.lin=nn.Linear(1000,10).to(device)
        
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv(x)),(2,2)).to(device)
        x=self.batchN1(x)
        x=F.relu(self.network(x)).to(device)
        x=self.batchN2(x)
        x=self.lin(x)
        return x

# GPU 설정 
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(device,"run!")

# 데이터 트랜스폼 
train_transform=transforms.Compose([transforms.ToTensor(),
transforms.RandomHorizontalFlip(),
transforms.RandomVerticalFlip(),
transforms.Normalize((0.5,),(0.5,))])

test_transform=transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.5,),(0.5,))])

# trainset, testset 구축 
trainset=datasets.FashionMNIST(root='content',
train=True,download=True,
transform=train_transform)

testset=datasets.FashionMNIST(root='content',
train=False,download=True,
transform=test_transform)
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

# TensorBoard 설정 및 기록
import MyTensorBoard as tb
tb.file_set('runs/fashion_mnist_experiment_1') # 로그 파일 생성 
tb.write(mynet,train_loader,'four_fashion_mnist_images') # 학습 이미지, 모델 기록 
print("make log!")

# weight 파일이 있으면 학습하지 않음 
try:
    path='./fashion_mnist.pth'
    mynet_=MyEfficientNet()
    mynet_.load_state_dict(torch.load(path)) # 없으면 여기서 오류!
    mynet_.to(device)
    print("weight file ok!")

except:
    # 목적 함수 및 옵티마이저
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(mynet.parameters(), lr=0.0001, eps=1e-08)
    print("loss & opt ok! training start!")

    # 학습 (epoch 수:100)
    epoch=3
    batch_size=128 
    steps_per_epoch=len(train_loader) # 배치 사이즈 기준 학습 수 (1 epoch)
    all_step=epoch*steps_per_epoch # 배치 사이즈 기준 학습 수 (100 epoch)

    for epoch in range(epoch):
        running_loss=0.0 # 초기 누적 오차 = 0 
        batch_acc_list=[] # batch 별 accuracy list

        # step (steps_per_epoch:469)
        # 128 (step) 128 (step) 128 (step) .... 469회
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
            # 누적 오차 
            running_loss +=loss.item()

            # batch_accuracy 계산 (한 스텝마다 accuracy 저장)
            _,predicted=torch.max(outputs.data,1)
            correct=(predicted==labels).sum().item()
            batch_acc = correct/batch_size * 100
            batch_acc_list.append(batch_acc)

            # 100 스텝마다 출력 (100x128개 데이터마다 출력)
            if (i+1) % 100 == 0:
                loss=running_loss / steps_per_epoch 
                step=epoch * len(train_loader) + (i+1)
                tb.loss_write(loss,step) # 학습 중 손실(running loss)을 기록
                print('Epoch: {}, Iter: {}, Loss:{}, Step:{}/{}'.format(epoch+1,i+1,loss,step,all_step))
                running_loss=0.0

        # epoch마다 accuracy 출력
        step=epoch * len(train_loader) + (i+1)
        epoch_acc=np.mean(batch_acc_list) # batch_accuracy_list의 평균으로 계산 (전체 데이터를 기준)
        print('Epoch: {}, Acc:{}, Step:{}/{}'.format(epoch+1,epoch_acc, step,all_step))
    
    print("training end!")

    # 모델 저장
    path='./fashion_mnist.pth'
    torch.save(mynet.state_dict(), path)

    # 모델 로드 (저장된 파라미터로 업데이트)
    mynet_=MyEfficientNet()
    mynet_.load_state_dict(torch.load(path)) 
    mynet_.to(device)
    print("model save!")

    
# 모델 테스트
correct=0
total=0
with torch.no_grad():
    for data in test_loader:
        images, labels=data
        images=images.to(device)
        labels=labels.to(device)
        outputs=mynet_(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print("test_acc:",correct/total*100)


# TensorBoard로 학습된 모델 평가 
classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
tb.eval_write(mynet_,test_loader,classes)
print("save log!")
