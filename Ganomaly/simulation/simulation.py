import sys,os
sys.path.append('/home/ahnsunghyun/pytorch/Ganomaly')
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from lib.networks import MyNetG

'''
ucsd_simulation
├── video
│   ├── ped1_24
│   │   └── 0.png
│   │   └── 1.png
│   │   ...
│   │   └── n.png
'''
# Dataloader (no answer)
def make_loader(path):
    splits = ['video']
    shuffle = {'video':False}

    transform = transforms.Compose([transforms.Resize(64),
                                    transforms.CenterCrop(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    dataset = {x: ImageFolder(os.path.join(path, x), transform) for x in splits}
    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                batch_size=64,
                                                shuffle=shuffle[x])
                    for x in splits}
    return dataloader


# Inference
with torch.no_grad():
    # make model
    netg=MyNetG()

    # get weight 
    path = "../output/{}/{}/train/weights/netG.pth".format('ganomaly','custom_ucsd')
    pretrained_dict = torch.load(path)['state_dict']

    # load weight
    try:
        netg.load_state_dict(pretrained_dict)
    except IOError:
        raise IOError("netG weights not found")
    print('Loaded weights.')

    # frame file path
    dir='./ucsd_simulation'

    # make dataloader
    dl=make_loader(dir)

    # inference
    batchsize=64
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    an_scores=torch.zeros(size=(len(dl['video'].dataset),), dtype=torch.float32, device=device)
    reals=[]
    fakes=[]
    for i, data in enumerate(dl['video'], 0):
        # data[0] is batch_tensors
        real=data[0]

        # inference
        fake, latent_i, latent_o = netg(real) 

        # save real and fake tensors
        for b in range(batchsize):
            if i*batchsize+b < 200:
                reals.append(real[b])
                fakes.append(fake[b])

        # get anomaly_scores
        error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1) 
        an_scores[i*batchsize : i*batchsize+error.size(0)] = error.reshape(error.size(0))

    # normalize
    an_scores = (an_scores - torch.min(an_scores)) / (torch.max(an_scores) - torch.min(an_scores))

    print('frame length:',len(an_scores))
    print('inference ok!')


# simulation prepare
# frame_name sorting
frame_names=[]
for f in os.listdir(os.path.join(dir,'video/ped1_24')):
    frame_names.append(f[:-4])
frame_names.sort()

# save frame_path using frame_names 
frame_paths=[]
for i in range(len(frame_names)):
    frame_paths.append(os.path.join(dir,'video/ped1_24',frame_names[i]+'.tif'))

# cv2 text font and color
red= (0, 0, 255)
font =  cv2.FONT_HERSHEY_PLAIN

# simulation start
anomaly=False
for i,fpath in enumerate(frame_paths):
    # check
    if an_scores[i] >= 0.5:
        anomaly=True
        print(i+1,') an_score:',an_scores[i],'(Anomaly) path:[',fpath,']')
    else:
        anomaly=False
        print(i+1,') an_score:',an_scores[i],'(Normal) path:[',fpath,']')

    # show
    img = cv2.imread(fpath,cv2.IMREAD_COLOR)
    if anomaly:  # Anomaly => putText
        img = cv2.putText(img, "Anomaly Frame", (100, 140), font, 2, red, 1, cv2.LINE_AA)
    cv2.imshow('ucsd ped1_24' , img)
    if cv2.waitKey(30) == 27: # esc => close
        break

    # save
    cv2.imwrite('./output/ped1_24/{}.tif'.format(i+1),img)
    
cv2.destroyAllWindows()
