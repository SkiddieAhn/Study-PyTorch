import numpy as np
import os
import time
import torch
import argparse
import cv2
from PIL import Image
import io
from sklearn import metrics
import matplotlib.pyplot as plt

from config import update_config
from Dataset import Label_loader
from utils import psnr_error
import Dataset
from network.unet import UNet
from network.proposed_model1 import PM1
from network.proposed_model2 import PM2
from multigpu import MultiGPU
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--generator', default='pm1', type=str, help='select generator network (pm1: proposed model 1, pm2: proposed model 2)')
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--trained_model', default=None, type=str, help='The pre-trained model to evaluate.')
parser.add_argument('--show_curve', action='store_true',
                    help='Show and save the psnr curve real-timely, this drops fps.')

def val(cfg, model=None):
    if model:  # This is for testing during training.
        generator = model
        generator.eval()
    else:
        # input size
        train_img_size=[4,cfg.img_size[0],cfg.img_size[1]]

        # generator
        if cfg.generator == 'pm1':
            generator =  MultiGPU(PM1(img_size=train_img_size, hidden_sizes=[48,96,192,384], downsample='pm'), dim=0).cuda()
        else:
            generator =  MultiGPU(PM2(img_size=train_img_size, hidden_sizes=[48,96,192,384], downsample='conv'), dim=0).cuda()

        generator.load_state_dict(torch.load('weights/' + cfg.trained_model)['net_g'])
        print(f'The pre-trained generator has been loaded from \'weights/{cfg.trained_model}\'.\n')

    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()
    video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]

    fps = 0
    psnr_group = []
    psnr_sum_group = []
    psnr_avg_group = []

    dataset_name = cfg.dataset

    with torch.no_grad():
        for i, folder in enumerate(video_folders):
            dataset = Dataset.test_dataset(cfg, folder)

            if not os.path.exists(f"results/{dataset_name}/f{i+1}"):
                os.makedirs(f"results/{dataset_name}/f{i+1}")

            psnrs = []
            save_num = 0

            for j, clip in enumerate(dataset):
                frame_1 = clip[0:3, :, :].unsqueeze(1).cuda()  # (3, 1, 64, 64) 
                frame_2 = clip[3:6, :, :].unsqueeze(1).cuda()  # (3, 1, 64, 64) 
                frame_3 = clip[6:9, :, :].unsqueeze(1).cuda()  # (3, 1, 64, 64) 
                frame_4 = clip[9:12, :, :].unsqueeze(1).cuda()  # (3, 1, 64, 64) 
                input_np = torch.cat([frame_1,frame_2, frame_3, frame_4], 1) # (3, 4, 64, 64) 
                input_frames = input_np.unsqueeze(0).cuda() # (1, 3, 4, 64, 64)

                target_np = clip[12:15, :, :] # (3, 64, 64)
                target_frame = target_np.unsqueeze(0).cuda() # (1, 3, 64, 64)

                G_frame = generator(input_frames)
                test_psnr = psnr_error(G_frame, target_frame).cpu().detach().numpy()
                psnrs.append(float(test_psnr))

                save_num=save_num+1

                torch.cuda.synchronize()
                end = time.time()
                if j > 1:  # Compute fps by calculating the time used in one completed iteration, this is more accurate.
                    fps = 1 / (end - temp)
                temp = end
                print(f'\rDetecting: [{i + 1:02d}] {j + 1}/{len(dataset)}, {fps:.2f} fps.', end='')

            psnr_group.append(np.array(psnrs))
            psnr_sum_group.append(np.sum(np.array(psnrs)))
            psnr_avg_group.append(np.mean(np.array(psnrs)))
    
    print()
    for i in range(len(psnr_sum_group)):
        print(i+1,': ',psnr_sum_group[i],'(Sum)',psnr_avg_group[i],'(Avg)')
    print(np.sum(psnr_sum_group),'(Sum)',np.mean(psnr_avg_group),'(Avg)')

    print('\nAll frames were detected, begin to compute AUC.')

    gt_loader = Label_loader(cfg, video_folders)  # Get gt labels.
    gt = gt_loader()

    assert len(psnr_group) == len(gt), f'Ground truth has {len(gt)} videos, but got {len(psnr_group)} detected videos.'

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)

    for i in range(len(psnr_group)):
        distance = psnr_group[i]

        distance -= min(distance)  # distance = (distance - min) / (max - min)
        distance /= max(distance)

        scores = np.concatenate((scores, distance), axis=0)
        labels = np.concatenate((labels, gt[i][4:]), axis=0)  # Exclude the first 4 unpredictable frames in gt.

    assert scores.shape == labels.shape, \
        f'Ground truth has {labels.shape[0]} frames, but got {scores.shape[0]} detected frames.'

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    print(f'AUC: {auc}\n')
    return auc


if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    val(test_cfg)

