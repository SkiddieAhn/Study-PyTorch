import os
from glob import glob
import cv2
import time
import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import random

from multigpu import MultiGPU
from utils import *
from losses import *
import Dataset
from network.unet import UNet
from network.proposed_model1 import PM1
from network.proposed_model2 import PM2
from network.non_generator.pix2pix_networks import PixelDiscriminator
from network.non_generator.liteFlownet import lite_flownet as lite_flow
from network.non_generator.flownet2.models import FlowNet2SD
from config import update_config
from show_gpu import showGPU


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def load_model_dict(sch, generator, discriminator, optimizer_G, optimizer_D, sch_G=None, sch_D=None):
    if sch:
        model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                    'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict(),
                    'sch_G': sch_G.state_dict(), 'sch_D': sch_D.state_dict()}
    else:
        model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                    'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
        
    return model_dict

def main():
    parser = argparse.ArgumentParser(description='Anomaly Prediction')
    parser.add_argument('--generator', default='pm1', type=str, help='select generator network (pm1: proposed model 1, pm2: proposed model 2)')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
    parser.add_argument('--iters', default=100000, type=int, help='The total iteration number.')
    parser.add_argument('--resume', default=None, type=str,
                        help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
    parser.add_argument('--save_interval', default=1000, type=int, help='Save the model every [save_interval] iterations.')
    parser.add_argument('--flownet', default='none', type=str, help='load flownet and use flownet loss too! (lite: LiteFlownet, 2sd: FlowNet2SD)')
    parser.add_argument('--work_num', default=0, type=int)
    parser.add_argument('--save_dir', default='sha', type=str, help='model save directory')
    parser.add_argument('--sch', default=True, type=str2bool, nargs='?', const=True, help='scheduler')

    args = parser.parse_args()
    train_cfg = update_config(args, mode='train')


    '''
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Data Loader
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''

    train_dataset = Dataset.train_dataset(train_cfg)

    # Remember to set drop_last=True, because we need to use 4 frames to predict one frame.
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size,
                                shuffle=True, num_workers=4, drop_last=True)
    train_cfg.epoch_size = train_cfg.iters // len(train_dataloader)
    train_cfg.print_cfg() # print config class!

    print('\n====================================================')
    print('Dataloader Ok!')
    print('----------------------------------------------------')
    print('[Data Size]:',len(train_dataloader.dataset))
    print('[Batch Size]:',train_cfg.batch_size)
    print('[One epoch]:',len(train_dataloader),'step   # (Data Size / Batch Size)')
    print('[Epoch & Iteration]:',train_cfg.epoch_size,'epoch &', train_cfg.iters,'step')
    print('----------------------------------------------------')
    print('====================================================')

    '''
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Model 
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''

    # input size
    train_img_size=[4,train_cfg.img_size[0],train_cfg.img_size[1]]

    # define model
    if train_cfg.generator == 'pm1':
        generator =  MultiGPU(PM1(img_size=train_img_size, hidden_sizes=[48,96,192,384], downsample='pm'), dim=0).cuda()
    else:
        generator =  MultiGPU(PM2(img_size=train_img_size, hidden_sizes=[48,96,192,384], downsample='conv'), dim=0).cuda()
    discriminator = PixelDiscriminator(input_nc=3).cuda()

    # define optimizer and scheduler
    optimizer_G = torch.optim.AdamW(generator.parameters(), lr=train_cfg.g_lr, weight_decay=train_cfg.l2)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=train_cfg.d_lr, weight_decay=train_cfg.l2)
    if train_cfg.sch:
        sch_G = torch.optim.lr_scheduler.PolynomialLR(optimizer=optimizer_G, total_iters=train_cfg.epoch_size)
        sch_D = torch.optim.lr_scheduler.PolynomialLR(optimizer=optimizer_D, total_iters=train_cfg.epoch_size)
    else:
        sch_G = sch_D = None

    # load model 
    if train_cfg.resume:
        generator.load_state_dict(torch.load(train_cfg.resume)['net_g'])
        discriminator.load_state_dict(torch.load(train_cfg.resume)['net_d'])
        optimizer_G.load_state_dict(torch.load(train_cfg.resume)['optimizer_g'])
        optimizer_D.load_state_dict(torch.load(train_cfg.resume)['optimizer_d'])
        if train_cfg.sch:
            sch_G.load_state_dict(torch.load(train_cfg.resume)['sch_G'])
            sch_D.load_state_dict(torch.load(train_cfg.resume)['sch_D'])
    else:
        generator.apply(proposed_weights_init)
        discriminator.apply(weights_init_normal)

    # define loss
    adversarial_loss = Adversarial_Loss().cuda()
    discriminate_loss = Discriminate_Loss().cuda()
    gradient_loss = Gradient_Loss(3).cuda()
    intensity_loss = Intensity_Loss().cuda()

    print('\n====================================================')
    print(f'Generator({train_cfg.generator}), Discriminator Ok!')
    print('====================================================')

    # define and load flownet
    if train_cfg.flownet != 'none':
        # flownet2
        if train_cfg.flownet == '2sd':
            flownet = FlowNet2SD().cuda()
            flownet.load_state_dict(torch.load('pretrained_flownet/FlowNet2-SD.pth')['state_dict'])

        # liteflow
        else:
            flownet = lite_flow.Network().cuda()
            flownet.load_state_dict(torch.load('pretrained_flownet/network-default.pytorch'))
        flownet.eval()
        flow_loss = Flow_Loss().cuda()

        print('\n====================================================')
        print(f'Pretrained FlowNet({train_cfg.flownet}) Ok!')
        print('====================================================')


    '''
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Training
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''

    # define writer
    writer = SummaryWriter(f'tensorboard_log/{train_cfg.dataset}_bs{train_cfg.batch_size}')
    start_iter = int(train_cfg.resume.split('_')[-1].split('.')[0]) if train_cfg.resume else 0
    training = True
    generator = generator.train()
    discriminator = discriminator.train()

    try:
        step = start_iter
        # [epoch: current step / (data size / batch size)] ex) 8/(16/4)
        epoch = int(step/len(train_dataloader))
        best_psnr = 0

        print('\n====================================================')
        print('Training Start!')
        print('====================================================')
        while training:
            # GPU Status Check!
            if epoch == 1:
                showGPU()
            
            # Training (1 epoch)
            for indice, clips in train_dataloader:
                frame_1 = clips[:, 0:3, :, :].unsqueeze(2).cuda()  # (n, 3, 1, 256, 256) 
                frame_2 = clips[:, 3:6, :, :].unsqueeze(2).cuda()  # (n, 3, 1, 256, 256) 
                frame_3 = clips[:, 6:9, :, :].unsqueeze(2).cuda()  # (n, 3, 1, 256, 256) 
                frame_4 = clips[:, 9:12, :, :].unsqueeze(2).cuda()  # (n, 3, 1, 256, 256) 
                f_target = clips[:, 12:15, :, :].cuda()  # (n, 3, 256, 256) 

                f_input = torch.cat([frame_1,frame_2, frame_3, frame_4], 2) # (n, 3, 4, 256, 256) 

                # pop() the used frame index, this can't work in train_dataset.__getitem__ because of multiprocessing.
                for index in indice:
                    train_dataset.all_seqs[index].pop()
                    if len(train_dataset.all_seqs[index]) == 0:
                        train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
                        random.shuffle(train_dataset.all_seqs[index])

                # Forward
                FG_frame = generator(f_input)
        
                # calculate Generator loss
                inte_fl = intensity_loss(FG_frame, f_target)
                grad_fl = gradient_loss(FG_frame, f_target)
                g_fl = adversarial_loss(discriminator(FG_frame))

                # use flownet
                if train_cfg.flownet != 'none':
                    f_input_last = clips[:, 9:12, :, :].cuda() # (n, 3, 256, 256)

                    # flownet2
                    if train_cfg.flownet == '2sd':
                        gt_flow_input = torch.cat([f_input_last.unsqueeze(2), f_target.unsqueeze(2)], 2) # (n, 3, 2, 256, 256) 
                        pred_flow_input = torch.cat([f_input_last.unsqueeze(2), FG_frame.unsqueeze(2)], 2)  # (n, 3, 2, 256, 256)   

                        flow_gt = (flownet(gt_flow_input * 255.) / 255.).detach()  # Input for flownet2sd is in (0, 255).
                        flow_pred = (flownet(pred_flow_input * 255.) / 255.).detach() 

                    # liteflow
                    else:
                        gt_flow_input = torch.cat([f_input_last, f_target], 1) # (n, 6, 256, 256)
                        pred_flow_input = torch.cat([f_input_last, FG_frame], 1) # (n, 6, 256, 256)

                        flow_gt = flownet.batch_estimate(gt_flow_input, flownet).detach() # (n, 2, 256, 256)
                        flow_pred = flownet.batch_estimate(pred_flow_input, flownet).detach() # (n, 2, 256, 256)

                    flow_fl = flow_loss(flow_pred, flow_gt)
                    G_fl_t = 1. * inte_fl + 1. * grad_fl + 2. * flow_fl + 0.05 * g_fl 

                # don't use flownet
                else:
                    G_fl_t = 1. * inte_fl + 1. * grad_fl + 0.05 * g_fl 


                # calculate Discriminator loss
                # (When training discriminator, don't train generator, so use .detach() to cut off gradients.)
                D_fl = discriminate_loss(discriminator(f_target), discriminator(FG_frame.detach()))

                # calculate Total loss
                G_l_t = G_fl_t 
                D_l = D_fl

                # Or just do .step() after all the gradients have been computed, like the following way:
                optimizer_G.zero_grad()
                G_l_t.backward()
                optimizer_G.step()
                
                optimizer_D.zero_grad()
                D_l.backward()
                optimizer_D.step()

                torch.cuda.synchronize()
                time_end = time.time()
                if step > start_iter:  # This doesn't include the testing time during training.
                    iter_t = time_end - temp
                temp = time_end
                    
                # check train status per 20 iteration!
                if step != start_iter:
                    if step % 20 == 0:
                        print(f'===========epoch:{epoch} (step:{step})============')

                        # calculate remained time
                        time_remain = (train_cfg.iters - step) * iter_t
                        eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]
                        
                        # calculate psnr
                        f_psnr = psnr_error(FG_frame, f_target)
                        if f_psnr >= best_psnr:
                            best_psnr = f_psnr
                            # early stopping (after 5000 step)
                            if step >= 5000:
                                model_dict = load_model_dict(train_cfg.sch, generator, discriminator, optimizer_G, optimizer_D, sch_G, sch_D)
                                torch.save(model_dict, f'weights/{train_cfg.work_num}_best_{train_cfg.dataset}.pth')
                                print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                                print(f'[best model] update! at {step} iteration!! [psnr: {f_psnr:.3f}]')
                                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

                        # current lr
                        lr_g = optimizer_G.param_groups[0]['lr']
                        lr_d = optimizer_D.param_groups[0]['lr']

                        # show psnr, loss, time, lr
                        print(f"[{step}] G_fl_total: {G_fl_t:.3f} | D_fl: {D_fl:.3f} | psnr: {f_psnr:.3f} | best: {best_psnr:.3f} | iter_t: {iter_t:.3f}s | remain_t: {eta} | lr_g: {lr_g:.7f} | lr_d: {lr_d:.7f}")

                        # write psnr
                        writer.add_scalar('psnr/forward/train_psnr', f_psnr, global_step=step)

                        # write loss
                        writer.add_scalar('D_loss_total/forward/d_loss', D_fl, global_step=step)
                        writer.add_scalar('G_loss_total/forward/g_loss', G_fl_t, global_step=step)
                        writer.add_scalar('G_loss_total/forward/inte_loss', inte_fl, global_step=step)
                        writer.add_scalar('G_loss_total/forward/grad_loss', grad_fl, global_step=step)
                        if train_cfg.flownet != 'none':
                            writer.add_scalar('G_loss_total/forward/flow_loss', grad_fl, global_step=step)

                        # write lr
                        writer.add_scalar('lr/forward/lr_g', lr_g, global_step=step)
                        writer.add_scalar('lr/forward/lr_d', lr_d, global_step=step)

                    # save current model per save_interval => Save Model to [Central Server]
                    if step % train_cfg.save_interval == 0:
                        model_dict = load_model_dict(train_cfg.sch, generator, discriminator, optimizer_G, optimizer_D, sch_G, sch_D)
                        torch.save(model_dict, f'/scratch/{train_cfg.save_dir}_weights/{train_cfg.work_num}_{train_cfg.dataset}_{step}.pth')
                        print(f'\nAlready saved: \'{train_cfg.work_num}_{train_cfg.dataset}_{step}.pth\'.')

                else:
                    best_psnr = psnr_error(FG_frame, f_target)

                # one iteration ok!
                step += 1

                # save last model
                if step > train_cfg.iters:
                    training = False
                    model_dict = load_model_dict(train_cfg.sch, generator, discriminator, optimizer_G, optimizer_D, sch_G, sch_D)
                    torch.save(model_dict, f'weights/latest_{train_cfg.dataset}_{step}.pth')
                    break
            
            # one epoch ok!
            epoch += 1
            if train_cfg.sch:
                sch_G.step()
                sch_D.step()
            

    except KeyboardInterrupt:
        print(f'\nStop early, model saved: \'latest_{train_cfg.dataset}_{step}.pth\'.\n')

        if glob(f'weights/latest*'):
            os.remove(glob(f'weights/latest*')[0])

        model_dict = load_model_dict(train_cfg.sch, generator, discriminator, optimizer_G, optimizer_D, sch_G, sch_D)
        torch.save(model_dict, f'weights/latest_{train_cfg.dataset}_{step}.pth')


if __name__=="__main__":
    main()