import math
import torch
from torch import nn
from unetr_pp.network_architecture.my_module import My_Transformer,My_EPA2
from monai.networks.layers.utils import get_norm_layer

'''
Residual Block (BN)
'''
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Residual Block
        self.residual_block = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm3d(num_features=out_dim),
                nn.ReLU(),
                nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm3d(num_features=out_dim),
            )            
        self.conv=nn.Conv3d(in_dim, out_dim, kernel_size=1)
        self.relu = nn.ReLU()
                  
    def forward(self, x):
        out = self.residual_block(x)  # (conv 3x3x3) *2
        out = out + self.conv(x)  # residual connection
        out = self.relu(out) # relu
        return out

'''
Residual Block (GN)
1x1x1 먼저 해준 버전 
'''
# class ResBlock(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.conv=nn.Conv3d(in_dim, out_dim, kernel_size=1)
#         # Residual Block
#         self.residual_block = nn.Sequential(
#                 nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1),
#                 nn.GroupNorm(num_channels=out_dim,num_groups=32),
#                 nn.ReLU(),
#                 nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1),
#                 nn.GroupNorm(num_channels=out_dim,num_groups=32),
#             )            
#         self.relu = nn.ReLU()
                  
#     def forward(self, x):
#         x = self.conv(x)
#         out = self.residual_block(x)  # (conv 3x3x3) *2
#         out = out + x  # residual connection
#         out = self.relu(out) # relu
#         return out

'''
Residual Block (GN)
'''
# class ResBlock(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         # Residual Block
#         self.residual_block = nn.Sequential(
#                 nn.Conv3d(in_dim, out_dim, kernel_size=3, padding=1),
#                 nn.GroupNorm(num_channels=out_dim,num_groups=32),
#                 nn.ReLU(),
#                 nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1),
#                 nn.GroupNorm(num_channels=out_dim,num_groups=32),
#             )            
#         self.conv=nn.Conv3d(in_dim, out_dim, kernel_size=1)
#         self.relu = nn.ReLU()
                  
#     def forward(self, x):
#         out = self.residual_block(x)  # (conv 3x3x3) *2
#         out = out + self.conv(x)  # residual connection
#         out = self.relu(out) # relu
#         return out


'''
ASTB (GN)
'''
class ASTB(nn.Module):
    def __init__(self,proj_size): 
        '''
        All Scale TIF Block
        '''
        super().__init__()

        channels=[32, 64, 128, 256]
        resolutions=[32*32*32, 16*16*16, 8*8*8, 4*4*4]

        # 1. norm & pooling module
        self.norm_set=nn.ModuleList([])
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        
        for crt_cnl in channels:
            self.norm_set.append(nn.LayerNorm(crt_cnl))

        # 2. channel control module
        self.control_cnl_set=nn.ModuleList([])

        for std_cnl in channels:
            control_cnl=nn.ModuleList([])
            for in_cnl in channels: 
                if in_cnl == std_cnl: # --> Identity
                    control_cnl.append(nn.Identity())

                else: # --> channel unify
                    control_cnl.append(nn.Linear(in_features=in_cnl,out_features=std_cnl))
            self.control_cnl_set.append(control_cnl)

        # 3. Transformer module
        self.transformer_set=nn.ModuleList([])
        
        for i,crt_cnl in enumerate(channels):
            crt_rst = resolutions[i]+3
            self.transformer_set.append(
                My_Transformer(input_size=crt_rst, proj_size=proj_size, dim=crt_cnl, depth=1, heads=4, dim_head=crt_cnl//4, mlp_dim=crt_cnl*2)
            )

        # 4. channel & resolution control module
        self.control_cnl_rst_set=nn.ModuleList([])

        for std_cnl in channels:
            control_cnl_rst=nn.ModuleList([])
            for in_cnl in channels: 
                if in_cnl == std_cnl: # --> Identity
                    control_cnl_rst.append(nn.Identity())

                elif in_cnl > std_cnl: # --> Upsampling
                    itr=int(math.log2(in_cnl//std_cnl))
                    self.up_layer=nn.Sequential()
                    cnl=in_cnl
                    for i in range(itr):
                        self.up_layer.add_module(f'upsample_{i+1}',nn.ConvTranspose3d(in_channels=cnl,out_channels=cnl//2,kernel_size=2,stride=2))
                        cnl=cnl//2
                    control_cnl_rst.append(self.up_layer)

                else: # --> downsampling
                    itr=int(math.log2(std_cnl//in_cnl))
                    self.down_layer=nn.Sequential()
                    cnl=in_cnl
                    for i in range(itr):
                        self.down_layer.add_module(f'downsample_{i+1}',nn.Conv3d(in_channels=cnl,out_channels=cnl*2,kernel_size=2,stride=2))
                        self.down_layer.add_module(f'group_norm{i+1}',get_norm_layer(name=("group", {"num_groups": cnl}), channels=cnl*2)) # <GN>
                        cnl=cnl*2
                    control_cnl_rst.append(self.down_layer)

            self.control_cnl_rst_set.append(control_cnl_rst)

        # 5. ResBlock module
        self.resblock_set=nn.ModuleList([])

        for crt_cnl in channels:
            self.resblock_set.append(ResBlock(in_dim=crt_cnl*4, out_dim=crt_cnl))

    def forward(self,x1,x2,x3,x4): 
        '''
        x1: H x W x D x C
        x2: H/2 x W/2 x D/2 x 2C
        x3: H/4 x W/4 x D/4 x 4C
        x4: H/8 x W/8 x D/8 x 8C
        '''
        save1, save2, save3, save4 = x1, x2, x3, x4       
        b_x1, c_x1, d_x1, h_x1, w_x1 = x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3], x1.shape[4]
        b_x2, c_x2, d_x2, h_x2, w_x2 = x2.shape[0], x2.shape[1], x2.shape[2], x2.shape[3], x2.shape[4]
        b_x3, c_x3, d_x3, h_x3, w_x3 = x3.shape[0], x3.shape[1], x3.shape[2], x3.shape[3], x3.shape[4]
        b_x4, c_x4, d_x4, h_x4, w_x4 = x4.shape[0], x4.shape[1], x4.shape[2], x4.shape[3], x4.shape[4]

        # reshape
        x1 = x1.reshape(b_x1, c_x1, -1).permute(0, 2, 1) # B, N, C (N=HWD)
        x2 = x2.reshape(b_x2, c_x2, -1).permute(0, 2, 1) # B, N/8, 2C 
        x3 = x3.reshape(b_x3, c_x3, -1).permute(0, 2, 1) # B, N/64, 4C 
        x4 = x4.reshape(b_x4, c_x4, -1).permute(0, 2, 1) # B, N/512, 8C 

        # norm & pool
        x1_p = torch.flatten(self.avgpool(self.norm_set[0](x1).transpose(1,2)), 1) # B, C
        x2_p = torch.flatten(self.avgpool(self.norm_set[1](x2).transpose(1,2)), 1) # B, 2C
        x3_p = torch.flatten(self.avgpool(self.norm_set[2](x3).transpose(1,2)), 1) # B, 4C
        x4_p = torch.flatten(self.avgpool(self.norm_set[3](x4).transpose(1,2)), 1) # B, 8C

        '''
        Transformer (Stage1)
        '''
        # channel control
        x2_p1 = self.control_cnl_set[0][1](x2_p).unsqueeze(1) # B, 1, C
        x3_p1 = self.control_cnl_set[0][2](x3_p).unsqueeze(1) # B, 1, C
        x4_p1 = self.control_cnl_set[0][3](x4_p).unsqueeze(1) # B, 1, C

        # concat & transformer & reshape
        X1 = self.transformer_set[0](torch.cat([x2_p1, x3_p1, x4_p1, x1],dim=1))[:, 3:, :] # B, N+3, C -> B, N, C
        X1 = X1.reshape(b_x1, c_x1, d_x1, h_x1, w_x1) # B, C, D, H, W

        '''
        Transformer (Stage2)
        '''
        # channel control
        x1_p2 = self.control_cnl_set[1][0](x1_p).unsqueeze(1) # B, 1, 2C
        x3_p2 = self.control_cnl_set[1][2](x3_p).unsqueeze(1) # B, 1, 2C
        x4_p2 = self.control_cnl_set[1][3](x4_p).unsqueeze(1) # B, 1, 2C

        # concat & transformer & reshape
        X2 = self.transformer_set[1](torch.cat([x1_p2, x3_p2, x4_p2, x2],dim=1))[:, 3:, :] # B, (N/8)+3, 2C -> B, N/8, 2C
        X2 = X2.reshape(b_x2, c_x2, d_x2, h_x2, w_x2) # B, 2C, D/2, H/2, W/2

        '''
        Transformer (Stage3)
        '''
        # channel control
        x1_p3 = self.control_cnl_set[2][0](x1_p).unsqueeze(1) # B, 1, 4C
        x2_p3 = self.control_cnl_set[2][1](x2_p).unsqueeze(1) # B, 1, 4C
        x4_p3 = self.control_cnl_set[2][3](x4_p).unsqueeze(1) # B, 1, 4C

        # concat & transformer & reshape
        X3 = self.transformer_set[2](torch.cat([x1_p3, x2_p3, x4_p3, x3],dim=1))[:, 3:, :] # B, (N/64)+3, 4C -> B, N/64, 4C
        X3 = X3.reshape(b_x3, c_x3, d_x3, h_x3, w_x3) # B, 4C, D/4, H/4, W/4

        '''
        Transformer (Stage4)
        '''
        # channel control
        x1_p4 = self.control_cnl_set[3][0](x1_p).unsqueeze(1) # B, 1, 8C
        x2_p4 = self.control_cnl_set[3][1](x2_p).unsqueeze(1) # B, 1, 8C
        x3_p4 = self.control_cnl_set[3][2](x3_p).unsqueeze(1) # B, 1, 8C

        # concat & transformer & reshape
        X4 = self.transformer_set[3](torch.cat([x1_p4, x2_p4, x3_p4, x4],dim=1))[:, 3:, :] # B, (N/512)+3, 8C -> B, N/512, 8C
        X4 = X4.reshape(b_x4, c_x4, d_x4, h_x4, w_x4) # B, 8C, D/8, H/8, W/8

        '''
        ResBlock (Stage1)
        '''
        # channel & resolution control
        X2_1 = self.control_cnl_rst_set[0][1](X2) # B, C, D, H, W
        X3_1 = self.control_cnl_rst_set[0][2](X3) # B, C, D, H, W
        X4_1 = self.control_cnl_rst_set[0][3](X4) # B, C, D, H, W

        # concat & resblock
        y1 = self.resblock_set[0](torch.cat([X2_1, X3_1, X4_1, X1], dim=1)) # B, C, D, H, W

        '''
        ResBlock (Stage2)
        '''
        # channel & resolution control
        X1_2 = self.control_cnl_rst_set[1][0](X1) # B, 2C, D/2, H/2, W/2
        X3_2 = self.control_cnl_rst_set[1][2](X3) # B, 2C, D/2, H/2, W/2
        X4_2 = self.control_cnl_rst_set[1][3](X4) # B, 2C, D/2, H/2, W/2

        # concat & resblock
        y2 = self.resblock_set[1](torch.cat([X1_2, X3_2, X4_2, X2], dim=1)) # B, 2C, D/2, H/2, W/2

        '''
        ResBlock (Stage3)
        '''
        # channel & resolution control
        X1_3 = self.control_cnl_rst_set[2][0](X1) # B, 4C, D/4, H/4, W/4
        X2_3 = self.control_cnl_rst_set[2][1](X2) # B, 4C, D/4, H/4, W/4
        X4_3 = self.control_cnl_rst_set[2][3](X4) # B, 4C, D/4, H/4, W/4

        # concat & resblock
        y3 = self.resblock_set[2](torch.cat([X1_3, X2_3, X4_3, X3], dim=1)) # B, 4C, D/4, H/4, W/4

        '''
        ResBlock (Stage4)
        '''
        # channel & resolution control
        X1_4 = self.control_cnl_rst_set[3][0](X1) # B, 8C, D/8, H/8, W/8
        X2_4 = self.control_cnl_rst_set[3][1](X2) # B, 8C, D/8, H/8, W/8
        X3_4 = self.control_cnl_rst_set[3][2](X3) # B, 8C, D/8, H/8, W/8

        # concat & resblock
        y4 = self.resblock_set[3](torch.cat([X1_4, X2_4, X3_4, X4], dim=1)) # B, 8C, D/8, H/8, W/8

        '''
        Skip Connection
        '''
        y1 = y1 + save1
        y2 = y2 + save2
        y3 = y3 + save3
        y4 = y4 + save4

        return y1, y2, y3, y4



class LSTB(nn.Module):
    '''
    Large Scale TIF Block
    '''
    def __init__(self,proj_size): 
        super().__init__()

        channels=[32, 64, 128, 256]
        resolutions=[32*32*32, 16*16*16, 8*8*8, 4*4*4]

        # 1. norm & pooling module
        self.norm_set=nn.ModuleList([])
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        
        for crt_cnl in channels:
            self.norm_set.append(nn.LayerNorm(crt_cnl))

        # 2. channel control module
        self.control_cnl_set=nn.ModuleList([])

        for std_cnl in channels:
            control_cnl=nn.ModuleList([])
            for in_cnl in channels: 
                if in_cnl == std_cnl: # --> Identity
                    control_cnl.append(nn.Identity())

                else: # --> channel unify
                    control_cnl.append(nn.Linear(in_features=in_cnl,out_features=std_cnl))
            self.control_cnl_set.append(control_cnl)

        # 3. Transformer module
        self.transformer_set=nn.ModuleList([])
        
        for i,crt_cnl in enumerate(channels):
            crt_rst = resolutions[i]+i # stage1 = NxC, stage2 = ((N/8)+1)x2C, stage3 = ((N/64)+2)x4C, stage4 = ((N/512)+3)x8C
            self.transformer_set.append(
                My_Transformer(input_size=crt_rst, proj_size=proj_size, dim=crt_cnl, depth=1, heads=4, dim_head=crt_cnl//4, mlp_dim=crt_cnl*2)
            )

        # 4. channel & resolution control module
        self.control_cnl_rst_set=nn.ModuleList([])

        for std_cnl in channels:
            control_cnl_rst=nn.ModuleList([])
            for in_cnl in channels: 
                if in_cnl == std_cnl: # --> Identity
                    control_cnl_rst.append(nn.Identity())

                elif in_cnl > std_cnl: # --> Upsampling
                    itr=int(math.log2(in_cnl//std_cnl))
                    self.up_layer=nn.Sequential()
                    cnl=in_cnl
                    for i in range(itr):
                        self.up_layer.add_module(f'upsample_{i+1}',nn.ConvTranspose3d(in_channels=cnl,out_channels=cnl//2,kernel_size=2,stride=2))
                        cnl=cnl//2
                    control_cnl_rst.append(self.up_layer)

                else: # --> downsampling
                    itr=int(math.log2(std_cnl//in_cnl))
                    self.down_layer=nn.Sequential()
                    cnl=in_cnl
                    for i in range(itr):
                        self.down_layer.add_module(f'downsample_{i+1}',nn.Conv3d(in_channels=cnl,out_channels=cnl*2,kernel_size=2,stride=2))
                        cnl=cnl*2
                    control_cnl_rst.append(self.down_layer)

            self.control_cnl_rst_set.append(control_cnl_rst)

        # 5. ResBlock module
        self.resblock_set=nn.ModuleList([])
        self.resblock_set.append(ResBlock(in_dim=32, out_dim=32)) # C -> C
        self.resblock_set.append(ResBlock(in_dim=128, out_dim=64)) # 4C -> 2C
        self.resblock_set.append(ResBlock(in_dim=384, out_dim= 128)) # 12C -> 4C
        self.resblock_set.append(ResBlock(in_dim=1024, out_dim=256)) # 32C -> 8C


    def forward(self,x1,x2,x3,x4): 
        '''
        x1: H x W x D x C
        x2: H/2 x W/2 x D/2 x 2C
        x3: H/4 x W/4 x D/4 x 4C
        x4: H/8 x W/8 x D/8 x 8C
        '''
        save1, save2, save3, save4 = x1, x2, x3, x4
        b_x1, c_x1, d_x1, h_x1, w_x1 = x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3], x1.shape[4]
        b_x2, c_x2, d_x2, h_x2, w_x2 = x2.shape[0], x2.shape[1], x2.shape[2], x2.shape[3], x2.shape[4]
        b_x3, c_x3, d_x3, h_x3, w_x3 = x3.shape[0], x3.shape[1], x3.shape[2], x3.shape[3], x3.shape[4]
        b_x4, c_x4, d_x4, h_x4, w_x4 = x4.shape[0], x4.shape[1], x4.shape[2], x4.shape[3], x4.shape[4]

        # reshape
        x1 = x1.reshape(b_x1, c_x1, -1).permute(0, 2, 1) # B, N, C (N=HWD)
        x2 = x2.reshape(b_x2, c_x2, -1).permute(0, 2, 1) # B, N/8, 2C 
        x3 = x3.reshape(b_x3, c_x3, -1).permute(0, 2, 1) # B, N/64, 4C 
        x4 = x4.reshape(b_x4, c_x4, -1).permute(0, 2, 1) # B, N/512, 8C 

        # norm & pool
        x1_p = torch.flatten(self.avgpool(self.norm_set[0](x1).transpose(1,2)), 1) # B, C
        x2_p = torch.flatten(self.avgpool(self.norm_set[1](x2).transpose(1,2)), 1) # B, 2C
        x3_p = torch.flatten(self.avgpool(self.norm_set[2](x3).transpose(1,2)), 1) # B, 4C
        x4_p = torch.flatten(self.avgpool(self.norm_set[3](x4).transpose(1,2)), 1) # B, 8C

        '''
        Transformer (Stage1)
        '''
        # transformer & reshape
        X1 = self.transformer_set[0](x1) # B, N, C
        X1 = X1.reshape(b_x1, c_x1, d_x1, h_x1, w_x1) # B, C, D, H, W

        '''
        Transformer (Stage2)
        '''
        # channel control
        x1_p2 = self.control_cnl_set[1][0](x1_p).unsqueeze(1) # B, 1, 2C

        # concat & transformer & reshape
        X2 = self.transformer_set[1](torch.cat([x1_p2, x2],dim=1))[:, 1:, :] # B, (N/8)+1, 2C -> B, N/8, 2C
        X2 = X2.reshape(b_x2, c_x2, d_x2, h_x2, w_x2) # B, 2C, D/2, H/2, W/2

        '''
        Transformer (Stage3)
        '''
        # channel control
        x1_p3 = self.control_cnl_set[2][0](x1_p).unsqueeze(1) # B, 1, 4C
        x2_p3 = self.control_cnl_set[2][1](x2_p).unsqueeze(1) # B, 1, 4C

        # concat & transformer & reshape
        X3 = self.transformer_set[2](torch.cat([x1_p3, x2_p3, x3],dim=1))[:, 2:, :] # B, (N/64)+2, 4C -> B, N/64, 4C
        X3 = X3.reshape(b_x3, c_x3, d_x3, h_x3, w_x3) # B, 4C, D/4, H/4, W/4

        '''
        Transformer (Stage4)
        '''
        # channel control
        x1_p4 = self.control_cnl_set[3][0](x1_p).unsqueeze(1) # B, 1, 8C
        x2_p4 = self.control_cnl_set[3][1](x2_p).unsqueeze(1) # B, 1, 8C
        x3_p4 = self.control_cnl_set[3][2](x3_p).unsqueeze(1) # B, 1, 8C

        # concat & transformer & reshape
        X4 = self.transformer_set[3](torch.cat([x1_p4, x2_p4, x3_p4, x4],dim=1))[:, 3:, :] # B, (N/512)+3, 8C -> B, N/512, 8C
        X4 = X4.reshape(b_x4, c_x4, d_x4, h_x4, w_x4) # B, 8C, D/8, H/8, W/8

        '''
        ResBlock (Stage1)
        '''
        # resblock
        y1 = self.resblock_set[0](X1) # B, C, D, H, W

        '''
        ResBlock (Stage2)
        '''
        # channel & resolution control
        X1_2 = self.control_cnl_rst_set[1][0](X1) # B, 2C, D/2, H/2, W/2

        # concat & resblock
        y2 = self.resblock_set[1](torch.cat([X1_2, X2], dim=1)) # B, 2C, D/2, H/2, W/2

        '''
        ResBlock (Stage3)
        '''
        # channel & resolution control
        X1_3 = self.control_cnl_rst_set[2][0](X1) # B, 4C, D/4, H/4, W/4
        X2_3 = self.control_cnl_rst_set[2][1](X2) # B, 4C, D/4, H/4, W/4

        # concat & resblock
        y3 = self.resblock_set[2](torch.cat([X1_3, X2_3, X3], dim=1)) # B, 4C, D/4, H/4, W/4

        '''
        ResBlock (Stage4)
        '''
        # channel & resolution control
        X1_4 = self.control_cnl_rst_set[3][0](X1) # B, 8C, D/8, H/8, W/8
        X2_4 = self.control_cnl_rst_set[3][1](X2) # B, 8C, D/8, H/8, W/8
        X3_4 = self.control_cnl_rst_set[3][2](X3) # B, 8C, D/8, H/8, W/8

        # concat & resblock
        y4 = self.resblock_set[3](torch.cat([X1_4, X2_4, X3_4, X4], dim=1)) # B, 8C, D/8, H/8, W/8

        '''
        Skip Connection
        '''
        y1 = y1 + save1
        y2 = y2 + save2
        y3 = y3 + save3
        y4 = y4 + save4

        return y1, y2, y3, y4
        

class SSTB(nn.Module):
    def __init__(self,proj_size): 
        '''
        Small Scale TIF Block
        '''
        super().__init__()

        channels=[32, 64, 128, 256]
        resolutions=[32*32*32, 16*16*16, 8*8*8, 4*4*4]

        # 1. norm & pooling module
        self.norm_set=nn.ModuleList([])
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        
        for crt_cnl in channels:
            self.norm_set.append(nn.LayerNorm(crt_cnl))

        # 2. channel control module
        self.control_cnl_set=nn.ModuleList([])

        for std_cnl in channels:
            control_cnl=nn.ModuleList([])
            for in_cnl in channels: 
                if in_cnl == std_cnl: # --> Identity
                    control_cnl.append(nn.Identity())

                else: # --> channel unify
                    control_cnl.append(nn.Linear(in_features=in_cnl,out_features=std_cnl))
            self.control_cnl_set.append(control_cnl)

        # 3. Transformer module
        self.transformer_set=nn.ModuleList([])
        
        for i,crt_cnl in enumerate(channels):
            crt_rst = resolutions[i]+(3-i) # stage1 = (N+3)xC, stage2 = ((N/8)+2)x2C, stage3 = ((N/64)+1)x4C, stage4 = (N/512)x8C
            self.transformer_set.append(
                My_Transformer(input_size=crt_rst, proj_size=proj_size, dim=crt_cnl, depth=1, heads=4, dim_head=crt_cnl//4, mlp_dim=crt_cnl*2)
            )

        # 4. channel & resolution control module
        self.control_cnl_rst_set=nn.ModuleList([])

        for std_cnl in channels:
            control_cnl_rst=nn.ModuleList([])
            for in_cnl in channels: 
                if in_cnl == std_cnl: # --> Identity
                    control_cnl_rst.append(nn.Identity())

                elif in_cnl > std_cnl: # --> Upsampling
                    itr=int(math.log2(in_cnl//std_cnl))
                    self.up_layer=nn.Sequential()
                    cnl=in_cnl
                    for i in range(itr):
                        self.up_layer.add_module(f'upsample_{i+1}',nn.ConvTranspose3d(in_channels=cnl,out_channels=cnl//2,kernel_size=2,stride=2))
                        cnl=cnl//2
                    control_cnl_rst.append(self.up_layer)

                else: # --> downsampling
                    itr=int(math.log2(std_cnl//in_cnl))
                    self.down_layer=nn.Sequential()
                    cnl=in_cnl
                    for i in range(itr):
                        self.down_layer.add_module(f'downsample_{i+1}',nn.Conv3d(in_channels=cnl,out_channels=cnl*2,kernel_size=2,stride=2))
                        cnl=cnl*2
                    control_cnl_rst.append(self.down_layer)

            self.control_cnl_rst_set.append(control_cnl_rst)

        # 5. ResBlock module
        self.resblock_set=nn.ModuleList([])
        self.resblock_set.append(ResBlock(in_dim=128, out_dim=32)) # 4C -> C
        self.resblock_set.append(ResBlock(in_dim=192, out_dim=64)) # 6C -> 2C
        self.resblock_set.append(ResBlock(in_dim=256, out_dim=128)) # 8C -> 4C
        self.resblock_set.append(ResBlock(in_dim=256, out_dim=256)) # 8C -> 8C

    def forward(self,x1,x2,x3,x4): 
        '''
        x1: H x W x D x C
        x2: H/2 x W/2 x D/2 x 2C
        x3: H/4 x W/4 x D/4 x 4C
        x4: H/8 x W/8 x D/8 x 8C
        '''
        save1, save2, save3, save4 = x1, x2, x3, x4
        b_x1, c_x1, d_x1, h_x1, w_x1 = x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3], x1.shape[4]
        b_x2, c_x2, d_x2, h_x2, w_x2 = x2.shape[0], x2.shape[1], x2.shape[2], x2.shape[3], x2.shape[4]
        b_x3, c_x3, d_x3, h_x3, w_x3 = x3.shape[0], x3.shape[1], x3.shape[2], x3.shape[3], x3.shape[4]
        b_x4, c_x4, d_x4, h_x4, w_x4 = x4.shape[0], x4.shape[1], x4.shape[2], x4.shape[3], x4.shape[4]

        # reshape
        x1 = x1.reshape(b_x1, c_x1, -1).permute(0, 2, 1) # B, N, C (N=HWD)
        x2 = x2.reshape(b_x2, c_x2, -1).permute(0, 2, 1) # B, N/8, 2C 
        x3 = x3.reshape(b_x3, c_x3, -1).permute(0, 2, 1) # B, N/64, 4C 
        x4 = x4.reshape(b_x4, c_x4, -1).permute(0, 2, 1) # B, N/512, 8C 

        # norm & pool
        x1_p = torch.flatten(self.avgpool(self.norm_set[0](x1).transpose(1,2)), 1) # B, C
        x2_p = torch.flatten(self.avgpool(self.norm_set[1](x2).transpose(1,2)), 1) # B, 2C
        x3_p = torch.flatten(self.avgpool(self.norm_set[2](x3).transpose(1,2)), 1) # B, 4C
        x4_p = torch.flatten(self.avgpool(self.norm_set[3](x4).transpose(1,2)), 1) # B, 8C

        '''
        Transformer (Stage1)
        '''
        # channel control
        x2_p1 = self.control_cnl_set[0][1](x2_p).unsqueeze(1) # B, 1, C
        x3_p1 = self.control_cnl_set[0][2](x3_p).unsqueeze(1) # B, 1, C
        x4_p1 = self.control_cnl_set[0][3](x4_p).unsqueeze(1) # B, 1, C

        # concat & transformer & reshape
        X1 = self.transformer_set[0](torch.cat([x2_p1, x3_p1, x4_p1, x1],dim=1))[:, 3:, :] # B, N+3, C -> B, N, C
        X1 = X1.reshape(b_x1, c_x1, d_x1, h_x1, w_x1) # B, C, D, H, W

        '''
        Transformer (Stage2)
        '''
        # channel control
        x3_p2 = self.control_cnl_set[1][2](x3_p).unsqueeze(1) # B, 1, 2C
        x4_p2 = self.control_cnl_set[1][3](x4_p).unsqueeze(1) # B, 1, 2C

        # concat & transformer & reshape
        X2 = self.transformer_set[1](torch.cat([x3_p2, x4_p2, x2],dim=1))[:, 2:, :] # B, (N/8)+2, 2C -> B, N/8, 2C
        X2 = X2.reshape(b_x2, c_x2, d_x2, h_x2, w_x2) # B, 2C, D/2, H/2, W/2

        '''
        Transformer (Stage3)
        '''
        # channel control
        x4_p3 = self.control_cnl_set[2][3](x4_p).unsqueeze(1) # B, 1, 4C

        # concat & transformer & reshape
        X3 = self.transformer_set[2](torch.cat([x4_p3, x3],dim=1))[:, 1:, :] # B, (N/64)+1, 4C -> B, N/64, 4C
        X3 = X3.reshape(b_x3, c_x3, d_x3, h_x3, w_x3) # B, 4C, D/4, H/4, W/4

        '''
        Transformer (Stage4)
        '''
        # transformer & reshape
        X4 = self.transformer_set[3](x4) # B, N/512, 8C
        X4 = X4.reshape(b_x4, c_x4, d_x4, h_x4, w_x4) # B, 8C, D/8, H/8, W/8

        '''
        ResBlock (Stage1)
        '''
        # channel & resolution control
        X2_1 = self.control_cnl_rst_set[0][1](X2) # B, C, D, H, W
        X3_1 = self.control_cnl_rst_set[0][2](X3) # B, C, D, H, W
        X4_1 = self.control_cnl_rst_set[0][3](X4) # B, C, D, H, W

        # concat & resblock
        y1 = self.resblock_set[0](torch.cat([X2_1, X3_1, X4_1, X1], dim=1)) # B, C, D, H, W

        '''
        ResBlock (Stage2)
        '''
        # channel & resolution control
        X3_2 = self.control_cnl_rst_set[1][2](X3) # B, 2C, D/2, H/2, W/2
        X4_2 = self.control_cnl_rst_set[1][3](X4) # B, 2C, D/2, H/2, W/2

        # concat & resblock
        y2 = self.resblock_set[1](torch.cat([X3_2, X4_2, X2], dim=1)) # B, 2C, D/2, H/2, W/2

        '''
        ResBlock (Stage3)
        '''
        # channel & resolution control
        X4_3 = self.control_cnl_rst_set[2][3](X4) # B, 4C, D/4, H/4, W/4

        # concat & resblock
        y3 = self.resblock_set[2](torch.cat([X4_3, X3], dim=1)) # B, 4C, D/4, H/4, W/4

        '''
        ResBlock (Stage4)
        '''
        # resblock
        y4 = self.resblock_set[3](X4) # B, 8C, D/8, H/8, W/8

        '''
        Skip Connection
        '''
        y1 = y1 + save1
        y2 = y2 + save2
        y3 = y3 + save3
        y4 = y4 + save4

        return y1, y2, y3, y4


class ASTB_ESA(nn.Module):
    def __init__(self,proj_size): 
        '''
        All Scale TIF Block
        '''
        super().__init__()

        channels=[32, 64, 128, 256]
        resolutions=[32*32*32, 16*16*16, 8*8*8, 4*4*4]

        # 1. norm & pooling module
        self.norm_set=nn.ModuleList([])
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        
        for crt_cnl in channels:
            self.norm_set.append(nn.LayerNorm(crt_cnl))

        # 2. channel control module
        self.control_cnl_set=nn.ModuleList([])

        for std_cnl in channels:
            control_cnl=nn.ModuleList([])
            for in_cnl in channels: 
                if in_cnl == std_cnl: # --> Identity
                    control_cnl.append(nn.Identity())

                else: # --> channel unify
                    control_cnl.append(nn.Linear(in_features=in_cnl,out_features=std_cnl))
            self.control_cnl_set.append(control_cnl)

        # 3. Transformer module
        self.transformer_set=nn.ModuleList([])
        
        for i,crt_cnl in enumerate(channels):
            crt_rst = resolutions[i]+3
            self.transformer_set.append(
                My_EPA2(input_size=crt_rst, hidden_size=crt_cnl, proj_size=proj_size)
            )

        # 4. channel & resolution control module
        self.control_cnl_rst_set=nn.ModuleList([])

        for std_cnl in channels:
            control_cnl_rst=nn.ModuleList([])
            for in_cnl in channels: 
                if in_cnl == std_cnl: # --> Identity
                    control_cnl_rst.append(nn.Identity())

                elif in_cnl > std_cnl: # --> Upsampling
                    itr=int(math.log2(in_cnl//std_cnl))
                    self.up_layer=nn.Sequential()
                    cnl=in_cnl
                    for i in range(itr):
                        self.up_layer.add_module(f'upsample_{i+1}',nn.ConvTranspose3d(in_channels=cnl,out_channels=cnl//2,kernel_size=2,stride=2))
                        cnl=cnl//2
                    control_cnl_rst.append(self.up_layer)

                else: # --> downsampling
                    itr=int(math.log2(std_cnl//in_cnl))
                    self.down_layer=nn.Sequential()
                    cnl=in_cnl
                    for i in range(itr):
                        self.down_layer.add_module(f'downsample_{i+1}',nn.Conv3d(in_channels=cnl,out_channels=cnl*2,kernel_size=2,stride=2))
                        cnl=cnl*2
                    control_cnl_rst.append(self.down_layer)

            self.control_cnl_rst_set.append(control_cnl_rst)

        # 5. ResBlock module
        self.resblock_set=nn.ModuleList([])

        for crt_cnl in channels:
            self.resblock_set.append(ResBlock(in_dim=crt_cnl*4, out_dim=crt_cnl))

    def forward(self,x1,x2,x3,x4): 
        '''
        x1: H x W x D x C
        x2: H/2 x W/2 x D/2 x 2C
        x3: H/4 x W/4 x D/4 x 4C
        x4: H/8 x W/8 x D/8 x 8C
        '''
        save1, save2, save3, save4 = x1, x2, x3, x4
        b_x1, c_x1, d_x1, h_x1, w_x1 = x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3], x1.shape[4]
        b_x2, c_x2, d_x2, h_x2, w_x2 = x2.shape[0], x2.shape[1], x2.shape[2], x2.shape[3], x2.shape[4]
        b_x3, c_x3, d_x3, h_x3, w_x3 = x3.shape[0], x3.shape[1], x3.shape[2], x3.shape[3], x3.shape[4]
        b_x4, c_x4, d_x4, h_x4, w_x4 = x4.shape[0], x4.shape[1], x4.shape[2], x4.shape[3], x4.shape[4]

        # reshape
        x1 = x1.reshape(b_x1, c_x1, -1).permute(0, 2, 1) # B, N, C (N=HWD)
        x2 = x2.reshape(b_x2, c_x2, -1).permute(0, 2, 1) # B, N/8, 2C 
        x3 = x3.reshape(b_x3, c_x3, -1).permute(0, 2, 1) # B, N/64, 4C 
        x4 = x4.reshape(b_x4, c_x4, -1).permute(0, 2, 1) # B, N/512, 8C 

        # norm & pool
        x1_p = torch.flatten(self.avgpool(self.norm_set[0](x1).transpose(1,2)), 1) # B, C
        x2_p = torch.flatten(self.avgpool(self.norm_set[1](x2).transpose(1,2)), 1) # B, 2C
        x3_p = torch.flatten(self.avgpool(self.norm_set[2](x3).transpose(1,2)), 1) # B, 4C
        x4_p = torch.flatten(self.avgpool(self.norm_set[3](x4).transpose(1,2)), 1) # B, 8C

        '''
        Transformer (Stage1)
        '''
        # channel control
        x2_p1 = self.control_cnl_set[0][1](x2_p).unsqueeze(1) # B, 1, C
        x3_p1 = self.control_cnl_set[0][2](x3_p).unsqueeze(1) # B, 1, C
        x4_p1 = self.control_cnl_set[0][3](x4_p).unsqueeze(1) # B, 1, C

        # concat & transformer & reshape
        X1 = self.transformer_set[0](torch.cat([x2_p1, x3_p1, x4_p1, x1],dim=1))[:, 3:, :] # B, N+3, C -> B, N, C
        X1 = X1.reshape(b_x1, c_x1, d_x1, h_x1, w_x1) # B, C, D, H, W

        '''
        Transformer (Stage2)
        '''
        # channel control
        x1_p2 = self.control_cnl_set[1][0](x1_p).unsqueeze(1) # B, 1, 2C
        x3_p2 = self.control_cnl_set[1][2](x3_p).unsqueeze(1) # B, 1, 2C
        x4_p2 = self.control_cnl_set[1][3](x4_p).unsqueeze(1) # B, 1, 2C

        # concat & transformer & reshape
        X2 = self.transformer_set[1](torch.cat([x1_p2, x3_p2, x4_p2, x2],dim=1))[:, 3:, :] # B, (N/8)+3, 2C -> B, N/8, 2C
        X2 = X2.reshape(b_x2, c_x2, d_x2, h_x2, w_x2) # B, 2C, D/2, H/2, W/2

        '''
        Transformer (Stage3)
        '''
        # channel control
        x1_p3 = self.control_cnl_set[2][0](x1_p).unsqueeze(1) # B, 1, 4C
        x2_p3 = self.control_cnl_set[2][1](x2_p).unsqueeze(1) # B, 1, 4C
        x4_p3 = self.control_cnl_set[2][3](x4_p).unsqueeze(1) # B, 1, 4C

        # concat & transformer & reshape
        X3 = self.transformer_set[2](torch.cat([x1_p3, x2_p3, x4_p3, x3],dim=1))[:, 3:, :] # B, (N/64)+3, 4C -> B, N/64, 4C
        X3 = X3.reshape(b_x3, c_x3, d_x3, h_x3, w_x3) # B, 4C, D/4, H/4, W/4

        '''
        Transformer (Stage4)
        '''
        # channel control
        x1_p4 = self.control_cnl_set[3][0](x1_p).unsqueeze(1) # B, 1, 8C
        x2_p4 = self.control_cnl_set[3][1](x2_p).unsqueeze(1) # B, 1, 8C
        x3_p4 = self.control_cnl_set[3][2](x3_p).unsqueeze(1) # B, 1, 8C

        # concat & transformer & reshape
        X4 = self.transformer_set[3](torch.cat([x1_p4, x2_p4, x3_p4, x4],dim=1))[:, 3:, :] # B, (N/512)+3, 8C -> B, N/512, 8C
        X4 = X4.reshape(b_x4, c_x4, d_x4, h_x4, w_x4) # B, 8C, D/8, H/8, W/8

        '''
        ResBlock (Stage1)
        '''
        # channel & resolution control
        X2_1 = self.control_cnl_rst_set[0][1](X2) # B, C, D, H, W
        X3_1 = self.control_cnl_rst_set[0][2](X3) # B, C, D, H, W
        X4_1 = self.control_cnl_rst_set[0][3](X4) # B, C, D, H, W

        # concat & resblock
        y1 = self.resblock_set[0](torch.cat([X2_1, X3_1, X4_1, X1], dim=1)) # B, C, D, H, W

        '''
        ResBlock (Stage2)
        '''
        # channel & resolution control
        X1_2 = self.control_cnl_rst_set[1][0](X1) # B, 2C, D/2, H/2, W/2
        X3_2 = self.control_cnl_rst_set[1][2](X3) # B, 2C, D/2, H/2, W/2
        X4_2 = self.control_cnl_rst_set[1][3](X4) # B, 2C, D/2, H/2, W/2

        # concat & resblock
        y2 = self.resblock_set[1](torch.cat([X1_2, X3_2, X4_2, X2], dim=1)) # B, 2C, D/2, H/2, W/2

        '''
        ResBlock (Stage3)
        '''
        # channel & resolution control
        X1_3 = self.control_cnl_rst_set[2][0](X1) # B, 4C, D/4, H/4, W/4
        X2_3 = self.control_cnl_rst_set[2][1](X2) # B, 4C, D/4, H/4, W/4
        X4_3 = self.control_cnl_rst_set[2][3](X4) # B, 4C, D/4, H/4, W/4

        # concat & resblock
        y3 = self.resblock_set[2](torch.cat([X1_3, X2_3, X4_3, X3], dim=1)) # B, 4C, D/4, H/4, W/4

        '''
        ResBlock (Stage4)
        '''
        # channel & resolution control
        X1_4 = self.control_cnl_rst_set[3][0](X1) # B, 8C, D/8, H/8, W/8
        X2_4 = self.control_cnl_rst_set[3][1](X2) # B, 8C, D/8, H/8, W/8
        X3_4 = self.control_cnl_rst_set[3][2](X3) # B, 8C, D/8, H/8, W/8

        # concat & resblock
        y4 = self.resblock_set[3](torch.cat([X1_4, X2_4, X3_4, X4], dim=1)) # B, 8C, D/8, H/8, W/8

        '''
        Skip Connection
        '''
        y1 = y1 + save1
        y2 = y2 + save2
        y3 = y3 + save3
        y4 = y4 + save4

        return y1, y2, y3, y4

class FinalConcat(nn.Module):
    def __init__(self):
        super().__init__()

        def Conv(in_cnl, out_cnl, ks=3, st=1):
            layers = []
            layers += [nn.Conv3d(in_channels=in_cnl,out_channels=out_cnl,kernel_size=ks,stride=st,padding=1)]
            layers += [nn.BatchNorm3d(num_features=out_cnl)]
            layers += [nn.ReLU(inplace=True)]
            conv = nn.Sequential(*layers)
            return conv

        def Upsample(in_cnl, out_cnl, ks, st):
            layers = []
            layers += [nn.ConvTranspose3d(in_channels=in_cnl,out_channels=out_cnl,kernel_size=ks,stride=st)]
            layers += [nn.BatchNorm3d(num_features=out_cnl)]
            layers += [nn.ReLU(inplace=True)]
            convT = nn.Sequential(*layers)
            return convT

        self.ds2x = Upsample(in_cnl=64, out_cnl=32, ks=2, st = 2)
        self.ds4x = nn.Sequential(
            Upsample(in_cnl=128, out_cnl=64, ks=2, st = 2),
            Upsample(in_cnl=64, out_cnl=32, ks=2, st = 2),
        )
        self.ds8x = nn.Sequential(
            Upsample(in_cnl=256, out_cnl=128, ks=2, st = 2),
            Upsample(in_cnl=128, out_cnl=64, ks=2, st = 2),
            Upsample(in_cnl=64, out_cnl=32, ks=2, st = 2),
        )

        self.conv = Conv(in_cnl=32*4, out_cnl=32)
        
    def forward(self, x1, x2, x3, x4):
        '''
        x1 : 32 x 32 x 32 x 32
        x2 : 16 x 16 x 16 x 64
        x3 : 8 x 8 x 8 x 128
        x4 : 4 x 4 x 4 x 256
        '''
        x2 = self.ds2x(x2)
        x3 = self.ds4x(x3)
        x4 = self.ds8x(x4)

        x = torch.cat([x1,x2,x3,x4],1) # B, 4C, D, H, W
        x = self.conv(x) # B, C, D, H, W
        
        return x

class MFA(nn.Module):
    def __init__(self):
        super().__init__()

        self.ds4x = nn.ConvTranspose3d(in_channels=32,out_channels=16,kernel_size=(2,4,4),stride=(2,4,4))

        self.ds8x = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64,out_channels=32,kernel_size=2,stride=2),
            nn.ConvTranspose3d(in_channels=32,out_channels=16,kernel_size=(2,4,4),stride=(2,4,4))
        )

        
    def forward(self, x1, x2, x3, x4):
        '''
        x1 : 128 x 128 x 64 x 16
        x2 : 128 x 128 x 64 x 16
        x3 : 32 x 32 x 32 x 32
        x4 : 16 x 16 x 16 x 64
        '''
        x3 = self.ds4x(x3)
        x4 = self.ds8x(x4)

        x = torch.cat([x1,x2,x3,x4],1) # B, 4C, D, H, W
                
        return x
