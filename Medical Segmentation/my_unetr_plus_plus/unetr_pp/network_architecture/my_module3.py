import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


'''
ConvNeXt Block 3D
'''
class NeXtBlock3D(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (B, C, D, H, W) -> (B, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) # (B, D, H, W, C) -> (B, C, D, H, W)

        x = input + x
        return x


'''
SAB : Serialized Attention Block
'''
class SerialAttn(nn.Module):
    '''
    Serialized Attention
    '''
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1)) # for channel attention
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1)) # for spatial attention

        # qkv are 3 linear layers (query, key, value)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.qkv2 = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        # projection matrices with shared weights used in attention module to project
        self.proj_k = self.proj_v = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop) 
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)
    
    def forward(self, x):
        B, N, C = x.shape # N=HWD
        
        '''
        Spatial Attention
        : K -> K(p), V -> V(p) [ Q x K_T(p) ]
        '''
        qkv2 = self.qkv2(x).reshape(B, N, 3, self.num_heads, C // self.num_heads) # B x N x 3 x h x C/h
        qkv2 = qkv2.permute(2, 0, 3, 1, 4) # 3 x B x h x N x C/h
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2] # B x h x N x C/h

        q2_t = q2.transpose(-2, -1) # B x h x C/h x N
        k2_t = k2.transpose(-2, -1) # B x h x C/h x N
        v2_t = v2.transpose(-2, -1) # B x h x C/h x N

        k2_t_projected = self.proj_k(k2_t) # B x h x C/h x p
        v2_t_projected = self.proj_v(v2_t) # B x h x C/h x p

        q2_t = torch.nn.functional.normalize(q2_t, dim=-1)
        k2_t = torch.nn.functional.normalize(k2_t, dim=-1)

        q2 = q2_t.permute(0, 1, 3, 2) # Q : B x h x N x C/h
        attn_SA = (q2 @ k2_t_projected) * self.temperature2  # [Q x K_T(p)] B x h x N x p
        
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA) # [Spatial Attn Map] B x h x N x p
        
        v2_projected = v2_t_projected.transpose(-2, -1) # V(p) : B x h x p x C/h

        # [Spatial Attn Map x V(p)] B x h x N x C/h -> B x C/h x h x N -> B x N x C
        x_SA = (attn_SA @ v2_projected).permute(0, 3, 1, 2).reshape(B, N, C) 
        
        '''
        Channel Attention
        : [ Q_T x K ]
        '''
        qkv = self.qkv(x_SA).reshape(B, N, 3, self.num_heads, C // self.num_heads) # B x N x 3 x h x C/h
        qkv = qkv.permute(2, 0, 3, 1, 4) # 3 x B x h x N x C/h
        q, k, v = qkv[0], qkv[1], qkv[2] # B x h x N x C/h

        q_t = q.transpose(-2, -1) # B x h x C/h x N
        k_t = k.transpose(-2, -1) # B x h x C/h x N
        v_t = v.transpose(-2, -1) # B x h x C/h x N

        q_t = torch.nn.functional.normalize(q_t, dim=-1)
        k_t = torch.nn.functional.normalize(k_t, dim=-1)
        
        k = k_t.transpose(-2, -1) # K : B x h x C/h x C/h
        attn_CA = (q_t @ k) * self.temperature # [Q_T x K] B x h x C/h x C/h 

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA) # [Channel Attn Map] B x h x C/h x C/h

        v = v_t.permute(0,1,3,2) # V : B x h x N x C/h

        # [V x Channel Attn Map] B x h x N x C/h -> B x C/h x h x N -> B x N x C
        x_CA = (v @ attn_CA).permute(0, 3, 1, 2).reshape(B, N, C)
        x = x_CA

        return x

class SAB(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        super().__init__()

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.MSA = SerialAttn(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
        self.NeXtBlock = NeXtBlock3D(dim = hidden_size)
        self.conv = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

    def forward(self, x):
        B, C, D, H, W = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.MSA(self.norm(x))

        attn_skip = attn.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        attn = self.NeXtBlock(attn_skip)
        x = attn_skip + self.conv(attn)

        return x


'''
PatchMerging 3D
'''
class PatchMerging3D(nn.Module):
    def __init__(self, dim):
        '''
        we remove layer norm. because we use GroupNorm outside.
        we assume that h,w,d are even numbers.
        '''
        super().__init__()
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x):
        '''
        x: B,C,D,H,W
        '''
        x = x.permute(0,2,3,4,1) # [B, D, H, W, C]
        
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.reduction(x)
        x = x.permute(0, 4, 1, 2, 3) # [B, C, D, H, W]
        
        return x


'''
PatchExpanding3D
'''
class PatchExpanding3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 4 * dim, bias=False)

    def forward(self, y):
        """
        y: B,C,D,H,W
        """
        y=y.permute(0,3,4,2,1) # [B, D, H, W, C]
        B, D, H, W, C = y.size()

        y=self.expand(y) # B, H, W, D, 4*C
    
        y=rearrange(y,'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c', p1=2, p2=2, p3=2, c=C//2) # B, 2*D, 2*H, 2*W, C//2

        y=y.permute(0,4,3,1,2) # B, C//2, 2*D, 2*H, 2*W
        
        return y


'''
ALC-MFA : All Local Concat MFA
'''
class ALC_MFA(nn.Module):
    def __init__(self,std_cnl): 
        super().__init__()
        '''
        std_cnl = standard_channel
        ex) 256
        '''
        self.control=nn.ModuleList([])
        for in_cnl in [32,64,128]: 
            itr=int(math.log2(std_cnl//in_cnl))
            self.down_layer=nn.Sequential()
            cnl=in_cnl
            for i in range(itr):
                # downsampling with Dilated Convolution
                self.down_layer.add_module(f'downsample_{i+1}',nn.Conv3d(in_channels=cnl,out_channels=cnl*2,kernel_size=2,stride=2,padding=1,dilation=3))
                cnl=cnl*2
            self.control.append(self.down_layer)

        self.NeXtBlock = NeXtBlock3D(dim = std_cnl)

    def forward(self,standard,x1,x2,x3):
        # control resolution and channel
        x1 = self.control[0](x1)
        x2 = self.control[1](x2)
        x3 = self.control[2](x3)

        # fusion
        x = x1 + x2 + x3 + standard
        x = self.NeXtBlock(x)

        return x


'''
Cross-MFA
'''
class CrossAttnModule(nn.Module):
    def __init__(self, N_1, N_2, proj_size, dim, mlp_dim, num_heads=4, attn_drop=0.1):
        '''
        N_1 = H_1 x W_1 x D_1
        N_2 = H_2 x W_2 x D_2
        dim = Channel
        '''
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkv layer
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)

        # projection layer
        self.proj_k = self.proj_v = nn.Linear(N_2, proj_size)

        # positional embedding layer 
        self.pos_embed_q = nn.Parameter(torch.zeros(1, N_1, dim))
        self.pos_embed_k = nn.Parameter(torch.zeros(1, N_2, dim))

        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)

        # Feed Forward Network
        self.ffn = nn.Sequential( 
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
          )

    def forward(self,x1,x2):
        B, N_1, C = x1.size()
        B, N_2, C = x2.size()

        '''
        Make Q, K, V
        '''
        q = q_skip = self.q(x1) # B x N_1 x C

        kv = self.kv(x2).reshape(B,N_2,2,C).permute(2,0,1,3) # 2 x B x N_2 x C
        k, v = kv[0], kv[1] # B x N_2 x C

        '''
        Add Positional Encoding
        '''
        q += self.pos_embed_q # B x N_1 x C
        k += self.pos_embed_k # B x N_2 x C

        '''
        Multi-Head Cross-Attention
        '''
        # reshape q,k,v
        q = q.reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B x h x N_1 x C/h
        k = k.reshape(B, N_2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B x h x N_2 x C/h
        v = v.reshape(B, N_2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B x h x N_2 x C/h

        k_t = k.transpose(-2,-1) # B x h x C/h x N_2
        v_t = v.transpose(-2,-1) # B x h x C/h x N_2

        k_t_projected = self.proj_k(k_t) # B x h x C/h x p
        v_t_projected = self.proj_v(v_t) # B x h x C/h x p

        attn = (q @ k_t_projected) * self.temperature # [Q x K_t(p)] B x h x N_1 x p

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        v_projected = v_t_projected.transpose(-2,-1) # B x h x p x C/h

        # [Attn Map x V(p)] B x h x N_1 x C/h -> B x C/h x h x N_1 -> B x N_1 x C
        x = (attn @ v_projected).permute(0, 3, 1, 2).reshape(B, N_1, C) 

        '''
        Add & Norm
        '''
        x += q_skip # B x N_1 X C
        x = x_save = torch.nn.functional.normalize(x, dim=-1) # B x N_1 x C

        '''
        FFN -> Add & Norm
        '''
        x = self.ffn(x)
        x += x_save
        x = torch.nn.functional.normalize(x, dim=-1) # B x N_1 x C

        return x

class CrossMFA(nn.Module):
    def __init__(self, HWD_l, HWD_g, proj_l, proj_g, dim_l, dim_g, itr):
        super().__init__()
        self.N_l = HWD_l
        self.N_g = HWD_g
        self.proj_l = proj_l
        self.proj_g = proj_g
        self.dim_l = dim_l
        self.dim_g = dim_g
        self.itr = itr

        # dim_l = local feature map channel (32,64,128)
        # dim_g = global feature map channel (64,128,256)

        self.linear_g = nn.Linear(dim_g, dim_l)

        self.crossAttn1 = CrossAttnModule(N_1=self.N_l, N_2=self.N_g, proj_size=self.proj_g, dim=self.dim_l, mlp_dim=self.dim_l*2) # Q: Local Feature, K,V: Global Feature
        self.crossAttn2 = CrossAttnModule(N_1=self.N_g, N_2=self.N_l, proj_size=self.proj_l, dim=self.dim_l, mlp_dim=self.dim_l*2) # Q: Global Feature, K,V: Local Feature

        self.upsample = nn.ConvTranspose3d(in_channels=dim_l,out_channels=dim_l,kernel_size=2,stride=2)
        self.NeXtBlock = NeXtBlock3D(dim = dim_l)

    def forward(self,lf, gf):
        '''
        lf: local feature 
        ex) 32 x 32 x 32 x 32, 16 x 16 x 16 x 64, 8 x 8 x 8 x 128

        fg: global feature
        ex) 16 x 16 x 16 x 64, 8 x 8 x 8 x 128, 4 x 4 x 4 x 256
        '''
        # save local feature
        lf_save = lf

        # 4D -> 2D
        B, C_l, D_l, H_l, W_l = lf.size()
        lf = lf.view(B, C_l, D_l * H_l * W_l).permute(0,2,1) # B, HWD_l, C_l

        B, C_g, D_g, H_g, W_g = gf.size()
        gf = gf.view(B, C_g, D_g * H_g * W_g).permute(0,2,1) # B, HWD_g, C_g

        # channel unify
        gf = self.linear_g(gf) # B, HWD_g, C_l

        '''
        lf: local feature (reshaped)
        ex) 32*32*32 x 32, 16*16*16 x 64, 8*8*8 x 128

        gf: global feature (reshaped)
        ex) 16*16x16 x 32, 8*8*8 x 64, 4*4*4 x 128
        '''

        # [Double Cross Attn] x itr
        in_1, in_2 = lf, gf
        for _ in range(self.itr):
            in_1 = out_1 = self.crossAttn1(in_1, in_2) # B, HWD_l, C_l 
            in_2 = out_2 = self.crossAttn2(in_2, in_1) # B, HWD_g, C_l

        # 2D -> 4D
        out_1_4d = out_1.reshape(B, D_l, H_l, W_l, C_l).permute(0, 4, 1, 2, 3) # B, C_l, D_l, H_l, W_l
        out_2_4d = out_2.reshape(B, D_g, H_g, W_g, C_l).permute(0, 4, 1, 2, 3) # B, C_l, D_g, H_g, W_g

        # Final Fusion with ConvNeXt Block
        last_in = out_1_4d + self.upsample(out_2_4d)
        out = self.NeXtBlock(last_in) # B, C_l, D_l, H_l, W_l

        # skip connection
        out += lf_save # B, C_l, D_l, H_l, W_l

        return out
