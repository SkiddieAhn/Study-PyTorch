import sys
import torch
from torch import nn, einsum
from einops import rearrange
sys.path.append('.')

'''
NFCE 수정 버전 (depthwise seperable conv를 이용함)
'''
class My_NFCE(nn.Module):
    def __init__(self,in_dim): 
        super().__init__()
        mid_dim=in_dim//4
        self.conv1=nn.Conv3d(in_channels=in_dim, out_channels=mid_dim, kernel_size=1, bias=False) # Conv 1x1x1
        self.norm1=nn.BatchNorm3d(mid_dim)

        # depthwise seperable convolution
        self.dsconv=nn.Sequential(
            nn.Conv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, padding=1, bias=False, groups=mid_dim), # Depth-wise Conv 3x3x3
            nn.Conv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=1, bias=False) # Point-wise Conv 1x1x1
        )
        self.norm2=nn.BatchNorm3d(mid_dim)

        self.conv3=nn.Conv3d(in_channels=mid_dim, out_channels=in_dim, kernel_size=1, bias=False) # Conv 1x1x1
        self.norm3=nn.BatchNorm3d(in_dim)

        self.relu=nn.ReLU()

    def forward(self,x):
        '''
        x: feature (H x W x D x C)
        ex) 16 x 16 x 16 x 64
        '''
        save=x # [B, C, D, H, W]
        
        # 1x1x1 conv -> [B, C//4, D, H, W]
        x=self.conv1(x) 
        x=self.norm1(x)
        x=self.relu(x)

        # depthwise seperable conv -> [B, C//4, D, H, W]
        x=self.dsconv(x) 
        x=self.norm2(x)
        x=self.relu(x)
        
        # 1x1x1 conv -> [B, C, D, H, W]
        x=self.conv3(x)
        x=self.norm3(x)
        
        # skip connection -> [B, C, D, H, W]
        x=x+save 
        x=self.relu(x)
        
        return x

'''
패치 합치기 (해상도 정보와 차원 정보를 채널 정보로 보냄)
'''
class My_PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(8 * dim)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x):
        """
        x: B,C,D,H,W
        """
        x=x.permute(0,3,4,2,1) # [B,H,W,D,C]
        B=x.shape[0];H=x.shape[1];W=x.shape[2];D=x.shape[3];C=x.shape[4]

        y=None
        for i in range(0,D,2):
            # process 2 slice
            x_=x[:, :, :, i:i+2, :] # B, H/2, W/2, 2, C
            
            x_0=x_[:, 0::2, 0::2, :, :] # B, H/2, W/2, 2, C
            x_1=x_[:, 0::2, 1::2, :, :] # B, H/2, W/2, 2, C 
            x_2=x_[:, 1::2, 0::2, :, :]  # B, H/2, W/2, 2, C 
            x_3=x_[:, 1::2, 1::2, :, :] # B, H/2, W/2, 2, C

            # width, height information -> channel information
            rst=torch.cat([x_0,x_1,x_2,x_3],-1) # B, H/2, W/2, 2, 4*C

            # dimension information -> channel information
            rst=rst.view(B, H//2, W//2, 1, 8*C) # B, H/2, W/2, 1, 8*C

            # concat 
            if i==0:
                y=rst.clone() # B, H/2, W/2, 1, 8*C
            else:
                y=torch.cat([y,rst],-2) # final shape -> [B, H/2, W/2, D/2, 8*C]
        
        # normalization
        y=self.norm(y) # B, H/2, W/2, D/2, 8*C
        
        # embedding
        y=self.reduction(y) # B, H/2, W/2, D/2, 2*C

        y=y.permute(0,4,3,1,2) # B, 2*C, D/2, H/2, W/2
        return y

'''
패치 확장하기 (채널 정보를 해상도 정보와 차원 정보로 보냄)
'''
class My_PatchExpanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim//2)
        self.expand = nn.Linear(dim, 4 * dim, bias=False)

    def forward(self, y):
        """
        y: B,C,D,H,W
        """
        y=y.permute(0,3,4,2,1) # [B,H,W,D,C]
        B=y.shape[0];H=y.shape[1];W=y.shape[2];D=y.shape[3];C=y.shape[4]

        # channel expand
        y=self.expand(y) # B, H, W, D, 4*C

        x=None
        for i in range(0,D):
            y_=y[:,:,:,i,:] # B, H, W, 1, 4*C
            y_=y_.view(B,H,W,1,4*C) 

            # channel information -> dimension information
            y_=y_.view(B, H, W, 2, 2*C) # B, H, W, 2, 2*C

            # channel informatinon -> width, height information
            rst=rearrange(y_,'b h w d (p1 p2 c)-> b (h p1) (w p2) d c', p1=2, p2=2, c=C//2) # B, 2*H, 2*W, 2, C//2
            
            # concat
            if i==0:
                x=rst.clone() # B, 2*H, 2*W, 2, C//2
            else:
                x=torch.cat([x,rst],-2) # final shape -> [B, 2*H, 2*W, 2*D, C//2]
                        
        # normalization
        x=self.norm(x) # B, 2*H, 2*W, 2*D, C//2

        x=x.permute(0,4,3,1,2) # B, C//2, 2*D, 2*H, 2*W
        return x


'''
EPA 직렬 버전
'''
class My_EPA(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1)) # for channel attention
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1)) # for spatial attention

        # qkv are 3 linear layers (query, key, value)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        # projection matrices with shared weights used in attention module to project
        self.proj_q = self.proj_k = self.proj_v = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop) 
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)
    
    def forward(self, x):
        '''
        Channel Attention
        : Q -> Q(p), K -> K(p) [ Q_T(p) x K(p) ]
        '''
        B, N, C = x.shape # N=HWD

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads) # B x N x 3 x h x C/h
        qkv = qkv.permute(2, 0, 3, 1, 4) # 3 x B x h x N x C/h
        q, k, v = qkv[0], qkv[1], qkv[2] # B x h x N x C/h

        q_t = q.transpose(-2, -1) # B x h x C/h x N
        k_t = k.transpose(-2, -1) # B x h x C/h x N
        v_t = v.transpose(-2, -1) # B x h x C/h x N

        q_t = torch.nn.functional.normalize(q_t, dim=-1)
        k_t = torch.nn.functional.normalize(k_t, dim=-1)
        
        q_t_projected = self.proj_q(q_t) # B x h x C/h x p
        k_t_projected = self.proj_k(k_t) # B x h x C/h x p

        k_projected = k_t_projected.transpose(-2, -1) # K(p) : B x h x p x C/h
        attn_CA = (q_t_projected @ k_projected) * self.temperature # [Q_T(p) x K(p)] B x h x C/h x C/h 

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA) # [Channel Attn Map] B x h x C/h x C/h

        v = v_t.permute(0,1,3,2) # V : B x h x N x C/h

        # [V x Channel Attn Map] B x h x N x C/h -> B x C/h x h x N -> B x N x C
        x_CA = (v @ attn_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        '''
        Spatial Attention
        : K -> K(p), V -> V(p) [ Q x K_T(p) ]
        '''
        qkv2 = self.qkv(x_CA).reshape(B, N, 3, self.num_heads, C // self.num_heads) # B x N x 3 x h x C/h
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
        x = x_SA

        return x


'''
TIF 3D 버전
'''
class My_Conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, groups):
        super(My_Conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True), # 수정
            nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True), # 수정
            nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class My_Attention(nn.Module):
    def __init__(self, input_size, proj_size, dim, heads, dim_head, dropout = 0.): # 수정
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) 

        self.E = self.F = nn.Linear(input_size+1, proj_size) # 수정

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        '''
        k, v projection (차원 축소하지 않으면 연산량이 너무 커짐)
        '''
        k=self.E(k.transpose(-2, -1)).transpose(-2,-1) # 수정
        v=self.F(v.transpose(-2, -1)).transpose(-2,-1) # 수정

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class My_Transformer(nn.Module):
    def __init__(self, input_size, proj_size, dim, depth, heads, dim_head, mlp_dim, dropout = 0.): # 수정
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, My_Attention(input_size, proj_size, dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class My_Cross_Att(nn.Module):
    def __init__(self, HWD_e, HWD_r, proj_size, dim_e, dim_r): # 수정
        super().__init__()
        self.transformer_e = My_Transformer(HWD_e, proj_size, dim=dim_e, depth=1, heads=4, dim_head=dim_e//4, mlp_dim=128) # UNETR++와 head, dim_head 통일 <local>
        self.transformer_r = My_Transformer(HWD_r, proj_size, dim=dim_r, depth=1, heads=4, dim_head=dim_r//4, mlp_dim=256) # UNETR++와 head, dim_head 통일 <global>
        self.norm_e = nn.LayerNorm(dim_e) 
        self.norm_r = nn.LayerNorm(dim_r) 
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear_e = nn.Linear(dim_e, dim_r)
        self.linear_r = nn.Linear(dim_r, dim_e)

    def forward(self, e, r):
       b_e, c_e, d_e, h_e, w_e = e.shape
       e = e.reshape(b_e, c_e, -1).permute(0, 2, 1) # B, N, C
       b_r, c_r, d_r, h_r, w_r = r.shape
       r = r.reshape(b_r, c_r, -1).permute(0, 2, 1)
       e_t = torch.flatten(self.avgpool(self.norm_e(e).transpose(1,2)), 1)
       r_t = torch.flatten(self.avgpool(self.norm_r(r).transpose(1,2)), 1)
       e_t = self.linear_e(e_t).unsqueeze(1)
       r_t = self.linear_r(r_t).unsqueeze(1)
       r = self.transformer_r(torch.cat([e_t, r],dim=1))[:, 1:, :]
       e = self.transformer_e(torch.cat([r_t, e],dim=1))[:, 1:, :]
       e = e.permute(0, 2, 1).reshape(b_e, c_e, d_e, h_e, w_e) 
       r = r.permute(0, 2, 1).reshape(b_r, c_r, d_r, h_r, w_r) 
       return e, r

class My_TIF(nn.Module):
    def __init__(self,HWD_e,HWD_r,proj_size,dim_e,dim_r): 
        super().__init__()
        # dim_e = local feature map channel (16,32,64,128)
        # dim_r = global feature map channel (32,64,128,256)
        self.cross_attn=My_Cross_Att(HWD_e,HWD_r,proj_size,dim_e,dim_r)
        self.up = nn.Upsample(scale_factor=2)
        self.conv=My_Conv_block(in_ch=dim_e+dim_r, out_ch=dim_e, groups=32)
        
    def forward(self,e,r):
        '''
        e: local feature (H x W x D x C)
        r: global feature (H/2 x W/2 x D/2 x 2C)
        '''
        skip=e
        e,r=self.cross_attn(e,r) # [B,C,D,H,W], [B,C,D/2,H/2,W/2]
        e = torch.cat([e,self.up(r)],1) # B,2C,D,H,W
        e=self.conv(e) # B,C,D,H,W
        e=skip+e # skip connection
        
        return e