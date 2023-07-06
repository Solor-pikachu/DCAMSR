import os
import sys
# import re
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
import copy
from functools import partial, reduce
import numpy as np
import itertools
import math
from collections import OrderedDict


class DCAT(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int = 64,
            num_heads: int = 8,
            dropout_rate: float = 0.1,
            pos_embed=True,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        # self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = DCA(input_size=input_size, input_size1=input_size,hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)

        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))
        else:
            self.pos_embed = None
            
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=1,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=1,stride=1),
        )

    def forward(self, x,ref):
        
        B, C, H, W = ref.shape
        ref = ref.reshape(B, C, H * W).permute(0, 2, 1)
        
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
            ref = ref + self.pos_embed
        attn = x + self.epa_block(self.norm(x),self.norm(ref))

        attn_skip = attn.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        attn_skip = self.ffn(attn_skip) + attn_skip
        return attn_skip

class DCA(nn.Module):

    def __init__(self, input_size,input_size1, hidden_size, proj_size, num_heads=4, qkv_bias=True,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Linear(hidden_size,hidden_size)
        
        self.kvv = nn.Linear(hidden_size,hidden_size*3)
        
        self.E = self.F = nn.Linear(input_size1, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x,ref):
        B, N, C = x.shape
        B1,N1,C1 = ref.shape
        
        x = self.q(x)
        q_shared = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        kvv = self.kvv(ref).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        kvv = kvv.permute(2, 0, 3, 1, 4)
        k_shared, v_CA, v_SA = kvv[0], kvv[1], kvv[2]

        #### 通道注意力
        q_shared = q_shared.transpose(-2, -1) #B,Head,C,N
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature # (B,Head,C,N) * (#B,Head,N,C) -> (B,Head,C,C)

        attn_CA = attn_CA.softmax(dim=-1) #B,Head,C,C
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C) # (B,Head,C,C) * (B,Head,C,N) -> (B,Head,C,N) -> (B,N,C)
        
        
        #### 位置注意力
        k_shared_projected = self.E(k_shared)
        v_SA_projected = self.F(v_SA)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2 # (B,Head,N,C) * (B,Head,C,64) -> (B,Head,N,64)

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C) # (B,Head,N,64) * (B,Head,64,C) -> (B,Head,N,C) -> (B,N,C)

        # Concat fusion
        x_CA = self.out_proj(x_CA)
        x_SA = self.out_proj2(x_SA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, dilation=1, act='relu'):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))

        return out + x


class SAM(nn.Module):
    def __init__(self, nf, use_residual=True, learnable=True):
        super(SAM, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        if self.learnable:
            self.conv_shared = nn.Sequential(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True),
                                             nn.ReLU(inplace=True))
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref):
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            style = self.conv_shared(torch.cat([lr, ref], dim=1))
            gamma = self.conv_gamma(style)
            beta = self.conv_beta(style)

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta

        return out


class Encoder(nn.Module):
    def __init__(self, in_chl, nf, n_blks=[1, 1, 1], act='relu'):
        super(Encoder, self).__init__()

        block = functools.partial(ResidualBlock, nf=nf)

        self.conv_L1 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block, n_layers=n_blks[0])

        self.conv_L2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.blk_L2 = make_layer(block, n_layers=n_blks[1])

        self.conv_L3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.blk_L3 = make_layer(block, n_layers=n_blks[2])

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        fea_L1 = self.blk_L1(self.act(self.conv_L1(x)))
        fea_L2 = self.blk_L2(self.act(self.conv_L2(fea_L1)))
        fea_L3 = self.blk_L3(self.act(self.conv_L3(fea_L2)))

        return [fea_L1, fea_L2, fea_L3]


class Decoder(nn.Module):
    def __init__(self, nf, out_chl, n_blks=[1, 1, 1, 1, 1, 1]):
        super(Decoder, self).__init__()

        block = functools.partial(ResidualBlock, nf=nf)

        self.conv_L3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.blk_L3 = make_layer(block, n_layers=n_blks[0])

        self.conv_L2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.blk_L2 = make_layer(block, n_layers=n_blks[1])

        self.conv_L1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block, n_layers=n_blks[2])

        self.merge_warp_x1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.blk_x1 = make_layer(block, n_blks[3])

        self.merge_warp_x2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.blk_x2 = make_layer(block, n_blks[4])

        self.merge_warp_x4 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.blk_x4 = make_layer(block, n_blks[5])

        self.conv_out = nn.Conv2d(64, out_chl, 3, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)

        self.pAda = SAM(nf, use_residual=True, learnable=True)

    def forward(self, lr_l, warp_ref_l):
        fea_L3 = self.act(self.conv_L3(lr_l[2]))
        fea_L3 = self.blk_L3(fea_L3)

        fea_L2 = self.act(self.conv_L2(fea_L3))
        fea_L2 = self.blk_L2(fea_L2)
        fea_L2_up = F.interpolate(fea_L2, scale_factor=2, mode='bilinear', align_corners=False)

        fea_L1 = self.act(self.conv_L1(torch.cat([fea_L2_up, lr_l[2]], dim=1)))
        fea_L1 = self.blk_L1(fea_L1)

        warp_ref_x1 = self.pAda(fea_L1, warp_ref_l[2])
        fea_x1 = self.act(self.merge_warp_x1(torch.cat([warp_ref_x1, fea_L1], dim=1)))
        fea_x1 = self.blk_x1(fea_x1)
        fea_x1_up = F.interpolate(fea_x1, scale_factor=2, mode='bilinear', align_corners=False)

        warp_ref_x2 = self.pAda(fea_x1_up, warp_ref_l[1])
        fea_x2 = self.act(self.merge_warp_x2(torch.cat([warp_ref_x2, fea_x1_up], dim=1)))
        fea_x2 = self.blk_x2(fea_x2)
        fea_x2_up = F.interpolate(fea_x2, scale_factor=2, mode='bilinear', align_corners=False)

        warp_ref_x4 = self.pAda(fea_x2_up, warp_ref_l[0])
        fea_x4 = self.act(self.merge_warp_x4(torch.cat([warp_ref_x4, fea_x2_up], dim=1)))
        fea_x4 = self.blk_x4(fea_x4)
        out = self.conv_out(fea_x4)

        return out

    
class DCAMSR(nn.Module):
    def __init__(self, args,scale):
        super().__init__()
        input_size = 256
        in_chl = 1
        nf = 64
        n_blks = [4, 4, 4]
        n_blks_dec = [2, 2, 2, 12, 8, 4]
        self.scale = scale
        depths = [1,1,1]

        self.enc = Encoder(in_chl=in_chl, nf=nf, n_blks=n_blks)
        self.decoder = Decoder(nf, in_chl, n_blks=n_blks_dec)

        self.trans_lv1 = nn.ModuleList([DCAT(input_size=input_size*input_size, hidden_size=64, proj_size=64, pos_embed=i!=0) for i in range(depths[0])] )
        self.trans_lv2 = nn.ModuleList([DCAT(input_size=input_size*input_size//4, hidden_size=64, proj_size=64, pos_embed=i!=0) for i in range(depths[1])] )
        self.trans_lv3 = nn.ModuleList([DCAT(input_size=input_size*input_size//16, hidden_size=64, proj_size=64, pos_embed=i!=0)  for i in range(depths[2])] )


    def forward(self, lr, ref, ref_down, gt=None):        
        
        lrsr = F.interpolate(lr, scale_factor=self.scale, mode='bilinear')
        
        fea_lrsr= self.enc(lrsr)
        fea_ref_l = self.enc(ref)
         
        warp_ref_patches_x4 = fea_lrsr[0] #320,320
        warp_ref_patches_x2 = fea_lrsr[1] #160,160
        warp_ref_patches_x1 = fea_lrsr[2] #80,80
        for transformer in self.trans_lv1:
            warp_ref_patches_x4 = transformer(warp_ref_patches_x4,fea_ref_l[0])
            fea_ref_l[0] = warp_ref_patches_x4
            
        for transformer in self.trans_lv2:
            warp_ref_patches_x2 = transformer(warp_ref_patches_x2,fea_ref_l[1])
            fea_ref_l[1] = warp_ref_patches_x2
            
        for transformer in self.trans_lv3:
            warp_ref_patches_x1 = transformer(warp_ref_patches_x1,fea_ref_l[2])
            fea_ref_l[2] = warp_ref_patches_x1

        warp_ref_l = [warp_ref_patches_x4, warp_ref_patches_x2, warp_ref_patches_x1]
        out = self.decoder(fea_lrsr, warp_ref_l)
        out = out + lrsr

        return out