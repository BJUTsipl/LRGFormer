# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
# from functools import partial
#
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.resnet import resnet26d, resnet50d
# from timm.models.registry import register_model
# from einops.layers.torch import Rearrange
import numpy as np
import time
from common.ops import (
    bchw_to_blc,
    blc_to_bchw,
    calculate_mask,
    calculate_mask_all,
    get_relative_coords_table_all,
    get_relative_position_index_simple,
)
from common.swin_v1_block import (
    build_last_conv,
)
# from mmcv_custom import load_checkpoint
# from mmseg.utils import get_root_logger
# from ..builder import BACKBONES

import torch.utils.checkpoint as checkpoint
from  common.mixed_attn_block_efficient  import EfficientMixAttnTransformerBlock#*


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """Not supported now, since we have cls_tokens now.....
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.idx = idx
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.split_size, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        elif idx == 2:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 3:
            H_sp, W_sp = self.resolution, self.resolution
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        self.H_sp_ = self.H_sp
        self.W_sp_ = self.W_sp

        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)
        
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, C, H, W = x.shape
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()#self.H_sp* self.W_sp
        return x

    def get_rpe(self, x, func):
        B, C, H, W = x.shape
        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        rpe = func(x) ### B', C, H', W'
        rpe = rpe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, rpe

    def forward(self, temp):
        """
        x: B N C
        mask: B N N
        """
        B, _, C, H, W = temp.shape

        idx = self.idx
        if idx == -1:
            H_sp, W_sp = H, W
        elif idx == 0:
            H_sp, W_sp = self.split_size, self.split_size
        elif idx == 1:
            H_sp, W_sp = H, self.split_size
        elif idx == 2:
            H_sp, W_sp = self.split_size, W
        elif idx == 3:
            H_sp, W_sp = self.resolution, self.resolution
        else:
            print ("ERROR MODE in forward", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        ### padding for split window
        H_pad = (self.H_sp - H % self.H_sp) % self.H_sp
        W_pad = (self.W_sp - W % self.W_sp) % self.W_sp
        top_pad = H_pad//2
        down_pad = H_pad - top_pad
        left_pad = W_pad//2
        right_pad = W_pad - left_pad
        H_ = H + H_pad
        W_ = W + W_pad

        qkv = F.pad(temp, (left_pad, right_pad, top_pad, down_pad)) ### B,3,C,H',W'
        qkv = qkv.permute(1, 0, 2, 3, 4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, rpe = self.get_rpe(v, self.get_v)

        ### Local attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)

        attn = self.attn_drop(attn)

        x = (attn @ v) + rpe
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H_, W_) # B H_ W_ C
        x = x[:, top_pad:H+top_pad, left_pad:W+left_pad, :]
        x = x.reshape(B, -1, C)

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, patches_resolution, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = patches_resolution
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.qkv11 = nn.Linear(dim // 4, dim // 4 * 3, bias=True)
        self.qkv22 = nn.Linear(dim // 4, dim // 4 * 3, bias=True)
        self.qkv33 = nn.Linear(dim // 4, dim // 4 * 3, bias=True)
        self.norm1 = norm_layer(dim)

        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 4
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 4, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 4, dim_out=dim // 4,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        # self.conv1= nn.Conv2d(dim // 4, dim // 4, 3, stride=1, padding=1)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

        atten_mask_matrix = None

        self.register_buffer("atten_mask_matrix", atten_mask_matrix)
        self.H = None
        self.W = None

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        # H = self.H
        # W = self.W
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        temp = self.qkv(img).reshape(B, H, W, 3, C).permute(0, 3, 4, 1, 2)

        if self.branch_num == 4:

            temp1=img.view(B, H, W, C).permute(0, 3, 1, 2)#(temp[:, :, :C // 4, :, :]) .view(B, H, W, C).permute(0, 3, 1, 2).contiguous() [:,:,:,:]
            x1 = (temp1[:, :C // 4, :, :])#self.conv1
            x1=x1.flatten(2).transpose(1, 2)
            # x1 = self.attns[0](temp[:, :, :C // 4, :, :])
            x11 = self.qkv11(x1)
            x11 = x11.reshape(B, H, W, 3, C // 4).permute(0, 3, 4, 1, 2)  # C//2
            x2 = self.attns[0](temp[:, :, C // 4:2 * C // 4, :, :] + x11)

            x22 = self.qkv22(x2)
            x22 = x22.reshape(B, H, W, 3, C // 4).permute(0, 3, 4, 1, 2)  # C//2
            x3 = self.attns[1](temp[:, :, 2 * C // 4:3 * C // 4, :, :] + x22)

            x33 = self.qkv33(x3)
            x33 = x33.reshape(B, H, W, 3, C // 4).permute(0, 3, 4, 1, 2)  # C//2
            x4 = self.attns[2](temp[:, :, 3 * C // 4:, :, :] + x33)

            attened_x = torch.cat([x1, x2, x3, x4], dim=2)
        else:
            attened_x = self.attns[0](temp)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
# class CSWinBlock(nn.Module):
#
#     def __init__(self, dim, patches_resolution, num_heads,
#                  split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
#                  drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm,
#                  last_stage=False):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.patches_resolution = patches_resolution
#         self.split_size = split_size
#         self.mlp_ratio = mlp_ratio
#         self.qkv = nn.Linear(dim, dim * 3, bias=True)
#         self.qkv11 = nn.Linear(dim//2, dim//2 * 3, bias=True)
#         self.norm1 = norm_layer(dim)
#
#         if last_stage:
#             self.branch_num = 1
#         else:
#             self.branch_num = 2
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(drop)
#
#         if last_stage:
#             self.attns = nn.ModuleList([
#                 LePEAttention(
#                     dim, resolution=self.patches_resolution, idx = -1,
#                     split_size=split_size, num_heads=num_heads, dim_out=dim,
#                     qkv_bias=qkv_bias, qk_scale=qk_scale,
#                     attn_drop=attn_drop, proj_drop=drop)
#                 for i in range(self.branch_num)])
#         else:
#             self.attns = nn.ModuleList([
#                 LePEAttention(
#                     dim//2, resolution=self.patches_resolution, idx = i,
#                     split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
#                     qkv_bias=qkv_bias, qk_scale=qk_scale,
#                     attn_drop=attn_drop, proj_drop=drop)
#                 for i in range(self.branch_num)])
#         mlp_hidden_dim = int(dim * mlp_ratio)
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
#         self.norm2 = norm_layer(dim)
#
#         atten_mask_matrix = None
#
#         self.register_buffer("atten_mask_matrix", atten_mask_matrix)
#         self.H = None
#         self.W = None
#
#     def forward(self, x,H,W):
#         """
#         x: B, H*W, C
#         """
#         B, L, C = x.shape
#         # H = self.H
#         # W = self.W
#         assert L == H * W, "flatten img_tokens has wrong size"
#         img = self.norm1(x)
#         temp = self.qkv(img).reshape(B, H, W, 3, C).permute(0, 3, 4, 1, 2)
#
#         if self.branch_num == 2:
#
#             x1 = self.attns[0](temp[:, :, :C // 2, :, :])
#             x11 = self.qkv11(x1)
#             x11 = x11.reshape(B, H, W, 3, C //2).permute(0, 3, 4, 1, 2)  # C//2
#             x2 = self.attns[1](temp[:, :, C // 2:, :, :] + x11)
#
#             # x22 = self.qkv11(x2)
#             # x22 = x22.reshape(B, H, W, 3, C // 3).permute(0, 3, 4, 1, 2)  # C//2
#             # x3 = self.attns[-1](temp[:, :, 2 * C // 3:, :, :] + x22)
#             attened_x = torch.cat([x1, x2], dim=2)#, x3
#         else:
#             attened_x = self.attns[0](temp)
#         attened_x = self.proj(attened_x)
#         x = x + self.drop_path(attened_x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#
#         return x

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x, H, W):
        B, new_HW, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        
        return x, H, W


class Up_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 1, 1, 0)
        self.norm = norm_layer(dim_out)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, H, W):
        B, new_HW, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.upsample(x)
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x, H, W

# @BACKBONES.register_module()

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x
def _get_stripe_info(stripe_size_in, stripe_groups_in, stripe_shift, input_resolution):
    stripe_size, shift_size = [], []
    for s, g, d in zip(stripe_size_in, stripe_groups_in, input_resolution):
        if g is None:
            stripe_size.append(s)
            shift_size.append(s // 2 if stripe_shift else 0)
        # else:
        #     stripe_size.append(d // g)
        #     shift_size.append(0 if g == 1 else d // (g * 2))
    return stripe_size, shift_size

class CSWin(nn.Module):
    """ Vis0ion Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=128, patch_size=4, in_chans=3, embed_dim=64, depth=[2,2,2], split_size = 7,
                 num_heads=[3,6,9], mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,patch_norm=True, use_chk=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_norm = patch_norm
        heads=num_heads
        self.use_chk = use_chk

        self.norm1 = nn.LayerNorm(embed_dim)
        curr_dim = embed_dim

        self.stage1 = nn.ModuleList(
            [EfficientMixAttnTransformerBlock(
                dim=curr_dim,
                input_resolution=img_size,
                num_heads_w=num_heads[0],
                num_heads_s=num_heads[0],
                window_size= [1,1,1,1,1,1,1,1],
                window_shift=i % 2 == 0,
                stripe_size=[8, 8],
                stripe_groups=[None, None],
                stripe_type="H" if i % 2 == 0 else "W",
                stripe_shift= i % 4 in [2, 3] ,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type="linear",
                anchor_proj_type="avgpool",
                anchor_one_stage=True,
                anchor_window_down_factor=1,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                pretrained_window_size=[0, 0],
                pretrained_stripe_size=[0, 0],
                res_scale=0.1  ,
                args=None,
            )
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim*(heads[1]//heads[0]))
        curr_dim = curr_dim*(heads[1]//heads[0])
        self.norm2 = nn.LayerNorm(curr_dim)
        self.stage2 = nn.ModuleList(
            [EfficientMixAttnTransformerBlock(
                dim=curr_dim,
                input_resolution=img_size,
                num_heads_w=num_heads[1],
                num_heads_s=num_heads[1],
                window_size= [1,1,1,1,1,1,1,1],
                window_shift=i % 2 == 0,
                stripe_size=[8, 8],
                stripe_groups=[None, None],
                stripe_type="H" if i % 2 == 0 else "W",
                stripe_shift=i % 4 in [2, 3] ,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type="linear",
                anchor_proj_type="avgpool",
                anchor_one_stage=True,
                anchor_window_down_factor=1,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                pretrained_window_size=[0, 0],
                pretrained_stripe_size=[0, 0],
                res_scale=0.1,
                args=None,
            )
             for i in range(depth[1])])
        
        self.merge2 = Merge_Block(curr_dim, curr_dim*(heads[2]//heads[1]))
        curr_dim = curr_dim*(heads[2]//heads[1])
        self.norm3 = nn.LayerNorm(curr_dim)
        temp_stage3 = []
        temp_stage3.extend(
            [EfficientMixAttnTransformerBlock(
                dim=curr_dim,
                input_resolution=img_size,
                num_heads_w=num_heads[2],
                num_heads_s=num_heads[2],
                window_size= [1,1,1,1,1,1,1,1],
                window_shift=i % 2 == 0,
                stripe_size=[8, 8],
                stripe_groups=[None, None],
                stripe_type="H" if i % 2 == 0 else "W",
                stripe_shift= i % 4 in [2, 3]  , #False,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type="linear",
                anchor_proj_type="avgpool",
                anchor_one_stage=True,
                anchor_window_down_factor=1,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                pretrained_window_size=[0, 0],
                pretrained_stripe_size=[0, 0],
                res_scale=0.1,
                args=None,
            )
             for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)
        self.merge3 = Merge_Block(curr_dim, curr_dim * (heads[3] // heads[2]))
        curr_dim = curr_dim * (heads[3] // heads[2])
        self.stage4 = nn.ModuleList(
            [EfficientMixAttnTransformerBlock(
                dim=curr_dim,
                input_resolution=img_size,
                num_heads_w=num_heads[3],
                num_heads_s=num_heads[3],
                window_size= [1,1,1,1,1,1,1,1],
                window_shift=i % 2 == 0,
                stripe_size=[8, 8],
                stripe_groups=[None, None],
                stripe_type="H" if i % 2 == 0 else "W",
                stripe_shift=i % 4 in [2, 3] ,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type="linear",
                anchor_proj_type="avgpool",
                anchor_one_stage=True,
                anchor_window_down_factor=1,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                pretrained_window_size=[0, 0],
                pretrained_stripe_size=[0, 0],
                res_scale=0.1,
                args=None,
            )
             for i in range(depth[3])])
        self.norm4 = nn.LayerNorm(curr_dim)

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.PositionalEncoding = PositionalEncoding(num_pos_feats_x=16, num_pos_feats_y=16,
                                                     num_pos_feats_z=16)  # (num_pos_feats_x=None, num_pos_feats_y=None, num_pos_feats_z=32)#
        self.DarkChannel = DarkChannel()

    stripe_size = [8, 8] # used for stripe window attention
    stripe_groups =[None, None] #[1, 1]
    window_size = [1, 1, 1, 1, 1, 1, 1, 1]#[3,3,3,3,3,3,3,3]#   [2, 2, 2, 2, 2, 2, 2] #
    stripe_shift =[w // 2 for w in  window_size] #False,

    pretrained_window_size = [0, 0]
    pretrained_stripe_size = [0, 0]
    conv_type = "1conv"
    init_method = "n"  # initialization method of the weight parameters used to train large scale models.
    fairscale_checkpoint = False   # fairscale activation checkpointing
    offload_to_cpu = False
    euclidean_dist = False
    window_size = to_2tuple(window_size)
    shift_size = [w // 2 for w in window_size]
    def set_table_index_mask(self, x_size):
        """
        Two used cases:
        1) At initialization: set the shared buffers.
        2) During forward pass: get the new buffers if the resolution of the input changes
        """
        # ss - stripe_size, sss - stripe_shift_size
        ss, sss = _get_stripe_info(self.stripe_size, self.stripe_groups, True, x_size)
        df =1 #self.anchor_window_down_factor

        table_w = get_relative_coords_table_all(
            self.window_size, self.pretrained_window_size
        )
        table_sh = get_relative_coords_table_all(ss, self.pretrained_stripe_size, df)
        table_sv = get_relative_coords_table_all(
            ss[::-1], self.pretrained_stripe_size, df
        )

        index_w = get_relative_position_index_simple(self.window_size)
        index_sh_a2w = get_relative_position_index_simple(ss, df, False)
        index_sh_w2a = get_relative_position_index_simple(ss, df, True)
        index_sv_a2w = get_relative_position_index_simple(ss[::-1], df, False)
        index_sv_w2a = get_relative_position_index_simple(ss[::-1], df, True)

        mask_w = calculate_mask(x_size, self.window_size, self.shift_size)
        mask_sh_a2w = calculate_mask_all(x_size, ss, sss, df, False)
        mask_sh_w2a = calculate_mask_all(x_size, ss, sss, df, True)
        mask_sv_a2w = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, False)
        mask_sv_w2a = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, True)
        return {
            "table_w": table_w,
            "table_sh": table_sh,
            "table_sv": table_sv,
            "index_w": index_w,
            "index_sh_a2w": index_sh_a2w,
            "index_sh_w2a": index_sh_w2a,
            "index_sv_a2w": index_sv_a2w,
            "index_sv_w2a": index_sv_w2a,
            "mask_w": mask_w,
            "mask_sh_a2w": mask_sh_a2w,
            "mask_sh_w2a": mask_sh_w2a,
            "mask_sv_a2w": mask_sv_a2w,
            "mask_sv_w2a": mask_sv_w2a,
        }

    def get_table_index_mask(self, device=None, input_resolution=None):
        # Used during forward pass
        if input_resolution == 128:
            return {
                "table_w": self.table_w,
                "table_sh": self.table_sh,
                "table_sv": self.table_sv,
                "index_w": self.index_w,
                "index_sh_a2w": self.index_sh_a2w,
                "index_sh_w2a": self.index_sh_w2a,
                "index_sv_a2w": self.index_sv_a2w,
                "index_sv_w2a": self.index_sv_w2a,
                "mask_w": self.mask_w,
                "mask_sh_a2w": self.mask_sh_a2w,
                "mask_sh_w2a": self.mask_sh_w2a,
                "mask_sv_a2w": self.mask_sv_a2w,
                "mask_sv_w2a": self.mask_sv_w2a,
            }
        else:
            table_index_mask = self.set_table_index_mask(input_resolution)
            for k, v in table_index_mask.items():
                table_index_mask[k] = v.to(device)
            return table_index_mask

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def save_out(self, x, norm, H, W):
        x = norm(x)
        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x


    def forward_features(self, x):
        # B = x.shape[0]
        depth_map = self.DarkChannel(x)
        # B, C, H, W = x.size()
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)

        depth_pool = F.interpolate(depth_map, size=(Wh, Ww), mode='bicubic')
        absolute_pos_embed = self.PositionalEncoding(x, depth_pool)
        x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C .reshape(B, C, -1).transpose(-1,-2).contiguous()#

        H, W=Wh, Ww
        x_size = (H, W)
        table_index_mask = self.get_table_index_mask(x.device, x_size)

        out = []
        for blk in self.stage1:
            blk.H = H
            blk.W = W
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x,x_size,table_index_mask)+x#,H, W

        out.append(self.save_out(x, self.norm1, H, W))#self.save_out(x, self.norm1, H, W)  #x

        for pre, blocks, norm in zip([self.merge1, self.merge2,self.merge3  ],
                                     [self.stage2, self.stage3, self.stage4 ],
                                     [self.norm2 , self.norm3, self.norm4 ]):

            x, H, W = pre(x, H, W)
            for blk in blocks:
                blk.H = H
                blk.W = W
                x_size = (H, W)
                table_index_mask = self.get_table_index_mask(x.device, x_size)
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x,x_size,table_index_mask)+x#, H, W

            out.append(self.save_out(x, norm, H, W))#self.save_out(x, norm, H, W) #x
        return tuple(out)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class CSWinDe(nn.Module):
    """ Vis0ion Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=64, depth=[2, 2, 2], split_size=7,
                 num_heads=[3, 6, 9], mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, patch_norm=True, use_chk=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_norm = patch_norm
        heads = num_heads
        self.use_chk = use_chk

        curr_dim = embed_dim

        self.up0 = Up_Block(curr_dim, curr_dim //(heads[0] // heads[1]))
        curr_dim = curr_dim // (heads[0] // heads[1])
        self.norm1 = nn.LayerNorm(curr_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList(
            [EfficientMixAttnTransformerBlock(
                dim=curr_dim,
                input_resolution=img_size,
                num_heads_w=num_heads[0],
                num_heads_s=num_heads[0],
                window_size=[1,1,1,1,1,1,1,1],
                window_shift=i % 2 == 0,
                stripe_size=[8, 8],
                stripe_groups=[None, None],
                stripe_type="H" if i % 2 == 0 else "W",
                stripe_shift=False,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type="linear",
                anchor_proj_type="avgpool",
                anchor_one_stage=True,
                anchor_window_down_factor=1,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                pretrained_window_size=[0, 0],
                pretrained_stripe_size=[0, 0],
                res_scale=0.1,
                args=None,

            )
             for i in range(depth[0])])

        self.up1 = Up_Block(curr_dim, curr_dim// (heads[0] // heads[1]))
        curr_dim = curr_dim // (heads[0] // heads[1])
        self.norm2 = nn.LayerNorm(curr_dim)
        self.stage2 = nn.ModuleList(
            [EfficientMixAttnTransformerBlock(
                dim=curr_dim,
                input_resolution=img_size,
                num_heads_w=num_heads[1],
                num_heads_s=num_heads[1],
                window_size=[1,1,1,1,1,1,1,1],
                window_shift=i % 2 == 0,
                stripe_size=[8, 8],
                stripe_groups=[None, None],
                stripe_type="H" if i % 2 == 0 else "W",
                stripe_shift=False,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type="linear",
                anchor_proj_type="avgpool",
                anchor_one_stage=True,
                anchor_window_down_factor=1,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                pretrained_window_size=[0, 0],
                pretrained_stripe_size=[0, 0],
                res_scale=0.1,
                args=None,
            )
             for i in range(depth[1])])

        self.up2 = Up_Block(curr_dim, curr_dim // (heads[1] // heads[2]))
        curr_dim = curr_dim //(heads[1] // heads[2])
        self.norm3 = nn.LayerNorm(curr_dim)
        temp_stage3 = []
        temp_stage3.extend(
            [EfficientMixAttnTransformerBlock(
                dim=curr_dim,
                input_resolution=img_size,
                num_heads_w=num_heads[2],
                num_heads_s=num_heads[2],
                window_size=[1,1,1,1,1,1,1,1],
                window_shift=i % 2 == 0,
                stripe_size=[8, 8],
                stripe_groups=[None, None],
                stripe_type="H" if i % 2 == 0 else "W",
                stripe_shift=False,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type="linear",
                anchor_proj_type="avgpool",
                anchor_one_stage=True,
                anchor_window_down_factor=1,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                pretrained_window_size=[0, 0],
                pretrained_stripe_size=[0, 0],
                res_scale=0.1,
                args=None,
            )
             for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)

    stripe_size = [8, 8]  # used for stripe window attention
    stripe_groups = [None, None]
    stripe_shift = False,
    window_size = [1, 1, 1, 1, 1, 1, 1, 1]
    pretrained_window_size = [0, 0]
    pretrained_stripe_size = [0, 0]
    conv_type = "1conv"
    init_method = "n"  # initialization method of the weight parameters used to train large scale models.
    fairscale_checkpoint = False  # fairscale activation checkpointing
    offload_to_cpu = False
    euclidean_dist = False
    window_size = to_2tuple(window_size)
    shift_size = [w // 2 for w in window_size]
    def set_table_index_mask(self, x_size):
        """
        Two used cases:
        1) At initialization: set the shared buffers.
        2) During forward pass: get the new buffers if the resolution of the input changes
        """
        # ss - stripe_size, sss - stripe_shift_size
        ss, sss = _get_stripe_info(self.stripe_size, self.stripe_groups, True, x_size)
        df = 1  # self.anchor_window_down_factor

        table_w = get_relative_coords_table_all(
            self.window_size, self.pretrained_window_size
        )
        table_sh = get_relative_coords_table_all(ss, self.pretrained_stripe_size, df)
        table_sv = get_relative_coords_table_all(
            ss[::-1], self.pretrained_stripe_size, df
        )

        index_w = get_relative_position_index_simple(self.window_size)
        index_sh_a2w = get_relative_position_index_simple(ss, df, False)
        index_sh_w2a = get_relative_position_index_simple(ss, df, True)
        index_sv_a2w = get_relative_position_index_simple(ss[::-1], df, False)
        index_sv_w2a = get_relative_position_index_simple(ss[::-1], df, True)

        mask_w = calculate_mask(x_size, self.window_size, self.shift_size)
        mask_sh_a2w = calculate_mask_all(x_size, ss, sss, df, False)
        mask_sh_w2a = calculate_mask_all(x_size, ss, sss, df, True)
        mask_sv_a2w = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, False)
        mask_sv_w2a = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, True)
        return {
            "table_w": table_w,
            "table_sh": table_sh,
            "table_sv": table_sv,
            "index_w": index_w,
            "index_sh_a2w": index_sh_a2w,
            "index_sh_w2a": index_sh_w2a,
            "index_sv_a2w": index_sv_a2w,
            "index_sv_w2a": index_sv_w2a,
            "mask_w": mask_w,
            "mask_sh_a2w": mask_sh_a2w,
            "mask_sh_w2a": mask_sh_w2a,
            "mask_sv_a2w": mask_sv_a2w,
            "mask_sv_w2a": mask_sv_w2a,
        }

    def get_table_index_mask(self, device=None, input_resolution=None):
        # Used during forward pass
        if input_resolution == 128:
            return {
                "table_w": self.table_w,
                "table_sh": self.table_sh,
                "table_sv": self.table_sv,
                "index_w": self.index_w,
                "index_sh_a2w": self.index_sh_a2w,
                "index_sh_w2a": self.index_sh_w2a,
                "index_sv_a2w": self.index_sv_a2w,
                "index_sv_w2a": self.index_sv_w2a,
                "mask_w": self.mask_w,
                "mask_sh_a2w": self.mask_sh_a2w,
                "mask_sh_w2a": self.mask_sh_w2a,
                "mask_sv_a2w": self.mask_sv_a2w,
                "mask_sv_w2a": self.mask_sv_w2a,
            }
        else:
            table_index_mask = self.set_table_index_mask(input_resolution)
            for k, v in table_index_mask.items():
                table_index_mask[k] = v.to(device)
            return table_index_mask

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def save_out(self, x, norm, H, W):
        x = norm(x)
        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    def save_image(self, x, norm, H, W):
        x = norm(x)
        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x
    def forward_features(self, csencoder):

        Wh, Ww = csencoder[3].size(2),  csencoder[3].size(3)
        x =csencoder[3].flatten(2).transpose(1, 2)
        # x = self.patch_embed(x)
        H, W = Wh, Ww
        out = []

        for pre, blocks, norm in zip([self.up0  ],
                                     [self.stage1  ],
                                     [self.norm1  ]):

            x, H, W = pre(x, H, W)
            for blk in blocks:
                blk.H = H
                blk.W = W
                x_size = (H, W)
                table_index_mask = self.get_table_index_mask(x.device, x_size)
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x+csencoder[2].flatten(2).transpose(1, 2),x_size, table_index_mask)+x#, H, W
            out.append(self.save_out(x, norm, H, W))#self.save_out(x, norm, H, W)
        for pre, blocks, norm in zip([ self.up1],
                                         [ self.stage2],
                                         [self.norm2]):

            x, H, W = pre(x, H, W)
            for blk in blocks:
                blk.H = H
                blk.W = W
                x_size = (H, W)
                table_index_mask = self.get_table_index_mask(x.device, x_size)
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:

                    x = blk(x+csencoder[1].flatten(2).transpose(1, 2),x_size, table_index_mask)+x#, H, W

            out.append(self.save_image(x, norm, H, W))#self.save_image(x, norm, H, W)
        for pre, blocks, norm in zip([ self.up2],
                                     [ self.stage3],
                                     [self.norm3]):

            x, H, W = pre(x, H, W)
            for blk in blocks:
                blk.H = H
                blk.W = W
                x_size = (H, W)
                table_index_mask = self.get_table_index_mask(x.device, x_size)
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x+csencoder[0].flatten(2).transpose(1, 2),x_size, table_index_mask)+x#, H, W

            out.append(self.save_image(x, norm, H, W))#self.save_image(x, norm, H, W)

        return tuple(out)

    def forward(self, csencoder):
        x = self.forward_features( csencoder)
        return x

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict



class DarkChannel(nn.Module):
    def __init__(self, kernel_size=15):
        super(DarkChannel, self).__init__()
        self.kernel_size = kernel_size
        self.pad_size = (self.kernel_size - 1) // 2
        self.unfold = nn.Unfold(self.kernel_size)
    def forward(self, x):
        # x : (B, 3, H, W), in [-1, 1]
        # x = (x + 1.0) / 2.0
        H, W = x.size()[2], x.size()[3]

        # maximum among three channels
        x, _ = x.min(dim=1, keepdim=True)  # (B, 1, H, W)
        x = nn.ReflectionPad2d(self.pad_size)(x)  # (B, 1, H+2p, W+2p)
        x = self.unfold(x)  # (B, k*k, H*W)
        x = x.unsqueeze(1)  # (B, 1, k*k, H*W)

        # maximum in (k, k) patch
        dark_map, _ = x.min(dim=2, keepdim=False)  # (B, 1, H*W)
        x = dark_map.view(-1, 1, H, W)
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x

import math
class PositionalEncoding(nn.Module):
    def __init__(self, num_pos_feats_x=64, num_pos_feats_y=64, num_pos_feats_z=128, temperature=10000, normalize=True,
                 scale=None):
        super().__init__()
        self.num_pos_feats_x = num_pos_feats_x
        self.num_pos_feats_y = num_pos_feats_y
        self.num_pos_feats_z = num_pos_feats_z
        self.num_pos_feats = max(num_pos_feats_x, num_pos_feats_y, num_pos_feats_z)
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, depth):
        b, c, h, w = x.size()
        b_d, c_d, h_d, w_d = depth.size()
        assert b == b_d and c_d == 1 and h == h_d and w == w_d

        if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
            y_embed = torch.arange(h, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(b, 1, w)
            x_embed = torch.arange(w, dtype=torch.float32, device=x.device).repeat(b, h, 1)
        z_embed = depth.squeeze().to(dtype=torch.float32, device=x.device)
        # z_embed2 = depth2.squeeze().to(dtype=torch.float32, device=x.device)

        if self.normalize:
            eps = 1e-6
            if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
                y_embed = y_embed / (y_embed.max() + eps) * self.scale
                x_embed = x_embed / (x_embed.max() + eps) * self.scale
            z_embed_max, _ = z_embed.reshape(b, -1).max(1)
            # z_embed_max2, _ = z_embed2.reshape(b, -1).max(1)
            z_embed = z_embed / (z_embed_max[:, None, None] + eps) * self.scale
            # z_embed2 = z_embed2 / (z_embed_max2[:, None, None] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
            pos_x = x_embed[:, :, :, None] / dim_t[:self.num_pos_feats_x]
            pos_y = y_embed[:, :, :, None] / dim_t[:self.num_pos_feats_y]
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos_z = z_embed[:, :, :, None] / dim_t[:self.num_pos_feats_z]
        pos_z = torch.stack((pos_z[:, :, :, 0::2].sin(), pos_z[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # pos_z2 = z_embed2[:, :, :, None] / dim_t[:self.num_pos_feats_z]
        # pos_z2 = torch.stack((pos_z2[:, :, :, 0::2].sin(), pos_z2[:, :, :, 1::2].cos()), dim=4).flatten(3)

        if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
            pos = torch.cat((pos_x, pos_y,  pos_z), dim=3).permute(0, 3, 1, 2)
        else:
            pos = pos_z.permute(0, 3, 1, 2)
        return pos
