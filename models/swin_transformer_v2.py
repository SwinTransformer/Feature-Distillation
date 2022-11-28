# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.utils import weight_norm
from torch import Tensor, Size
from typing import Union, List
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np

_shape_t = Union[int, List[int], Size]


class LayerNormFP32(nn.LayerNorm):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(LayerNormFP32, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).type_as(input)


class LinearFP32(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearFP32, self).__init__(in_features, out_features, bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input.float(), self.weight.float(),
                        self.bias.float() if self.bias is not None else None)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 norm_layer=None, mlpfp32=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlpfp32 = mlpfp32

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        if norm_layer is not None:
            self.norm = norm_layer(hidden_features)
        else:
            self.norm = None

    def forward(self, x):
        x = self.fc1(x)
        if self.norm:
            x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        if self.mlpfp32:
            x = self.fc2.float()(x.type(torch.float32))
            x = self.drop.float()(x)
            # print(f"======>[MLP FP32]")
        else:
            x = self.fc2(x)
            x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class CRPB(nn.Module):
    def __init__(self, num_heads, window_size, relative_coords_table_type='norm8_log', rpe_hidden_dim=512, rpe_output_type='normal'):
        super().__init__()

        self.relative_coords_table_type = relative_coords_table_type
        self.window_size = window_size
        self.num_heads = num_heads
        self.rpe_output_type = rpe_output_type
        if self.relative_coords_table_type != "none":
            # mlp to generate table of relative position bias
            self.rpe_mlp = nn.Sequential(nn.Linear(2, rpe_hidden_dim, bias=True),
                                         nn.ReLU(inplace=True),
                                         LinearFP32(rpe_hidden_dim, num_heads, bias=False))

            # get relative_coords_table
            relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            relative_coords_table = torch.stack(
                torch.meshgrid([relative_coords_h,
                                relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
            if relative_coords_table_type == 'linear':
                relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            elif relative_coords_table_type == 'norm8_log':
                relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
            elif relative_coords_table_type == 'norm8_log_192to224':
                if self.window_size[0] == 14:
                    relative_coords_table[:, :, :, 0] /= (11)
                    relative_coords_table[:, :, :, 1] /= (11)
                elif self.window_size[0] == 7:
                    relative_coords_table[:, :, :, 0] /= (5)
                    relative_coords_table[:, :, :, 1] /= (5)
                else:
                    raise NotImplementedError
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
            elif relative_coords_table_type == 'norm8_log_192to256':
                if self.window_size[0] == 16:
                    relative_coords_table[:, :, :, 0] /= (11)
                    relative_coords_table[:, :, :, 1] /= (11)
                elif self.window_size[0] == 8:
                    relative_coords_table[:, :, :, 0] /= (5)
                    relative_coords_table[:, :, :, 1] /= (5)
                else:
                    raise NotImplementedError
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
            elif relative_coords_table_type == 'norm8_log_192to384':
                if self.window_size[0] == 24:
                    relative_coords_table[:, :, :, 0] /= (11)
                    relative_coords_table[:, :, :, 1] /= (11)
                elif self.window_size[0] == 12:
                    relative_coords_table[:, :, :, 0] /= (5)
                    relative_coords_table[:, :, :, 1] /= (5)
                else:
                    raise NotImplementedError
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
            elif relative_coords_table_type == 'norm8_log_256to384':
                if self.window_size[0] == 24:
                    relative_coords_table[:, :, :, 0] /= (15)
                    relative_coords_table[:, :, :, 1] /= (15)
                elif self.window_size[0] == 12:
                    relative_coords_table[:, :, :, 0] /= (7)
                    relative_coords_table[:, :, :, 1] /= (7)
                else:
                    raise NotImplementedError
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
            else:
                raise NotImplementedError
            self.register_buffer("relative_coords_table", relative_coords_table)
        else:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            trunc_normal_(self.relative_position_bias_table, std=.02)
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias_table = self.rpe_mlp(self.relative_coords_table).view(-1, self.num_heads)

        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # befault we use sigmoid
        if self.rpe_output_type == 'normal':
            pass
        elif self.rpe_output_type == 'sigmoid':
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        else:
            raise NotImplementedError
        return relative_position_bias


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 relative_coords_table_type='norm8_log', rpe_hidden_dim=512,
                 rpe_output_type='normal', attn_type='normal', mlpfp32=False, use_shared_rel_pos_bias=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.mlpfp32 = mlpfp32
        self.attn_type = attn_type
        self.rpe_output_type = rpe_output_type
        self.relative_coords_table_type = relative_coords_table_type
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias

        if self.attn_type == 'cosine_mh':
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        elif self.attn_type == 'normal':
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim ** -0.5
        else:
            raise NotImplementedError()
        if not use_shared_rel_pos_bias and self.relative_coords_table_type != "none":
            # mlp to generate table of relative position bias
            self.rpe_mlp = nn.Sequential(nn.Linear(2, rpe_hidden_dim, bias=True),
                                         nn.ReLU(inplace=True),
                                         LinearFP32(rpe_hidden_dim, num_heads, bias=False))

            # get relative_coords_table
            relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            relative_coords_table = torch.stack(
                torch.meshgrid([relative_coords_h,
                                relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
            if relative_coords_table_type == 'linear':
                relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            elif relative_coords_table_type == 'norm8_log':
                relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
            else:
                raise NotImplementedError
            self.register_buffer("relative_coords_table", relative_coords_table)
        else:
            if not use_shared_rel_pos_bias:
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
                trunc_normal_(self.relative_position_bias_table, std=.02)
        if not use_shared_rel_pos_bias:
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, relative_position_bias=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
            relative_position_bias: pass in when using shared rpb between blocks
        """
        B_, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        if self.attn_type == 'cosine_mh':
            q = F.normalize(q.float(), dim=-1)
            k = F.normalize(k.float(), dim=-1)
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to(self.logit_scale.device))).exp()
            attn = (q @ k.transpose(-2, -1)) * logit_scale.float()
        elif self.attn_type == 'normal':
            q = q * self.scale
            attn = (q.float() @ k.float().transpose(-2, -1))
        else:
            raise NotImplementedError()

        if not self.use_shared_rel_pos_bias and self.relative_coords_table_type != "none":
            # relative_position_bias_table: 2*Wh-1 * 2*Ww-1, nH
            relative_position_bias_table = self.rpe_mlp(self.relative_coords_table).view(-1, self.num_heads)
        else:
            if not self.use_shared_rel_pos_bias:
                relative_position_bias_table = self.relative_position_bias_table
        if not self.use_shared_rel_pos_bias:
            relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            if self.rpe_output_type == 'normal':
                pass
            elif self.rpe_output_type == 'sigmoid':
                relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
            else:
                raise NotImplementedError

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = attn.type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        if self.mlpfp32:
            x = self.proj.float()(x.type(torch.float32))
            x = self.proj_drop.float()(x)
            # print(f"======>[ATTN FP32]")
        else:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 use_mlp_norm=False, endnorm=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 relative_coords_table_type='norm8_log', rpe_hidden_dim=512,
                 rpe_output_type='normal', attn_type='normal', mlpfp32=False, use_shared_rel_pos_bias=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_mlp_norm = use_mlp_norm
        self.endnorm = endnorm
        self.mlpfp32 = mlpfp32

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            relative_coords_table_type=relative_coords_table_type, rpe_output_type=rpe_output_type,
            rpe_hidden_dim=rpe_hidden_dim, attn_type=attn_type, mlpfp32=mlpfp32, use_shared_rel_pos_bias=use_shared_rel_pos_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_layer=norm_layer if self.use_mlp_norm else None, mlpfp32=mlpfp32)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        if self.endnorm:
            self.enorm = norm_layer(dim)
        else:
            self.enorm = None

    def forward(self, x, relative_position_bias=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, with L = {L}, H = {H}, W = {W}"

        shortcut = x

        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        orig_type = x.dtype  # attn may force to fp32
        attn_windows = self.attn(x_windows, mask=self.attn_mask, relative_position_bias=relative_position_bias)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        if self.mlpfp32:
            x = self.norm1.float()(x)
            x = x.type(orig_type)
        else:
            x = self.norm1(x)
        x = shortcut + self.drop_path(x)
        shortcut = x

        orig_type = x.dtype
        x = self.mlp(x)
        if self.mlpfp32:
            x = self.norm2.float()(x)
            x = x.type(orig_type)
        else:
            x = self.norm2(x)
        x = shortcut + self.drop_path(x)

        if self.endnorm:
            x = self.enorm(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        init_values (float | None, optional): post norm init value. Default: None
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, checkpoint_blocks=255,
                 init_values=None, endnorm_interval=-1, use_mlp_norm=False,
                 relative_coords_table_type='norm8_log', rpe_hidden_dim=512,
                 rpe_output_type='normal', attn_type='normal', mlpfp32_blocks=[-1], use_shared_rel_pos_bias=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.checkpoint_blocks = checkpoint_blocks
        self.init_values = init_values if init_values is not None else 0.0
        self.endnorm_interval = endnorm_interval
        self.mlpfp32_blocks = mlpfp32_blocks
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                     num_heads=num_heads, window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer,
                                     use_mlp_norm=use_mlp_norm,
                                     endnorm=True if ((i + 1) % endnorm_interval == 0) and (
                                             endnorm_interval > 0) else False,
                                     relative_coords_table_type=relative_coords_table_type,
                                     rpe_hidden_dim=rpe_hidden_dim,
                                     rpe_output_type=rpe_output_type,
                                     use_shared_rel_pos_bias=use_shared_rel_pos_bias,
                                     attn_type=attn_type,
                                     mlpfp32=True if i in mlpfp32_blocks else False)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # build rpe
        if use_shared_rel_pos_bias:
            if min(self.input_resolution) <= window_size:
                # if window size is larger than input resolution, we don't partition windows
                window_size = min(self.input_resolution)

            self.rel_pos_bias = CRPB(num_heads, to_2tuple(window_size), relative_coords_table_type, rpe_hidden_dim, rpe_output_type)

    def forward(self, x):
        if self.use_shared_rel_pos_bias:
            rel_pos_bias = self.rel_pos_bias()
        else:
            rel_pos_bias = None
        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint and idx < self.checkpoint_blocks:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_block_norm_weights(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, self.init_values)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, self.init_values)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, init_values=1e-5, init_std=0.02,
                 endnorm_interval=-1, use_mlp_norm_layers=[],
                 relative_coords_table_type='norm8_log', rpe_hidden_dim=512,
                 attn_type='cosine_mh', rpe_output_type='sigmoid', rpe_wd=False, fix_init=True,
                 checkpoint_blocks=[255, 255, 255, 255],
                 use_shared_rel_pos_bias=False,
                 mlpfp32_layer_blocks=[[-1], [-1], [-1], [-1]], **kwargs):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.init_std = init_std
        self.endnorm_interval = endnorm_interval
        self.use_mlp_norm_layers = use_mlp_norm_layers
        self.relative_coords_table_type = relative_coords_table_type
        self.rpe_hidden_dim = rpe_hidden_dim
        self.rpe_output_type = rpe_output_type
        self.rpe_wd = rpe_wd
        self.attn_type = attn_type
        self.fix_init = fix_init

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=self.init_std)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                checkpoint_blocks=checkpoint_blocks[i_layer],
                init_values=init_values,
                endnorm_interval=endnorm_interval,
                use_mlp_norm=True if i_layer in use_mlp_norm_layers else False,
                relative_coords_table_type=self.relative_coords_table_type,
                rpe_hidden_dim=self.rpe_hidden_dim,
                rpe_output_type=self.rpe_output_type,
                attn_type=self.attn_type,
                use_shared_rel_pos_bias=use_shared_rel_pos_bias,
                mlpfp32_blocks=mlpfp32_layer_blocks[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if self.fix_init:
            self.fix_init_weight()
        for bly in self.layers:
            bly._init_block_norm_weights()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        block_id = 1
        for layer in self.layers:
            for block in layer.blocks:
                rescale(block.attn.proj.weight.data, block_id)
                rescale(block.mlp.fc2.weight.data, block_id)
                block_id += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if self.rpe_wd:
            return {"logit_scale"}
        else:
            return {"rpe_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head.float()(x.float())
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


def build_swin_v2(config):
    model = SwinTransformerV2(
        img_size=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.SWIN.PATCH_SIZE,
        in_chans=config.MODEL.SWIN.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=config.MODEL.SWIN.EMBED_DIM,
        depths=config.MODEL.SWIN.DEPTHS,
        num_heads=config.MODEL.SWIN.NUM_HEADS,
        window_size=config.MODEL.SWIN.WINDOW_SIZE,
        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        qk_scale=config.MODEL.SWIN.QK_SCALE,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        ape=config.MODEL.SWIN.APE,
        patch_norm=config.MODEL.SWIN.PATCH_NORM,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        relative_coords_table_type=config.DEV.RCT_TYPE,
        use_shared_rel_pos_bias=config.MODEL.SWIN.USE_SHARED_RPB,
        checkpoint_blocks=config.DEV.CHECKPOINT_BLOCKS)

    return model