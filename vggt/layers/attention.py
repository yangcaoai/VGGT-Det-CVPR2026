# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
from torch import softmax
import torch.nn.functional as F
import torch
XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None, 
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, save_vis_attn=False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        # debug #
        # self.fused_attn=False
        # if B==40 or B==1: # bs=1  or Global attn
        #     self.fused_attn=False
        #########

        if save_vis_attn:
            # import time
            # import torch
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)

            # start_event.record()
            # self.last_attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)

            seq_len = q.size(-2)  # 773
            identity_v = torch.eye(
                seq_len,
                device=q.device,
                dtype=q.dtype
            )  # shape [773, 773]

            identity_v = identity_v.view(1, 1, seq_len, seq_len)             # [1, 1, 773, 773]
            identity_v = identity_v.expand(q.shape[0], q.shape[1], seq_len, seq_len) 

            self.last_attn = F.scaled_dot_product_attention(
                q,
                k,
                identity_v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            del identity_v
            # end_event.record()
            # torch.cuda.synchronize()
            # print(1)
            # print(f"Attention computation time: {start_event.elapsed_time(end_event):.3f} ms")

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

            # if B==40:   # frame attn
            #     from torch import save
            #     save_dir = "attn_layer_visualization/frame_attn"
            #     os.makedirs(save_dir, exist_ok=True)

            #     if not hasattr(self, 'current_layer'):
            #         self.current_layer = 0
                    
            #     for view_n in observe_view:
            #         filename = f"frame_attn_view{view_n}_layer{self.current_layer}.pth"
            #         save_path = os.path.join(save_dir, filename)
            #         save(attn_mean.cpu(), save_path)

            #     self.current_layer += 1

            # if B==1:   # global attn
            #     from torch import save
            #     save_dir = "attn_layer_visualization/global_attn"
            #     os.makedirs(save_dir, exist_ok=True)

            #     if not hasattr(self, 'current_layer'):
            #         self.current_layer = 0

            #     for view_n in observe_view:
            #         filename = f"global_attn_view{view_n}_layer{self.current_layer}.pth"
            #         save_path = os.path.join(save_dir, filename)
            #         save(attn_mean.cpu(), save_path)

            #     self.current_layer += 1

        
            
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, save_vis_attn=False) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x,save_vis_attn=save_vis_attn)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
