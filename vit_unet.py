#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


def _build_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int,
                               device: torch.device, dtype: torch.dtype) -> Tensor:
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for sin-cos positional embedding"
    omega = torch.arange(embed_dim // 4, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 4)))

    y = torch.arange(grid_h, device=device, dtype=dtype)
    x = torch.arange(grid_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    yy = yy.reshape(-1)[:, None] * omega[None, :]
    xx = xx.reshape(-1)[:, None] * omega[None, :]

    pos_embed = torch.cat((torch.sin(yy), torch.cos(yy), torch.sin(xx), torch.cos(xx)), dim=1)
    return pos_embed.unsqueeze(0)


class ViTSegmenter(nn.Module):
    """Pure ViT encoder with lightweight convolutional decoder for CT segmentation.

    The model tokenizes patches, processes them with a ViT-B/16 encoder, and
    relies on progressive transposed convolutions to reconstruct full-resolution
    predictions without UNet-style skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        dropout: float = 0.1,
        **unused_kwargs,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size,
                                     padding=0,
                                     bias=False)
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.decoder_blocks = nn.ModuleList()
        in_c = embed_dim
        for _ in range(4):  # 2^4 = 16 to recover patch_size strides
            out_c = max(in_c // 2, max(num_classes, 32))
            block = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_c),
                nn.GELU(),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.GELU(),
            )
            self.decoder_blocks.append(block)
            in_c = out_c

        self.seg_head = nn.Conv2d(in_c, num_classes, kernel_size=1)

    def _pad_input(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (pad_h, pad_w)

    def forward(self, x: Tensor) -> Tensor:
        orig_h, orig_w = x.shape[-2:]
        x, pads = self._pad_input(x)

        patches = self.patch_embed(x)
        b, embed_dim, h, w = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)
        pos_embed = _build_2d_sincos_pos_embed(self.embed_dim, h, w, tokens.device, tokens.dtype)
        tokens = self.pos_drop(tokens + pos_embed)
        tokens = self.transformer(tokens)
        features = tokens.transpose(1, 2).reshape(b, self.embed_dim, h, w)

        for block in self.decoder_blocks:
            features = block(features)

        logits = self.seg_head(features)
        pad_h, pad_w = pads
        if pad_h or pad_w:
            logits = logits[..., :orig_h, :orig_w]
        return logits

    def init_weights(self, *args, **kwargs) -> None:
        def _init_fn(module: nn.Module) -> None:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.apply(_init_fn)
