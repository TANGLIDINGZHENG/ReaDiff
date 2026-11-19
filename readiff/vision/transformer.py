"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""

from einops import rearrange
import torch
from torch import nn

from .attention import TCAttentionBlock


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, design_details):
        super().__init__()
        self.T = design_details['temporal_length']
        self.resblocks = nn.Sequential(*[
            TCAttentionBlock(width, heads, None, i, design_details) for i in range(layers)
        ])

    def forward(self, x, tome_info):
        frames = self.T
        patches = x.size(1) // frames
        for block in self.resblocks:
            x, _ = block(x, tome_info=tome_info)
        frame_tokens = rearrange(x[:, :frames * patches], "B (T N) D -> B T N D",
                                 T=frames, N=patches)
        context = x[:, frames * patches:]
        tokens = torch.cat([frame_tokens[:, :, 0], context], dim=1)
        return tokens.unsqueeze(1), context
