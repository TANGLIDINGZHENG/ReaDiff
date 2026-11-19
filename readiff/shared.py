"""Shared visual backbone, motion decoder and feature normalization."""
import torch
from torch import nn

from .decoder import Decoder
from .vision.encoder import TCVisionTransformer


class SharedModules(nn.Module):
    def __init__(self):
        super().__init__()
        design = dict(temporal_length=16, positional_embedding_type="space",
                      local_global_bias=True, context_token_k=96, seed_token_a=0.3,
                      tome_r=100, tome_d=0)
        self.image_encoder = TCVisionTransformer(
            input_resolution=224, patch_size=16, num_frames=16, width=768,
            layers=12, heads=12, output_dim=512, design_details=design
        )
        self.decoder = Decoder()
        self.register_buffer("mean", torch.empty(263))
        self.register_buffer("std", torch.empty(263))

    def encode(self, frames):
        rgb, context = self.image_encoder(frames[:, :, :3].contiguous())
        flow, _ = self.image_encoder(frames[:, :, 3:].contiguous())
        global_video = torch.cat((rgb.mean(1), flow.mean(1)), dim=-1)
        local_video = torch.cat((rgb, flow), dim=-1)
        return global_video, dict(local=local_video, context=context)
