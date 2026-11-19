"""Noise-adaptive conditional temporal denoiser for ReaDiff-S/M/L."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal(positions, width):
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=positions.device) / max(half - 1, 1)
    )
    phase = positions.float().unsqueeze(-1) * frequencies
    return torch.cat([phase.sin(), phase.cos()], dim=-1)


def valid_mask(lengths, size):
    if lengths.ndim != 1 or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("Token lengths must be a one-dimensional integer tensor")
    if bool(((lengths < 1) | (lengths > size)).any()):
        raise ValueError(f"Token lengths must be between 1 and {size}")
    return torch.arange(size, device=lengths.device)[None] < lengths[:, None]


class VideoFeatureAdapter(nn.Module):
    """Shared residual adapter for global/local tokens and RGB/flow streams."""
    def __init__(self, width=512, bottleneck=64):
        super().__init__()
        if not isinstance(bottleneck, int) or not 1 <= bottleneck <= width:
            raise ValueError("Video adapter bottleneck must be an integer in [1, feature width]")
        self.width = width
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, packed):
        if packed.shape[-1] not in (self.width, 2 * self.width):
            raise ValueError("Expected one or two packed TC-CLIP feature streams")
        # Separate the feature dimension, applying the SAME small module to
        # each frozen encoder stream before the existing flow fusion.
        features = packed.reshape(*packed.shape[:-1], -1, self.width)
        update = self.up(self.activation(self.down(self.norm(features))))
        return (features + update).reshape_as(packed)


class ReactionAdapter(nn.Module):
    def __init__(self, width, rank=64):
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(width, rank), nn.SiLU(), nn.Linear(rank, 9 * width))
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)
        self.local_norm = nn.LayerNorm(width)
        self.local_in = nn.Linear(width, rank)
        self.short = nn.Conv1d(rank, rank, 3, padding=1, groups=rank)
        self.long = nn.Conv1d(rank, rank, 7, padding=3, groups=rank)
        self.route = nn.Linear(width, 2 * rank)
        self.local_out = nn.Linear(rank, width)
        nn.init.zeros_(self.route.weight)
        nn.init.zeros_(self.route.bias)
        nn.init.zeros_(self.local_out.weight)
        nn.init.zeros_(self.local_out.bias)


    def coefficients(self, condition):
        return self.condition(condition).unsqueeze(1).chunk(9, dim=-1)

    def local_update(self, hidden, condition, valid):
        # Mask after biased projection and between temporal operators: padding
        # must never become evidence for a neighboring valid motion token.
        value = F.silu(self.local_in(self.local_norm(hidden)))
        value = value.masked_fill(~valid[..., None], 0).transpose(1, 2)
        weights = self.route(condition).view(len(hidden), 2, -1).softmax(dim=1)
        mixed = weights[:, 0, :, None] * self.short(value)
        mixed = mixed + weights[:, 1, :, None] * self.long(value)
        return self.local_out(F.silu(mixed.transpose(1, 2))).masked_fill(~valid[..., None], 0)

class ContextTokens(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, width)

    def forward(self, context):
        return self.projection(self.norm(context))


class Denoiser(nn.Module):
    def __init__(self, latent_dim, ff_size, n_layers, n_heads):
        super().__init__()
        code_dim, clip_dim, evidence_width = 512, 512, 128
        dropout = 0.2
        self.latent_dim = latent_dim
        self.input_proj = nn.Linear(code_dim, latent_dim)
        self.video_proj = nn.Linear(clip_dim, latent_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim)
        )
        layer = nn.TransformerDecoderLayer(
            latent_dim, n_heads, ff_size, dropout, activation="gelu", batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.output_proj = nn.Linear(latent_dim, code_dim)
        self.video_adapter = VideoFeatureAdapter(clip_dim, 64)
        self.video_fusion = nn.Linear(2 * clip_dim, clip_dim)
        self.adapters = nn.ModuleList([ReactionAdapter(latent_dim) for _ in range(n_layers)])
        self.video_evidence = ContextTokens(evidence_width)
        self.evidence_query = nn.Linear(self.latent_dim, evidence_width)
        self.evidence_key = nn.Linear(evidence_width, evidence_width)
        self.evidence_value = nn.Linear(evidence_width, evidence_width)
        self.evidence_output = nn.Linear(evidence_width, self.latent_dim)
        self.evidence_gates = nn.Parameter(torch.zeros(len(self.decoder.layers)))
        self.evidence_lag = nn.Parameter(torch.zeros(len(self.decoder.layers)))
        self.evidence_precision = nn.Parameter(torch.zeros(len(self.decoder.layers)))

    def prepare_condition(self, raw):
        return dict(local=raw["local"], memory=self.video_evidence(raw["context"]))

    def primary_condition(self, global_video, condition_data, drop_mask):
        memory = torch.cat((global_video[:, None], condition_data['local']), dim=1)
        memory = self.video_fusion(self.video_adapter(memory)).masked_fill(drop_mask[:, None, None], 0)
        projected = self.video_proj(memory)
        memory = projected + sinusoidal(torch.arange(projected.shape[1], device=projected.device), self.latent_dim)
        return projected.mean(dim=1), memory

    def forward(self, noisy, times, global_video, local_video, lengths, force_uncond=False):
        condition_data = local_video
        valid = valid_mask(lengths, noisy.shape[1])
        noisy = noisy.masked_fill(~valid[..., None], 0)
        drop_mask = torch.full((len(noisy),), force_uncond, dtype=torch.bool, device=noisy.device)
        summary, memory = self.primary_condition(global_video, condition_data, drop_mask)
        time = self.time_embed(sinusoidal(times, self.latent_dim))
        condition = F.layer_norm(time + summary, (self.latent_dim,))
        hidden = self.input_proj(noisy)
        hidden = hidden + sinusoidal(torch.arange(noisy.shape[1], device=noisy.device), self.latent_dim)
        hidden = hidden + time[:, None]
        evidence = condition_data['memory']
        b, k, width = evidence.shape
        keys = self.evidence_key(evidence).reshape(b, k, 4, width // 4).transpose(1, 2)
        values = self.evidence_value(evidence).reshape(b, k, 4, width // 4).transpose(1, 2)
        for index, (layer, adapter) in enumerate(zip(self.decoder.layers, self.adapters)):
            s1, b1, g1, s2, b2, g2, s3, b3, g3 = adapter.coefficients(condition)
            attended = layer.self_attn(hidden, hidden, hidden, key_padding_mask=~valid, need_weights=False)[0]
            hidden = layer.norm1(hidden + (1 + g1) * layer.dropout1(attended))
            hidden = hidden * (1 + s1) + b1
            attended = layer.multihead_attn(hidden, memory, memory, need_weights=False)[0]
            hidden = layer.norm2(hidden + (1 + g2) * layer.dropout2(attended))
            hidden = hidden * (1 + s2) + b2
            queries = self.evidence_query(hidden).reshape(b, -1, 4, width // 4).transpose(1, 2)
            extra = F.scaled_dot_product_attention(queries, keys, values, attn_mask=None)
            extra = extra.transpose(1, 2).reshape(b, noisy.shape[1], width)
            extra = self.evidence_output(extra).masked_fill(drop_mask[:, None, None], 0)
            hidden = hidden + self.evidence_gates[index].tanh() * extra
            feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(hidden))))
            hidden = layer.norm3(hidden + (1 + g3) * layer.dropout3(feedforward))
            hidden = hidden * (1 + s3) + b3
            hidden = hidden + adapter.local_update(hidden, condition, valid)
        return self.output_proj(hidden).masked_fill(~valid[..., None], 0)


class ReaDiff(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.denoiser = Denoiser(**architecture)
        self.register_buffer("alpha_bar", torch.empty(1000))
        self.register_buffer("latent_mean", torch.empty(512))
        self.register_buffer("latent_std", torch.empty(512))
