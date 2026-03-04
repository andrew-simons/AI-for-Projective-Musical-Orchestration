# models/stage1_encoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        if T > self.pos.size(0):
            raise ValueError(f"T={T} exceeds max_len={self.pos.size(0)}. Increase cfg.max_len or chunk_len.")
        return x + self.pos[:T].unsqueeze(0)


@dataclass
class Stage1Config:
    d_in: int = 256
    d_out: int = 129
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 2048
    use_learned_pos: bool = True


class Stage1Encoder(nn.Module):
    """
    Encoder-only transformer that predicts instrument activity logits per frame.

    Input:  x (B, T, d_in)
    Output: logits (B, T, d_out)
    """
    def __init__(self, cfg: Stage1Config):
        super().__init__()
        self.cfg = cfg

        self.x_proj = nn.Linear(cfg.d_in, cfg.d_model)

        if cfg.use_learned_pos:
            self.x_pos = LearnedPositionalEncoding(cfg.max_len, cfg.d_model)
        else:
            self.x_pos = nn.Identity()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.out = nn.Linear(cfg.d_model, cfg.d_out)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_emb = self.x_pos(self.x_proj(x))
        h = self.encoder(x_emb, src_key_padding_mask=src_key_padding_mask)
        return self.out(h)