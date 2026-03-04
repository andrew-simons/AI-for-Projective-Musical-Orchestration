# models/baseline_encoder_decoder.py
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
        return x + self.pos[:T].unsqueeze(0)


@dataclass
class BaselineConfig:
    d_in: int = 256          # piano roll(128) + onset(128)
    d_out: int = 129         # instrument_activity
    d_model: int = 256
    nhead: int = 8
    num_enc_layers: int = 4
    num_dec_layers: int = 4
    dim_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 2048      # must be >= your chunk length
    use_learned_pos: bool = True


class BaselineEncoderDecoder(nn.Module):
    """
    Encoder-decoder transformer for continuous sequences.

    Input:
      x: (B, T, d_in) float
    Target:
      y: (B, T, d_out) float in [0,1]

    During training (teacher forcing):
      decoder_input = shift_right(y) with a learned start token
      output logits: (B, T, d_out)
    """

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg

        self.x_proj = nn.Linear(cfg.d_in, cfg.d_model)
        self.y_proj = nn.Linear(cfg.d_out, cfg.d_model)

        self.start_token = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.normal_(self.start_token, mean=0.0, std=0.02)

        if cfg.use_learned_pos:
            self.x_pos = LearnedPositionalEncoding(cfg.max_len, cfg.d_model)
            self.y_pos = LearnedPositionalEncoding(cfg.max_len, cfg.d_model)
        else:
            self.x_pos = nn.Identity()
            self.y_pos = nn.Identity()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_dec_layers)

        self.out = nn.Linear(cfg.d_model, cfg.d_out)

    @staticmethod
    def _shift_right(y_emb: torch.Tensor, start_token: torch.Tensor) -> torch.Tensor:
        # y_emb: (B, T, D)
        B, T, D = y_emb.shape
        start = start_token.view(1, 1, D).expand(B, 1, D)
        return torch.cat([start, y_emb[:, :-1, :]], dim=1)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # standard subsequent mask for decoder self-attn
        # shape (T, T) with True meaning "blocked" in PyTorch Transformer
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B,T) True for pad
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # (B,T) True for pad
    ) -> torch.Tensor:
        """
        Returns logits: (B, T, d_out)
        """
        B, T, _ = x.shape
        device = x.device

        # Encoder
        x_emb = self.x_pos(self.x_proj(x))
        memory = self.encoder(x_emb, src_key_padding_mask=src_key_padding_mask)

        # Decoder with teacher forcing
        y_emb = self.y_pos(self.y_proj(y))
        y_in = self._shift_right(y_emb, self.start_token)

        tgt_mask = self._causal_mask(T, device=device)
        dec = self.decoder(
            tgt=y_in,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.out(dec)
        return logits