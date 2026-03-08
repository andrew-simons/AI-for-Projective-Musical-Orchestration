from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from models.stage1_encoder import Stage1Config, Stage1Encoder


@dataclass
class Stage1ModelBundle:
    model: Stage1Encoder
    cfg: Stage1Config
    device: torch.device


def _resolve_checkpoint_path(explicit: Optional[str] = None) -> Path:
    """
    Resolve the checkpoint path to use for Stage-1 inference.

    Preference order:
      1) Explicit path if provided.
      2) STAGE1_CHECKPOINT environment variable.
      3) Default path under checkpoints/stage1_encoder_v2/best.pt
         or checkpoints/stage1_encoder_v1/best.pt (whichever exists first).
    """
    if explicit:
        return Path(explicit).expanduser().resolve()

    import os

    env_path = os.environ.get("STAGE1_CHECKPOINT")
    if env_path:
        return Path(env_path).expanduser().resolve()

    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "checkpoints" / "stage1_encoder_v2" / "best.pt",
        root / "checkpoints" / "stage1_encoder_v1" / "best.pt",
        root / "checkpoints" / "stage1_encoder_v2" / "last.pt",
        root / "checkpoints" / "stage1_encoder_v1" / "last.pt",
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "No Stage-1 checkpoint found. Set STAGE1_CHECKPOINT or provide an explicit path."
    )


def load_stage1_model(
    ckpt_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Stage1ModelBundle:
    """
    Load Stage-1 encoder model + config from a training checkpoint.

    This centralises checkpoint loading so both CLI scripts and the HTTP backend
    share identical behaviour.
    """
    path = _resolve_checkpoint_path(ckpt_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device)
    cfg = Stage1Config(**ckpt["config"])
    model = Stage1Encoder(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return Stage1ModelBundle(model=model, cfg=cfg, device=device)


@torch.no_grad()
def predict_instrument_activity_chunked(
    bundle: Stage1ModelBundle,
    roll: np.ndarray,
    onset: Optional[np.ndarray],
    batch_chunk: int = 512,
) -> np.ndarray:
    """
    Run Stage-1 encoder over a full piano roll (and optional onset) sequence.

    Args:
        bundle: Loaded model/config/device.
        roll: (T, 128) float32 piano roll.
        onset: (T, 128) float32 onset roll or None.
        batch_chunk: Maximum temporal chunk size, must be <= cfg.max_len.

    Returns:
        instrument_activity_hat: (T, 129) float32 in [0,1].
    """
    model, cfg, device = bundle.model, bundle.cfg, bundle.device

    T = int(roll.shape[0])
    if T == 0:
        return np.zeros((0, cfg.d_out), dtype=np.float32)

    if cfg.d_in == 256:
        if onset is None:
            raise ValueError("Stage1Config.d_in=256 requires onset features, but onset is None.")
        x_full = np.concatenate([roll, onset], axis=1).astype(np.float32)
    else:
        x_full = roll.astype(np.float32)

    out = np.zeros((T, cfg.d_out), dtype=np.float32)

    # Learned positional encoding requires chunk_len <= cfg.max_len
    chunk = min(int(batch_chunk), int(cfg.max_len))

    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        x_t = torch.from_numpy(x_full[s:e]).unsqueeze(0).to(device)  # (1, chunk, D)
        logits = model(x_t)  # (1, chunk, d_out)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy().astype(np.float32)
        out[s:e] = probs

    return out

