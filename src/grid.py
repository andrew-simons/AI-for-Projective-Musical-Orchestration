# src/grid.py
from __future__ import annotations

import numpy as np


def make_time_grid(duration_s: float, hop_s: float) -> np.ndarray:
    """
    Returns an array of frame start times: [0, hop, 2*hop, ..., < duration]
    """
    if duration_s <= 0:
        return np.zeros((0,), dtype=np.float32)
    n_frames = int(np.ceil(duration_s / hop_s))
    return (np.arange(n_frames, dtype=np.float32) * hop_s).astype(np.float32)


def time_to_frame(t: float, hop_s: float) -> int:
    return int(round(t / hop_s))