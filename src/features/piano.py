# src/features/piano.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pretty_midi

from src.grid import time_to_frame


@dataclass
class PianoFeatures:
    hop_s: float
    duration_s: float
    roll: np.ndarray          # (T, 128) float32 in [0,1]
    onset: np.ndarray         # (T, 128) float32 {0,1}
    active_notes: np.ndarray  # (T,) float32 count of active pitches
    onset_count: np.ndarray   # (T,) float32 count of onsets


def extract_piano_features(midi_path: str, hop_s: float = 0.05) -> PianoFeatures:
    pm = pretty_midi.PrettyMIDI(midi_path)

    # duration: use end of last event if possible
    duration_s = pm.get_end_time()
    if duration_s <= 0:
        # fallback
        duration_s = 0.0

    fs = 1.0 / hop_s  # frames per second

    # PrettyMIDI returns shape (128, T). We'll transpose to (T, 128).
    roll = pm.get_piano_roll(fs=fs).T.astype(np.float32)  # velocities 0..127
    if roll.size == 0:
        roll = np.zeros((0, 128), dtype=np.float32)

    # Normalize velocities to [0,1]
    roll = np.clip(roll / 127.0, 0.0, 1.0)

    # Onset roll: mark note starts
    onset = np.zeros_like(roll, dtype=np.float32)

    # Piano reductions may be stored as one instrument, but we'll aggregate across all
    for inst in pm.instruments:
        for note in inst.notes:
            t0 = note.start
            p = int(note.pitch)
            if 0 <= p < 128 and roll.shape[0] > 0:
                f0 = time_to_frame(t0, hop_s)
                if 0 <= f0 < onset.shape[0]:
                    onset[f0, p] = 1.0

    active_notes = (roll > 0).sum(axis=1).astype(np.float32) if roll.shape[0] > 0 else np.zeros((0,), dtype=np.float32)
    onset_count = onset.sum(axis=1).astype(np.float32) if onset.shape[0] > 0 else np.zeros((0,), dtype=np.float32)

    return PianoFeatures(
        hop_s=hop_s,
        duration_s=float(duration_s),
        roll=roll,
        onset=onset,
        active_notes=active_notes,
        onset_count=onset_count,
    )


def piano_features_to_npz_dict(pf: PianoFeatures) -> Dict[str, Any]:
    return {
        "hop_s": np.array([pf.hop_s], dtype=np.float32),
        "duration_s": np.array([pf.duration_s], dtype=np.float32),
        "roll": pf.roll.astype(np.float32),
        "onset": pf.onset.astype(np.float32),
        "active_notes": pf.active_notes.astype(np.float32),
        "onset_count": pf.onset_count.astype(np.float32),
    }
