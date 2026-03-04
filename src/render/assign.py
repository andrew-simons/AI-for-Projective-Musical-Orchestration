# src/render/assign.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from src.io.musicxml_io import NoteEvent


@dataclass
class PartSpec:
    name: str
    gm_program: int
    pitch_range: Tuple[int, int]
    polyphony_ok: bool = True


DEFAULT_PARTS: List[PartSpec] = [
    PartSpec("Flute", 73, (60, 96)),
    PartSpec("Oboe", 68, (58, 91)),
    PartSpec("Clarinet", 71, (50, 94)),
    PartSpec("Bassoon", 70, (34, 79)),
    PartSpec("Horn", 60, (40, 84)),
    PartSpec("Trumpet", 56, (55, 90)),
    PartSpec("Trombone", 57, (40, 80)),
    PartSpec("Tuba", 58, (28, 64)),
    PartSpec("Violin", 40, (55, 103)),
    PartSpec("Viola", 41, (48, 96)),
    PartSpec("Cello", 42, (36, 84)),
    PartSpec("Contrabass", 43, (28, 60)),
]


def _t2frame(t: float, hop_s: float, T: int) -> int:
    if T <= 0:
        return 0
    return int(np.clip(int(np.round(t / hop_s)), 0, T - 1))


def _range_score(pitches: List[int], lo: int, hi: int) -> float:
    """
    Soft range match: 1.0 if in-range, decays smoothly if not.
    Uses median pitch as a representative.
    """
    if not pitches:
        return 0.0
    m = float(np.median(pitches))
    if lo <= m <= hi:
        return 1.0
    # distance outside range (in semitones)
    d = (lo - m) if m < lo else (m - hi)
    return float(np.exp(-(d * d) / (2 * (6.0 ** 2))))  # sigma ~ 6 semitones


def assign_events_to_parts(
    events: List[NoteEvent],
    instrument_activity_hat: Optional[np.ndarray],  # (T,129) sigmoid probs in [0,1] OR None
    hop_s: float,
    parts: List[PartSpec] = DEFAULT_PARTS,
    activity_thresh: float = 0.20,
    topk: int = 4,

    # NEW knobs:
    hysteresis_on: float = 0.22,     # higher = harder to turn on
    hysteresis_off: float = 0.14,    # lower = easier to stay on once active
    usage_half_life_frames: int = 400,   # penalize recent overuse; ~20s if hop=0.05
    usage_strength: float = 0.35,        # bigger => more anti-hogging
    max_active_parts_per_frame: int = 6, # global cap
) -> Dict[str, List[NoteEvent]]:
    """
    Strategy:
      - At each note onset frame, decide a set of ACTIVE parts (global selection).
      - Then assign chord tones among active parts by register + best range match.
      - Apply hysteresis so parts don't flicker.
      - Apply usage penalty so one part can't eat the whole piece.
    """
    out: Dict[str, List[NoteEvent]] = {p.name: [] for p in parts}

    # If no model, do pure range/register fallback.
    use_model = instrument_activity_hat is not None and instrument_activity_hat.size > 0
    T_model = int(instrument_activity_hat.shape[0]) if use_model else 0

    # State: which parts are currently active (for hysteresis)
    active: Dict[str, bool] = {p.name: False for p in parts}

    # State: recent usage score per part (EMA-ish)
    usage: Dict[str, float] = {p.name: 0.0 for p in parts}
    decay = 0.5 ** (1.0 / max(1, int(usage_half_life_frames)))  # per-frame decay

    # Group events by onset frame so we can do global per-frame selection
    events_by_frame: Dict[int, List[NoteEvent]] = {}
    if use_model:
        for ev in events:
            f = _t2frame(ev.start_s, hop_s, T_model)
            events_by_frame.setdefault(f, []).append(ev)
        frames = sorted(events_by_frame.keys())
    else:
        # still need deterministic ordering
        frames = [0]
        events_by_frame[0] = list(events)

    # Precompute part centers for sorting by register
    part_center = {p.name: 0.5 * (p.pitch_range[0] + p.pitch_range[1]) for p in parts}

    def choose_active_parts(f: int, pitches_in_frame: List[int]) -> List[PartSpec]:
        # Fallback: pick strings/brass by register if no model
        if not use_model:
            # keep it simple: always allow strings + 1 brass
            base = [p for p in parts if p.name in ["Violin", "Viola", "Cello", "Contrabass", "Horn"]]
            return base[: max_active_parts_per_frame]

        # decay usage each frame we visit
        for k in usage:
            usage[k] *= decay

        scored: List[Tuple[float, PartSpec]] = []
        for ps in parts:
            lo, hi = ps.pitch_range
            r = _range_score(pitches_in_frame, lo, hi)
            if r < 0.10:
                continue

            act = float(instrument_activity_hat[f, ps.gm_program])

            # hysteresis thresholds
            thr_on = max(activity_thresh, hysteresis_on)
            thr_off = min(activity_thresh, hysteresis_off)

            # if already active, allow lower threshold
            thr = thr_off if active[ps.name] else thr_on
            if act < thr:
                continue

            # anti-hogging penalty (recent overuse)
            penalty = float(np.exp(-usage_strength * usage[ps.name]))
            score = act * r * penalty
            scored.append((score, ps))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [ps for _, ps in scored[: max(1, min(max_active_parts_per_frame, topk * 2))]]

        # Also enforce a hard cap of max_active_parts_per_frame
        return chosen[: max_active_parts_per_frame]

    def assign_chord_to_parts(pitches: List[int], active_parts: List[PartSpec]) -> List[PartSpec]:
        # If nothing active, fallback to strings by register
        if not active_parts:
            active_parts = [p for p in parts if p.name in ["Violin", "Viola", "Cello", "Contrabass"]]

        # Sort parts low->high
        active_parts = sorted(active_parts, key=lambda ps: part_center[ps.name])

        mapped: List[PartSpec] = []
        pitches_sorted = sorted(pitches)

        # Map lowest pitches to lowest parts, etc.
        for i, p in enumerate(pitches_sorted):
            j = int(np.floor(i * (len(active_parts) / max(1, len(pitches_sorted)))))
            j = min(j, len(active_parts) - 1)
            mapped.append(active_parts[j])

        return mapped

    for f in frames:
        frame_events = events_by_frame[f]
        # collect all onset pitches in this frame
        pitches_in_frame: List[int] = []
        for ev in frame_events:
            pitches_in_frame.extend([p for p in ev.pitches if 0 <= p < 128])
        pitches_in_frame = sorted(pitches_in_frame)

        chosen_active = choose_active_parts(f, pitches_in_frame)

        # update hysteresis active states
        chosen_names = {ps.name for ps in chosen_active}
        for ps in parts:
            if ps.name in chosen_names:
                active[ps.name] = True
            else:
                # turn off slowly: only if model exists and activity is low
                if use_model:
                    act = float(instrument_activity_hat[f, ps.gm_program])
                    if act < hysteresis_off:
                        active[ps.name] = False
                else:
                    active[ps.name] = False

        # assign each event in frame
        for ev in frame_events:
            pitches = sorted([p for p in ev.pitches if 0 <= p < 128])
            if not pitches:
                continue

            assigned_parts = assign_chord_to_parts(pitches, chosen_active)

            # emit one NoteEvent per pitch
            for p, ps in zip(sorted(pitches), assigned_parts):
                out[ps.name].append(NoteEvent(ev.start_s, ev.end_s, [int(p)], ev.velocity01))
                usage[ps.name] += 1.0  # count usage (recent)

    for k in out:
        out[k].sort(key=lambda e: e.start_s)
    return out