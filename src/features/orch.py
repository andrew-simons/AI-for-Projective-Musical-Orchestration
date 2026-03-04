# ## Orchestration features (`src/features/orch.py`)

# This module extracts a mix of **dense frame-level summaries** and **sparse onset events** from an orchestral MIDI file using `pretty_midi`. It is designed for training orchestration models with a “starter” feature set that is easy to consume, while still capturing important orchestration concepts such as **doubling**, **dynamics**, **instrument family usage**, and **registration/voicing**.

# > MIDI limitation: Standard MIDI does **not** explicitly encode many score-level orchestration markings (e.g., *sul ponticello*, *harmonics*, *Schalltrichter auf*). The “special techniques” features here are **heuristics** unless your MIDI files use consistent keyswitch/articulation conventions.

# ### Output: `OrchFeatures`

# All time-varying dense features use a fixed hop size `hop_s` and have length `T = ceil(duration_s / hop_s)`.

# #### Core metadata
# - `hop_s: float` — seconds per frame (default `0.05`)
# - `duration_s: float` — end time of MIDI in seconds

# #### Dense frame-level summaries (baseline-friendly)
# - `instrument_activity: (T, 129) float32`
#   - Per-frame **max velocity (0..1)** for each instrument ID.
#   - Instrument IDs:
#     - `0..127` = GM programs
#     - `128` = drums
# - `pitch_activity: (T, 128) float32`
#   - Per-frame **max velocity (0..1)** for each MIDI pitch (aggregated across instruments).
# - `family_activity: (T, N_FAM) float32`
#   - Per-frame **max velocity (0..1)** for each GM family bucket.
#   - Family buckets follow GM groups (program // 8) plus a dedicated `"drums"` family.

# #### Sparse onset events (note-on style)
# These arrays all have the same length `K` (# of note onsets).
# - `sparse_t: (K,) int32` — onset frame index
# - `sparse_i: (K,) int16` — instrument ID (0..128)
# - `sparse_p: (K,) int16` — MIDI pitch (0..127)
# - `sparse_v: (K,) float32` — onset velocity in [0,1]
# - `sparse_f: (K,) int16` — instrument family ID (0..N_FAM-1)

# **Interpretation:** Sparse events are the primary source for “who played what at onset time”. Dense arrays provide easy-to-train summaries over time.

# #### Simple counts
# - `total_notes_per_inst: (129,) int64`
# - `total_notes: int`
# - `total_notes_per_family: (N_FAM,) int64`

# ---

# ## Added orchestration-specific features

# ### 1) Doubling detection
# Doubling is computed from **near-simultaneous onsets** (by default: same frame only).
# - `doubling_unison_counts: (129, 129) int64`
#   - Symmetric matrix: how often instrument A and B hit **the same pitch** at (nearly) the same time.
#   - Diagonal is 0.
# - `doubling_interval_hist: (129, 129, 12) int64`
#   - Symmetric on (i, j), counts of interval classes **mod 12** between simultaneous onsets.
#   - `interval_class = (p_j - p_i) mod 12`
#   - Useful for learning octave/third/sixth doublings, etc.

# **Windowing:** Controlled by `doubling_window_frames`:
# - `0` = same-frame only (strict)
# - `1` = allow ±1 frame (looser), etc.

# > Note: This is onset-based doubling, not sustained overlap. If you need overlap doubling, we can add it, but it’s heavier.

# ### 2) Dynamics (per instrument)
# Dynamics are approximated with MIDI velocity (0..1).
# - `mean_vel_per_inst: (129,) float32`
# - `max_vel_per_inst: (129,) float32`

# > If your MIDI contains expressive CC lanes (e.g., CC11 expression, CC1 modwheel), those can be integrated later for better dynamics modeling.

# ### 3) Instrumentation / family usage & predominance
# Family activity and family note counts provide a coarse but robust summary of instrumentation.
# - `family_activity: (T, N_FAM) float32`
# - `total_notes_per_family: (N_FAM,) int64`

# This supports features like:
# - “strings predominate in this section”
# - “brass enters strongly here”

# ### 4) Registration / voicing
# Voicing and registration are computed from **onsets per frame** (sparse events), giving a per-frame snapshot of vertical spacing.
# Global (all instruments):
# - `global_low_pitch: (T,) float32` — lowest onset pitch in frame (NaN if none)
# - `global_high_pitch: (T,) float32` — highest onset pitch in frame (NaN if none)
# - `global_pitch_centroid: (T,) float32` — mean onset pitch in frame (NaN if none)
# - `global_pitch_spread: (T,) float32` — high - low (NaN if none)

# Per-family registration (onsets only):
# - `family_low_pitch: (T, N_FAM) float32` (NaN if none)
# - `family_high_pitch: (T, N_FAM) float32` (NaN if none)
# - `family_pitch_centroid: (T, N_FAM) float32` (NaN if none)

# **Why onset-based?** It’s stable, cheap, and aligns well with orchestration “events” (new attacks). Sustained registration can be added if needed.

# ### 5) Special techniques (heuristics)
# Since MIDI rarely encodes explicit techniques, we provide a lightweight hint:
# - `onset_out_of_range: (K,) uint8`
#   - 1 if the onset pitch lies outside a rough “typical range” for its **GM family**; else 0.

# This can sometimes correlate with “effects” writing (extreme registers), but it is NOT technique-aware in the score sense.

# > Recommended upgrade path:
# > - If you have **MusicXML**, parse explicit technique markings directly.
# > - If you have **keyswitch MIDI**, add an articulation-map layer keyed to program/channel/notes.

# ---

# ## Serialization (`orch_features_to_npz_dict`)
# The helper `orch_features_to_npz_dict(of)` returns a dictionary ready for `np.savez`, including:
# - dense arrays (`instrument_activity`, `pitch_activity`, `family_activity`)
# - sparse arrays (`sparse_*`)
# - doubling features
# - dynamics stats
# - registration/voicing stats
# - counts

# This format is intended to be stable across training runs.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pretty_midi

from src.grid import time_to_frame

N_GM = 128
DRUM_ID = 128
N_INST = 129  # 0..127 + drums as 128

# ----------------------------
# GM family mapping (0..127)
# ----------------------------
# GM program groups (0-indexed programs):
# 0-7 Piano, 8-15 Chrom Perc, 16-23 Organ, 24-31 Guitar,
# 32-39 Bass, 40-47 Strings, 48-55 Ensemble, 56-63 Brass,
# 64-71 Reed, 72-79 Pipe, 80-87 Synth Lead, 88-95 Synth Pad,
# 96-103 Synth FX, 104-111 Ethnic, 112-119 Percussive, 120-127 SFX
FAMILY_NAMES: List[str] = [
    "piano", "chrom_perc", "organ", "guitar",
    "bass", "strings", "ensemble", "brass",
    "reed", "pipe", "synth_lead", "synth_pad",
    "synth_fx", "ethnic", "percussive", "sfx",
    "drums",
]
N_FAM = len(FAMILY_NAMES)


def gm_family(program: int, is_drum: bool) -> int:
    if is_drum:
        return FAMILY_NAMES.index("drums")
    # clamp just in case
    p = int(np.clip(program, 0, 127))
    return p // 8  # 0..15


# ----------------------------
# Very rough "typical ranges"
# (MIDI pitch numbers)
# Edit these as you like, or load from config.
# ----------------------------
# Note: This is NOT instrument-accurate GM mapping. It's a heuristic.
FAMILY_PITCH_RANGES: Dict[int, Tuple[int, int]] = {
    FAMILY_NAMES.index("piano"): (21, 108),        # A0..C8
    FAMILY_NAMES.index("strings"): (36, 96),       # C2..C7-ish
    FAMILY_NAMES.index("ensemble"): (36, 96),
    FAMILY_NAMES.index("brass"): (34, 82),
    FAMILY_NAMES.index("reed"): (46, 94),
    FAMILY_NAMES.index("pipe"): (48, 96),
    FAMILY_NAMES.index("guitar"): (40, 88),
    FAMILY_NAMES.index("bass"): (28, 72),
    FAMILY_NAMES.index("organ"): (36, 96),
    FAMILY_NAMES.index("chrom_perc"): (48, 96),
    FAMILY_NAMES.index("ethnic"): (36, 96),
    FAMILY_NAMES.index("percussive"): (36, 96),
    FAMILY_NAMES.index("sfx"): (0, 127),
    FAMILY_NAMES.index("synth_lead"): (36, 108),
    FAMILY_NAMES.index("synth_pad"): (24, 96),
    FAMILY_NAMES.index("synth_fx"): (0, 127),
    FAMILY_NAMES.index("drums"): (0, 127),
}


@dataclass
class OrchFeatures:
    hop_s: float
    duration_s: float

    # Dense summaries (easy to train with early baselines)
    instrument_activity: np.ndarray  # (T, 129) float32 : per-frame max vel01 for each instrument
    pitch_activity: np.ndarray       # (T, 128) float32 : per-frame max vel01 for each pitch (any inst)

    # Sparse note onset events: arrays of equal length K
    # (t_idx, inst_id, pitch, vel01)
    sparse_t: np.ndarray
    sparse_i: np.ndarray
    sparse_p: np.ndarray
    sparse_v: np.ndarray
    sparse_f: np.ndarray            # (K,) int16 family id (0..N_FAM-1)

    # Simple analytics
    total_notes_per_inst: np.ndarray   # (129,)
    total_notes: int

    # ----------------------------
    # Added: dynamics per instrument
    # ----------------------------
    mean_vel_per_inst: np.ndarray      # (129,) float32
    max_vel_per_inst: np.ndarray       # (129,) float32

    # ----------------------------
    # Added: instrumentation / family usage
    # ----------------------------
    family_activity: np.ndarray        # (T, N_FAM) float32 : per-frame max vel01 per family
    total_notes_per_family: np.ndarray # (N_FAM,) int64

    # ----------------------------
    # Added: registration / voicing
    # Using onsets (sparse) as a light-weight proxy
    # ----------------------------
    # Global per-frame voicing stats from onsets at that frame
    global_low_pitch: np.ndarray       # (T,) float32 (nan if none)
    global_high_pitch: np.ndarray      # (T,) float32 (nan if none)
    global_pitch_centroid: np.ndarray  # (T,) float32 (nan if none)
    global_pitch_spread: np.ndarray    # (T,) float32 (nan if none) high-low

    # Per-family per-frame registration stats (onsets)
    family_low_pitch: np.ndarray       # (T, N_FAM) float32 (nan if none)
    family_high_pitch: np.ndarray      # (T, N_FAM) float32 (nan if none)
    family_pitch_centroid: np.ndarray  # (T, N_FAM) float32 (nan if none)

    # ----------------------------
    # Added: doubling detection (onset-based)
    # ----------------------------
    # Pairwise unison doubling counts: how often two instruments hit the same pitch at (nearly) same time
    doubling_unison_counts: np.ndarray     # (N_INST, N_INST) int64 symmetric, diagonal 0

    # Pairwise interval hist (mod 12) for simultaneous onsets: counts per interval class
    doubling_interval_hist: np.ndarray     # (N_INST, N_INST, 12) int64 symmetric on (i,j), intervals computed as (p_j - p_i) mod 12

    # ----------------------------
    # Added: special technique hints (heuristics)
    # ----------------------------
    onset_out_of_range: np.ndarray        # (K,) uint8 : 1 if pitch outside rough family range else 0


def _inst_id(inst: pretty_midi.Instrument) -> int:
    if inst.is_drum:
        return DRUM_ID
    return int(inst.program)  # 0..127


def extract_orch_features(
    midi_path: str,
    hop_s: float = 0.05,
    doubling_window_frames: int = 0,  # 0 = same-frame only; 1 = +/-1 frame, etc.
) -> OrchFeatures:
    pm = pretty_midi.PrettyMIDI(midi_path)
    duration_s = pm.get_end_time()
    T = int(np.ceil(duration_s / hop_s)) if duration_s > 0 else 0

    instrument_activity = np.zeros((T, N_INST), dtype=np.float32)
    pitch_activity = np.zeros((T, 128), dtype=np.float32)

    family_activity = np.zeros((T, N_FAM), dtype=np.float32)
    total_notes_per_inst = np.zeros((N_INST,), dtype=np.int64)
    total_notes_per_family = np.zeros((N_FAM,), dtype=np.int64)

    # sparse onsets
    sparse_t: List[int] = []
    sparse_i: List[int] = []
    sparse_p: List[int] = []
    sparse_v: List[float] = []
    sparse_f: List[int] = []

    # dynamics stats
    vel_sum_per_inst = np.zeros((N_INST,), dtype=np.float64)
    vel_max_per_inst = np.zeros((N_INST,), dtype=np.float32)

    total_notes = 0

    # Build per-frame onset buckets for voicing + doubling
    # frame -> list of (iid, pitch)
    onsets_by_frame: List[List[Tuple[int, int]]] = [[] for _ in range(T)] if T > 0 else []

    # Iterate notes
    for inst in pm.instruments:
        iid = _inst_id(inst)
        fam = gm_family(inst.program, inst.is_drum)

        for note in inst.notes:
            total_notes += 1
            total_notes_per_inst[iid] += 1
            total_notes_per_family[fam] += 1

            p = int(note.pitch)
            if not (0 <= p < 128):
                continue

            if T == 0:
                continue

            f0 = time_to_frame(note.start, hop_s)
            f1 = time_to_frame(note.end, hop_s)
            f0 = max(0, min(T - 1, f0))
            f1 = max(0, min(T - 1, f1))
            if f1 < f0:
                f1 = f0

            v01 = float(np.clip(note.velocity / 127.0, 0.0, 1.0))

            # Dense summaries over note span
            instrument_activity[f0 : f1 + 1, iid] = np.maximum(instrument_activity[f0 : f1 + 1, iid], v01)
            pitch_activity[f0 : f1 + 1, p] = np.maximum(pitch_activity[f0 : f1 + 1, p], v01)
            family_activity[f0 : f1 + 1, fam] = np.maximum(family_activity[f0 : f1 + 1, fam], v01)

            # Sparse onset event
            sparse_t.append(f0)
            sparse_i.append(iid)
            sparse_p.append(p)
            sparse_v.append(v01)
            sparse_f.append(fam)

            onsets_by_frame[f0].append((iid, p))

            # Dynamics stats per instrument (note-based)
            vel_sum_per_inst[iid] += v01
            if v01 > vel_max_per_inst[iid]:
                vel_max_per_inst[iid] = v01

    # Finalize mean/max vel per inst
    mean_vel_per_inst = np.zeros((N_INST,), dtype=np.float32)
    nonzero = total_notes_per_inst > 0
    mean_vel_per_inst[nonzero] = (vel_sum_per_inst[nonzero] / total_notes_per_inst[nonzero]).astype(np.float32)
    max_vel_per_inst = vel_max_per_inst.astype(np.float32)

    # ----------------------------
    # Registration / voicing (onset-based)
    # ----------------------------
    global_low = np.full((T,), np.nan, dtype=np.float32)
    global_high = np.full((T,), np.nan, dtype=np.float32)
    global_centroid = np.full((T,), np.nan, dtype=np.float32)
    global_spread = np.full((T,), np.nan, dtype=np.float32)

    fam_low = np.full((T, N_FAM), np.nan, dtype=np.float32)
    fam_high = np.full((T, N_FAM), np.nan, dtype=np.float32)
    fam_centroid = np.full((T, N_FAM), np.nan, dtype=np.float32)

    # We’ll also need families at onsets; easiest is to use sparse arrays after creation
    sparse_t_arr = np.array(sparse_t, dtype=np.int32) if sparse_t else np.zeros((0,), dtype=np.int32)
    sparse_f_arr = np.array(sparse_f, dtype=np.int16) if sparse_f else np.zeros((0,), dtype=np.int16)
    sparse_p_arr = np.array(sparse_p, dtype=np.int16) if sparse_p else np.zeros((0,), dtype=np.int16)

    # For quick per-frame onset pitch lists, re-use onsets_by_frame for global,
    # and compute family splits by scanning sparse arrays per frame (still okay).
    if T > 0 and sparse_t_arr.size > 0:
        # Build indices per frame for sparse events (onsets)
        idxs_by_frame: List[List[int]] = [[] for _ in range(T)]
        for k, t in enumerate(sparse_t_arr):
            if 0 <= t < T:
                idxs_by_frame[int(t)].append(k)

        for t in range(T):
            idxs = idxs_by_frame[t]
            if not idxs:
                continue
            pitches = sparse_p_arr[idxs].astype(np.float32)
            global_low[t] = float(np.min(pitches))
            global_high[t] = float(np.max(pitches))
            global_centroid[t] = float(np.mean(pitches))
            global_spread[t] = float(global_high[t] - global_low[t])

            # per-family
            fams = sparse_f_arr[idxs]
            for f in range(N_FAM):
                mask = fams == f
                if not np.any(mask):
                    continue
                fp = pitches[mask]
                fam_low[t, f] = float(np.min(fp))
                fam_high[t, f] = float(np.max(fp))
                fam_centroid[t, f] = float(np.mean(fp))

    # ----------------------------
    # Doubling detection (onset-based)
    # ----------------------------
    unison_counts = np.zeros((N_INST, N_INST), dtype=np.int64)
    interval_hist = np.zeros((N_INST, N_INST, 12), dtype=np.int64)

    if T > 0:
        # helper to gather events in a window around t
        def gather_events(t: int) -> List[Tuple[int, int]]:
            if doubling_window_frames <= 0:
                return onsets_by_frame[t]
            out: List[Tuple[int, int]] = []
            t0 = max(0, t - doubling_window_frames)
            t1 = min(T - 1, t + doubling_window_frames)
            for tt in range(t0, t1 + 1):
                out.extend(onsets_by_frame[tt])
            return out

        for t in range(T):
            events = gather_events(t)
            if len(events) < 2:
                continue

            # For each instrument, allow multiple onsets in window; keep them all.
            # Compute pairwise relations among events.
            for a in range(len(events)):
                i1, p1 = events[a]
                for b in range(a + 1, len(events)):
                    i2, p2 = events[b]
                    if i1 == i2:
                        continue

                    # Unison doubling
                    if p1 == p2:
                        unison_counts[i1, i2] += 1
                        unison_counts[i2, i1] += 1

                    # Interval class (mod 12)
                    ic = int((p2 - p1) % 12)
                    interval_hist[i1, i2, ic] += 1
                    ic2 = int((p1 - p2) % 12)
                    interval_hist[i2, i1, ic2] += 1

        # zero diagonal explicitly
        np.fill_diagonal(unison_counts, 0)
        for ic in range(12):
            np.fill_diagonal(interval_hist[:, :, ic], 0)

    # ----------------------------
    # Special techniques (heuristic)
    # ----------------------------
    onset_out_of_range = np.zeros((len(sparse_p_arr),), dtype=np.uint8)
    if len(sparse_p_arr) > 0:
        for k in range(len(sparse_p_arr)):
            f = int(sparse_f_arr[k])
            p = int(sparse_p_arr[k])
            lo, hi = FAMILY_PITCH_RANGES.get(f, (0, 127))
            onset_out_of_range[k] = 1 if (p < lo or p > hi) else 0

    return OrchFeatures(
        hop_s=float(hop_s),
        duration_s=float(duration_s),

        instrument_activity=instrument_activity,
        pitch_activity=pitch_activity,

        sparse_t=sparse_t_arr,
        sparse_i=np.array(sparse_i, dtype=np.int16) if sparse_i else np.zeros((0,), dtype=np.int16),
        sparse_p=sparse_p_arr,
        sparse_v=np.array(sparse_v, dtype=np.float32) if sparse_v else np.zeros((0,), dtype=np.float32),
        sparse_f=sparse_f_arr,

        total_notes_per_inst=total_notes_per_inst.astype(np.int64),
        total_notes=int(total_notes),

        mean_vel_per_inst=mean_vel_per_inst,
        max_vel_per_inst=max_vel_per_inst,

        family_activity=family_activity.astype(np.float32),
        total_notes_per_family=total_notes_per_family.astype(np.int64),

        global_low_pitch=global_low,
        global_high_pitch=global_high,
        global_pitch_centroid=global_centroid,
        global_pitch_spread=global_spread,

        family_low_pitch=fam_low,
        family_high_pitch=fam_high,
        family_pitch_centroid=fam_centroid,

        doubling_unison_counts=unison_counts,
        doubling_interval_hist=interval_hist,

        onset_out_of_range=onset_out_of_range,
    )


def orch_features_to_npz_dict(of: OrchFeatures) -> Dict[str, Any]:
    return {
        "hop_s": np.array([of.hop_s], dtype=np.float32),
        "duration_s": np.array([of.duration_s], dtype=np.float32),

        "instrument_activity": of.instrument_activity.astype(np.float32),
        "pitch_activity": of.pitch_activity.astype(np.float32),

        "sparse_t": of.sparse_t,
        "sparse_i": of.sparse_i,
        "sparse_p": of.sparse_p,
        "sparse_v": of.sparse_v,
        "sparse_f": of.sparse_f,

        "total_notes_per_inst": of.total_notes_per_inst.astype(np.int64),
        "total_notes": np.array([of.total_notes], dtype=np.int64),

        "mean_vel_per_inst": of.mean_vel_per_inst.astype(np.float32),
        "max_vel_per_inst": of.max_vel_per_inst.astype(np.float32),

        "family_activity": of.family_activity.astype(np.float32),
        "total_notes_per_family": of.total_notes_per_family.astype(np.int64),

        "global_low_pitch": of.global_low_pitch.astype(np.float32),
        "global_high_pitch": of.global_high_pitch.astype(np.float32),
        "global_pitch_centroid": of.global_pitch_centroid.astype(np.float32),
        "global_pitch_spread": of.global_pitch_spread.astype(np.float32),

        "family_low_pitch": of.family_low_pitch.astype(np.float32),
        "family_high_pitch": of.family_high_pitch.astype(np.float32),
        "family_pitch_centroid": of.family_pitch_centroid.astype(np.float32),

        "doubling_unison_counts": of.doubling_unison_counts.astype(np.int64),
        "doubling_interval_hist": of.doubling_interval_hist.astype(np.int64),

        "onset_out_of_range": of.onset_out_of_range.astype(np.uint8),

        # metadata (nice to have)
        "family_names": np.array(FAMILY_NAMES, dtype=object),
    }