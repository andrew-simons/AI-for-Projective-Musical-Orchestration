#!/usr/bin/env python3
# scripts/20_xml_demo_orchestrate.py
from __future__ import annotations

# --- make repo root importable no matter where this is run from ---
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -----------------------------------------------------------------

import argparse
from typing import Optional

import numpy as np
import torch

from src.io.musicxml_io import (
    load_piano_xml_events,
    events_to_roll_and_onset,
    infer_tempo_and_timesig,
)
from src.render.assign import assign_events_to_parts, DEFAULT_PARTS
from src.render.write_xml import write_orchestral_musicxml

# NEW MODEL
from models.stage1_encoder import Stage1Encoder, Stage1Config


def load_stage1_model(
    ckpt_path: str,
    device: torch.device,
) -> tuple[Stage1Encoder, Stage1Config]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Stage1Config(**ckpt["config"])
    model = Stage1Encoder(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_instrument_activity_chunked(
    model: Stage1Encoder,
    cfg: Stage1Config,
    roll: np.ndarray,
    onset: np.ndarray,
    device: torch.device,
    batch_chunk: int = 512,
) -> np.ndarray:
    """
    Returns instrument_activity_hat: (T,129) in [0,1]
    """
    T = roll.shape[0]
    if T == 0:
        return np.zeros((0, cfg.d_out), dtype=np.float32)

    # Build input
    if cfg.d_in == 256:
        x = np.concatenate([roll, onset], axis=1).astype(np.float32)
    else:
        x = roll.astype(np.float32)

    out = np.zeros((T, cfg.d_out), dtype=np.float32)

    # IMPORTANT: learned positional encoding requires chunk_len <= cfg.max_len
    chunk = min(int(batch_chunk), int(cfg.max_len))

    for s in range(0, T, chunk):
        e = min(T, s + chunk)

        x_t = torch.from_numpy(x[s:e]).unsqueeze(0).to(device)  # (1,chunk,D)
        logits = model(x_t)                                     # (1,chunk,129)

        out[s:e] = (
            torch.sigmoid(logits)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    return out


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("piano_xml", type=str)
    ap.add_argument("--out_xml", type=str, default="out_orch.musicxml")
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--hop_s", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--activity_thresh", type=float, default=0.20)
    ap.add_argument("--topk", type=int, default=4)

    # NEW assign knobs (match src/render/assign.py)
    ap.add_argument("--hysteresis_on", type=float, default=0.22)
    ap.add_argument("--hysteresis_off", type=float, default=0.14)
    ap.add_argument("--usage_half_life_frames", type=int, default=400)
    ap.add_argument("--usage_strength", type=float, default=0.35)
    ap.add_argument("--max_active_parts_per_frame", type=int, default=6)

    # NEW: discourage note-to-note instrument switching (requires assign.py change too)
    ap.add_argument("--switch_penalty", type=float, default=0.35)

    # Back-compat convenience knob (maps to new knobs)
    ap.add_argument("--continuity", type=float, default=None)

    # Infer by default
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--time_signature", type=str, default=None)

    args = ap.parse_args()

    # Infer tempo / timesig
    inferred_bpm, inferred_ts = infer_tempo_and_timesig(
        args.piano_xml,
        default_bpm=120.0,
        default_time_signature="4/4",
    )

    bpm = float(args.bpm) if args.bpm is not None else float(inferred_bpm)
    time_signature = (
        str(args.time_signature)
        if args.time_signature is not None
        else str(inferred_ts)
    )

    print(
        f"[debug] output bpm={bpm}, time_signature={time_signature} "
        f"(inferred bpm={inferred_bpm}, ts={inferred_ts})"
    )

    # Parse MusicXML
    events, duration_s = load_piano_xml_events(args.piano_xml)

    roll, onset = events_to_roll_and_onset(
        events,
        duration_s=duration_s,
        hop_s=args.hop_s,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stage-1 prediction
    instrument_activity_hat: Optional[np.ndarray] = None

    if args.checkpoint:
        model, cfg = load_stage1_model(args.checkpoint, device)

        instrument_activity_hat = predict_instrument_activity_chunked(
            model,
            cfg,
            roll,
            onset,
            device,
            batch_chunk=args.chunk,
        )

        print(f"[ok] predicted instrument activity: {instrument_activity_hat.shape}")

        m = (
            instrument_activity_hat.mean(axis=0)
            if instrument_activity_hat.shape[0]
            else np.zeros((cfg.d_out,), dtype=np.float32)
        )
        top = np.argsort(-m)[:10]
        print(
            "[debug] top mean activity programs:",
            [(int(i), float(m[i])) for i in top],
        )

    # Map continuity -> new knobs (optional)
    if args.continuity is not None:
        c = float(np.clip(args.continuity, 0.0, 1.0))

        base_on = float(args.hysteresis_on)
        base_off = float(args.hysteresis_off)

        mid = 0.5 * (base_on + base_off)
        gap = (base_on - base_off) + 0.10 * c

        args.hysteresis_on = mid + 0.5 * gap
        args.hysteresis_off = mid - 0.5 * gap

        args.max_active_parts_per_frame = max(
            1, int(round(args.max_active_parts_per_frame - 2 * c))
        )

    # Assign to parts
    parts_to_events = assign_events_to_parts(
        events=events,
        instrument_activity_hat=instrument_activity_hat,
        hop_s=args.hop_s,
        parts=DEFAULT_PARTS,
        activity_thresh=args.activity_thresh,
        topk=args.topk,
        hysteresis_on=args.hysteresis_on,
        hysteresis_off=args.hysteresis_off,
        usage_half_life_frames=args.usage_half_life_frames,
        usage_strength=args.usage_strength,
        max_active_parts_per_frame=args.max_active_parts_per_frame,
        switch_penalty=args.switch_penalty,  # <-- requires assign.py signature support
    )

    # Debug part usage
    counts = {k: len(v) for k, v in parts_to_events.items()}
    used = sorted([(c, k) for k, c in counts.items() if c > 0], reverse=True)

    print("[debug] part usage (notes):")
    for c, k in used:
        print(f"  {k:12s} {c}")

    # Write MusicXML
    write_orchestral_musicxml(
        parts_to_events=parts_to_events,
        out_xml_path=args.out_xml,
        bpm=bpm,
        time_signature=time_signature,
        quantize_denom=96,
        non_transposing=True,
    )

    print(f"[ok] wrote orchestral xml: {args.out_xml}")


if __name__ == "__main__":
    main()