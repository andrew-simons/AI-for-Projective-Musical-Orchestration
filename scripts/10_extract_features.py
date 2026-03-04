#!/usr/bin/env python3
# scripts/10_extract_features.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.piano import extract_piano_features, piano_features_to_npz_dict
from src.features.orch import extract_orch_features, orch_features_to_npz_dict


def safe_pair_id_to_filename(pair_id: str) -> str:
    # "bouliane/12" -> "bouliane__12"
    return pair_id.replace("/", "__")


def load_manifest() -> pd.DataFrame:
    p = Path("data/processed/lop_manifest.parquet")
    if p.exists():
        return pd.read_parquet(p)
    p = Path("data/processed/lop_manifest.csv")
    if p.exists():
        return pd.read_csv(p)
    raise SystemExit("Manifest not found. Run scripts/00_build_lop_manifest.py first.")


def nan_frac(x: np.ndarray) -> float:
    # robust even if empty
    if x.size == 0:
        return 1.0
    return float(np.isnan(x).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop_s", type=float, default=0.05, help="Frame hop in seconds (e.g., 0.05 = 50ms)")
    ap.add_argument("--doubling_window_frames", type=int, default=0,
                    help="Doubling detection window in frames (0 = same frame, 1 = +/-1 frame, etc.)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of pairs for quick testing (0 = all)")
    ap.add_argument("--only_usable", action="store_true", help="Only process rows where usable == True")
    ap.add_argument("--exclude_imslp", action="store_true", help="Skip source == imslp")
    args = ap.parse_args()

    df = load_manifest()
    if args.only_usable:
        df = df[df["usable"] == True].copy()
    if args.exclude_imslp:
        df = df[df.get("is_imslp", False) == False].copy()

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    out_piano = Path("data/features/piano")
    out_orch = Path("data/features/orch")
    out_meta = Path("data/features/meta")
    out_piano.mkdir(parents=True, exist_ok=True)
    out_orch.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    records = []
    failures = 0

    for _, row in df.iterrows():
        pair_id = str(row.get("pair_id", ""))
        piano_midi = row.get("piano_midi")
        orch_midi = row.get("orch_midi")

        if not pair_id or not isinstance(piano_midi, str) or not isinstance(orch_midi, str):
            failures += 1
            records.append(
                {"pair_id": pair_id, "error": "missing pair_id/piano_midi/orch_midi"}
            )
            continue

        piano_path = Path(piano_midi)
        orch_path = Path(orch_midi)
        if not piano_path.exists() or not orch_path.exists():
            failures += 1
            records.append(
                {
                    "pair_id": pair_id,
                    "error": "midi path does not exist",
                    "piano_midi": str(piano_path),
                    "orch_midi": str(orch_path),
                }
            )
            continue

        fname = safe_pair_id_to_filename(pair_id)

        try:
            pf = extract_piano_features(str(piano_path), hop_s=args.hop_s)
            of = extract_orch_features(
                str(orch_path),
                hop_s=args.hop_s,
                doubling_window_frames=args.doubling_window_frames,
            )

            piano_npz = out_piano / f"{fname}.npz"
            orch_npz = out_orch / f"{fname}.npz"

            np.savez_compressed(piano_npz, **piano_features_to_npz_dict(pf))
            np.savez_compressed(orch_npz, **orch_features_to_npz_dict(of))

            # Extra quick stats for indexing/debugging
            T_piano = int(pf.roll.shape[0])
            T_orch = int(of.instrument_activity.shape[0])
            K_orch = int(of.sparse_t.shape[0])

            # Doubling summary scalars (safe for empty)
            unison_total = int(of.doubling_unison_counts.sum()) if hasattr(of, "doubling_unison_counts") else 0
            interval_total = int(of.doubling_interval_hist.sum()) if hasattr(of, "doubling_interval_hist") else 0

            # Registration NaN rates (how often frames have no onsets)
            reg_nan_global_centroid = nan_frac(of.global_pitch_centroid) if hasattr(of, "global_pitch_centroid") else 1.0
            reg_nan_global_low = nan_frac(of.global_low_pitch) if hasattr(of, "global_low_pitch") else 1.0
            reg_nan_global_high = nan_frac(of.global_high_pitch) if hasattr(of, "global_high_pitch") else 1.0

            # Family count (should match N_FAM)
            n_fam = int(of.family_activity.shape[1]) if hasattr(of, "family_activity") and of.family_activity.ndim == 2 else 0

            records.append(
                {
                    "pair_id": pair_id,
                    "composer_guess": row.get("composer_guess"),
                    "source": row.get("source"),
                    "piano_npz": str(piano_npz),
                    "orch_npz": str(orch_npz),
                    "T_piano": T_piano,
                    "T_orch": T_orch,
                    "piano_duration_s": float(pf.duration_s),
                    "orch_duration_s": float(of.duration_s),
                    "orch_sparse_K": K_orch,
                    "orch_n_families": n_fam,
                    "doubling_window_frames": int(args.doubling_window_frames),
                    "doubling_unison_total": unison_total,
                    "doubling_interval_total": interval_total,
                    "reg_nan_frac_global_centroid": reg_nan_global_centroid,
                    "reg_nan_frac_global_low": reg_nan_global_low,
                    "reg_nan_frac_global_high": reg_nan_global_high,
                }
            )

        except Exception as e:
            failures += 1
            records.append(
                {
                    "pair_id": pair_id,
                    "error": repr(e),
                    "piano_midi": str(piano_path),
                    "orch_midi": str(orch_path),
                }
            )

    idx = pd.DataFrame(records)
    idx_path = out_meta / "features_index.csv"
    idx.to_csv(idx_path, index=False)

    print(f"[ok] wrote feature index: {idx_path}")
    print(f"[ok] processed rows: {len(idx)}")
    print(f"[warn] failures: {failures}")


if __name__ == "__main__":
    main()