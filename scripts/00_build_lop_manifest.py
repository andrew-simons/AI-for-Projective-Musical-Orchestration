#!/usr/bin/env python3
"""
Build a paired manifest for LOP_database_06_09_17-style directory layout.

Expected layout (your tree):
ROOT/
  bouliane/0/*.mid *.csv
  bouliane/1/*.mid *.csv
  ...
  bouliane.csv
  imslp/... + imslp.csv
  hand_picked_Spotify/... + hand_picked_spotify.csv
  liszt_classical_archives/... + liszt_classical_archives.csv
  debug/... (subset)
  README.md

Outputs:
  data/processed/lop_manifest.parquet (or .csv fallback)
  reports/spot_check.csv
"""

from __future__ import annotations

import argparse
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd

# Optional MIDI stats via mido
try:
    import mido  # type: ignore
except Exception:
    mido = None


# ---------- Helpers ----------

SOURCE_DIR_BLACKLIST = {
    ".DS_Store",
    "__MACOSX",
}

ROOT_FILE_BLACKLIST_SUFFIXES = {".csv", ".md", ".txt", ".json", ".parquet"}


def is_numeric_dir(p: Path) -> bool:
    return p.is_dir() and p.name.isdigit()


def guess_composer_from_filename(stem: str) -> Optional[str]:
    """
    Heuristic: take leading token before first underscore, if it looks like a Name.
    Examples:
      Debussy_La Mer_i(...) -> Debussy
      StraussJ_Danube(...)  -> StraussJ (we keep as-is)
      Moussorgsky_TableauxProm -> Moussorgsky
    If no underscore, return None.
    """
    if "_" not in stem:
        return None
    lead = stem.split("_", 1)[0].strip()
    if not lead:
        return None
    # reject overly generic leads
    bad = {"orch", "solo", "piano", "keyboard", "symphony"}
    if lead.lower() in bad:
        return None
    return lead


def _score_piano_candidate(name: str) -> int:
    n = name.lower()
    s = 0
    # strong indicators
    if "solo" in n:
        s += 5
    if "piano" in n or "keyboard" in n:
        s += 3
    # avoid orchestra files
    if "orch" in n:
        s -= 6
    return s


def _score_orch_candidate(name: str) -> int:
    n = name.lower()
    s = 0
    if "orch" in n:
        s += 5
    # avoid solo/piano reductions
    if "solo" in n:
        s -= 6
    if "piano" in n or "keyboard" in n:
        s -= 3
    return s


def pick_best_pair_files(pair_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path], Dict[str, Any]]:
    """
    Returns (piano_mid, orch_mid, piano_csv, orch_csv, debug_info)

    We select candidates by scoring filenames. If multiple match, pick max score.
    """
    mids = sorted(pair_dir.glob("*.mid"))
    csvs = sorted(pair_dir.glob("*.csv"))

    debug: Dict[str, Any] = {
        "mid_files": [p.name for p in mids],
        "csv_files": [p.name for p in csvs],
    }

    piano_mid = None
    orch_mid = None
    if mids:
        piano_mid = max(mids, key=lambda p: _score_piano_candidate(p.name))
        orch_mid = max(mids, key=lambda p: _score_orch_candidate(p.name))

        # If both selectors picked the same file (can happen in weird edge cases), try second-best
        if piano_mid == orch_mid and len(mids) > 1:
            piano_sorted = sorted(mids, key=lambda p: _score_piano_candidate(p.name), reverse=True)
            orch_sorted = sorted(mids, key=lambda p: _score_orch_candidate(p.name), reverse=True)
            piano_mid = piano_sorted[0]
            orch_mid = orch_sorted[0]
            if piano_mid == orch_mid:
                # fallback: choose by "solo" and "orch" existence
                solo = [p for p in mids if "solo" in p.name.lower()]
                orch = [p for p in mids if "orch" in p.name.lower()]
                if solo:
                    piano_mid = solo[0]
                if orch:
                    orch_mid = orch[0]

    piano_csv = None
    orch_csv = None
    if csvs:
        piano_csv = max(csvs, key=lambda p: _score_piano_candidate(p.name))
        orch_csv = max(csvs, key=lambda p: _score_orch_candidate(p.name))
        if piano_csv == orch_csv and len(csvs) > 1:
            piano_sorted = sorted(csvs, key=lambda p: _score_piano_candidate(p.name), reverse=True)
            orch_sorted = sorted(csvs, key=lambda p: _score_orch_candidate(p.name), reverse=True)
            piano_csv = piano_sorted[0]
            orch_csv = orch_sorted[0]
            if piano_csv == orch_csv:
                solo = [p for p in csvs if "solo" in p.name.lower()]
                orch = [p for p in csvs if "orch" in p.name.lower()]
                if solo:
                    piano_csv = solo[0]
                if orch:
                    orch_csv = orch[0]

    debug.update(
        {
            "picked_piano_mid": piano_mid.name if piano_mid else None,
            "picked_orch_mid": orch_mid.name if orch_mid else None,
            "picked_piano_csv": piano_csv.name if piano_csv else None,
            "picked_orch_csv": orch_csv.name if orch_csv else None,
        }
    )

    return piano_mid, orch_mid, piano_csv, orch_csv, debug


def midi_quick_stats(path: Path) -> Dict[str, Optional[float]]:
    """
    Cheap stats: track count and approximate duration in seconds.
    Requires mido; otherwise returns None values.
    """
    if mido is None:
        return {"tracks": None, "duration_s": None}

    try:
        mid = mido.MidiFile(path)
        # Mido provides .length (seconds) for standard MIDIs
        return {"tracks": float(len(mid.tracks)), "duration_s": float(getattr(mid, "length", None) or 0.0)}
    except Exception:
        return {"tracks": None, "duration_s": None}


def duration_mismatch_flag(piano_s: Optional[float], orch_s: Optional[float], ratio_thresh: float = 1.6, abs_thresh_s: float = 12.0) -> bool:
    """i2325
    
    Flag if durations disagree a lot. This is a heuristic.
    """
    if piano_s is None or orch_s is None:
        return False
    if piano_s <= 0 or orch_s <= 0:
        return False
    big_abs = abs(piano_s - orch_s) >= abs_thresh_s
    big_ratio = (max(piano_s, orch_s) / max(1e-6, min(piano_s, orch_s))) >= ratio_thresh
    return bool(big_abs and big_ratio)


# ---------- Data model ----------

@dataclass
class PairRow:
    pair_id: str                  # e.g. "bouliane/12"
    source: str                   # e.g. "bouliane"
    pair_index: int               # 12
    pair_dir: str                 # absolute path
    piano_midi: Optional[str]
    orch_midi: Optional[str]
    piano_csv: Optional[str]
    orch_csv: Optional[str]
    composer_guess: Optional[str]
    raw_stem_guess: Optional[str]
    is_debug: bool
    is_imslp: bool
    has_missing_files: bool
    bad_duration_mismatch: bool
    piano_tracks: Optional[float]
    orch_tracks: Optional[float]
    piano_duration_s: Optional[float]
    orch_duration_s: Optional[float]


# ---------- Main build ----------

def build_manifest(root: Path, include_debug: bool = False) -> pd.DataFrame:
    rows: List[PairRow] = []

    # source dirs are subdirs that are not numeric and not blacklisted
    for source_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        if source_dir.name in SOURCE_DIR_BLACKLIST:
            continue

        # skip weird dirs
        if source_dir.name.startswith("."):
            continue

        source = source_dir.name
        is_debug_source = (source.lower() == "debug")
        if is_debug_source and not include_debug:
            continue

        # only proceed if it contains numeric subdirs
        numeric_children = [p for p in source_dir.iterdir() if is_numeric_dir(p)]
        if not numeric_children:
            # this might be some other helper dir
            continue

        for pair_dir in sorted(numeric_children, key=lambda p: int(p.name)):
            pair_idx = int(pair_dir.name)
            pair_id = f"{source}/{pair_idx}"

            piano_mid, orch_mid, piano_csv, orch_csv, dbg = pick_best_pair_files(pair_dir)

            # Guess composer + stem from whichever file exists
            stem_source = None
            if orch_mid:
                stem_source = orch_mid.stem
            elif piano_mid:
                stem_source = piano_mid.stem
            elif orch_csv:
                stem_source = orch_csv.stem
            elif piano_csv:
                stem_source = piano_csv.stem

            composer = guess_composer_from_filename(stem_source or "") if stem_source else None

            piano_stats = midi_quick_stats(piano_mid) if piano_mid else {"tracks": None, "duration_s": None}
            orch_stats = midi_quick_stats(orch_mid) if orch_mid else {"tracks": None, "duration_s": None}

            piano_dur = piano_stats.get("duration_s")
            orch_dur = orch_stats.get("duration_s")

            has_missing = any(x is None for x in [piano_mid, orch_mid, piano_csv, orch_csv])

            bad_mismatch = duration_mismatch_flag(
                piano_dur if isinstance(piano_dur, (int, float)) else None,
                orch_dur if isinstance(orch_dur, (int, float)) else None,
            )

            row = PairRow(
                pair_id=pair_id,
                source=source,
                pair_index=pair_idx,
                pair_dir=str(pair_dir.resolve()),
                piano_midi=str(piano_mid.resolve()) if piano_mid else None,
                orch_midi=str(orch_mid.resolve()) if orch_mid else None,
                piano_csv=str(piano_csv.resolve()) if piano_csv else None,
                orch_csv=str(orch_csv.resolve()) if orch_csv else None,
                composer_guess=composer,
                raw_stem_guess=stem_source,
                is_debug=is_debug_source,
                is_imslp=(source.lower() == "imslp"),
                has_missing_files=has_missing,
                bad_duration_mismatch=bad_mismatch,
                piano_tracks=piano_stats.get("tracks"),
                orch_tracks=orch_stats.get("tracks"),
                piano_duration_s=piano_dur if isinstance(piano_dur, (int, float)) else None,
                orch_duration_s=orch_dur if isinstance(orch_dur, (int, float)) else None,
            )
            rows.append(row)

    df = pd.DataFrame([asdict(r) for r in rows])

    # A simple "usable" flag for downstream filtering
    df["usable"] = (~df["has_missing_files"]) & (~df["bad_duration_mismatch"])
    return df


def write_outputs(df: pd.DataFrame, out_dir: Path, reports_dir: Path, seed: int = 42) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Prefer parquet; fall back to csv if pyarrow missing
    parquet_path = out_dir / "lop_manifest.parquet"
    csv_path = out_dir / "lop_manifest.csv"

    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[ok] wrote {parquet_path}")
    except Exception as e:
        print(f"[warn] could not write parquet ({e}); writing csv instead")
        df.to_csv(csv_path, index=False)
        print(f"[ok] wrote {csv_path}")

    # Spot check sample
    rng = random.Random(seed)
    sample_n = min(25, len(df))
    sample_idx = rng.sample(list(df.index), sample_n) if sample_n > 0 else []
    spot = df.loc[sample_idx].sort_values(["source", "pair_index"])
    spot_path = reports_dir / "spot_check.csv"
    spot.to_csv(spot_path, index=False)
    print(f"[ok] wrote {spot_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lop_root", required=True, help="Path to LOP_database_06_09_17 root directory")
    ap.add_argument("--out_dir", default="data/processed", help="Output directory for processed artifacts")
    ap.add_argument("--reports_dir", default="reports", help="Output directory for reports")
    ap.add_argument("--include_debug", action="store_true", help="Include the debug/ source folder")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.lop_root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"LOP root not found: {root}")

    df = build_manifest(root, include_debug=args.include_debug)

    # Print summary
    print("\n=== Manifest summary ===")
    print(f"rows: {len(df)}")
    if len(df) > 0:
        print(df.groupby("source")["pair_id"].count().sort_values(ascending=False).to_string())
        print("\nusable rows:", int(df["usable"].sum()))
        print("missing files:", int(df["has_missing_files"].sum()))
        print("bad duration mismatch:", int(df["bad_duration_mismatch"].sum()))
        print("\nTop composers (guessed):")
        print(df["composer_guess"].value_counts(dropna=False).head(15).to_string())

    write_outputs(df, Path(args.out_dir), Path(args.reports_dir), seed=args.seed)


if __name__ == "__main__":
    main()
