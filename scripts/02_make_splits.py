#!/usr/bin/env python3
"""
Create train/valid/test split files from lop_manifest.

Outputs:
  data/processed/splits/train.txt
  data/processed/splits/valid.txt
  data/processed/splits/test.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import pandas as pd


def read_manifest() -> pd.DataFrame:
    p = Path("data/processed/lop_manifest.parquet")
    if p.exists():
        return pd.read_parquet(p)
    p = Path("data/processed/lop_manifest.csv")
    if p.exists():
        return pd.read_csv(p)
    raise SystemExit("Manifest not found. Run scripts/00_build_lop_manifest.py first.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_imslp", action="store_true", help="Exclude IMSLP source entirely")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--valid_frac", type=float, default=0.1)
    args = ap.parse_args()

    df = read_manifest()

    df = df[df["usable"] == True].copy()
    if args.exclude_imslp:
        df = df[df["is_imslp"] == False].copy()

    ids = list(df["pair_id"].values)
    rng = random.Random(args.seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * args.train_frac)
    n_valid = int(n * args.valid_frac)
    n_test = n - n_train - n_valid

    train_ids = ids[:n_train]
    valid_ids = ids[n_train : n_train + n_valid]
    test_ids = ids[n_train + n_valid :]

    splits_dir = Path("data/processed/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    (splits_dir / "train.txt").write_text("\n".join(train_ids) + "\n")
    (splits_dir / "valid.txt").write_text("\n".join(valid_ids) + "\n")
    (splits_dir / "test.txt").write_text("\n".join(test_ids) + "\n")

    print("[ok] wrote splits:")
    print(f"  train: {len(train_ids)}")
    print(f"  valid: {len(valid_ids)}")
    print(f"  test : {len(test_ids)}")


if __name__ == "__main__":
    main()
