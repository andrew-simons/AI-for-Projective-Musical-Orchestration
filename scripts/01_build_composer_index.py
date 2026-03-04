#!/usr/bin/env python3
"""
Build a composer index + quality summary from lop_manifest.

Outputs:
  data/processed/composer_index.csv
  data/processed/composer_quality.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/lop_manifest.parquet")
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        # fallback to csv
        manifest_path = Path("data/processed/lop_manifest.csv")
    if not manifest_path.exists():
        raise SystemExit("Manifest not found. Run scripts/00_build_lop_manifest.py first.")

    df = read_manifest(manifest_path)

    # composer_index: counts per composer
    comp = (
        df.groupby("composer_guess", dropna=False)
        .agg(
            n_pairs=("pair_id", "count"),
            n_usable=("usable", "sum"),
            n_imslp=("is_imslp", "sum"),
            sources=("source", lambda s: ",".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .rename(columns={"composer_guess": "composer"})
        .sort_values("n_pairs", ascending=False)
    )
    comp["usable_frac"] = comp["n_usable"] / comp["n_pairs"].clip(lower=1)
    comp["imslp_frac"] = comp["n_imslp"] / comp["n_pairs"].clip(lower=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    composer_index_path = out_dir / "composer_index.csv"
    comp.to_csv(composer_index_path, index=False)
    print(f"[ok] wrote {composer_index_path}")

    # A "quality" view: per composer, highlight problematic rates
    qual = comp.copy()
    qual["notes"] = ""
    qual.loc[qual["usable_frac"] < 0.8, "notes"] += "low_usable_frac;"
    qual.loc[qual["imslp_frac"] > 0.0, "notes"] += "contains_imslp;"

    composer_quality_path = out_dir / "composer_quality.csv"
    qual.to_csv(composer_quality_path, index=False)
    print(f"[ok] wrote {composer_quality_path}")

    print("\nTop composers:")
    print(comp.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
