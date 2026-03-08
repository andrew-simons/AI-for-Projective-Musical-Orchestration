from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class LatentPoint:
    """
    Single data-point in latent space, used for background samples / landmarks.
    """

    id: str
    coords_2d: np.ndarray  # shape (2,)
    latent: np.ndarray  # shape (D,)
    label: Optional[str] = None
    meta: Dict[str, str] | None = None


@dataclass
class LatentSpace:
    """
    PCA-based latent space over time-averaged instrument activity vectors.
    """

    mean: np.ndarray  # (D,)
    components: np.ndarray  # (2, D)
    points: List[LatentPoint]

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def project(self, z: np.ndarray) -> np.ndarray:
        """
        Map a true latent vector z (D,) to 2D coordinates using PCA components.
        """
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        return (self.components @ (z - self.mean)).astype(np.float32)

    def invert(self, xy: Tuple[float, float]) -> np.ndarray:
        """
        Approximate inverse mapping from 2D coordinates back into latent space.

        Returns:
            z: (D,) vector in the PCA subspace.
        """
        x, y = float(xy[0]), float(xy[1])
        v = np.array([x, y], dtype=np.float32)
        return (self.mean + self.components.T @ v).astype(np.float32)

    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Return (x_min, x_max), (y_min, y_max) over all sample points.
        """
        if not self.points:
            return (0.0, 1.0), (0.0, 1.0)
        xs = np.array([p.coords_2d[0] for p in self.points], dtype=np.float32)
        ys = np.array([p.coords_2d[1] for p in self.points], dtype=np.float32)
        return (float(xs.min()), float(xs.max())), (float(ys.min()), float(ys.max()))


def _load_instrument_activity_vector(npz_path: Path) -> np.ndarray:
    """
    Load a single orchestration feature NPZ and extract a 129-D mean instrument activity vector.
    """
    data = np.load(str(npz_path))
    # training features: instrument_activity: (T, 129)
    ia = data["instrument_activity"].astype(np.float32)
    if ia.ndim != 2:
        raise ValueError(f"instrument_activity has unexpected shape {ia.shape} in {npz_path}")
    if ia.shape[1] != 129:
        raise ValueError(f"Expected 129 instrument channels, got {ia.shape[1]} in {npz_path}")
    if ia.shape[0] == 0:
        return np.zeros((129,), dtype=np.float32)
    return ia.mean(axis=0).astype(np.float32)


def _fit_pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a 2D PCA projection using a simple SVD (no external dependencies).

    Args:
        X: (N, D) matrix of latent vectors.

    Returns:
        mean: (D,)
        components: (2, D) top-2 principal components (rows).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    N, D = X.shape
    if N == 0:
        raise ValueError("Cannot fit PCA with N=0")

    mean = X.mean(axis=0)
    Xc = X - mean

    # Economy SVD on covariance proxy
    # Shape: Xc: (N, D). We want top-2 principal directions in feature space.
    # Use SVD directly on Xc to avoid forming full covariance.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Vt: (min(N,D), D). Take first 2 rows as principal components.
    k = min(2, Vt.shape[0])
    components = Vt[:k].astype(np.float32)
    if components.shape[0] < 2:
        # pad with zeros if dataset is degenerate
        pad = np.zeros((2 - components.shape[0], D), dtype=np.float32)
        components = np.concatenate([components, pad], axis=0)
    return mean.astype(np.float32), components


def build_latent_space_from_index(
    index_csv: Path,
    root: Optional[Path] = None,
    max_items: int = 512,
) -> LatentSpace:
    """
    Build a LatentSpace instance from the training feature index CSV.

    Each row contributes one latent vector by time-averaging the 'instrument_activity'
    array in its orchestral NPZ file.
    """
    root = root or index_csv.parents[2]
    df = pd.read_csv(index_csv)

    # Require both piano and orch npz paths; mirror training/train_stage1.py
    df = df[df["piano_npz"].notna() & df["orch_npz"].notna()].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No valid rows in feature index: {index_csv}")

    if max_items is not None and max_items > 0:
        df = df.iloc[: int(max_items)].copy()

    latents: List[np.ndarray] = []
    points: List[LatentPoint] = []

    for _, row in df.iterrows():
        pair_id = str(row["pair_id"])
        orch_rel = str(row["orch_npz"])
        label = str(row.get("composer_guess", "")) if "composer_guess" in df.columns else ""

        npz_path = (root / orch_rel).resolve()
        if not npz_path.exists():
            continue

        z = _load_instrument_activity_vector(npz_path)
        latents.append(z)
        points.append(
            LatentPoint(
                id=pair_id,
                coords_2d=np.zeros((2,), dtype=np.float32),  # filled after PCA
                latent=z,
                label=label or None,
                meta={"source": str(row.get("source", ""))},
            )
        )

    if not latents:
        raise ValueError("No latent vectors could be constructed from feature index.")

    X = np.stack(latents, axis=0)  # (N, 129)
    mean, components = _fit_pca_2d(X)

    # Project all points
    for p in points:
        p.coords_2d = (components @ (p.latent - mean)).astype(np.float32)

    return LatentSpace(mean=mean, components=components, points=points)

