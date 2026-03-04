# training/train_stage1.py
from __future__ import annotations

# --- make repo root importable no matter where this is run from ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # projective_orchestration/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -----------------------------------------------------------------

import argparse
import json
import math
from dataclasses import asdict
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.stage1_encoder import Stage1Config, Stage1Encoder


# -----------------------
# Dataset
# -----------------------
class PianoOrchChunkDataset(Dataset):
    """
    Returns:
      x: (L, d_in)
      y: (L, 129)
      pad_mask: (L,) bool  (True = padded)
    """
    def __init__(
        self,
        index_df: pd.DataFrame,
        chunk_len: int = 256,
        use_onset: bool = True,
        seed: int = 0,
    ):
        self.df = index_df.reset_index(drop=True)
        self.chunk_len = int(chunk_len)
        self.use_onset = bool(use_onset)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        p = np.load(row["piano_npz"])
        o = np.load(row["orch_npz"])

        roll = p["roll"].astype(np.float32)  # (T,128)
        if self.use_onset:
            onset = p["onset"].astype(np.float32)
            x_full = np.concatenate([roll, onset], axis=1).astype(np.float32)  # (T,256)
        else:
            x_full = roll.astype(np.float32)

        y_full = o["instrument_activity"].astype(np.float32)  # (T,129)

        T = min(x_full.shape[0], y_full.shape[0])
        L = self.chunk_len

        if T <= 1:
            x = np.zeros((L, x_full.shape[1]), dtype=np.float32)
            y = np.zeros((L, y_full.shape[1]), dtype=np.float32)
            pad = np.ones((L,), dtype=bool)
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(pad)

        if T >= L:
            start = int(self.rng.integers(0, T - L + 1))
            x = x_full[start : start + L]
            y = y_full[start : start + L]
            pad = np.zeros((L,), dtype=bool)
        else:
            x = np.zeros((L, x_full.shape[1]), dtype=np.float32)
            y = np.zeros((L, y_full.shape[1]), dtype=np.float32)
            pad = np.ones((L,), dtype=bool)
            x[:T] = x_full[:T]
            y[:T] = y_full[:T]
            pad[:T] = False

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(pad)


# -----------------------
# Metrics (masked)
# -----------------------
@torch.no_grad()
def compute_metrics_masked(
    logits: torch.Tensor,  # (B,L,C)
    y: torch.Tensor,       # (B,L,C)
    pad: torch.Tensor,     # (B,L) True for padded
    thr: float = 0.2
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)

    mask = (~pad).unsqueeze(-1).float()  # (B,L,1)
    denom = mask.sum() * y.size(-1) + 1e-9

    mse = (((probs - y) ** 2) * mask).sum().item() / float(denom)
    mae = (torch.abs(probs - y) * mask).sum().item() / float(denom)

    yb = (y >= thr) & (~pad).unsqueeze(-1)
    pb = (probs >= thr) & (~pad).unsqueeze(-1)

    tp = (pb & yb).sum().item()
    fp = (pb & ~yb).sum().item()
    fn = (~pb & yb).sum().item()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {"mse": float(mse), "mae": float(mae), "precision": float(precision), "recall": float(recall), "f1": float(f1)}


@torch.no_grad()
def per_instrument_f1_masked(
    logits: torch.Tensor,  # (B,L,C)
    y: torch.Tensor,       # (B,L,C)
    pad: torch.Tensor,     # (B,L)
    thr: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      f1: (C,)
      pos_frames: (C,) number of positive frames in y (masked)
    """
    probs = torch.sigmoid(logits)
    valid = (~pad).unsqueeze(-1)  # (B,L,1)

    pb = (probs >= thr) & valid
    yb = (y >= thr) & valid

    tp = (pb & yb).sum(dim=(0, 1)).cpu().numpy()
    fp = (pb & ~yb).sum(dim=(0, 1)).cpu().numpy()
    fn = (~pb & yb).sum(dim=(0, 1)).cpu().numpy()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    pos_frames = yb.sum(dim=(0, 1)).cpu().numpy()
    return f1.astype(np.float32), pos_frames.astype(np.int64)


def macro_f1_over_active(f1: np.ndarray, pos_frames: np.ndarray, min_pos_frames: int) -> float:
    active = pos_frames >= int(min_pos_frames)
    if active.sum() == 0:
        return float("nan")
    return float(np.mean(f1[active]))


# -----------------------
# Utilities
# -----------------------
def load_index_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["piano_npz"].notna() & df["orch_npz"].notna()].copy()


def make_splits(df: pd.DataFrame, seed: int, frac: float, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    if frac < 1.0:
        n = max(1, int(math.floor(len(df) * frac)))
        idx = idx[:n]

    n_val = max(1, int(math.floor(len(idx) * val_frac)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def compute_pos_weight(train_df: pd.DataFrame, max_w: float = 50.0) -> np.ndarray:
    """
    pos_weight[c] = neg/pos for BCEWithLogitsLoss.
    Computed over full sequences (not chunks) to be stable.
    """
    pos = np.zeros((129,), dtype=np.float64)
    tot = 0.0

    for _, row in train_df.iterrows():
        o = np.load(row["orch_npz"])
        y = o["instrument_activity"].astype(np.float32)
        if y.ndim != 2 or y.shape[1] != 129:
            continue
        pos += y.sum(axis=0)
        tot += float(y.shape[0])

    neg = np.maximum(0.0, tot - pos)
    w = neg / (pos + 1e-6)

    # clamp to avoid absurd weights for ultra-rare instruments
    w = np.clip(w, 1.0, float(max_w)).astype(np.float32)
    return w


def save_checkpoint(path: Path, model: nn.Module, optim: torch.optim.Optimizer, step: int, epoch: int, cfg: Stage1Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optim": optim.state_dict(), "step": step, "epoch": epoch, "config": asdict(cfg)},
        str(path),
    )


# -----------------------
# Main train
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", type=str, default="data/features/meta/features_index.csv")
    ap.add_argument("--subset_frac", type=float, default=1.0)
    ap.add_argument("--val_frac", type=float, default=0.1)

    ap.add_argument("--chunk_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.2)
    ap.add_argument("--min_pos_frames", type=int, default=200, help="For macro-F1: only instruments with >= this many positive frames.")

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dim_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--use_onset", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--smooth_lambda", type=float, default=0.01)

    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--run_name", type=str, default="stage1_encoder_v1")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    df = load_index_csv(args.index_csv)
    train_df, val_df = make_splits(df, seed=args.seed, frac=args.subset_frac, val_frac=args.val_frac)

    d_in = 256 if args.use_onset else 128
    cfg = Stage1Config(
        d_in=d_in,
        d_out=129,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        max_len=max(2048, args.chunk_len),
        use_learned_pos=True,
    )

    model = Stage1Encoder(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- pos_weight ----
    pos_w_np = compute_pos_weight(train_df, max_w=50.0)
    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pos_weight.npy", pos_w_np)

    # debug print: largest weights = rarest instruments
    topw = np.argsort(-pos_w_np)[:10]
    print("[debug] pos_weight top10:", [(int(i), float(pos_w_np[i])) for i in topw])

    pos_weight = torch.from_numpy(pos_w_np).to(device)

    train_ds = PianoOrchChunkDataset(train_df, chunk_len=args.chunk_len, use_onset=args.use_onset, seed=args.seed)
    val_ds = PianoOrchChunkDataset(val_df, chunk_len=args.chunk_len, use_onset=args.use_onset, seed=args.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    # AMP new API
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    (out_dir / "config.json").write_text(json.dumps({"args": vars(args), "cfg": asdict(cfg)}, indent=2))

    step = 0
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        for xb, yb, pad in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pad = pad.to(device)  # (B,L)

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb, src_key_padding_mask=pad)

                # elementwise BCE so we can mask pads
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none", pos_weight=pos_weight
                )
                mask = (~pad).unsqueeze(-1).float()  # (B,L,1)
                bce = (bce * mask).sum() / (mask.sum() * yb.size(-1) + 1e-9)

                # smoothness on probabilities (L1 works well)
                p = torch.sigmoid(logits)
                dp = torch.abs(p[:, 1:, :] - p[:, :-1, :])
                dp_mask = (~pad[:, 1:] & ~pad[:, :-1]).unsqueeze(-1).float()
                smooth = (dp * dp_mask).sum() / (dp_mask.sum() * yb.size(-1) + 1e-9)

                loss = bce + float(args.smooth_lambda) * smooth

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

            if step % args.log_every == 0:
                model.eval()
                with torch.no_grad():
                    m = compute_metrics_masked(logits, yb, pad, thr=args.thr)
                model.train()
                print(
                    f"epoch={epoch} step={step} loss={loss.item():.4f} "
                    f"bce={bce.item():.4f} smooth={smooth.item():.4f} f1={m['f1']:.3f}"
                )

            if step % args.ckpt_every == 0 and step > 0:
                save_checkpoint(out_dir / "last.pt", model, optim, step, epoch, cfg)

            step += 1

        # Validation
        model.eval()
        val_bces: List[float] = []
        agg_logits: List[torch.Tensor] = []
        agg_y: List[torch.Tensor] = []
        agg_pad: List[torch.Tensor] = []

        with torch.no_grad():
            for xb, yb, pad in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pad = pad.to(device)

                logits = model(xb, src_key_padding_mask=pad)

                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none", pos_weight=pos_weight
                )
                mask = (~pad).unsqueeze(-1).float()
                bce = (bce * mask).sum() / (mask.sum() * yb.size(-1) + 1e-9)

                val_bces.append(float(bce.item()))
                agg_logits.append(logits.cpu())
                agg_y.append(yb.cpu())
                agg_pad.append(pad.cpu())

        vloss = float(np.mean(val_bces)) if val_bces else float("inf")

        if agg_logits:
            logits_all = torch.cat(agg_logits, dim=0)
            y_all = torch.cat(agg_y, dim=0)
            pad_all = torch.cat(agg_pad, dim=0)

            vm = compute_metrics_masked(logits_all, y_all, pad_all, thr=args.thr)

            f1_inst, pos_frames = per_instrument_f1_masked(logits_all, y_all, pad_all, thr=args.thr)
            macro_active = macro_f1_over_active(f1_inst, pos_frames, min_pos_frames=args.min_pos_frames)

            order = np.argsort(-f1_inst)
            top = order[:10]
            bot = order[-10:]

            print(
                f"[val] epoch={epoch} val_bce={vloss:.4f} "
                f"micro_f1={vm['f1']:.3f} macro_f1_active={macro_active:.3f} "
                f"mse={vm['mse']:.4f} mae={vm['mae']:.4f}"
            )
            print("[val] top instruments by F1:", [(int(i), float(f1_inst[i]), int(pos_frames[i])) for i in top])
            print("[val] bottom instruments by F1:", [(int(i), float(f1_inst[i]), int(pos_frames[i])) for i in bot])

            pd.DataFrame(
                {"inst_id": np.arange(len(f1_inst)), "f1": f1_inst, "pos_frames": pos_frames}
            ).to_csv(out_dir / "val_f1_by_instrument.csv", index=False)
        else:
            print(f"[val] epoch={epoch} val_bce={vloss:.4f} (no data)")

        save_checkpoint(out_dir / "last.pt", model, optim, step, epoch, cfg)
        if vloss < best_val:
            best_val = vloss
            save_checkpoint(out_dir / "best.pt", model, optim, step, epoch, cfg)
            print(f"[ok] new best checkpoint: {out_dir / 'best.pt'} (val_bce={best_val:.4f})")

    print(f"[done] checkpoints in {out_dir}")


if __name__ == "__main__":
    main()