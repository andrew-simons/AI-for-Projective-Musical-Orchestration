# training/train_baseline.py
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.baseline_encoder_decoder import BaselineConfig, BaselineEncoderDecoder


# -----------------------
# Dataset
# -----------------------
class PianoOrchChunkDataset(Dataset):
    """
    Loads (piano_npz, orch_npz) pairs from features_index.csv and returns random chunks.

    Output:
      x: (L, d_in)
      y: (L, 129)
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        p_path = row["piano_npz"]
        o_path = row["orch_npz"]

        p = np.load(p_path)
        o = np.load(o_path)

        roll = p["roll"].astype(np.float32)          # (T,128)
        if self.use_onset:
            onset = p["onset"].astype(np.float32)    # (T,128)
            x_full = np.concatenate([roll, onset], axis=1)  # (T,256)
        else:
            x_full = roll  # (T,128)

        y_full = o["instrument_activity"].astype(np.float32)  # (T,129)

        T = min(x_full.shape[0], y_full.shape[0])
        if T <= 1:
            # edge case: return tiny padded chunk (rare)
            x = np.zeros((self.chunk_len, x_full.shape[1]), dtype=np.float32)
            y = np.zeros((self.chunk_len, y_full.shape[1]), dtype=np.float32)
            return torch.from_numpy(x), torch.from_numpy(y)

        L = self.chunk_len
        if T >= L:
            start = int(self.rng.integers(0, T - L + 1))
            x = x_full[start : start + L]
            y = y_full[start : start + L]
        else:
            # pad if piece shorter than chunk
            x = np.zeros((L, x_full.shape[1]), dtype=np.float32)
            y = np.zeros((L, y_full.shape[1]), dtype=np.float32)
            x[:T] = x_full[:T]
            y[:T] = y_full[:T]

        return torch.from_numpy(x), torch.from_numpy(y)


# -----------------------
# Metrics
# -----------------------
@torch.no_grad()
def compute_metrics(logits: torch.Tensor, y: torch.Tensor, thr: float = 0.2) -> Dict[str, float]:
    """
    logits, y: (B,T,C)
    """
    probs = torch.sigmoid(logits)
    mse = torch.mean((probs - y) ** 2).item()
    mae = torch.mean(torch.abs(probs - y)).item()

    yb = (y >= thr)
    pb = (probs >= thr)

    tp = (pb & yb).sum().item()
    fp = (pb & ~yb).sum().item()
    fn = (~pb & yb).sum().item()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {"mse": mse, "mae": mae, "precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def per_instrument_f1(logits: torch.Tensor, y: torch.Tensor, thr: float = 0.2) -> np.ndarray:
    """
    Returns F1 per instrument: (C,)
    """
    probs = torch.sigmoid(logits)
    pb = (probs >= thr)
    yb = (y >= thr)

    # Sum over batch and time
    tp = (pb & yb).sum(dim=(0, 1)).cpu().numpy()
    fp = (pb & ~yb).sum(dim=(0, 1)).cpu().numpy()
    fn = (~pb & yb).sum(dim=(0, 1)).cpu().numpy()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return f1.astype(np.float32)


# -----------------------
# Utilities
# -----------------------
def load_index_csv(path: str = "data/features/meta/features_index.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Keep only rows with both npz
    df = df[df["piano_npz"].notna() & df["orch_npz"].notna()].copy()
    return df


def make_splits(df: pd.DataFrame, seed: int = 0, frac: float = 0.25, val_frac: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def save_checkpoint(path: Path, model: nn.Module, optim: torch.optim.Optimizer, step: int, epoch: int, cfg: BaselineConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "step": step,
            "epoch": epoch,
            "config": asdict(cfg),
        },
        str(path),
    )


# -----------------------
# Main train
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", type=str, default="data/features/meta/features_index.csv")
    ap.add_argument("--subset_frac", type=float, default=0.25)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--chunk_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.2)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--dec_layers", type=int, default=4)
    ap.add_argument("--dim_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--use_onset", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--run_name", type=str, default="baseline_v1")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    df = load_index_csv(args.index_csv)
    train_df, val_df = make_splits(df, seed=args.seed, frac=args.subset_frac, val_frac=args.val_frac)

    d_in = 256 if args.use_onset else 128
    cfg = BaselineConfig(
        d_in=d_in,
        d_out=129,
        d_model=args.d_model,
        nhead=args.nhead,
        num_enc_layers=args.enc_layers,
        num_dec_layers=args.dec_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        max_len=max(2048, args.chunk_len),
        use_learned_pos=True,
    )

    model = BaselineEncoderDecoder(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = PianoOrchChunkDataset(train_df, chunk_len=args.chunk_len, use_onset=args.use_onset, seed=args.seed)
    val_ds = PianoOrchChunkDataset(val_df, chunk_len=args.chunk_len, use_onset=args.use_onset, seed=args.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"args": vars(args), "cfg": asdict(cfg)}, indent=2))

    step = 0
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb, yb)  # teacher forcing
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

            if step % args.log_every == 0:
                model.eval()
                with torch.no_grad():
                    m = compute_metrics(logits, yb, thr=args.thr)
                model.train()
                print(f"epoch={epoch} step={step} train_loss={loss.item():.4f} f1={m['f1']:.3f} mse={m['mse']:.4f}")

            if step % args.ckpt_every == 0 and step > 0:
                save_checkpoint(out_dir / "last.pt", model, optim, step, epoch, cfg)

            step += 1

        # Validation
        model.eval()
        val_losses: List[float] = []
        agg_logits = []
        agg_y = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb, yb)
                loss = loss_fn(logits, yb)
                val_losses.append(loss.item())
                agg_logits.append(logits.detach().cpu())
                agg_y.append(yb.detach().cpu())

        vloss = float(np.mean(val_losses)) if val_losses else float("inf")
        logits_all = torch.cat(agg_logits, dim=0) if agg_logits else torch.empty(0)
        y_all = torch.cat(agg_y, dim=0) if agg_y else torch.empty(0)
        vm = compute_metrics(logits_all, y_all, thr=args.thr) if logits_all.numel() > 0 else {"mse": float("nan"), "mae": float("nan"), "precision": 0.0, "recall": 0.0, "f1": 0.0}

        print(f"[val] epoch={epoch} val_loss={vloss:.4f} f1={vm['f1']:.3f} mse={vm['mse']:.4f} mae={vm['mae']:.4f}")

        # Per-instrument F1 report
        if logits_all.numel() > 0:
            f1_inst = per_instrument_f1(logits_all, y_all, thr=args.thr)
            order = np.argsort(-f1_inst)
            top = order[:10]
            bot = order[-10:]
            print("[val] top instruments by F1:", [(int(i), float(f1_inst[i])) for i in top])
            print("[val] bottom instruments by F1:", [(int(i), float(f1_inst[i])) for i in bot])

            # save CSV
            pd.DataFrame({"inst_id": np.arange(len(f1_inst)), "f1": f1_inst}).to_csv(out_dir / "val_f1_by_instrument.csv", index=False)

        # Checkpoints
        save_checkpoint(out_dir / "last.pt", model, optim, step, epoch, cfg)
        if vloss < best_val:
            best_val = vloss
            save_checkpoint(out_dir / "best.pt", model, optim, step, epoch, cfg)
            print(f"[ok] new best checkpoint: {out_dir / 'best.pt'} (val_loss={best_val:.4f})")

    print(f"[done] checkpoints in {out_dir}")


if __name__ == "__main__":
    main()