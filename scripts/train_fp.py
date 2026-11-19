#!/usr/bin/env python3
"""Train FPNet on the local split. Usage:

    python scripts/train_fp.py --epochs 3 --batch-size 256 --out runs/fpnet.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fingerprint import FPNet, train_fpnet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/local_spectra.parquet")
    ap.add_argument("--out", default="runs/fpnet.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    train_rows = df[df["split"] == "train"].to_dict("records")
    val_rows = df[df["split"] == "val"].to_dict("records")
    if args.max_train:
        train_rows = train_rows[: args.max_train]
    print(f"train spectra: {len(train_rows)}, val spectra: {len(val_rows)}")

    device = torch.device(args.device)
    model = FPNet(d_model=args.d_model)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    history = train_fpnet(model, train_rows, val_rows, device,
                          epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, seed=args.seed, log=print)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"d_model": args.d_model, "n_bits": model.n_bits},
        "history": history,
    }, args.out)
    print(f"saved {args.out}")
    print(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
