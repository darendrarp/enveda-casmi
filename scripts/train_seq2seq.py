#!/usr/bin/env python3
"""Train the spectra->SMILES transformer on the local split. Usage:

    python scripts/train_seq2seq.py --epochs 3 --out runs/seq2seq.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.seq2seq import SmilesTokenizer, SpectraTransformer, train_seq2seq  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/local_spectra.parquet")
    ap.add_argument("--out", default="runs/seq2seq.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
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

    tokenizer = SmilesTokenizer.fit([r["normalized_smiles"] for r in train_rows])
    print(f"tokenizer vocab: {len(tokenizer)}")
    device = torch.device(args.device)
    model = SpectraTransformer(len(tokenizer), d_model=args.d_model,
                               nhead=args.nhead,
                               num_encoder_layers=args.layers,
                               num_decoder_layers=args.layers)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    history = train_seq2seq(model, tokenizer, train_rows, val_rows, device,
                            epochs=args.epochs, batch_size=args.batch_size,
                            lr=args.lr, seed=args.seed, log=print)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "vocab": tokenizer.vocab,
        "config": {"d_model": args.d_model, "nhead": args.nhead,
                   "num_encoder_layers": args.layers,
                   "num_decoder_layers": args.layers},
        "history": history,
    }, args.out)
    print(f"saved {args.out}")
    print(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
