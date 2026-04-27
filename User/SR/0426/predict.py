"""
Inference for validation/test bearings.

Given a directory of .tdms files (one bearing run), this script:
  1) extracts the same per-file features used at training
  2) builds the last-window input (T frames ending at the last file)
  3) loads the trained checkpoint and predicts normalized RUL
  4) converts back to seconds using the lifetime statistics from training

Output: predicted RUL [seconds] for each input run dir, written to a CSV.

Usage:
  python predict.py /path/to/Validation_Bearing1 /path/to/Validation_Bearing2 \
                    --ckpt checkpoints/best.pt --out submission.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import AdaptiveMAGNN_TCN
from preprocess import features_one_run


def build_last_window(df: pd.DataFrame, feat_cols: list[str], window: int,
                      mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    df = df.sort_values("idx").reset_index(drop=True)
    F = df[feat_cols].to_numpy(dtype=np.float32)
    F = (F - mean) / np.maximum(std, 1e-6)
    F = F[-window:]
    if F.shape[0] < window:
        pad = np.zeros((window - F.shape[0], F.shape[1]), dtype=np.float32)
        F = np.concatenate([pad, F], axis=0)
    return F.T  # (n_vars, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="bearing TDMS dirs to predict")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--feat-cache", default="features/predict_cache",
                    help="dir for caching per-run feature CSVs")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    feat_cols = ck["feat_cols"]
    mean = np.asarray(ck["scaler_mean"], dtype=np.float32)
    std  = np.asarray(ck["scaler_std"],  dtype=np.float32)
    train_args = ck["args"]
    lifetimes = ck["lifetimes"]                      # minutes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptiveMAGNN_TCN(
        n_vars=len(feat_cols), t_in=train_args["window"],
        c_hidden=train_args["c_hidden"], n_scales=train_args["n_scales"],
        top_k=train_args["top_k"], tcn_hidden=train_args["tcn_hidden"],
        tcn_blocks=train_args["tcn_blocks"]).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()

    # use median train lifetime as denormalization scale for unseen bearings.
    # Training labels are normalized RUL ∈ [0,1] of *each run's own* lifetime,
    # so at the last frame the typical RUL is exactly 0. The model's prediction
    # at the last frame is therefore "fraction of remaining life relative to a
    # typical run" and the most natural denorm is the median train lifetime.
    L_ref_min = float(np.median(list(lifetimes.values())))
    print(f"[ckpt] feat_cols={len(feat_cols)}, lifetimes={lifetimes}")
    print(f"[denorm] using L_ref = {L_ref_min:.0f} min (median train lifetime)")

    cache = Path(args.feat_cache); cache.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in args.run_dirs:
        run = Path(run_dir)
        cache_csv = cache / f"{run.name}_features.csv"
        if cache_csv.exists():
            print(f"[cache] using {cache_csv}")
            df = pd.read_csv(cache_csv)
        else:
            print(f"[extract] {run}")
            df = features_one_run(run, samples_per_rev=256, verbose=False)
            df.to_csv(cache_csv, index=False)

        # ensure missing columns are zero-filled (e.g. if a feat failed in some files)
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0

        x = build_last_window(df, feat_cols, train_args["window"], mean, std)
        xb = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            yhat = float(model(xb).cpu().numpy().squeeze())
        rul_min = yhat * L_ref_min
        rul_sec = rul_min * 60.0
        rows.append({"run": run.name, "rul_norm": yhat,
                     "rul_min": rul_min, "rul_sec": rul_sec})
        print(f"  pred: rul_norm={yhat:.4f}  rul_min={rul_min:.1f}  "
              f"rul_sec={rul_sec:.0f}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\n[save] {args.out}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
