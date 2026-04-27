"""
Sanity-check the trained model by predicting on the held-out training run
(treating each prefix of files as a 'submission point') and compare against
ground-truth lifetime.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import math
import numpy as np
import pandas as pd
import torch

from model import AdaptiveMAGNN_TCN

ROOT = Path(__file__).parent
FEAT_DIR = ROOT / "features"


def comp_score(rul_pred_sec: np.ndarray, rul_true_sec: np.ndarray) -> np.ndarray:
    """Per-sample competition score in (0,1]; matches the official spec."""
    err = 100.0 * (rul_true_sec - rul_pred_sec) / np.maximum(rul_true_sec, 1e-2)
    # over-pred (err<=0): S = exp(-ln(0.5)*err/20)  -> harsh
    # under-pred (err>0): S = exp(+ln(0.5)*err/50)  -> mild
    ln_half = math.log(0.5)
    s = np.where(err <= 0,
                 np.exp(-ln_half * err / 20.0),
                 np.exp( ln_half * err / 50.0))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--run", required=True, help="Train run name e.g. Train4")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    feat_cols = ck["feat_cols"]
    mean = np.asarray(ck["scaler_mean"], dtype=np.float32)
    std  = np.asarray(ck["scaler_std"],  dtype=np.float32)
    args_t = ck["args"]
    lifetimes = ck["lifetimes"]

    df = pd.read_csv(FEAT_DIR / f"{args.run}_features.csv")
    df = df.sort_values("idx").reset_index(drop=True)
    F = df[feat_cols].to_numpy(dtype=np.float32)
    F = (F - mean) / np.maximum(std, 1e-6)

    L_min = lifetimes[args.run] if args.run in lifetimes \
        else float(np.median(list(lifetimes.values())))
    L_ref = float(np.median(list(lifetimes.values())))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptiveMAGNN_TCN(
        n_vars=len(feat_cols), t_in=args_t["window"],
        c_hidden=args_t["c_hidden"], n_scales=args_t["n_scales"],
        top_k=args_t["top_k"], tcn_hidden=args_t["tcn_hidden"],
        tcn_blocks=args_t["tcn_blocks"]).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()

    W = args_t["window"]
    rows = []
    for i in range(len(F)):
        start = max(0, i - W + 1)
        seg = F[start:i + 1]
        if seg.shape[0] < W:
            pad = np.zeros((W - seg.shape[0], seg.shape[1]), dtype=np.float32)
            seg = np.concatenate([pad, seg], axis=0)
        x = torch.from_numpy(seg.T).unsqueeze(0).to(device)
        with torch.no_grad():
            yhat = float(model(x).cpu().numpy().squeeze())
        t = i * 10.0
        rul_true_min = max(L_min - t, 0.0)
        rul_pred_min = yhat * L_ref
        rows.append({"idx": i, "t_min": t,
                     "rul_norm_pred": yhat,
                     "rul_pred_sec": rul_pred_min * 60.0,
                     "rul_true_sec": rul_true_min * 60.0})
    out = pd.DataFrame(rows)
    out["score"] = comp_score(out["rul_pred_sec"].values,
                              out["rul_true_sec"].values)
    print(f"\n=== {args.run}: lifetime={L_min:.0f} min, L_ref={L_ref:.0f} min ===")
    print(out.tail(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  mean score across all submission points: {out['score'].mean():.3f}")
    print(f"  score at last frame (the actual submission): {out['score'].iloc[-1]:.3f}")
    out.to_csv(ROOT / f"sanity_{args.run}.csv", index=False)


if __name__ == "__main__":
    main()
