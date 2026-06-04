"""
Adaptive Ensemble — per-cycle RUL time-series plot
===================================================
feat_sim (per-cycle line) + k-NN (horizontal dashed) + Adaptive blend (per-cycle line)
Adaptive blend at each cycle: blend(feat_sim_t, knn_final, hi_at_t)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE          = Path("/data/home/ksphm/2026-challenge-KSPHM")
FEAT_SIM_DIR  = BASE / "User/SR/0605/output/rul_feat_sim"
KNN_CSV       = BASE / "User/SR/0605/output_knn/test_summary.csv"
HI_TEST_DIR   = BASE / "User/SR/0604/output/test"
OUT_DIR       = BASE / "User/SR/0605/output_adaptive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HI_THRESHOLD  = 0.05
W_SHORT       = 0.60
W_LONG        = 0.40
TEST_IDS      = [1, 2, 3, 4, 5, 6]
INTERVAL_SEC  = 600


def adaptive_blend(fs_rul, knn_rul, hi_val):
    if hi_val >= HI_THRESHOLD:
        return min(fs_rul, knn_rul)
    else:
        short = min(fs_rul, knn_rul)
        long_ = max(fs_rul, knn_rul)
        return W_LONG * long_ + W_SHORT * short


knn_df = pd.read_csv(KNN_CSV)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, tid in zip(axes, TEST_IDS):
    fs_df   = pd.read_csv(FEAT_SIM_DIR / f"Test{tid}_RUL.csv")
    hi_df   = pd.read_csv(HI_TEST_DIR  / f"Test{tid}_HI.csv")
    knn_row = knn_df[knn_df["test_id"] == tid].iloc[0]

    knn_cyc  = float(knn_row["rul_biased_cycles"])
    hi_arr   = hi_df["HI"].values.astype(float)

    cycles   = fs_df["obs_cycle"].values
    fs_cyc   = fs_df["preds_final"].values

    # per-cycle adaptive blend using current HI
    ada_cyc = np.array([
        adaptive_blend(fs_cyc[i], knn_cyc, hi_arr[c - 1])
        for i, c in enumerate(cycles)
    ])

    hi_end    = float(hi_arr[-1])
    rule      = "min" if hi_end >= HI_THRESHOLD else "60:40"
    ada_final = ada_cyc[-1]

    ax.plot(cycles, fs_cyc,  color="#4C9BE8", lw=1.5, label="feat_sim")
    ax.axhline(knn_cyc,      color="#F4A460", lw=1.5, ls="--", label=f"k-NN {knn_cyc:.1f}c")
    ax.plot(cycles, ada_cyc, color="#2E8B57", lw=2.2, label="Adaptive")

    ax.scatter([cycles[-1]], [ada_final], color="#2E8B57", s=60, zorder=5)
    ax.annotate(f"{ada_final:.1f}c",
                xy=(cycles[-1], ada_final),
                xytext=(cycles[-1] - 6, ada_final + max(fs_cyc) * 0.08),
                fontsize=9, color="#2E8B57", fontweight="bold")

    ax.set_title(f"Test{tid}  (HI={hi_end:.3f}, {rule})", fontsize=10)
    ax.set_xlabel("Obs Cycle")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle(
    "Adaptive Ensemble v1: feat_sim + k-NN\n"
    f"HI >= {HI_THRESHOLD}: min(feat_sim, k-NN)  |  HI < {HI_THRESHOLD}: 60:40 blend",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
out_path = OUT_DIR / "adaptive_test_rul_timeseries.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved: {out_path}")
