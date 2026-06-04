"""Plot all 7 features over time for all 4 train bearings.
Each feature gets one row; each bearing a different color.
Low-speed regime points shown as circles, high-speed as triangles.
Also shows per-bearing min-max normalized version in a second figure.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

BASE      = "/data/home/ksphm/2026-challenge-KSPHM/User/SR/0603_v3"
OUT_TRAIN = os.path.join(BASE, "output/train")

BEARINGS = [1, 2, 3, 4]
INTERVAL_SEC = 600
ALL_FEATS = [
    "ch3_high_band", "ch4_high_band",
    "ch3_total_power", "ch3_energy", "ch3_rms",
    "ch3_std", "ch3_p2p",
]
FEAT_LABELS = {
    "ch3_high_band":   "CH3 High-band ratio\n(3–8 kHz / total PSD)",
    "ch4_high_band":   "CH4 High-band ratio\n(3–8 kHz / total PSD)",
    "ch3_total_power": "CH3 Total Power\n(Welch PSD sum)",
    "ch3_energy":      "CH3 Energy\n(∑x²)",
    "ch3_rms":         "CH3 RMS",
    "ch3_std":         "CH3 Std Dev",
    "ch3_p2p":         "CH3 Peak-to-Peak",
}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Load data
dfs    = {}
hi_dfs = {}
for bid in BEARINGS:
    dfs[bid]    = pd.read_csv(os.path.join(OUT_TRAIN, f"Bearing{bid}_features_raw.csv"))
    hi_dfs[bid] = pd.read_csv(os.path.join(OUT_TRAIN, f"Bearing{bid}_HI.csv"))


def make_figure(normalize: bool, outname: str):
    fig, axes = plt.subplots(len(ALL_FEATS), 1, figsize=(15, 3.5 * len(ALL_FEATS)),
                              sharex=False)
    title_suffix = "(min-max normalized per bearing)" if normalize else "(raw values)"
    fig.suptitle(f"All Features over Time — Train B1–B4  {title_suffix}",
                 fontsize=13, fontweight="bold")

    for fi, feat in enumerate(ALL_FEATS):
        ax = axes[fi]
        for bi, bid in enumerate(BEARINGS):
            df    = dfs[bid]
            hi_df = hi_dfs[bid]
            n     = len(df)
            t     = np.arange(n) * INTERVAL_SEC / 3600
            y     = df[feat].values.astype(float)

            if normalize:
                lo, hi_v = y.min(), y.max()
                y = (y - lo) / (hi_v - lo + 1e-12)

            regime = hi_df["regime"].values[:n]
            col    = COLORS[bi]

            # line
            ax.plot(t, y, color=col, lw=0.9, alpha=0.45)

            # scatter by regime
            ax.scatter(t[regime == 0], y[regime == 0],
                       s=6, color=col, marker="o", alpha=0.7, zorder=3,
                       label=f"B{bid} low" if fi == 0 else None)
            ax.scatter(t[regime == 1], y[regime == 1],
                       s=6, color=col, marker="^", alpha=0.7, zorder=3,
                       label=f"B{bid} high" if fi == 0 else None)

        ax.set_ylabel(FEAT_LABELS[feat], fontsize=9)
        ax.set_xlabel("Time [hr]", fontsize=9)
        ax.grid(True, ls="--", alpha=0.3)
        ax.tick_params(labelsize=8)

        # bearing label at end of each line
        for bi, bid in enumerate(BEARINGS):
            df = dfs[bid]
            n  = len(df)
            t_end = (n - 1) * INTERVAL_SEC / 3600
            y_end = df[feat].values[-1]
            if normalize:
                lo, hi_v = df[feat].values.min(), df[feat].values.max()
                y_end = (y_end - lo) / (hi_v - lo + 1e-12)
            ax.annotate(f"B{bid}", xy=(t_end, y_end),
                        fontsize=7, color=COLORS[bi],
                        xytext=(3, 0), textcoords="offset points", va="center")

    # legend in first subplot
    handles = []
    for bi, bid in enumerate(BEARINGS):
        handles.append(plt.Line2D([0], [0], color=COLORS[bi], lw=1.5, label=f"Bearing {bid}"))
    handles += [
        plt.scatter([], [], s=10, color="gray", marker="o", label="Low speed (<850 rpm)"),
        plt.scatter([], [], s=10, color="gray", marker="^", label="High speed (≥850 rpm)"),
    ]
    axes[0].legend(handles=handles, fontsize=8, ncol=6,
                   loc="upper left", framealpha=0.8)

    plt.tight_layout()
    outpath = os.path.join(OUT_TRAIN, outname)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


make_figure(normalize=False, outname="all_features_raw.png")
make_figure(normalize=True,  outname="all_features_normalized.png")
