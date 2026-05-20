"""
Generate A-full specific figures for SR/0520/progress.md.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add rul/code to path so we can import from rul_th742_afull.py
sys.path.append(str(Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0520/rul/code")))
import rul_th742_afull as afull

# ── Paths ──────────────────────────────────────────────────────────────────
BASE      = Path("/data/home/ksphm/2026-challenge-KSPHM")
OUT_FIGS  = BASE / "User/SR/0520/figures"
OUT_FIGS.mkdir(exist_ok=True)

# ── Colors ──────────────────────────────────────────────────────────────────
C = {"B1": "#4C72B0", "B2": "#DD8452", "B3": "#55A868", "B4": "#C44E52",
     "T1": "#8172B2", "T2": "#937860", "T3": "#DA8BC3",
     "T4": "#8C8C8C", "T5": "#CCB974", "T6": "#64B5CD"}

# ============================================================================
# 1. Figure 7: A-full Train HI (B1~B4 recalculated under global Train-baseline)
# ============================================================================
print("Generating Fig 7 (A-full Train HIs)...")
params, sigma, feat_w, grp_w, aux_w, aux_s, aux_features = afull.load_th_params()
train_dfs  = afull.load_train_features()
train_regs = afull.load_train_regimes()

train_baseline_all     = afull.compute_train_baseline(train_dfs, train_regs, afull.MAIN_ALL_FEATS)
train_aux_baseline_all = afull.compute_train_baseline(train_dfs, train_regs, aux_features)

hi_train_all = {}
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
fig.suptitle("Exp D (A-full) — Recalculated Train HI Trajectories (Global Train-baseline)", fontsize=13, fontweight="bold")

for ax, (b, color) in zip(axes, [(1, C["B1"]), (2, C["B2"]), (3, C["B3"]), (4, C["B4"])]):
    hi_corr, _, _, _ = afull.compute_bearing_hi(
        train_dfs[b], train_regs[b], params, sigma, feat_w, grp_w,
        aux_features, aux_w, aux_s, train_baseline_all, train_aux_baseline_all)
    hi_train_all[b] = hi_corr
    
    ax.plot(hi_corr, color=color, lw=2)
    ax.fill_between(range(len(hi_corr)), 0, hi_corr, color=color, alpha=0.15)
    ax.axhline(hi_corr.max(), color=color, ls="--", lw=1.0, alpha=0.7)
    ax.text(len(hi_corr) * 0.02, hi_corr.max() + 0.015, f"max={hi_corr.max():.3f}", fontsize=9, color=color)
    ax.set_title(f"Bearing {b}  (n={len(hi_corr)})", fontsize=11)
    ax.set_xlabel("Observation index")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, 140)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Recalculated HI")
plt.tight_layout()
fig.savefig(OUT_FIGS / "fig7_train_hi_afull.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 7 done")

# ============================================================================
# 2. Figure 8: A-full LOOCV RUL Predictions (validation LOO RUL curves)
# ============================================================================
print("Copying LOOCV predictions to figures...")
import shutil
shutil.copy(
    BASE / "User/SR/0520/rul/output/th742_afull/loocv_predictions.png",
    OUT_FIGS / "fig8_loocv_predictions_afull.png"
)
print("Fig 8 done")

# ============================================================================
# 3. Figure 9: A-full Test HIs vs own HIs & Test RUL Predictions
# ============================================================================
print("Copying Test predictions to figures...")
shutil.copy(
    BASE / "User/SR/0520/rul/output/th742_afull/test_predictions.png",
    OUT_FIGS / "fig9_test_predictions_afull.png"
)
print("Fig 9 done")

print("\nAll A-full specific figures generated successfully!")
