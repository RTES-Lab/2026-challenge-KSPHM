"""
Generate A-full specific figures for SR/0528/progress.md.
Requires rul_th742_afull.py to have been run first (generates loocv_predictions.png
and test_predictions.png under rul/output/afull/).
"""
import sys
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0528/rul/code")))
import rul_th742_afull as afull

BASE     = Path("/data/home/ksphm/2026-challenge-KSPHM")
OUT_FIGS = BASE / "User/SR/0528/figures"
OUT_FIGS.mkdir(exist_ok=True)
AFULL_OUT = BASE / "User/SR/0528/rul/output/afull"

C = {
    "B1": "#4C72B0", "B2": "#DD8452", "B3": "#55A868", "B4": "#C44E52",
    "T1": "#8172B2", "T2": "#937860", "T3": "#DA8BC3",
    "T4": "#8C8C8C", "T5": "#CCB974", "T6": "#64B5CD",
}

# ── Fig 1: Train HI trajectories (global Train-baseline) ────────────────────
print("Generating Fig 1 (Train HI, global Train-baseline)...")
params, sigma, feat_w, grp_w, aux_w, aux_s, aux_features = afull.load_th_params()
train_dfs  = afull.load_train_features()
train_regs = afull.load_train_regimes()

train_baseline_all     = afull.compute_train_baseline(train_dfs, train_regs, afull.MAIN_ALL_FEATS)
train_aux_baseline_all = afull.compute_train_baseline(train_dfs, train_regs, aux_features)

hi_train_all = {}
fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
fig.suptitle(
    "A-full — Train HI Trajectories (Global Train-baseline)",
    fontsize=13, fontweight="bold"
)
for ax, (b, color) in zip(axes, [(1, C["B1"]), (2, C["B2"]), (3, C["B3"]), (4, C["B4"])]):
    hi_corr, _, _, _ = afull.compute_bearing_hi(
        train_dfs[b], train_regs[b], params, sigma, feat_w, grp_w,
        aux_features, aux_w, aux_s, train_baseline_all, train_aux_baseline_all)
    hi_train_all[b] = hi_corr

    ax.plot(hi_corr, color=color, lw=2)
    ax.fill_between(range(len(hi_corr)), 0, hi_corr, color=color, alpha=0.15)
    ax.axhline(hi_corr.max(), color=color, ls="--", lw=1.0, alpha=0.7)
    ax.text(len(hi_corr) * 0.02, hi_corr.max() + 0.015,
            f"max={hi_corr.max():.3f}", fontsize=9, color=color)
    ax.set_title(f"Bearing {b}  (n={len(hi_corr)})", fontsize=11)
    ax.set_xlabel("Observation index")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, 145)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Recalculated HI (A-full)")
plt.tight_layout()
fig.savefig(OUT_FIGS / "fig1_train_hi_afull.png", dpi=150, bbox_inches="tight")
plt.close()
print("  done → fig1_train_hi_afull.png")

# ── Fig 2: LOOCV RUL predictions (copied from pipeline output) ──────────────
print("Generating Fig 2 (LOOCV RUL predictions)...")
shutil.copy(AFULL_OUT / "loocv_predictions.png", OUT_FIGS / "fig2_loocv_rul_afull.png")
print("  done → fig2_loocv_rul_afull.png")

# ── Fig 3: Test HI + Test RUL predictions (copied from pipeline output) ─────
print("Generating Fig 3 (Test HI & RUL predictions)...")
shutil.copy(AFULL_OUT / "test_predictions.png", OUT_FIGS / "fig3_test_predictions_afull.png")
print("  done → fig3_test_predictions_afull.png")

print("\nAll figures generated successfully.")
print(f"Output: {OUT_FIGS}")
