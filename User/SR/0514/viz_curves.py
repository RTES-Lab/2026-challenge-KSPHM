"""HI and RUL curve visualization for train/test data."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

BASE = Path(__file__).parent
HI_TRAIN = BASE / "hi/output/train"
HI_TRAIN_B = BASE / "hi/output/train_hib"
HI_TEST = BASE / "hi/output/test_v4"
HI_TEST_B = BASE / "hi/output/test_hib_v4"
RUL_TRAIN = BASE / "rul/output/train"
RUL_ENS = BASE / "rul/output/ensemble"

COLOR_A = "#2196F3"
COLOR_B = "#FF5722"
COLOR_TRUE = "#4CAF50"
COLOR_ENS = "#9C27B0"
COLOR_LA = "#2196F3"
COLOR_LB = "#FF5722"
COLOR_LGBM = "#FF9800"


# ── Figure 1: HI Curves (Train 2×2 + Test 2×3) ────────────────────────────
fig1 = plt.figure(figsize=(20, 10))
fig1.suptitle("Health Index (HI) Curves — Train & Test  [v4: Train-Anchored Absolute Scaling]", fontsize=15, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 10, figure=fig1, hspace=0.45, wspace=0.4)

# Train: 4 bearings → columns 0–7 split into 4 pairs (2-wide each), but simpler: 4 subplots in 4 cols
train_axes = []
for i, b in enumerate(range(1, 5)):
    col = i * 2 + 1  # center 4 subplots with padding
    ax = fig1.add_subplot(gs[0, col * 10 // 10 : col * 10 // 10 + 1])
    train_axes.append((ax, b))

# Simpler: use GridSpecFromSubplotSpec for clean sections
fig1.clear()
fig1.suptitle("Health Index (HI) Curves — Train & Test", fontsize=15, fontweight="bold", y=0.99)

outer = gridspec.GridSpec(2, 1, figure=fig1, hspace=0.55, top=0.93, bottom=0.07)

# Top row: Train (4 subplots)
inner_train = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.35)
# Bottom row: Test (6 subplots)
inner_test = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[1], wspace=0.35)

# Train HI
for i, b in enumerate(range(1, 5)):
    ax = fig1.add_subplot(inner_train[i])
    hi_a = pd.read_csv(HI_TRAIN / f"Bearing{b}_best.csv")["HI"].values
    hi_b = pd.read_csv(HI_TRAIN_B / f"Bearing{b}_best.csv")["HI"].values
    x_a = np.arange(len(hi_a))
    x_b = np.arange(len(hi_b))
    ax.plot(x_a, hi_a, color=COLOR_A, lw=1.5, label="HI-A")
    ax.plot(x_b, hi_b, color=COLOR_B, lw=1.5, ls="--", alpha=0.8, label="HI-B")
    ax.set_title(f"Bearing {b}", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Cycle", fontsize=8)
    ax.set_ylabel("HI", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, loc="upper left")

# Section label for train
fig1.text(0.5, 0.965, "[ TRAIN ]", ha="center", fontsize=10, color="gray",
          transform=fig1.transFigure)

# Test HI
for i, t in enumerate(range(1, 7)):
    ax = fig1.add_subplot(inner_test[i])
    hi_a = pd.read_csv(HI_TEST / f"Test{t}_best.csv")["HI"].values
    hi_b = pd.read_csv(HI_TEST_B / f"Test{t}_best.csv")["HI"].values
    x_a = np.arange(len(hi_a))
    x_b = np.arange(len(hi_b))
    ax.plot(x_a, hi_a, color=COLOR_A, lw=1.5, label="HI-A")
    ax.plot(x_b, hi_b, color=COLOR_B, lw=1.5, ls="--", alpha=0.8, label="HI-B")
    ax.set_title(f"Test {t}", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Cycle", fontsize=8)
    ax.set_ylabel("HI", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, loc="upper right")

fig1.text(0.5, 0.49, "[ TEST ]", ha="center", fontsize=10, color="gray",
          transform=fig1.transFigure)

out1 = BASE / "viz_hi_curves.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {out1}")


# ── Figure 2: RUL Curves (Train LOOCV 2×2 + Test Ensemble 2×3) ───────────
fig2 = plt.figure(figsize=(20, 10))
fig2.suptitle("RUL Prediction Curves — Train (LOOCV) & Test (Ensemble)", fontsize=15,
              fontweight="bold", y=0.99)

outer2 = gridspec.GridSpec(2, 1, figure=fig2, hspace=0.55, top=0.93, bottom=0.07)
inner_train2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer2[0], wspace=0.35)
inner_test2 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer2[1], wspace=0.35)

# Train LOOCV RUL
rul_train_df = pd.read_csv(RUL_TRAIN / "LSTM_RUL_results.csv")
for i, b in enumerate(range(1, 5)):
    ax = fig2.add_subplot(inner_train2[i])
    df = rul_train_df[rul_train_df["test_bearing"] == b].copy()
    ax.plot(df["obs_cycle"], df["rul_true"], color=COLOR_TRUE, lw=2, label="True RUL")
    ax.plot(df["obs_cycle"], df["rul_pred"], color=COLOR_A, lw=1.8, ls="--", label="Pred RUL")
    ax.set_title(f"Bearing {b}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Obs Cycle", fontsize=8)
    ax.set_ylabel("RUL (cycles)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, loc="upper right")

fig2.text(0.5, 0.965, "[ TRAIN — LOOCV ]", ha="center", fontsize=10, color="gray",
          transform=fig2.transFigure)

# Test Ensemble RUL
for i, t in enumerate(range(1, 7)):
    ax = fig2.add_subplot(inner_test2[i])
    df = pd.read_csv(RUL_ENS / f"Test{t}_ensemble_RUL.csv")
    ax.plot(df["obs_cycle"], df["rul_pred_lstm_a"], color=COLOR_LA, lw=1.2, alpha=0.7, label="LSTM-A")
    ax.plot(df["obs_cycle"], df["rul_pred_lstm_b"], color=COLOR_LB, lw=1.2, alpha=0.7, ls="--", label="LSTM-B")
    ax.plot(df["obs_cycle"], df["rul_pred_lgbm"], color=COLOR_LGBM, lw=1.2, alpha=0.7, ls=":", label="LGBM")
    ax.plot(df["obs_cycle"], df["rul_pred_ensemble"], color=COLOR_ENS, lw=2.2, label="Ensemble")
    ax.set_title(f"Test {t}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Obs Cycle", fontsize=8)
    ax.set_ylabel("RUL (cycles)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    # final RUL annotation
    final_rul = df["rul_pred_ensemble"].iloc[-1]
    ax.annotate(f"{final_rul:.1f}c", xy=(df["obs_cycle"].iloc[-1], final_rul),
                xytext=(-5, 8), textcoords="offset points", fontsize=7,
                color=COLOR_ENS, fontweight="bold")
    if i == 0:
        ax.legend(fontsize=6, loc="upper right")

fig2.text(0.5, 0.49, "[ TEST — Ensemble ]", ha="center", fontsize=10, color="gray",
          transform=fig2.transFigure)

out2 = BASE / "viz_rul_curves.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")
