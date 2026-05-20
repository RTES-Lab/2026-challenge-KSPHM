"""Generate figures for SR/0518/progress.md."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0518/figures")
OUT.mkdir(exist_ok=True)

TH_HI_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM/User/TH/FI/07_v7/output/v7_4_2_conditional_aux_boost")
SR_TRAIN_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0514/hi/output/train")
SR_TEST_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0514/hi/output/test")

# ── colors ──────────────────────────────────────────────────────────────────
C = {"B1": "#4C72B0", "B2": "#DD8452", "B3": "#55A868", "B4": "#C44E52",
     "T1": "#8172B2", "T2": "#937860", "T3": "#DA8BC3",
     "T4": "#8C8C8C", "T5": "#CCB974", "T6": "#64B5CD"}

LGBM_C = "#E64B35"
LSTM_C  = "#4DBBD5"
ENS_C   = "#00A087"
BASE_C  = "#7E6148"

# ============================================================
# Fig 1: TH Train HI trajectories (4 bearings)
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
fig.suptitle("TH v7_4_2 — Train Bearing HI Trajectories", fontsize=13, fontweight="bold")

for ax, (i, color) in zip(axes, [(1, C["B1"]), (2, C["B2"]), (3, C["B3"]), (4, C["B4"])]):
    df = pd.read_csv(TH_HI_DIR / f"v7_4_2_Bearing{i}_HI.csv")
    hi = df["HI_v7_4_2"].values
    ax.plot(hi, color=color, lw=2)
    ax.fill_between(range(len(hi)), 0, hi, color=color, alpha=0.15)
    ax.axhline(hi.max(), color=color, ls="--", lw=1.0, alpha=0.7)
    ax.text(len(hi) * 0.02, hi.max() + 0.015, f"max={hi.max():.3f}", fontsize=9, color=color)
    ax.set_title(f"Bearing {i}  (n={len(hi)})", fontsize=11)
    ax.set_xlabel("Observation index")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, 140)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Health Index (HI)")

# Highlight B3 low max
axes[2].add_patch(mpatches.FancyArrowPatch(
    (89 * 0.6, 0.62), (89 * 0.6, 0.59),
    arrowstyle="-|>", color="red", mutation_scale=12))
axes[2].text(5, 0.64, "Low max\n(scale issue)", fontsize=8, color="red")

plt.tight_layout()
fig.savefig(OUT / "fig1_train_hi.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig1 done")

# ============================================================
# Fig 2: TH Test HI trajectories (6 tests)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharey=True)
fig.suptitle("TH v7_4_2 — Test HI Trajectories  (50 obs each, RUL unknown)", fontsize=13, fontweight="bold")

test_rul_v4 = [7.40, 8.58, 5.21, 9.78, 8.31, 6.79]  # Exp C-1: start_obs estimated
test_rul_base = [5.05, 5.08, 4.69, 3.43, 7.46, 5.40]

for idx, (ax, (i, tcolor)) in enumerate(zip(axes.flat,
        [(1, C["T1"]), (2, C["T2"]), (3, C["T3"]),
         (4, C["T4"]), (5, C["T5"]), (6, C["T6"])])):
    df = pd.read_csv(TH_HI_DIR / f"v7_4_2_Test{i}_HI.csv")
    hi = df["HI_v7_4_2"].values
    ax.plot(hi, color=tcolor, lw=2.5)
    ax.fill_between(range(len(hi)), 0, hi, color=tcolor, alpha=0.12)
    h_start = hi[0]; h_end = hi[-1]
    ax.set_title(f"Test {i}   hi: {h_start:.3f} → {h_end:.3f}", fontsize=10.5)
    ax.set_xlabel("Observation")
    ax.set_ylim(-0.02, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # annotate predicted RUL
    ax.text(2, 0.88, f"RUL(v4,start-est)={test_rul_v4[idx]:.1f}h", fontsize=8.5,
            color=ENS_C, fontweight="bold")
    ax.text(2, 0.78, f"RUL(0514)={test_rul_base[idx]:.1f}h", fontsize=8.5,
            color=BASE_C)

axes[0, 0].set_ylabel("Health Index (HI)")
axes[1, 0].set_ylabel("Health Index (HI)")
plt.tight_layout()
fig.savefig(OUT / "fig2_test_hi.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig2 done")

# ============================================================
# Fig 3: LOOCV score progression (all experiments)
# ============================================================
exps = ["SR 0514\n(baseline)", "Exp A\n(TH HI\n+LGBM+LSTM)", "Exp B\n(+window\nnorm)", "Exp C\n(+obs\nfraction)", "Exp C-1\n(+start_obs\nestimate)"]
lgbm_scores   = [np.nan, 0.3625, 0.3345, 0.4102, 0.4102]
lstm_scores   = [np.nan, 0.4261, 0.4261, 0.4261, 0.4261]
ens_scores    = [0.4326,  0.4529, 0.4551, 0.5004, 0.5004]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(exps))

ax.bar(x[1:], lgbm_scores[1:], width=0.22, label="LGBM", color=LGBM_C, alpha=0.8, align="center")
ax.bar(x[1:] + 0.24, lstm_scores[1:], width=0.22, label="LSTM-A", color=LSTM_C, alpha=0.8)

for i, v in enumerate(ens_scores):
    ax.scatter(x[i], v, color=ENS_C, s=120, zorder=5, edgecolors="white", linewidths=1.5)
    ax.text(x[i], v + 0.008, f"{v:.4f}", ha="center", va="bottom", fontsize=9,
            color=ENS_C, fontweight="bold")

ax.plot(x, ens_scores, color=ENS_C, lw=2.5, marker="o", markersize=9,
        markerfacecolor="white", markeredgecolor=ENS_C, markeredgewidth=2,
        label="Ensemble (final)", zorder=4)

# Baseline horizontal line
ax.axhline(0.4326, color=BASE_C, ls="--", lw=1.5, alpha=0.8)
ax.text(4.05, 0.434, "SR 0514 baseline\n0.4326", fontsize=8.5, color=BASE_C, va="bottom")

ax.set_xticks(x)
ax.set_xticklabels(exps, fontsize=10)
ax.set_ylabel("LOOCV Score", fontsize=11)
ax.set_title("LOOCV Score Progression Across Experiments", fontsize=13, fontweight="bold")
ax.set_ylim(0.0, 0.60)
ax.legend(fontsize=9, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate improvement arrow
ax.annotate("", xy=(3, 0.5004), xytext=(0, 0.4326),
            arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.5, linestyle="dashed"))
ax.text(1.5, 0.48, "+15.7%", fontsize=10, color="gray", rotation=10)
# C-1 note: LOOCV same, Test bias improved
ax.annotate("Same LOOCV\n(Test bias reduced)", xy=(4, 0.5004), xytext=(3.3, 0.545),
            fontsize=8, color="#666666",
            arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0))

plt.tight_layout()
fig.savefig(OUT / "fig3_loocv_progression.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig3 done")

# ============================================================
# Fig 4: Per-fold LGBM vs Ensemble score — v1 vs v3
# ============================================================
bearings = ["B1", "B2", "B3", "B4"]
lgbm_v1 = [0.4345, 0.4313, 0.1344, 0.4498]
lgbm_v3 = [0.6071, 0.5618, 0.0705, 0.4015]
ens_v1  = [0.3639, 0.5415, 0.5054, 0.4009]
ens_v3  = [0.5246, 0.6405, 0.4666, 0.3696]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.suptitle("Per-Fold Score: Exp A (v1) vs Exp C (v3)", fontsize=13, fontweight="bold")

x = np.arange(4)
w = 0.35
colors = [C["B1"], C["B2"], "#FF4444", C["B4"]]  # B3 red to highlight

for ax, (v1, v3, title) in zip(axes, [
        (lgbm_v1, lgbm_v3, "LGBM Fold Score"),
        (ens_v1,  ens_v3,  "Ensemble Fold Score")]):
    bars1 = ax.bar(x - w/2, v1, w, label="v1 (baseline TH HI)", color=[c + "99" for c in ["#4C72B0","#DD8452","#FF4444","#C44E52"]], edgecolor="white")
    bars2 = ax.bar(x + w/2, v3, w, label="v3 (+obs_fraction)", color=colors, edgecolor="white")

    for bar, val in zip(bars1, v1):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)
    for bar, val in zip(bars2, v3):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(bearings, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 0.80)
    ax.axhline(0, color="black", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)

axes[0].set_ylabel("LOOCV Score", fontsize=11)

# Mark B3 collapse
for ax in axes:
    ax.annotate("B3\ncollapse!", xy=(2 + 0.17, 0.0705), xytext=(2.6, 0.18),
                fontsize=9, color="red",
                arrowprops=dict(arrowstyle="-|>", color="red", lw=1.2))

plt.tight_layout()
fig.savefig(OUT / "fig4_fold_scores.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig4 done")

# ============================================================
# Fig 5: Test RUL predictions across all versions
# ============================================================
tests = [f"Test{i}" for i in range(1, 7)]
rul_0514 = [5.05, 5.08, 4.69, 3.43, 7.46, 5.40]
rul_v1   = [8.13, 10.89, 7.44, 6.52, 9.94, 8.61]
rul_v2   = [8.58, 10.79, 5.60, 5.06, 7.51, 5.61]
rul_v3   = [9.32, 10.04, 8.12, 10.43, 9.33, 9.42]   # Exp C (start=0)
rul_v4   = [7.40,  8.58, 5.21,  9.78, 8.31, 6.79]   # Exp C-1 (start estimated)

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(6)
w = 0.15

ax.bar(x - 2*w, rul_0514, w, label="SR 0514 (baseline)", color=BASE_C, alpha=0.85)
ax.bar(x - 1*w, rul_v1,   w, label="Exp A: TH HI+LGBM+LSTM", color="#4472C4", alpha=0.85)
ax.bar(x + 0*w, rul_v2,   w, label="Exp B: +window-norm", color="#ED7D31", alpha=0.85)
ax.bar(x + 1*w, rul_v3,   w, label="Exp C: +obs_frac (start=0)", color=ENS_C, alpha=0.85)
ax.bar(x + 2*w, rul_v4,   w, label="Exp C-1: +start_obs estimate", color="#9B59B6", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(tests, fontsize=11)
ax.set_ylabel("Predicted RUL (hours)", fontsize=11)
ax.set_title("Test RUL Predictions: Version Comparison", fontsize=13, fontweight="bold")
ax.set_ylim(0, 13)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate Test4 v3 spike
ax.annotate("Test4 spike!\n(obs_frac bias)", xy=(3 + 1*w, 10.43), xytext=(4.1, 11.5),
            fontsize=9, color="red",
            arrowprops=dict(arrowstyle="-|>", color="red", lw=1.2))
# Annotate Test4 v4 improvement
ax.annotate("v4: improved\n(9.78)", xy=(3 + 2*w, 9.78), xytext=(4.5, 10.8),
            fontsize=8.5, color="#9B59B6",
            arrowprops=dict(arrowstyle="->", color="#9B59B6", lw=1.0))

# Add hi_end as text
hi_ends = [0.436, 0.108, 0.336, 0.571, 0.529, 0.447]
for i, h in enumerate(hi_ends):
    ax.text(i, -0.7, f"hi_end\n{h:.3f}", ha="center", fontsize=7.5, color="gray")

plt.tight_layout()
fig.savefig(OUT / "fig5_test_rul.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig5 done")

# ============================================================
# Fig 6: obs_fraction dilemma — LOOCV vs Test
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("obs_fraction: Effective in LOOCV but Biased on Test", fontsize=13, fontweight="bold")

# Left: LOOCV — B3 lifecycle (89 cycles)
ax = axes[0]
ax.set_title("LOOCV (Train) — obs_fraction is CORRECT", fontsize=11, color="green")
mean_life = 126  # B1/B2/B4 average

# Show B3 full lifecycle
cycles_b3 = np.arange(89)
obs_frac_b3 = cycles_b3 / mean_life
ax.plot(cycles_b3, obs_frac_b3, color=C["B3"], lw=2.5, label="B3 obs_frac (actual)")
ax.axhline(1.0, color="gray", ls=":", lw=1)
ax.axvline(89, color=C["B3"], ls="--", lw=1.5, alpha=0.6, label="B3 end-of-life")

# Show mean life reference
ax.axhline(89/mean_life, color=C["B3"], ls=":", lw=1.2, alpha=0.5)
ax.text(91, 89/mean_life - 0.03, f"B3 max\n={89/mean_life:.2f}", fontsize=8, color=C["B3"])

ax.fill_between(cycles_b3, obs_frac_b3, alpha=0.15, color=C["B3"])
ax.set_xlabel("Observation cycle")
ax.set_ylabel("obs_fraction = cycle / mean_train_life (126)")
ax.set_xlim(0, 145)
ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=9)
ax.text(5, 0.95, "LGBM sees full lifecycle\n=> learns correct scale", fontsize=9,
        color="green", bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: Test — we don't know where the 50 obs start
ax = axes[1]
ax.set_title("Test — obs_fraction is WRONG (start unknown)", fontsize=11, color="red")

# Hypothetical: Test bearing, true start might be at cycle 40 of a 90-cycle life
true_start = 40
true_obs = np.arange(50) + true_start  # cycles 40~89
obs_frac_true = true_obs / mean_life

# What LGBM assumes (start=0)
assumed_obs = np.arange(50)
obs_frac_assumed = assumed_obs / mean_life

ax.plot(range(50), obs_frac_assumed, color=LGBM_C, lw=2.5, ls="--",
        label="LGBM assumes: start=0")
ax.plot(range(50), obs_frac_true, color="green", lw=2.5,
        label="True: starts at cycle 40")

ax.fill_between(range(50), obs_frac_assumed, obs_frac_true, alpha=0.2, color="red")
ax.text(20, 0.28, "Gap = bias\n(RUL over-estimate)", fontsize=9, color="red",
        ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee"))

ax.set_xlabel("Observation index (in test window)")
ax.set_ylabel("obs_fraction")
ax.legend(fontsize=9)
ax.set_ylim(-0.05, 0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "fig6_obs_frac_dilemma.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig6 done")

print("\nAll figures saved to:", OUT)
