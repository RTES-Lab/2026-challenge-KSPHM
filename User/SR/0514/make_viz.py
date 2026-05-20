"""0514 additional visualizations — pipeline overview, asymmetric loss, normalization bug comparison"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

BASE = "/data/home/ksphm/2026-challenge-KSPHM/User/SR/0514"

# ──────────────────────────────────────────────────────────
# Fig A: Full pipeline overview — large fonts, no whitespace waste
# ──────────────────────────────────────────────────────────
# Key insight: keep coordinate space SMALL (0-10 x 0-10) so matplotlib's
# point-based font sizes map to a large fraction of the canvas.
# figsize=20x12 @ dpi=120  →  2400x1440 px.  fontsize=20 → ~33px clearly readable.
fig_a, ax = plt.subplots(figsize=(20, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
fig_a.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")
ax.patch.set_visible(False)
ax.set_title("0514 Full Pipeline Overview",
             fontsize=28, fontweight="bold", pad=14, color="#111")


def box(ax, x, y, w, h, text, fc, ec, fs=18, bold=False):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.06",
                       facecolor=fc, edgecolor=ec, linewidth=3, zorder=3)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=4,
            multialignment="center", linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color="#444"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=3, mutation_scale=24), zorder=5)


# ── Layer 1: Input  (y=9.1) ────────────────────────────────
# 3 boxes w=2.8 h=1.1 | centers x=1.5, 4.5, 7.5 | gaps=0.6 ✓
L1Y = 9.1
box(ax, 1.5, L1Y, 2.8, 1.1,
    "Raw TDMS\n(Train / Test)",
    "#E3F2FD", "#1565C0", 18, True)
box(ax, 4.5, L1Y, 2.8, 1.1,
    "Feature Extraction\n(RMS, Kurtosis,\nHigh-Band, P2P)",
    "#E8F5E9", "#2E7D32", 16)
box(ax, 7.5, L1Y, 2.8, 1.1,
    "SSM Regime Label\nFFT RPM → Low/High\n(850 RPM threshold)",
    "#FFF8E1", "#F57F17", 16)

arrow(ax, 2.9, L1Y, 3.1, L1Y)
arrow(ax, 5.9, L1Y, 6.1, L1Y)

# ── HI section divider ─────────────────────────────────────
ax.axhline(8.05, xmin=0.0, xmax=1.0, color="#1A237E", lw=2.5, ls="--", alpha=0.5)
ax.text(5.0, 8.27, "HI Generation  (FDR · LOO External Baseline)",
        ha="center", fontsize=20, fontweight="bold", color="#1A237E")

# ── Layer 2: HI boxes  (y=6.75) ────────────────────────────
# 2 boxes w=4.5 h=1.45 | centers x=2.4, 7.6 | gap=0.7 ✓
L2Y = 6.75
box(ax, 2.4, L2Y, 4.5, 1.45,
    "HI-A  (Frequency Domain)\nhighfreq / energy / variation\nFDR → Weighted Sum → EMA",
    "#E3F2FD", "#1565C0", 17)
box(ax, 7.6, L2Y, 4.5, 1.45,
    "HI-B  (Time Domain)\nimpulse / amplitude\n(Kurt, Crest, RMS, P2P)",
    "#FCE4EC", "#880E4F", 17)

# SSM bottom (7.5, 8.55) → HI-A top (2.4, 7.47)  and HI-B top (7.6, 7.47)
arrow(ax, 7.5, 8.55, 2.4, 7.47)
arrow(ax, 7.5, 8.55, 7.6, 7.47)

# ── RUL section divider ────────────────────────────────────
ax.axhline(5.6, xmin=0.0, xmax=1.0, color="#4A148C", lw=2.5, ls="--", alpha=0.5)
ax.text(5.0, 5.82, "RUL Prediction  (3-Model Ensemble)",
        ha="center", fontsize=20, fontweight="bold", color="#4A148C")

# ── Layer 3: RUL models  (y=4.3) ───────────────────────────
# 3 boxes w=2.9 h=1.35 | centers x=1.6, 5.0, 8.4 | gaps=0.45 ✓
L3Y = 4.3
box(ax, 1.6, L3Y, 2.9, 1.35,
    "LSTM-A\n(HI-A)  5 seeds\nAsymLoss",
    "#EDE7F6", "#4A148C", 17)
box(ax, 5.0, L3Y, 2.9, 1.35,
    "LSTM-B\n(HI-B)  5 seeds\nAsymLoss",
    "#FCE4EC", "#880E4F", 17)
box(ax, 8.4, L3Y, 2.9, 1.35,
    "LightGBM\n(HI-A flat)\nAsym custom obj",
    "#E8F5E9", "#1B5E20", 17)

# HI-A bottom (2.4, 6.03) → LSTM-A top (1.6, 4.97)
arrow(ax, 2.4, 6.03, 1.6, 4.97)
# HI-A bottom → LSTM-B top (5.0, 4.97)
arrow(ax, 2.4, 6.03, 5.0, 4.97)
# HI-B bottom (7.6, 6.03) → LSTM-B top
arrow(ax, 7.6, 6.03, 5.0, 4.97)
# HI-B bottom → LGBM top (8.4, 4.97)
arrow(ax, 7.6, 6.03, 8.4, 4.97)

# ── Layer 4: Ensemble  (y=2.55) ────────────────────────────
# 1 wide box w=9.4 h=1.15 centered at x=5.0
L4Y = 2.55
box(ax, 5.0, L4Y, 9.5, 1.15,
    "Ensemble + Calibration\n"
    "LOOCV score-based weights  [0.1 – 0.5]    "
    "Calib factor grid search  0.7–1.0  →  best = 1.00",
    "#FFF3E0", "#E65100", 17)

arrow(ax, 1.6, 3.62, 2.5, 3.12)   # LSTM-A → Ensemble
arrow(ax, 5.0, 3.62, 5.0, 3.12)   # LSTM-B → Ensemble
arrow(ax, 8.4, 3.62, 7.5, 3.12)   # LGBM   → Ensemble

plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.0)
_out_path = f"{BASE}/viz_pipeline_overview.png"
plt.savefig(_out_path, dpi=120, bbox_inches="tight", pad_inches=0, facecolor="#F8F9FA")
plt.close()

# crop bottom whitespace: detect rows that are all background colour
from PIL import Image
_img = Image.open(_out_path).convert("RGB")
_arr = np.array(_img)
_bg = np.array([248, 249, 250])  # #F8F9FA
_non_bg_rows = np.where(~np.all(_arr == _bg, axis=(1, 2)))[0]
if len(_non_bg_rows):
    _bottom = _non_bg_rows[-1] + 1
    _img = _img.crop((0, 0, _img.width, _bottom))
    _img.save(_out_path)
print("Saved: viz_pipeline_overview.png")


# ── Separate Key Changes figure ─────────────────────────────────────
# Stands alone so each card gets big readable text
fig_kc, ax_kc = plt.subplots(figsize=(14, 10))
ax_kc.set_xlim(0, 10)
ax_kc.set_ylim(0, 10)
ax_kc.axis("off")
fig_kc.patch.set_facecolor("#FFFDE7")
ax_kc.set_facecolor("#FFFDE7")
ax_kc.set_title("0514 Key Changes", fontsize=30, fontweight="bold",
                pad=14, color="#333")

changes = [
    ("Bug Fix 1",
     "z-score + FDR double-norm removed  →  ratio 1e+8 explosion fixed",
     "#FFCDD2", "#C62828"),
    ("Bug Fix 2",
     "Train baseline source unified  →  old PCA  to  signal_transform_v2",
     "#C8E6C9", "#2E7D32"),
    ("Bug Fix 3",
     "Test baseline contamination fixed  →  self first-10%  to  Train external baseline",
     "#BBDEFB", "#1565C0"),
    ("Loss Function",
     "MSE  →  AsymmetricRULLoss   (heavier penalty on over-prediction)",
     "#E1BEE7", "#6A1B9A"),
    ("Ensemble",
     "LSTM-A + LSTM-B + LightGBM   5-seed median + weighted ensemble",
     "#FFE0B2", "#E65100"),
]
for k, (title, desc, fc, ec) in enumerate(changes):
    yy = 9.1 - k * 1.75
    bk = FancyBboxPatch((0.15, yy - 0.75), 9.7, 1.5,
                        boxstyle="round,pad=0.08",
                        facecolor=fc, edgecolor=ec, linewidth=3.5, zorder=3)
    ax_kc.add_patch(bk)
    ax_kc.text(0.5, yy + 0.28, title, fontsize=22, fontweight="bold",
               va="center", zorder=4, color="#111")
    ax_kc.text(0.5, yy - 0.26, desc, fontsize=18,
               va="center", zorder=4, color="#333", linespacing=1.4)

plt.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.02)
plt.savefig(f"{BASE}/viz_key_changes.png", dpi=120, bbox_inches="tight",
            facecolor="#FFFDE7")
plt.close()
print("Saved: viz_key_changes.png")


# ──────────────────────────────────────────────────────────
# Fig B: Asymmetric loss vs MSE comparison
# ──────────────────────────────────────────────────────────
fig_b, axes_b = plt.subplots(1, 2, figsize=(14, 6))
fig_b.suptitle("Loss Function Comparison: MSE vs AsymmetricRULLoss", fontsize=15, fontweight="bold")
fig_b.patch.set_facecolor("#F8F9FA")

Er = np.linspace(-150, 150, 600)

# Competition scoring function (higher score = better)
score_over  = np.exp(-np.log(0.5) * Er / 20.0)   # Er <= 0 (over-prediction)
score_under = np.exp( np.log(0.5) * Er / 50.0)   # Er > 0  (under-prediction)
asym_score  = np.where(Er <= 0, score_over, score_under)

# Loss = -Score (lower is better)
asym_loss   = -asym_score
mse_loss    = (Er / 100.0) ** 2 * 3 - 1   # scaled for visual alignment

ax_l, ax_s = axes_b

# ── Panel 1: Loss function curves ──────────────────────────────────
ax_l.plot(Er, mse_loss, "b--", lw=2.5, label="MSE (∝ Er²)", alpha=0.7)
ax_l.plot(Er, asym_loss, "r-", lw=2.5, label="AsymmetricRULLoss (-Score)")

ax_l.axvline(0, color="black", lw=1, ls=":")
ax_l.fill_betweenx(np.linspace(-1.1, 2.5, 100),
                    -150, 0, alpha=0.06, color="red", label="Over-prediction zone (Er≤0)")
ax_l.fill_betweenx(np.linspace(-1.1, 2.5, 100),
                    0, 150, alpha=0.06, color="blue", label="Under-prediction zone (Er>0)")

ax_l.annotate("slope: Er/20\n(steep penalty)", xy=(-80, asym_loss[200]),
              xytext=(-140, 0.5), fontsize=9,
              arrowprops=dict(arrowstyle="->", color="red"),
              color="red", fontweight="bold")
ax_l.annotate("slope: Er/50\n(mild penalty)", xy=(80, asym_loss[440]),
              xytext=(30, 1.8), fontsize=9,
              arrowprops=dict(arrowstyle="->", color="navy"),
              color="navy", fontweight="bold")

ax_l.set_xlim(-150, 150)
ax_l.set_ylim(-1.1, 2.5)
ax_l.set_xlabel("Er = 100×(RUL_true − RUL_pred)/RUL_true (%)", fontsize=10)
ax_l.set_ylabel("Loss (lower is better)", fontsize=10)
ax_l.set_title("Loss Function Curves", fontsize=12, fontweight="bold")
ax_l.legend(fontsize=9)
ax_l.text(0.02, 0.97, "Er<0: Over-prediction\n(predicted longer than actual → missed early PM)",
          transform=ax_l.transAxes, fontsize=8.5, color="red",
          va="top", bbox=dict(boxstyle="round", fc="#FFEBEE", alpha=0.8))
ax_l.text(0.58, 0.97, "Er>0: Under-prediction\n(predicted shorter than actual → unnecessary swap)",
          transform=ax_l.transAxes, fontsize=8.5, color="navy",
          va="top", bbox=dict(boxstyle="round", fc="#E3F2FD", alpha=0.8))

# ── Panel 2: Competition score function ─────────────────────
ax_s.plot(Er, asym_score, "r-", lw=3, label="Competition Score function")
ax_s.fill_between(Er[Er <= 0], 0, asym_score[Er <= 0],
                   alpha=0.2, color="red", label="Over-prediction zone")
ax_s.fill_between(Er[Er >= 0], 0, asym_score[Er >= 0],
                   alpha=0.2, color="blue", label="Under-prediction zone")
ax_s.axvline(0, color="black", lw=1.5, ls="--", label="Er=0 (perfect prediction)")
ax_s.axhline(1.0, color="gray", lw=1, ls=":", alpha=0.7)
ax_s.axhline(0.5, color="orange", lw=1.5, ls="--", alpha=0.8, label="Score=0.5")
ax_s.scatter([-20], [np.exp(-np.log(0.5)*(-20)/20)], s=120, zorder=5, color="red")
ax_s.annotate("Er=-20 → Score=2.0\n(20% over-pred = 2x penalty)",
              xy=(-20, np.exp(-np.log(0.5)*(-20)/20)),
              xytext=(-140, 2.5), fontsize=9,
              arrowprops=dict(arrowstyle="->", color="red"), color="red")
ax_s.scatter([50], [np.exp(np.log(0.5)*50/50)], s=120, zorder=5, color="navy")
ax_s.annotate("Er=+50 → Score=0.5\n(50% under-pred = half score)",
              xy=(50, np.exp(np.log(0.5)*50/50)),
              xytext=(55, 1.5), fontsize=9,
              arrowprops=dict(arrowstyle="->", color="navy"), color="navy")

ax_s.set_xlim(-150, 150)
ax_s.set_ylim(0, 4)
ax_s.set_xlabel("Er = 100×(RUL_true − RUL_pred)/RUL_true (%)", fontsize=10)
ax_s.set_ylabel("Competition Score", fontsize=10)
ax_s.set_title("Competition Scoring Function (higher = better)\nAsymmetric: over-prediction penalty > under-prediction penalty",
               fontsize=12, fontweight="bold")
ax_s.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{BASE}/viz_asymmetric_loss.png", dpi=150, bbox_inches="tight",
            facecolor="#F8F9FA")
plt.close()
print("Saved: viz_asymmetric_loss.png")


# ──────────────────────────────────────────────────────────
# Fig C: Double-normalization bug (Before/After)
# ──────────────────────────────────────────────────────────
fig_c, axes_c = plt.subplots(2, 3, figsize=(18, 10))
fig_c.suptitle("HI Normalization Bug Fix: Before (double-norm) vs After (FDR-only)",
               fontsize=15, fontweight="bold")
fig_c.patch.set_facecolor("#F8F9FA")

np.random.seed(42)
N = 100
t = np.arange(N)

raw_feat = 0.5 + 0.003 * t + 0.05 * np.random.randn(N)
raw_feat[60:] += 0.2 * (t[60:] - 60) / 40

mu_true = raw_feat[:10].mean()

z_scored = (raw_feat - raw_feat.mean()) / (raw_feat.std() + 1e-8)
mu_after_zscore = z_scored[:10].mean()
# Simulate actual bug: baseline collapses to ~0 after z-score → near-zero denominator → explosion
mu_bugged_fdr = 0.01
ratio_bugged = (z_scored - mu_bugged_fdr) / (abs(mu_bugged_fdr) + 1e-8)

mu_train = raw_feat[:10].mean()
ratio_fixed = (raw_feat - mu_train) / (abs(mu_train) + 1e-8)


def safe_minmax(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)


hi_bugged = safe_minmax(np.clip(ratio_bugged, -50, 50))
hi_fixed  = safe_minmax(ratio_fixed)

axes_c[0, 0].plot(t, raw_feat, "b-", lw=2)
axes_c[0, 0].axvspan(0, 10, alpha=0.2, color="green", label="Train baseline zone")
axes_c[0, 0].axhline(mu_true, color="green", ls="--", lw=1.5, label=f"mu_train={mu_true:.3f}")
axes_c[0, 0].set_title("(1) Raw Feature (ch3_rms)", fontsize=11, fontweight="bold")
axes_c[0, 0].set_ylabel("Feature value")
axes_c[0, 0].legend(fontsize=8)

axes_c[0, 1].plot(t, z_scored, "r-", lw=2)
axes_c[0, 1].axhline(mu_after_zscore, color="red", ls="--", lw=1.5,
                      label=f"z-score mu={mu_after_zscore:.4f} ~= 0!")
axes_c[0, 1].axvspan(0, 10, alpha=0.2, color="red")
axes_c[0, 1].set_title("(2) After z-score (mu ~= 0)", fontsize=11, fontweight="bold", color="red")
axes_c[0, 1].set_ylabel("z-scored value")
axes_c[0, 1].legend(fontsize=9)
axes_c[0, 1].text(0.5, 0.85, "WARNING: baseline collapses to 0!\n(FDR denominator explosion root cause)",
                   transform=axes_c[0, 1].transAxes, ha="center",
                   fontsize=10, color="red", fontweight="bold",
                   bbox=dict(boxstyle="round", fc="#FFEBEE", alpha=0.9))

safe_ratio_bugged = np.clip(ratio_bugged, -50, 50)
axes_c[0, 2].plot(t, safe_ratio_bugged, "r-", lw=2)
axes_c[0, 2].set_title("(3) FDR ratio (bug: 1e+8 explosion -> clipped)\n[Before: z-score -> FDR]",
                        fontsize=11, fontweight="bold", color="red")
axes_c[0, 2].set_ylabel("FDR ratio (clipped to ±50)")
axes_c[0, 2].text(0.5, 0.85, "Actual ratio explodes to 1e+8 scale\n-> meaningless HI output",
                   transform=axes_c[0, 2].transAxes, ha="center",
                   fontsize=10, color="red", fontweight="bold",
                   bbox=dict(boxstyle="round", fc="#FFEBEE", alpha=0.9))

axes_c[1, 0].plot(t, ratio_fixed, "g-", lw=2)
axes_c[1, 0].axhline(0, color="black", ls=":", lw=1)
axes_c[1, 0].axvspan(0, 10, alpha=0.2, color="green", label="Train baseline zone")
axes_c[1, 0].set_title("(4) FDR ratio (fixed)\n[After: FDR-only, Train baseline used]",
                        fontsize=11, fontweight="bold", color="green")
axes_c[1, 0].set_ylabel("FDR ratio")
axes_c[1, 0].legend(fontsize=8)

axes_c[1, 1].plot(t, hi_bugged, "r--", lw=2, label="Before (z-score->FDR, bugged)")
axes_c[1, 1].plot(t, hi_fixed,  "g-",  lw=2.5, label="After (FDR-only, fixed)")
axes_c[1, 1].axvline(60, color="orange", ls="--", lw=1.5, label="Degradation onset")
axes_c[1, 1].set_title("(5) Final HI Comparison", fontsize=11, fontweight="bold")
axes_c[1, 1].set_ylabel("Health Index [0, 1]")
axes_c[1, 1].legend(fontsize=9)

ax_bias = axes_c[1, 2]
t2 = np.arange(60)
np.random.seed(42)
# Already-degraded bearing: starts at 0.40, slowly rises
raw_deg = 0.40 + 0.006 * t2 + 0.010 * np.random.randn(60)

mu_self  = raw_deg[:6].mean()   # ~0.40: contaminated baseline
mu_train = 0.10                  # healthy Train baseline

# HI = relative deviation from baseline (ratio)
hi_contam  = (raw_deg - mu_self)  / (mu_self  + 1e-9)  # starts ~0
hi_correct = (raw_deg - mu_train) / (mu_train + 1e-9)  # starts ~3.0

threshold = 0.5  # 50% deviation from baseline → degraded

cross_contam  = np.where(hi_contam  > threshold)[0]
cross_correct = np.where(hi_correct > threshold)[0]
t_contam  = int(cross_contam[0])  if len(cross_contam)  else None
t_correct = int(cross_correct[0]) if len(cross_correct) else None

ax_bias.plot(t2, hi_contam,  "r--", lw=2.5,
             label=f"Self baseline (contaminated, mu={mu_self:.2f})")
ax_bias.plot(t2, hi_correct, "g-",  lw=2.5,
             label=f"Train baseline (correct, mu={mu_train:.2f})")
ax_bias.axhline(threshold, color="orange", ls=":", lw=2.5,
                label=f"Alarm threshold ({threshold})")
ax_bias.fill_betweenx([0, threshold], 0, t_contam or 60,
                      alpha=0.12, color="red", label="Missed degradation window")

if t_correct == 0:
    ax_bias.annotate("Correct: alarm at t=0\n(already degraded from start!)",
                     xy=(1, threshold + 0.1), xytext=(10, 1.2),
                     fontsize=9, color="green", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="green"))
if t_contam is not None:
    ax_bias.axvline(t_contam, color="red", ls="--", lw=1.5, alpha=0.8)
    ax_bias.annotate(f"Contaminated: alarm at t={t_contam}\n(missed first {t_contam} steps!)",
                     xy=(t_contam, threshold), xytext=(t_contam - 22, 2.8),
                     fontsize=9, color="red", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="red"))

ax_bias.set_title("(6) Baseline Contamination — Alarm Timing\n(same bearing, different baseline → different alarm time)",
                  fontsize=11, fontweight="bold")
ax_bias.set_ylabel("HI (relative deviation from baseline)")
ax_bias.set_ylim(-0.5, 7)
ax_bias.legend(fontsize=8, loc="upper left")

for ax in axes_c.flatten():
    ax.set_xlabel("Time slot")

plt.tight_layout()
plt.savefig(f"{BASE}/viz_normalization_bugfix.png", dpi=150, bbox_inches="tight",
            facecolor="#F8F9FA")
plt.close()
print("Saved: viz_normalization_bugfix.png")


# ──────────────────────────────────────────────────────────
# Fig D: HI-A vs HI-B comparison (4 Train bearings)
# ──────────────────────────────────────────────────────────
fig_d, axes_d = plt.subplots(2, 2, figsize=(16, 10))
fig_d.suptitle("HI-A (Frequency Domain) vs HI-B (Time Domain) — Train Bearings 1-4",
               fontsize=14, fontweight="bold")
fig_d.patch.set_facecolor("#F8F9FA")

colors_a = ["#1565C0", "#2E7D32", "#F57F17", "#880E4F"]

for idx, bid in enumerate([1, 2, 3, 4]):
    ax = axes_d.flatten()[idx]
    hi_a = pd.read_csv(f"{BASE}/hi/output/train/Bearing{bid}_best.csv")["HI"].values
    hi_b = pd.read_csv(f"{BASE}/hi/output/train_hib/Bearing{bid}_best.csv")["HI"].values
    t = np.arange(len(hi_a))

    ax.plot(t, hi_a, color=colors_a[idx], lw=2.5, label="HI-A (freq: highband/energy/variation)")
    ax.plot(t, hi_b, color=colors_a[idx], lw=1.5, ls="--", alpha=0.7,
             label="HI-B (time: kurtosis/crest/rms/p2p)")
    ax.fill_between(t, hi_a, hi_b, alpha=0.1, color=colors_a[idx])
    ax.set_title(f"Bearing {bid}  (life={len(hi_a)-1} cycles)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Health Index")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8)
    ax.scatter([len(hi_a)-1], [hi_a[-1]], color="red", s=80, zorder=5)

plt.tight_layout()
plt.savefig(f"{BASE}/viz_hi_ab_comparison.png", dpi=150, bbox_inches="tight",
            facecolor="#F8F9FA")
plt.close()
print("Saved: viz_hi_ab_comparison.png")


# ──────────────────────────────────────────────────────────
# Fig E: Ensemble model LOOCV results
# ──────────────────────────────────────────────────────────
fig_e, axes_e = plt.subplots(1, 2, figsize=(18, 8))
fig_e.suptitle(
    "3-Model Ensemble LOOCV Results\n"
    "Left: final scores (5-seed median, 4-fold LOOCV) │ "
    "Right: LSTM-A development history (single seed, same 4-fold LOOCV)",
    fontsize=12, fontweight="bold"
)
fig_e.patch.set_facecolor("#F8F9FA")

# ── Left panel: per-model comparison ─────────────────────────────────
ax_fold = axes_e[0]

# Detailed x-axis labels explaining each model
model_names = [
    "LSTM-A\n(HI-A: freq-domain)\n5-seed median",
    "LSTM-B\n(HI-B: time-domain)\n5-seed median",
    "LGBM\n(HI-A flat)\nsingle model",
    "Ensemble\n(weighted)\nLSTM-A+B+LGBM",
]
scores_per_model = [0.6068, 0.4867, 0.4800, 0.5937]
colors_m = ["#1565C0", "#880E4F", "#2E7D32", "#E65100"]

bars = ax_fold.bar(np.arange(4), scores_per_model, color=colors_m, alpha=0.85,
                    edgecolor="black", lw=0.8)
for bar, sc in zip(bars, scores_per_model):
    ax_fold.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f"{sc:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax_fold.set_xticks(np.arange(4))
ax_fold.set_xticklabels(model_names, fontsize=9)
ax_fold.set_ylabel("LOOCV Score (4-fold avg, 5-seed median)", fontsize=10)
ax_fold.set_ylim(0, 0.85)
ax_fold.set_title("Per-Model LOOCV Score (higher is better)\n[5-seed median — differs from right panel single-seed scores]",
                   fontsize=11, fontweight="bold")
ax_fold.axhline(0.6068, color="#1565C0", ls="--", lw=1.5, alpha=0.6,
                label="LSTM-A 5-seed median = 0.6068")
ax_fold.legend(fontsize=9)
ax_fold.annotate("Best\nSingle Model\n(5-seed median)", xy=(0, 0.6068),
                  xytext=(0.6, 0.70),
                  fontsize=9, color="#1565C0", fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="#1565C0"))

# Model legend box inside the plot
legend_text = (
    "HI-A (freq): highfreq/energy/variation → FDR weighted sum → LSTM-A\n"
    "HI-B (time): kurtosis/crest/rms/p2p → FDR weighted sum → LSTM-B\n"
    "LGBM: HI-A as flat input (no sequence) → tree ensemble\n"
    "Ensemble: LOOCV score-based weighted avg (LSTM-A highest weight)"
)
ax_fold.text(0.01, 0.99, legend_text, transform=ax_fold.transAxes,
              fontsize=8, va="top", ha="left", linespacing=1.5,
              bbox=dict(boxstyle="round", fc="#EEF2FF", ec="#1565C0", alpha=0.9))

# ── Right panel: LSTM-A development history ──────────────────────────
ax_hist = axes_e[1]
run_labels = ["Run1\nMSE Loss", "Run2\nAsym Loss\n(initial)", "Run3~5\nexploration",
              "Run6\nAsym+LOO\n+RUL norm\n(final)"]
run_scores = [0.4579, 0.5749, 0.4780, 0.5956]
run_colors = ["#B0BEC5", "#90A4AE", "#78909C", "#42A5F5"]

bars2 = ax_hist.bar(np.arange(4), run_scores, color=run_colors, alpha=0.9,
                     edgecolor="black", lw=0.8, width=0.6)
for bar, sc in zip(bars2, run_scores):
    ax_hist.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                  f"{sc:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax_hist.set_xticks(np.arange(4))
ax_hist.set_xticklabels(run_labels, fontsize=9)
ax_hist.set_ylabel("LOOCV Score (4-fold avg)", fontsize=10)
ax_hist.set_ylim(0, 0.80)
ax_hist.set_title(
    "LSTM-A Development History (single seed, 4-fold LOOCV)\n"
    "Note: single-seed final=0.5956 → 5-seed median=0.6068 (left panel)",
    fontsize=11, fontweight="bold"
)

# Reference lines
ax_hist.axhline(0.4579, color="gray", ls=":", lw=1.5, label="Initial (MSE loss, single seed): 0.4579")
ax_hist.axhline(0.5956, color="#42A5F5", ls="--", lw=2, label="Final (Asym+LOO, single seed): 0.5956")
ax_hist.axhline(0.6068, color="#1565C0", ls="-", lw=2, alpha=0.7,
                label="5-seed median final: 0.6068 (left panel)")

delta = 0.5956 - 0.4579
ax_hist.annotate("", xy=(3, 0.5956), xytext=(0, 0.4579),
                  arrowprops=dict(arrowstyle="-|>", color="#1B5E20", lw=2.5))
ax_hist.text(2, 0.535, f"+{delta:.4f}\n(+{delta/0.4579*100:.1f}%)\n(single seed)",
              ha="center", fontsize=10, color="#1B5E20", fontweight="bold",
              bbox=dict(boxstyle="round", fc="#E8F5E9", alpha=0.85))

# Annotation explaining 5-seed gap
ax_hist.annotate(
    "5-seed median\n(0.6068)\n+0.0112 vs\nsingle seed",
    xy=(3.45, 0.6068), xytext=(3.35, 0.70),
    fontsize=8.5, color="#1565C0", fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color="#1565C0"),
    bbox=dict(boxstyle="round", fc="#E3F2FD", alpha=0.85)
)

ax_hist.legend(fontsize=8.5, loc="upper left")

plt.tight_layout()
plt.savefig(f"{BASE}/viz_ensemble_loocv.png", dpi=150, bbox_inches="tight",
            facecolor="#F8F9FA")
plt.close()
print("Saved: viz_ensemble_loocv.png")

print("\nAll additional visualizations done!")
