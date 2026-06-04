"""
RUL Adaptive Ensemble v1
========================
Combines feat_sim (LGBM+LSTM, leaky start_frac) and k-NN v1 (honest, biased)
using HI-conditional blending:

  HI_end >= 0.05  →  min(feat_sim, knn)          # degradation confirmed → conservative
  HI_end <  0.05  →  0.4 * max + 0.6 * min       # normal op → 60:40 toward shorter

Rationale:
  - HI >= 0.05: k-NN matches degradation-phase windows well; min is safe
  - HI <  0.05: k-NN bias toward B2 short life (const-RUL=22) → too aggressive
               feat_sim start_frac based on feature similarity → more grounded
               60:40 blend keeps conservatism without fully trusting k-NN
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
BASE         = Path("/data/home/ksphm/2026-challenge-KSPHM")
FEAT_SIM_CSV = BASE / "User/SR/0605/output/rul_feat_sim/test_summary_v5.csv"
KNN_CSV      = BASE / "User/SR/0605/output_knn/test_summary.csv"
OUT_DIR      = BASE / "User/SR/0605/output_adaptive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HI_THRESHOLD = 0.05   # above: degradation confirmed → use min
W_SHORT      = 0.60   # weight on shorter prediction (normal operation zone)
W_LONG       = 0.40   # weight on longer prediction


# ══════════════════════════════════════════════════════════════════
def adaptive_blend(feat_rul, knn_rul, hi_end):
    """Apply HI-conditional blending rule."""
    if hi_end >= HI_THRESHOLD:
        return min(feat_rul, knn_rul)
    else:
        short = min(feat_rul, knn_rul)
        long  = max(feat_rul, knn_rul)
        return W_LONG * long + W_SHORT * short


if __name__ == "__main__":
    print("=" * 72)
    print("  RUL Adaptive Ensemble v1")
    print(f"  HI threshold: {HI_THRESHOLD}  |  weights (long:short) = {W_LONG:.2f}:{W_SHORT:.2f}")
    print("=" * 72)

    feat_df = pd.read_csv(FEAT_SIM_CSV)
    knn_df  = pd.read_csv(KNN_CSV)

    # ── Per-test adaptive blend ────────────────────────────────────
    rows = []
    for _, fs_row in feat_df.iterrows():
        tid      = int(fs_row["test_id"])
        knn_row  = knn_df[knn_df["test_id"] == tid].iloc[0]

        fs_rul   = float(fs_row["final_rul_hours"])
        knn_rul  = float(knn_row["rul_biased_hours"])
        hi_end   = float(knn_row["hi_end"])
        kurt_max = float(knn_row["kurt_max"])

        blended  = adaptive_blend(fs_rul, knn_rul, hi_end)
        rule     = "min" if hi_end >= HI_THRESHOLD else f"60:40"

        rows.append({
            "test_id":      tid,
            "hi_end":       round(hi_end, 4),
            "kurt_max":     round(kurt_max, 3),
            "feat_sim_rul": round(fs_rul, 3),
            "knn_rul":      round(knn_rul, 3),
            "adaptive_rul": round(blended, 3),
            "rule":         rule,
        })
        print(f"  T{tid}: HI={hi_end:.3f}  feat_sim={fs_rul:.2f}hr  "
              f"knn={knn_rul:.2f}hr  [{rule}] → {blended:.2f}hr")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "adaptive_test_summary.csv", index=False)

    # ── Bar chart comparison ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    w = 0.26

    bars_fs  = ax.bar(x - w,     df["feat_sim_rul"], w, label="feat_sim",    color="#4C9BE8", alpha=0.8, edgecolor="navy")
    bars_knn = ax.bar(x,         df["knn_rul"],      w, label="k-NN v1",     color="#F4A460", alpha=0.8, edgecolor="saddlebrown")
    bars_ada = ax.bar(x + w,     df["adaptive_rul"], w, label="Adaptive v1", color="#2E8B57", alpha=0.9, edgecolor="darkgreen")

    for bar, row in zip(bars_ada, df.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{row.adaptive_rul:.2f}h",
                ha="center", va="bottom", fontsize=9, color="darkgreen", fontweight="bold")

    # HI annotations
    for i, row in enumerate(df.itertuples()):
        ax.text(x[i], -0.35, f"HI={row.hi_end:.3f}\n{row.rule}",
                ha="center", va="top", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([f"T{r['test_id']}" for _, r in df.iterrows()])
    ax.set_xlabel("Test Bearing")
    ax.set_ylabel("Predicted RUL (hours)")
    ax.set_title(
        "Adaptive Ensemble v1: feat_sim + k-NN\n"
        f"HI >= {HI_THRESHOLD}: min  |  HI < {HI_THRESHOLD}: 60:40 blend (short-weighted)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=-0.8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "adaptive_test_rul.png", dpi=150)
    plt.close()

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  Final adaptive RUL predictions:")
    print(df[["test_id", "hi_end", "feat_sim_rul", "knn_rul",
              "adaptive_rul", "rule"]].to_string(index=False))
    print(f"\n  Output: {OUT_DIR}/")
    print("=" * 72)
