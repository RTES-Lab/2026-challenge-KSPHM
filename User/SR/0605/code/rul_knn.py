"""
RUL k-NN v1 — Window-Feature k-NN RUL estimation
=================================================
No LSTM, no start_frac estimation, no information leakage.

Why k-NN over LSTM:
  - 61-71% of LOOCV training sequences have constant-RUL targets
    → LSTM minimizes MSE by predicting the mean (~25-35 cycles) for everything
  - k-NN: const-RUL windows map to const-RUL neighbors naturally
  - k-NN: degradation windows find similar degradation-state neighbors
  - No obs_frac needed → LOOCV and test conditions identical

Algorithm:
  1. Build lookup table: all 50-cycle windows in training bearings
     → (feature_vector, true_RUL) rows
  2. Query: extract 8-9 features from a 50-cycle window
  3. k-NN (k=10, inverse-distance weighting) on z-score normalized features
  4. Conservative bias search to exploit score asymmetry
     (Er > 0 penalty is 2.5x milder than Er < 0)

Features per window (9 total):
  hi_mean, hi_slope, hi_end, hi_std, hi_range,
  kurt_mean, kurt_max, kurt_std, regime_frac
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
BASE       = Path("/data/home/ksphm/2026-challenge-KSPHM")
HI_TRAIN   = BASE / "User/SR/0604/output/train"
HI_TEST    = BASE / "User/SR/0604/output/test"
OUT_DIR    = BASE / "User/SR/0605/output_knn"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS     = [1, 2, 3, 4]
TEST_IDS     = [1, 2, 3, 4, 5, 6]
WIN_SIZE     = 50
STRIDE       = 1
INTERVAL_SEC = 600
K_NEIGHBORS  = 10

EOL          = {1: 126, 2: 114, 3: 89, 4: 137}
NORMAL_UNTIL = {1: 89,  2: 92,  3: 62, 4: 78}

FEAT_NAMES = [
    "hi_mean", "hi_slope", "hi_end", "hi_std", "hi_range",
    "kurt_mean", "kurt_max", "kurt_std", "regime_frac",
]
N_FEAT = len(FEAT_NAMES)


# ══════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════
def load_bearing_data(hi_dir, ids, prefix):
    data = {}
    for bid in ids:
        hi_df   = pd.read_csv(hi_dir / f"{prefix}{bid}_HI.csv")
        feat_df = pd.read_csv(hi_dir / f"{prefix}{bid}_features_raw.csv")
        n = min(len(hi_df), len(feat_df))
        kurt = np.log1p(np.maximum(
            feat_df["ch1_kurt_log"].values[:n],
            feat_df["ch2_kurt_log"].values[:n],
        ))
        data[bid] = {
            "hi":     hi_df["HI"].values[:n].astype(float),
            "regime": hi_df["regime"].values[:n].astype(int),
            "kurt":   kurt,
            "n":      n,
        }
    return data


def rul_labels(n_total, bid):
    nu, eol = NORMAL_UNTIL[bid], EOL[bid]
    idx = np.arange(n_total)
    return np.where(idx <= nu, float(eol - nu),
                    np.maximum(eol - idx, 0).astype(float))


# ══════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════
def extract_features(hi_win, kurt_win, regime_win):
    """Extract N_FEAT scalar features from a fixed-size window."""
    n = len(hi_win)
    t = np.arange(n, dtype=float)
    slope = float(np.polyfit(t, hi_win, 1)[0]) if n > 1 else 0.0
    seg = max(1, min(10, n // 4))
    return np.array([
        float(np.mean(hi_win)),                                     # hi_mean
        slope,                                                      # hi_slope (per cycle)
        float(hi_win[-1]),                                          # hi_end
        float(np.std(hi_win)),                                      # hi_std
        float(hi_win[-seg:].mean() - hi_win[:seg].mean()),          # hi_range
        float(np.mean(kurt_win)),                                   # kurt_mean
        float(np.max(kurt_win)),                                    # kurt_max
        float(np.std(kurt_win)),                                    # kurt_std
        float(np.mean(regime_win)),                                 # regime_frac
    ], dtype=float)


# ══════════════════════════════════════════════════════════════════
# Lookup table & k-NN
# ══════════════════════════════════════════════════════════════════
def build_lookup_table(data, bids):
    rows = []
    for b in bids:
        hi     = data[b]["hi"]
        kurt   = data[b]["kurt"]
        regime = data[b]["regime"].astype(float)
        rul    = rul_labels(len(hi), b)
        n = len(hi)
        for t_s in range(0, n - WIN_SIZE + 1):
            t_e = t_s + WIN_SIZE
            feats = extract_features(hi[t_s:t_e], kurt[t_s:t_e], regime[t_s:t_e])
            rows.append({
                **{FEAT_NAMES[i]: feats[i] for i in range(N_FEAT)},
                "bearing": b,
                "t_end":   t_e - 1,
                "rul":     float(rul[t_e - 1]),
            })
    return pd.DataFrame(rows)


def fit_knn(lookup_df, k=K_NEIGHBORS):
    X = lookup_df[FEAT_NAMES].values.astype(float)
    y = lookup_df["rul"].values.astype(float)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    k_eff  = min(k, len(y))
    knn = KNeighborsRegressor(
        n_neighbors=k_eff, weights="distance", metric="euclidean"
    )
    knn.fit(X_norm, y)
    return knn, scaler


def predict_knn(knn, scaler, feats):
    f_norm = scaler.transform(feats.reshape(1, -1))
    return max(float(knn.predict(f_norm)[0]), 0.0)


# ══════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════
def comp_score(rul_true, rul_pred):
    if rul_true <= 0:
        return np.nan
    Er = 100.0 * (rul_true - rul_pred) / rul_true
    return (np.exp(-np.log(0.5) * Er / 20.0) if Er <= 0
            else np.exp(np.log(0.5) * Er / 50.0))


def avg_score(true_ruls, preds):
    scores = [comp_score(t, p) for t, p in zip(true_ruls, preds)]
    valid  = [s for s in scores if not np.isnan(s)]
    return float(np.mean(valid)) if valid else np.nan


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 72)
    print("  RUL k-NN v1 — Window-Feature k-NN (no LSTM, no obs_frac)")
    print(f"  Features ({N_FEAT}): {FEAT_NAMES}")
    print(f"  WIN={WIN_SIZE}  k={K_NEIGHBORS}")
    print("=" * 72)

    train_data = load_bearing_data(HI_TRAIN, BEARINGS, "Bearing")
    test_data  = load_bearing_data(HI_TEST,  TEST_IDS, "Test")

    # ── LOOCV ──────────────────────────────────────────────────────
    print("\n[1] LOOCV")
    fold_results = {}   # bid → (t_ends, true_ruls, preds, score)

    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        lookup_df  = build_lookup_table(train_data, train_bids)
        knn, scaler = fit_knn(lookup_df)

        hi     = train_data[test_bid]["hi"]
        kurt   = train_data[test_bid]["kurt"]
        regime = train_data[test_bid]["regime"].astype(float)
        rul    = rul_labels(len(hi), test_bid)
        N = len(hi)

        t_ends, true_ruls, preds = [], [], []
        for t_s in range(0, N - WIN_SIZE + 1, STRIDE):
            t_e   = t_s + WIN_SIZE
            feats = extract_features(hi[t_s:t_e], kurt[t_s:t_e], regime[t_s:t_e])
            pred  = predict_knn(knn, scaler, feats)
            t_ends.append(t_e - 1)
            true_ruls.append(float(rul[t_e - 1]))
            preds.append(pred)

        sc = avg_score(true_ruls, preds)
        fold_results[test_bid] = (t_ends, true_ruls, preds, sc)
        print(f"  B{test_bid}  train={train_bids}  "
              f"lookup={len(lookup_df)} windows  "
              f"query={len(preds)} windows  score={sc:.4f}")

    loocv_mean = float(np.mean([v[3] for v in fold_results.values()]))
    print(f"\n  LOOCV (bias=1.0): {loocv_mean:.4f}")
    print(f"  Baselines:  no_obs_frac=0.342  kurtosis_feat(leaky)=0.491")

    # ── Conservative bias search ────────────────────────────────────
    all_true_cv = [r for _, tv, _, _ in fold_results.values() for r in tv]
    all_pred_cv = [p for _, _, pv, _ in fold_results.values() for p in pv]
    best_bias, best_sc = 1.0, avg_score(all_true_cv, all_pred_cv)
    bias_search = {}
    for bias in np.arange(0.65, 1.01, 0.025):
        sc_b = avg_score(all_true_cv, [p * bias for p in all_pred_cv])
        bias_search[round(float(bias), 3)] = round(sc_b, 4)
        if sc_b > best_sc:
            best_sc, best_bias = sc_b, float(bias)

    print(f"\n  Bias search (top 5):")
    for b_val, sc_val in sorted(bias_search.items(), key=lambda x: -x[1])[:5]:
        marker = " ← optimal" if abs(b_val - best_bias) < 0.001 else ""
        print(f"    bias={b_val:.3f}  LOOCV={sc_val:.4f}{marker}")

    # Per-bearing LOOCV with optimal bias
    print(f"\n  Per-bearing LOOCV (bias={best_bias:.3f}):")
    biased_fold_scores = {}
    for bid in BEARINGS:
        t_ends, true_ruls, preds, _ = fold_results[bid]
        biased = [p * best_bias for p in preds]
        sc_b = avg_score(true_ruls, biased)
        biased_fold_scores[bid] = sc_b
        print(f"    B{bid}: raw={fold_results[bid][3]:.4f}  biased={sc_b:.4f}")
    print(f"    Mean (biased): {best_sc:.4f}")

    # ── LOOCV individual plots ──────────────────────────────────────
    for bid in BEARINGS:
        t_ends, true_ruls, preds, sc_raw = fold_results[bid]
        biased = [p * best_bias for p in preds]
        sc_b   = biased_fold_scores[bid]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_ends, true_ruls, "k-",  lw=1.5, label="True RUL")
        ax.plot(t_ends, preds,     "b--", lw=1.5, alpha=0.85,
                label=f"k-NN raw   {sc_raw:.3f}")
        ax.plot(t_ends, biased,    "r-",  lw=1.5, alpha=0.85,
                label=f"k-NN biased {sc_b:.3f}")
        ax.set_title(
            f"Bearing{bid} LOOCV — k-NN (WIN={WIN_SIZE}, k={K_NEIGHBORS})",
            fontsize=10)
        ax.set_xlabel("Window End (cycle)")
        ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"Bearing{bid}_loocv.png", dpi=150)
        plt.close()

    # ── Combined LOOCV plot ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, bid in zip(axes.flatten(), BEARINGS):
        t_ends, true_ruls, preds, sc_raw = fold_results[bid]
        biased = [p * best_bias for p in preds]
        sc_b   = biased_fold_scores[bid]
        ax.plot(t_ends, true_ruls, "k-",  lw=1.5, label="True RUL")
        ax.plot(t_ends, preds,     "b--", lw=1.4, alpha=0.8,
                label=f"k-NN {sc_raw:.3f}")
        ax.plot(t_ends, biased,    "r-",  lw=1.4, alpha=0.8,
                label=f"biased {sc_b:.3f}")
        ax.set_title(f"Bearing{bid}", fontsize=10)
        ax.set_xlabel("Window End (cycle)")
        ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"LOOCV k-NN  WIN={WIN_SIZE}  k={K_NEIGHBORS}\n"
        f"Mean raw={loocv_mean:.3f}  biased={best_sc:.3f} (bias={best_bias:.2f})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loocv_all.png", dpi=150)
    plt.close()

    # ── Test RUL ───────────────────────────────────────────────────
    print("\n[2] Test RUL (full-train lookup table)")
    lookup_full  = build_lookup_table(train_data, BEARINGS)
    knn_full, scaler_full = fit_knn(lookup_full)
    print(f"  Lookup table size: {len(lookup_full)}")

    summary_rows = []
    for tid in TEST_IDS:
        hi     = test_data[tid]["hi"]
        kurt   = test_data[tid]["kurt"]
        regime = test_data[tid]["regime"].astype(float)
        feats  = extract_features(hi, kurt, regime)

        pred_raw    = predict_knn(knn_full, scaler_full, feats)
        pred_biased = pred_raw * best_bias
        pred_hr     = pred_biased * INTERVAL_SEC / 3600

        # Nearest-neighbor provenance (top 3)
        f_norm = scaler_full.transform(feats.reshape(1, -1))
        dists, idxs = knn_full.kneighbors(f_norm, n_neighbors=3)
        nn_info = [
            (int(lookup_full.iloc[i]["bearing"]),
             int(lookup_full.iloc[i]["t_end"]),
             round(lookup_full.iloc[i]["rul"], 1))
            for i in idxs[0]
        ]
        nn_str = " | ".join(f"B{b}@t{t}(r={r})" for b, t, r in nn_info)

        print(f"  T{tid}: HI_end={hi[-1]:.4f}  kurt_max={kurt.max():.3f}  "
              f"RUL={pred_raw:.1f}c → {pred_hr:.2f}hr  NN: {nn_str}")
        summary_rows.append({
            "test_id":          tid,
            "hi_end":           round(float(hi[-1]), 4),
            "kurt_max":         round(float(kurt.max()), 3),
            "rul_raw_cycles":   round(pred_raw, 2),
            "rul_biased_cycles": round(pred_biased, 2),
            "rul_biased_hours": round(pred_hr, 2),
            "nn1_bearing":      nn_info[0][0],
            "nn1_t_end":        nn_info[0][1],
            "nn1_rul":          nn_info[0][2],
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary.csv", index=False)

    # Test RUL bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = [f"T{r['test_id']}" for _, r in df_sum.iterrows()]
    bars = ax.bar(x_labels, df_sum["rul_biased_hours"],
                  color="steelblue", alpha=0.8, edgecolor="navy")
    for bar, row in zip(bars, df_sum.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                f"{row.rul_biased_hours:.2f}h\n(HI={row.hi_end:.3f})",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Test Bearing")
    ax.set_ylabel("Predicted RUL (hours)")
    ax.set_title(
        f"Test RUL — k-NN (WIN={WIN_SIZE}, k={K_NEIGHBORS}, bias={best_bias:.2f})\n"
        f"LOOCV raw={loocv_mean:.3f}  biased={best_sc:.3f}",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_rul_all.png", dpi=150)
    plt.close()

    # ── Final summary ───────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  LOOCV raw  (bias=1.0):     {loocv_mean:.4f}")
    print(f"  LOOCV biased (bias={best_bias:.3f}): {best_sc:.4f}")
    print(f"  Reference: no_obs_frac=0.342  |  kurtosis_feat(leaky)=0.491")
    print(f"\n  Test RUL predictions:")
    print(df_sum[["test_id", "hi_end", "kurt_max",
                  "rul_raw_cycles", "rul_biased_hours"]].to_string(index=False))
    print(f"\n  Output: {OUT_DIR}/")
    print("=" * 72)
