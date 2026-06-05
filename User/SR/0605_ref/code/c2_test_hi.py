"""
c2_test_hi.py  —  Approach B: Direct Feature-Based Test HI
===========================================================
Computes HI for test bearings directly from extracted features (c1 output),
without any neural network. Steps:

  1. Condition classification per file via k-means on {ch1_rms, ch3_rms}
     — both channels tend to be high at high RPM and low at low RPM.
     k=2 clusters; higher-RMS cluster → condition 1 (high speed).

  2. Signal transformation parameters estimated from training bearings
     (c2_hi_pipeline.py FIX-1 approach, averaged across LOO folds).

  3. FDR HI constructed using ALL training bearings as baseline
     (no LOO needed: test bearings are independent of training data).

  4. Scale normalization: test raw HI divided by the 95th-percentile of
     all training-bearing HI, so test HI ≈ [0, 1] on the same scale.

Output:
  output/hi/Test{t}_HI_Direct.csv  — feature-based HI  (Approach B)
  output/hi/test_hi_direct.png
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# reuse helpers from c2_hi_pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent))
from c2_hi_pipeline import (
    BEARING_IDS, ALL_FEATS, FEATURE_Q, FEATURE_GROUPS,
    NORMAL_RATIO,
    load_operation, align_rpm, discretize_rpm,
    estimate_transform_params, transform_signal,
    compute_bearing_baseline,
    v3_pipeline_raw, apply_global_scale,
    minmax_scale, moving_average,
    monotonicity, trendability,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM")
DATA_DIR = BASE_DIR / "dataset"
HI_DIR   = BASE_DIR / "User/SR/0605_ref/output/hi"
HI_DIR.mkdir(parents=True, exist_ok=True)


# ── Condition classification without RPM ──────────────────────────────────────

def classify_condition_rms(df_feat: pd.DataFrame) -> np.ndarray:
    """
    Assign condition (0=low, 1=high) to each file using k-means on
    {ch1_rms, ch3_rms}.  Higher-RMS cluster = high RPM = condition 1.
    Falls back to condition 0 if clustering fails.
    """
    rms_cols = [c for c in ("ch1_rms", "ch3_rms") if c in df_feat.columns]
    if not rms_cols:
        return np.zeros(len(df_feat), dtype=int)
    X = df_feat[rms_cols].values.astype(np.float64)
    if len(np.unique(X[:, 0])) < 2:
        return np.zeros(len(df_feat), dtype=int)
    try:
        km    = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        # cluster with higher mean RMS = high RPM = condition 1
        c0_mean = X[labels == 0].mean()
        c1_mean = X[labels == 1].mean()
        if c0_mean > c1_mean:           # swap labels
            labels = 1 - labels
        return labels.astype(int)
    except Exception:
        return np.zeros(len(df_feat), dtype=int)


# ── Average transformation parameters from training ───────────────────────────

def get_avg_transform_params(feat_cols: list) -> dict:
    """
    Estimate signal transformation parameters from each training bearing (LOO
    style for robustness), then average the (a, b) values per feature.
    Returns: {feat: (a_avg, b_avg)}  for non-baseline condition (cond=1).
    """
    a_vals = {f: [] for f in feat_cols}
    b_vals = {f: [] for f in feat_cols}

    for b in BEARING_IDS:
        df_feat = pd.read_csv(HI_DIR / f"Bearing{b}_features.csv"
                              if (HI_DIR / f"Bearing{b}_features.csv").exists()
                              else BASE_DIR / "User/SR/0605_ref/output/hi"
                                             / f"Bearing{b}_features.csv")
        op = load_operation(b)
        n  = len(df_feat)
        rpm  = align_rpm(op, n)
        cond = discretize_rpm(rpm)

        for feat in feat_cols:
            y_raw  = df_feat[feat].values.astype(np.float64)
            params = estimate_transform_params(y_raw, cond, baseline_cond=0, n_cond=2)
            a, b_  = params.get(1, (1.0, 0.0))
            a_vals[feat].append(a)
            b_vals[feat].append(b_)

    return {f: (float(np.mean(a_vals[f])), float(np.mean(b_vals[f])))
            for f in feat_cols}


# ── Training reference scale ──────────────────────────────────────────────────

def compute_training_hi_scale() -> float:
    """
    95th percentile of ALL training bearing HI values → use as normalization
    reference for test HI so test is on the same [0,~1] scale as training.
    """
    all_hi = []
    for b in BEARING_IDS:
        csv = HI_DIR / f"Bearing{b}_HI.csv"
        if csv.exists():
            all_hi.extend(pd.read_csv(csv)["HI"].values.tolist())
    return float(np.percentile(all_hi, 95)) if all_hi else 1.0


# ── Test HI computation ───────────────────────────────────────────────────────

def compute_test_hi(test_id: int,
                    avg_params: dict,
                    training_baseline: dict,
                    best_br: float,
                    best_alpha: float,
                    g_min: float,
                    g_max: float,
                    feat_cols: list) -> np.ndarray:
    """
    Compute FDR HI for one test / validation bearing.

    Normalisation uses (g_min, g_max) derived from ALL four training bearings
    via compute_global_scale() — the same scale that the test pipeline should use.

    This guarantees:
      • Training HI  normalised with LOO scale  (other 3 bearings)
      • Test/Val HI  normalised with global scale (all 4 training bearings)
    Both scales are derived from training data only → no leakage from test.
    """
    feat_csv = HI_DIR / f"Test{test_id}_features.csv"
    if not feat_csv.exists():
        raise FileNotFoundError(f"{feat_csv} — run c1_extract_features.py first")
    df_feat = pd.read_csv(feat_csv)

    # 1. Classify condition
    cond  = classify_condition_rms(df_feat)
    n_low = int((cond == 0).sum()); n_high = int((cond == 1).sum())

    # 2. Apply signal transformation using averaged training params
    meta_cols = [c for c in ("file_idx", "time_sec", "bearing_id") if c in df_feat.columns]
    df_trans  = df_feat[meta_cols].copy()
    for feat in feat_cols:
        if feat not in df_feat.columns:
            continue
        y_raw = df_feat[feat].values.astype(np.float64)
        a, b  = avg_params.get(feat, (1.0, 0.0))
        df_trans[feat] = np.where(cond == 0, y_raw, a * y_raw + b)
    df_trans["cond"] = cond

    # 3. Raw FDR score (same pipeline as training, but using all-training baseline)
    raw_hi = v3_pipeline_raw(df_trans, best_br, best_alpha, training_baseline)

    # 4. Apply the GLOBAL scale (from all 4 training bearings)
    hi = apply_global_scale(raw_hi, g_min, g_max)

    print(f"  Test{test_id}: cond low={n_low}, high={n_high}  "
          f"HI range [{hi.min():.3f}, {hi.max():.3f}]")
    return hi


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  Approach B — Direct Feature-Based Test HI")
    print("=" * 60)

    # ── Load scale params saved by c2_hi_pipeline ─────────────────────────
    scale_csv = HI_DIR / "scale_params.csv"
    if not scale_csv.exists():
        raise FileNotFoundError(f"{scale_csv} — run c2_hi_pipeline.py first.")
    sp = pd.read_csv(scale_csv).iloc[0]
    best_br    = float(sp["br"])
    best_alpha = float(sp["alpha"])
    g_min      = float(sp["g_min"])
    g_max      = float(sp["g_max"])
    print(f"  Loaded scale_params: br={best_br}, alpha={best_alpha},"
          f" g_min={g_min:.4f}, g_max={g_max:.4f}")

    # feature column names
    sample_df = pd.read_csv(HI_DIR / "Bearing1_features.csv")
    feat_cols = [c for c in sample_df.columns
                 if c not in ("file_idx", "time_sec", "bearing_id")]

    # ── Average transformation params from training ────────────────────────
    print("\n[1/3] Estimating average signal transformation params …")
    avg_params = get_avg_transform_params(feat_cols)
    print(f"  Done. e.g. ch3_rms: a={avg_params.get('ch3_rms', (1,0))[0]:.4f}, "
          f"b={avg_params.get('ch3_rms', (1,0))[1]:.4f}")

    # ── Training baseline (all 4 bearings) — consistent with global scale ──
    print("\n[2/3] Building training baseline (all 4 bearings) …")
    dfs_train = {}
    for b in BEARING_IDS:
        df_t = pd.read_csv(HI_DIR / f"Bearing{b}_features_transformed.csv")
        op   = load_operation(b)
        cond = discretize_rpm(align_rpm(op, len(df_t)))
        df_t["cond"] = cond
        dfs_train[b] = df_t
    # All 4 bearings in baseline (test/val bearings are independent of training)
    training_baseline = compute_bearing_baseline(dfs_train, best_br, exclude_bid=0)

    # ── Compute test HI ───────────────────────────────────────────────────
    print("\n[3/3] Computing test HI …")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, t in zip(axes.flat, range(1, 7)):
        hi = compute_test_hi(t, avg_params, training_baseline,
                             best_br, best_alpha, g_min, g_max, feat_cols)
        pd.DataFrame({"file": np.arange(1, len(hi)+1), "HI_Direct": hi}
                     ).to_csv(HI_DIR / f"Test{t}_HI_Direct.csv", index=False)
        ax.plot(hi, "b-", lw=1.5)
        ax.set_title(f"Test {t}  (Approach B)"); ax.grid(alpha=0.3)

    plt.suptitle(
        f"Approach B: Test Bearings — Direct Feature HI\n"
        f"(global scale: g_min={g_min:.3f}, g_max={g_max:.3f})", fontsize=11)
    plt.tight_layout()
    plt.savefig(HI_DIR / "test_hi_direct.png", dpi=120)
    plt.close()
    print(f"  Plot → {HI_DIR / 'test_hi_direct.png'}")
    print("\nApproach B complete.")


if __name__ == "__main__":
    run()
