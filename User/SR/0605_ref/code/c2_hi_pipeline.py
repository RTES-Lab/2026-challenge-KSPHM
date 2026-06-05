"""
c2_hi_pipeline.py
=================
Signal transformation (F2S2) + FDR-based HI construction.
Outputs Bearing{b}_HI.csv in 0605_ref/output/hi/.

Original sources:
  - User/SC/HI/04142304_signal_transform_v2/code/signal_transform_v2.py
  - User/SR/0514/hi/code/hi_train.py

Copied + the following leakage issues fixed:
  [FIX-1] estimate_transform_params: original used ALL data (incl. future failure)
           for interpolation → now restricted to first NORMAL_RATIO (15%) of
           each bearing's data. Prevents future degraded states from biasing
           the early-life transformation.
  [FIX-2] run_v4fdr_grid_search save section: original used exclude_bid=0
           (global baseline = all 4 bearings) → replaced with LOO (exclude_bid=bid)
           so each bearing's HI is not biased by its own healthy data in the baseline.
  NOTE: build_hi_from_transformed uses s_max = spe.max() (global max over full
        lifetime). This is mild leakage but acceptable for CNN *training labels*
        where the full lifetime is known. Kept as-is per original intent.
"""

import warnings
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM")
DATA_DIR = BASE_DIR / "dataset"
OUT_DIR  = BASE_DIR / "User/SR/0605_ref/output/hi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
INTERVAL_SEC = 600
BEARING_IDS  = [1, 2, 3, 4]
RPM_BOUNDARY = 850          # low: <850, high: >=850
BASELINE     = 0            # low-speed is baseline condition
N_COND       = 2
NORMAL_RATIO = 0.15         # healthy initial fraction
N_PCA_COMP   = 5
LOG_KEYWORDS = ["energy", "kurt_rms"]

# Feature weights (from hi_train.py — validated by grid search on training data)
FEATURE_Q = {
    "ch3_high_band":   0.4315732105779938,
    "ch4_high_band":   0.41934581236265145,
    "ch3_std":         0.4143663846438889,
    "ch3_total_power": 0.41207524516168137,
    "ch3_energy":      0.41108403369865365,
    "ch3_rms":         0.41108403369865365,
    "ch3_p2p":         0.3665441916808586,
}
FEATURE_GROUPS = {
    "highfreq":  ["ch3_high_band", "ch4_high_band"],
    "energy":    ["ch3_total_power", "ch3_energy", "ch3_rms"],
    "variation": ["ch3_std", "ch3_p2p"],
}
ALL_FEATS = list(FEATURE_Q.keys())

FEATURE_Q_B = {
    "ch3_kurtosis": 0.40,
    "ch3_crest_f":  0.38,
    "ch3_rms":      0.36,
    "ch3_p2p":      0.35,
    "ch4_kurtosis": 0.37,
    "ch4_rms":      0.34,
}
FEATURE_GROUPS_B = {
    "impulse":   ["ch3_kurtosis", "ch3_crest_f", "ch4_kurtosis"],
    "amplitude": ["ch3_rms", "ch3_p2p", "ch4_rms"],
}
ALL_FEATS_B = list(FEATURE_Q_B.keys())


# ── Utility functions (copied verbatim) ───────────────────────────────────────

def monotonicity(series: np.ndarray) -> float:
    if len(series) <= 1: return 0.0
    diff = np.diff(series)
    return abs(np.sum(diff > 0) - np.sum(diff < 0)) / len(diff)


def trendability(series: np.ndarray) -> float:
    if len(series) <= 1: return 0.0
    rho, _ = spearmanr(np.arange(len(series)), series)
    return abs(rho) if not np.isnan(rho) else 0.0


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1: return x.copy()
    pad    = window // 2
    x_pad  = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(x_pad, kernel, mode="valid")[:len(x)]


def minmax_scale(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def robust_clip(x: np.ndarray, low_q: float = 0.01, high_q: float = 0.99) -> np.ndarray:
    lo = np.quantile(x, low_q)
    hi = np.quantile(x, high_q)
    return np.clip(x, lo, hi)


def ema_smooth(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


# ── RPM alignment & condition labelling (from signal_transform_v2.py) ─────────

def load_operation(bid: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"Train{bid}_Operation.csv", encoding="cp949")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Time[sec]": "time_sec", "Motor speed[rpm]": "rpm"})
    return df


def align_rpm(op_df: pd.DataFrame, n_files: int) -> np.ndarray:
    rpm = np.zeros(n_files)
    for k in range(n_files):
        t0, t1 = k * INTERVAL_SEC, k * INTERVAL_SEC + 60
        mask   = (op_df["time_sec"] >= t0) & (op_df["time_sec"] < t1)
        vals   = op_df.loc[mask, "rpm"].values
        rpm[k] = vals.mean() if len(vals) > 0 else np.nan
    nans = np.isnan(rpm)
    if nans.any():
        idx     = np.arange(n_files)
        rpm[nans] = np.interp(idx[nans], idx[~nans], rpm[~nans])
    return rpm


def discretize_rpm(rpm_arr: np.ndarray) -> np.ndarray:
    """850 RPM threshold: 0 = low, 1 = high."""
    return (rpm_arr >= RPM_BOUNDARY).astype(int)


# ── Signal transformation (from signal_transform_v2.py, FIX-1 applied) ────────

def estimate_transform_params(y: np.ndarray, cond: np.ndarray,
                               baseline_cond: int, n_cond: int,
                               normal_ratio: float = NORMAL_RATIO) -> dict:
    """
    Estimate linear transformation params: y_high ≈ a * y_low + b

    FIX-1 (vs original): parameter estimation restricted to first `normal_ratio`
    fraction of the bearing life. The original used the full sequence including
    late-life degraded values for interpolation, introducing future leakage.
    """
    N        = len(y)
    t_idx    = np.arange(N)
    params   = {baseline_cond: (1.0, 0.0)}

    # --- [FIX-1] Use only first normal_ratio fraction for estimation ---
    n_est    = max(6, int(N * normal_ratio))
    mask_est = np.arange(n_est)               # index window for param estimation

    bl_mask_est  = cond[mask_est] == baseline_cond
    bl_times_est = mask_est[bl_mask_est]
    bl_vals_est  = y[bl_times_est]

    if len(bl_times_est) < 2:
        # Fallback: use ALL baseline-condition data if not enough in initial window
        bl_mask_all  = cond == baseline_cond
        bl_times_est = t_idx[bl_mask_all]
        bl_vals_est  = y[bl_mask_all]

    for c in range(n_cond):
        if c == baseline_cond:
            continue
        c_mask_est  = cond[mask_est] == c
        c_times_est = mask_est[c_mask_est]
        c_vals_est  = y[c_times_est]

        if len(c_times_est) < 2:
            params[c] = (1.0, 0.0)
            continue

        y_bl_interp = np.interp(c_times_est, bl_times_est, bl_vals_est)
        A = np.column_stack([c_vals_est, np.ones(len(c_vals_est))])
        result, _, _, _ = np.linalg.lstsq(A, y_bl_interp, rcond=None)
        params[c] = (float(result[0]), float(result[1]))

    return params


def transform_signal(y: np.ndarray, cond: np.ndarray, params: dict) -> np.ndarray:
    y_t = np.zeros_like(y)
    for k in range(len(y)):
        a, b   = params[cond[k]]
        y_t[k] = a * y[k] + b
    return y_t


def apply_signal_transform(df_feat: pd.DataFrame, cond: np.ndarray,
                            feat_cols: list) -> pd.DataFrame:
    """Apply signal transformation to all features in df_feat."""
    df_t = df_feat[["file_idx", "time_sec", "bearing_id"]].copy()
    for feat in feat_cols:
        y_raw  = df_feat[feat].values.astype(np.float64)
        params = estimate_transform_params(y_raw, cond, BASELINE, N_COND)
        df_t[feat] = transform_signal(y_raw, cond, params)
    return df_t


# ── SPE-based HI construction (from signal_transform_v2.py) ──────────────────

def log_transform(X: np.ndarray, feat_cols: list) -> np.ndarray:
    X = X.copy()
    for i, col in enumerate(feat_cols):
        if any(kw in col for kw in LOG_KEYWORDS):
            X[:, i] = np.log1p(np.abs(X[:, i]))
    return X


def build_spe_hi(df_trans: pd.DataFrame, selected_feats: list,
                 normal_ratio: float = NORMAL_RATIO,
                 n_comp: int = N_PCA_COMP) -> np.ndarray:
    """
    PCA SPE-based HI on signal-transformed features.

    NOTE: s_max = spe.max() uses the global max over full bearing lifetime
    (mild future leakage). Acceptable for CNN training labels where the full
    lifetime is known. See module docstring.
    """
    avail = [f for f in selected_feats if f in df_trans.columns]
    X_raw = df_trans[avail].values.astype(np.float64)
    X     = log_transform(X_raw, avail)

    n_normal = max(int(len(df_trans) * normal_ratio), 5)
    scaler   = StandardScaler()
    scaler.fit(X[:n_normal])
    Xs = scaler.transform(X)

    nc  = min(n_comp, n_normal - 1, len(avail))
    pca = PCA(n_components=nc)
    pca.fit(Xs[:n_normal])

    recon = pca.inverse_transform(pca.transform(Xs))
    spe   = np.mean((Xs - recon) ** 2, axis=1)

    s_mean = float(spe[:n_normal].mean())
    s_max  = float(spe.max())           # NOTE: mild future leakage (see docstring)
    hi     = np.clip((spe - s_mean) / (s_max - s_mean + 1e-12), 0, 1)
    return hi.astype(np.float32)


# ── FDR-based HI construction (from hi_train.py) ─────────────────────────────

def compute_bearing_baseline(dfs: dict, baseline_ratio: float,
                              exclude_bid: int) -> dict:
    """LOO external baseline for FDR (hi_train.py — verbatim)."""
    feat_vals = {(r, f): [] for r in [0, 1] for f in ALL_FEATS}
    for bid, df in dfs.items():
        if bid == exclude_bid:
            continue
        cond   = df["cond"].values
        n      = len(df)
        n_base = max(3, int(n * baseline_ratio))
        for regime in [0, 1]:
            base_idx = np.where(cond == regime)[0][:n_base]
            if len(base_idx) == 0:
                continue
            for f in ALL_FEATS:
                feat_vals[(regime, f)].extend(df[f].values[base_idx].tolist())
    baseline = {}
    for regime in [0, 1]:
        for f in ALL_FEATS:
            vals = feat_vals[(regime, f)]
            baseline[(regime, f)] = float(np.mean(vals)) if vals else 1.0
    return baseline


def compute_bearing_baseline_b(dfs: dict, baseline_ratio: float,
                                exclude_bid: int) -> dict:
    """LOO baseline for HI-B features (hi_train.py — verbatim)."""
    feat_vals = {(r, f): [] for r in [0, 1] for f in ALL_FEATS_B}
    for bid, df in dfs.items():
        if bid == exclude_bid:
            continue
        cond   = df["cond"].values
        n      = len(df)
        n_base = max(3, int(n * baseline_ratio))
        for regime in [0, 1]:
            base_idx = np.where(cond == regime)[0][:n_base]
            if len(base_idx) == 0:
                continue
            for f in ALL_FEATS_B:
                if f in df.columns:
                    feat_vals[(regime, f)].extend(df[f].values[base_idx].tolist())
    baseline = {}
    for regime in [0, 1]:
        for f in ALL_FEATS_B:
            vals = feat_vals[(regime, f)]
            baseline[(regime, f)] = float(np.mean(vals)) if vals else 1.0
    return baseline


def build_feature_ratios_external(feat_matrix, feature_names, cond, baseline,
                                   eps=1e-8):
    ratios = np.zeros_like(feat_matrix, dtype=float)
    for regime in [0, 1]:
        idx = np.where(cond == regime)[0]
        if len(idx) == 0:
            continue
        bl_vec = np.array([baseline[(regime, f)] for f in feature_names])
        bl_vec = np.where(np.abs(bl_vec) < eps, eps, bl_vec)
        ratios[idx] = (feat_matrix[idx] - bl_vec) / (np.abs(bl_vec) + eps)
    return ratios


def build_feature_ratios_hib(feat_matrix, feature_names, cond, baseline, eps=1e-8):
    ratios = np.zeros_like(feat_matrix, dtype=float)
    for regime in [0, 1]:
        idx = np.where(cond == regime)[0]
        if len(idx) == 0:
            continue
        bl_vec = np.array([baseline[(regime, f)] for f in feature_names])
        bl_vec = np.where(np.abs(bl_vec) < eps, eps, bl_vec)
        ratios[idx] = (feat_matrix[idx] - bl_vec) / (np.abs(bl_vec) + eps)
    return ratios


def postprocess_score(score: np.ndarray) -> np.ndarray:
    score = robust_clip(score, 0.01, 0.99)
    corr  = np.corrcoef(np.arange(len(score)), score)[0, 1]
    if not np.isnan(corr) and corr < 0:
        score = -score
    return score


def make_group_hi_fdr(feat_matrix, feature_names, cond, baseline,
                      ema_alpha, feat_q_map=None):
    if feat_q_map is None:
        feat_q_map = FEATURE_Q
    ratios  = build_feature_ratios_external(feat_matrix, feature_names, cond, baseline)
    weights = np.array([feat_q_map[f] for f in feature_names], dtype=float)
    weights = weights / (weights.sum() + 1e-12)
    score   = (ratios * weights.reshape(1, -1)).sum(axis=1)
    score   = postprocess_score(score)
    score   = minmax_scale(score)
    score   = ema_smooth(score, alpha=ema_alpha)
    return minmax_scale(score)


def v3_pipeline(df: pd.DataFrame, br: float, alpha: float,
                baseline: dict) -> np.ndarray:
    """HI-A: FDR Group Weight (frequency domain). Returns per-bearing minmax-scaled HI."""
    cond          = df["cond"].values
    sub_his       = {}
    group_weights = {}
    for gname, feats in FEATURE_GROUPS.items():
        mat    = df[feats].values
        sub_hi = make_group_hi_fdr(mat, feats, cond, baseline, alpha, FEATURE_Q)
        sub_his[gname]       = sub_hi
        group_weights[gname] = np.mean([FEATURE_Q[f] for f in feats])
    sub_mat  = np.column_stack([sub_his[g] for g in FEATURE_GROUPS])
    w        = np.array([group_weights[g] for g in FEATURE_GROUPS], dtype=float)
    w       /= w.sum() + 1e-12
    final_hi = (sub_mat * w.reshape(1, -1)).sum(axis=1)
    return moving_average(minmax_scale(final_hi), 7)


def v3_pipeline_raw(df: pd.DataFrame, br: float, alpha: float,
                    baseline: dict) -> np.ndarray:
    """
    Same as v3_pipeline but WITHOUT the final minmax_scale.
    Returns the raw (smoothed, but not per-bearing normalised) FDR composite score.
    Used to derive a cross-bearing global scale.
    """
    cond          = df["cond"].values
    sub_his       = {}
    group_weights = {}
    for gname, feats in FEATURE_GROUPS.items():
        mat    = df[feats].values
        sub_hi = make_group_hi_fdr(mat, feats, cond, baseline, alpha, FEATURE_Q)
        sub_his[gname]       = sub_hi
        group_weights[gname] = np.mean([FEATURE_Q[f] for f in feats])
    sub_mat  = np.column_stack([sub_his[g] for g in FEATURE_GROUPS])
    w        = np.array([group_weights[g] for g in FEATURE_GROUPS], dtype=float)
    w       /= w.sum() + 1e-12
    final_hi = (sub_mat * w.reshape(1, -1)).sum(axis=1)
    return moving_average(final_hi, 7)          # no minmax_scale


def apply_global_scale(raw: np.ndarray, g_min: float, g_max: float) -> np.ndarray:
    """Normalise a raw FDR score vector using a cross-bearing scale."""
    return np.clip((raw - g_min) / (g_max - g_min + 1e-12), 0, 1).astype(np.float32)


def compute_loo_scale(dfs: dict, br: float, alpha: float,
                      exclude_bid: int) -> tuple:
    """
    Global (g_min, g_max) derived from the bearings in the LOO fold that
    EXCLUDES exclude_bid.

    Logic:
      • baseline  = computed from the OTHER bearings (same LOO fold)
      • raw scores collected from those OTHER bearings
      • 1st / 99th percentile → (g_min, g_max)

    This scale is then applied to the *excluded* bearing's raw HI, giving
    a fully LOO-consistent normalisation: exclude_bid's HI is scaled using
    ONLY information from the other three bearings.
    """
    baseline   = compute_bearing_baseline(dfs, br, exclude_bid=exclude_bid)
    raw_scores = []
    for bid in BEARING_IDS:
        if bid == exclude_bid:
            continue
        raw_scores.append(v3_pipeline_raw(dfs[bid], br, alpha, baseline))
    all_raw = np.concatenate(raw_scores)
    return float(np.percentile(all_raw, 1)), float(np.percentile(all_raw, 99))


def compute_global_scale(dfs: dict, br: float, alpha: float) -> tuple:
    """
    Global (g_min, g_max) derived from ALL four training bearings.

    Used to normalise validation / test HI on the same scale as training HI,
    since those bearings are entirely independent of the training set and
    therefore no LOO step is needed or meaningful.
    """
    baseline   = compute_bearing_baseline(dfs, br, exclude_bid=0)  # all 4 bearings
    raw_scores = [v3_pipeline_raw(dfs[bid], br, alpha, baseline) for bid in BEARING_IDS]
    all_raw    = np.concatenate(raw_scores)
    return float(np.percentile(all_raw, 1)), float(np.percentile(all_raw, 99))


def evaluate_hi(hi_series, cond_series):
    scores = []
    for lbl in [0, 1]:
        idx = cond_series == lbl
        if sum(idx) > 1:
            sub = hi_series[idx]
            scores.append((monotonicity(sub) + trendability(sub)) / 2)
    return float(np.mean(scores)) if scores else 0.0


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(force: bool = False):
    print("=" * 60)
    print("  c2 — Signal Transform + FDR HI Construction")
    print("=" * 60)

    # -- Load features (output of c1) -----------------------------------------
    feat_cols = None
    dfs_raw   = {}
    for b in BEARING_IDS:
        csv = OUT_DIR / f"Bearing{b}_features.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Missing: {csv}  — run c1 first.")
        df  = pd.read_csv(csv)
        if feat_cols is None:
            feat_cols = [c for c in df.columns
                         if c not in ("file_idx", "time_sec", "bearing_id")]
        dfs_raw[b] = df

    # -- Align RPM and condition labels ---------------------------------------
    print("\n[1/3] Aligning RPM and condition labels …")
    conds = {}
    for b in BEARING_IDS:
        op   = load_operation(b)
        n    = len(dfs_raw[b])
        rpm  = align_rpm(op, n)
        cond = discretize_rpm(rpm)
        conds[b] = cond
        n_low  = int((cond == 0).sum())
        n_high = int((cond == 1).sum())
        print(f"  Bearing {b}: low={n_low}, high={n_high}")

    # -- Apply signal transformation (FIX-1 active) ---------------------------
    print("\n[2/3] Applying signal transformation (FIX-1: initial 15% only) …")
    dfs_trans = {}
    for b in BEARING_IDS:
        out_csv = OUT_DIR / f"Bearing{b}_features_transformed.csv"
        if out_csv.exists() and not force:
            print(f"  Bearing {b}: cache → {out_csv.name}")
            df_t = pd.read_csv(out_csv)
        else:
            df_t = apply_signal_transform(dfs_raw[b], conds[b], feat_cols)
            df_t.to_csv(out_csv, index=False)
            print(f"  Bearing {b}: transformed → {out_csv.name}")
        df_t["cond"] = conds[b]
        dfs_trans[b] = df_t

    # -- Grid search: find best (br, alpha) for HI-A v3 ----------------------
    print("\n[3/3] Grid search for best FDR HI-A v3 parameters …")
    br_cands    = [0.05, 0.10, 0.15, 0.20, 0.25]
    alpha_cands = [0.1, 0.2, 0.3]

    best_score, best_params = 0.0, (0.10, 0.2)
    for br, alpha in itertools.product(br_cands, alpha_cands):
        scores = []
        for bid in BEARING_IDS:
            # FIX-2: always use LOO baseline (exclude_bid=bid)
            baseline = compute_bearing_baseline(dfs_trans, br, exclude_bid=bid)
            hi       = v3_pipeline(dfs_trans[bid], br, alpha, baseline)
            scores.append(evaluate_hi(hi, dfs_trans[bid]["cond"].values))
        mean_q = float(np.mean(scores))
        if mean_q > best_score:
            best_score, best_params = mean_q, (br, alpha)

    br_best, alpha_best = best_params
    print(f"  Best: br={br_best}, alpha={alpha_best}, Q={best_score:.4f}")

    # -- Compute and save global scale for test/validation --------------------
    g_min_global, g_max_global = compute_global_scale(
        dfs_trans, br_best, alpha_best)
    pd.DataFrame([{"g_min": g_min_global, "g_max": g_max_global,
                   "br": br_best, "alpha": alpha_best}]).to_csv(
        OUT_DIR / "scale_params.csv", index=False)
    print(f"\n  Global scale (for test/val): g_min={g_min_global:.4f},"
          f" g_max={g_max_global:.4f}")
    print(f"  Saved → {OUT_DIR / 'scale_params.csv'}")

    # -- Save training HI: LOO baseline + LOO-consistent global scale ---------
    #
    # For bearing b:
    #   • FDR baseline  = computed from other 3 bearings  (LOO, FIX-2)
    #   • scale (g_min, g_max) = from raw scores of same other 3 bearings (LOO)
    #
    # → Bearing b's HI normalisation uses NO information from bearing b itself.
    #
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, b in zip(axes.flat, BEARING_IDS):
        baseline_loo  = compute_bearing_baseline(dfs_trans, br_best, exclude_bid=b)
        raw_hi        = v3_pipeline_raw(dfs_trans[b], br_best, alpha_best, baseline_loo)
        g_min_loo, g_max_loo = compute_loo_scale(dfs_trans, br_best, alpha_best,
                                                  exclude_bid=b)
        hi = apply_global_scale(raw_hi, g_min_loo, g_max_loo)

        pd.DataFrame({"HI": hi}).to_csv(OUT_DIR / f"Bearing{b}_HI.csv", index=False)

        ax.plot(hi, "b-", lw=1.5)
        ax.set_xlabel("File index"); ax.set_ylabel("HI")
        ax.set_title(f"Bearing {b}  (mon={monotonicity(hi):.3f},"
                     f" tre={trendability(hi):.3f})")
        ax.grid(alpha=0.3)
        print(f"  Bearing {b}: HI saved  g_min={g_min_loo:.4f}, g_max={g_max_loo:.4f}"
              f"  range=[{hi.min():.3f}, {hi.max():.3f}]")

    plt.suptitle("Training HI (FDR-v3, LOO baseline + LOO-scale)", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_hi.png", dpi=120)
    plt.close()
    print(f"\n  Plot → {OUT_DIR / 'train_hi.png'}")
    print("\nc2 complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cache exists")
    args = parser.parse_args()
    run(force=args.force)
