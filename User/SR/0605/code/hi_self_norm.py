"""
HI Self-Normalized + Failure-Anchored v1
=========================================
Addresses the two fundamental HI scale problems in the FDR approach:

Problem A — B4 hi_start=0.55:
  FDR uses pooled LOO baseline (B1+B2+B3 healthy mean).
  B4's healthy features differ from that fleet mean → looks "degraded" at t=0.
  Fix: each bearing uses its OWN first-10% data as FDR baseline (self-baseline).
  → B4 self-FDR starts near 0 by construction.

Problem B — B3 hi_end=0.14:
  p5/p95 normalization uses the distribution of all LOO time steps.
  B3's feature changes are small relative to B1/B2/B4 → gets compressed to low HI.
  Fix: normalize each training bearing's HI so its EOL value = 1.0 (failure anchor).
  → All training bearings end at HI≈1.0.

For test inference:
  - Self-baseline from test bearing's own first 10% (min 5 obs)
  - Failure anchor = median of training bearing EOL raw HI values
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kurtosis as scipy_kurtosis
from scipy.signal import welch
import nptdms  # needed for regime classification
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
BASE        = "/data/home/ksphm/2026-challenge-KSPHM"
SR_BASE     = f"{BASE}/User/SR/0605"
DATASET_DIR = os.path.join(BASE, "dataset")
TEST_DIR    = os.path.join(DATASET_DIR, "Test")

# Reuse 0604 feature caches (includes ch1/ch2 impulse features)
V4_TRAIN_CACHE = f"{BASE}/User/SR/0604/output/train"
V4_TEST_CACHE  = f"{BASE}/User/SR/0604/output/test"

OUT_TRAIN = os.path.join(SR_BASE, "output/train")
OUT_TEST  = os.path.join(SR_BASE, "output/test")
os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_TEST,  exist_ok=True)

BEARINGS       = [1, 2, 3, 4]
TEST_IDS       = [1, 2, 3, 4, 5, 6]
FS             = 25600
INTERVAL_SEC   = 600
MEAS_WIN_SEC   = 60
RPM_BOUNDARY   = 850
BASELINE_RATIO = 0.10
EOL_RATIO      = 0.05   # last 5% of bearing data used for failure anchor
EOL_FLOOR_RATIO = 0.25  # min calibration anchor as fraction of fleet anchor
                         # prevents small-degradation bearings (B3) from
                         # saturating at HI=1.0 too early → confuses LSTM
EMA_ALPHA      = 0.10
MIN_REGIME_WIN = 5
SMOOTH_WIN     = 7

# Known end-of-life for each training bearing (cycles)
EOL_CYCLES = {1: 126, 2: 114, 3: 89, 4: 137}

FEATURE_GROUPS = {
    "highfreq":  ["ch3_high_band", "ch4_high_band"],
    "energy":    ["ch3_total_power", "ch3_energy", "ch3_rms"],
    "impulse":   ["ch1_kurt_log", "ch2_kurt_log", "ch1_crest", "ch2_crest"],
    "variation": ["ch3_std", "ch3_p2p"],
}
ALL_FEATS = [f for feats in FEATURE_GROUPS.values() for f in feats]

# Apply log1p post-load to energy features (0604 cache stores raw values)
LOG_TRANSFORM_FEATS = ["ch3_total_power", "ch3_energy", "ch3_rms"]

FEATURE_Q_GLOBAL = {
    "ch3_high_band": 0.43, "ch4_high_band": 0.42,
    "ch3_total_power": 0.45, "ch3_energy": 0.45, "ch3_rms": 0.45,
    "ch1_kurt_log": 0.40, "ch2_kurt_log": 0.40,
    "ch1_crest": 0.38, "ch2_crest": 0.38,
    "ch3_std": 0.41, "ch3_p2p": 0.37,
}


# ══════════════════════════════════════════════════════════════════
# Feature loading (reuse 0604 cache — includes ch1/ch2 impulse features)
# ══════════════════════════════════════════════════════════════════
def _apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to energy features (0604 cache stores raw, v4 used log1p)."""
    df = df.copy()
    for f in LOG_TRANSFORM_FEATS:
        if f in df.columns:
            df[f] = np.log1p(df[f])
    return df


def extract_train_features(bid: int) -> pd.DataFrame:
    cache = os.path.join(V4_TRAIN_CACHE, f"Bearing{bid}_features_raw.csv")
    if not os.path.exists(cache):
        raise FileNotFoundError(f"Feature cache missing: {cache}\n"
                                f"Run 0604/code/hi_loo_regime_v1.py first.")
    return _apply_log_transforms(pd.read_csv(cache))


def extract_test_features(tid: int) -> pd.DataFrame:
    cache = os.path.join(V4_TEST_CACHE, f"Test{tid}_features_raw.csv")
    if not os.path.exists(cache):
        raise FileNotFoundError(f"Feature cache missing: {cache}\n"
                                f"Run 0604/code/hi_loo_regime_v1.py first.")
    return _apply_log_transforms(pd.read_csv(cache))


def get_train_cond(bid: int, n: int) -> np.ndarray:
    op = pd.read_csv(
        os.path.join(DATASET_DIR, f"Train{bid}_Operation.csv"), encoding="cp949"
    )
    op.columns = [c.strip() for c in op.columns]
    op = op.rename(columns={"Time[sec]": "time_sec", "Motor speed[rpm]": "rpm"})
    rpm = np.zeros(n)
    for k in range(n):
        t0   = k * INTERVAL_SEC
        t1   = t0 + MEAS_WIN_SEC
        mask = (op["time_sec"] >= t0) & (op["time_sec"] < t1)
        vals = op.loc[mask, "rpm"].values
        rpm[k] = vals.mean() if len(vals) > 0 else np.nan
    nans = np.isnan(rpm)
    if nans.any():
        idx = np.arange(n)
        rpm[nans] = np.interp(idx[nans], idx[~nans], rpm[~nans])
    return (rpm >= RPM_BOUNDARY).astype(int)


def classify_regime_fft(test_id: int) -> np.ndarray:
    tdms_dir = os.path.join(TEST_DIR, f"Test{test_id}")
    files    = sorted(glob.glob(os.path.join(tdms_dir, "*.tdms")))
    cond = np.zeros(len(files), dtype=int)
    for k, fp in enumerate(files):
        f = nptdms.TdmsFile(fp)
        x = f["Vibration"]["CH2"][:]
        nperseg = min(65536, len(x) // 4)
        freqs, psd = welch(x, fs=FS, nperseg=nperseg)
        mask = (freqs >= 8) & (freqs <= 20)
        rpm  = float(freqs[mask][np.argmax(psd[mask])] * 60)
        cond[k] = 1 if rpm >= RPM_BOUNDARY else 0
    return cond


def load_bearing(bid: int) -> pd.DataFrame:
    feat = extract_train_features(bid)
    n    = len(feat)
    cond = get_train_cond(bid, n)
    df   = feat[ALL_FEATS].copy().reset_index(drop=True)
    df["cond"] = cond
    return df


# ══════════════════════════════════════════════════════════════════
# Self-baseline (key change A: per-bearing own early data)
# ══════════════════════════════════════════════════════════════════
def compute_self_baseline(df: pd.DataFrame, baseline_ratio: float = BASELINE_RATIO,
                          min_obs: int = 5) -> dict:
    """
    Per-bearing, per-regime baseline from bearing's own early observations.
    Each bearing's healthy state is defined by ITSELF, not the fleet average.
    """
    baseline = {}
    for regime in [0, 1]:
        idx_r  = np.where(df["cond"].values == regime)[0]
        n_base = max(min_obs, int(len(idx_r) * baseline_ratio))
        n_base = min(n_base, len(idx_r))
        for f in ALL_FEATS:
            if n_base > 0:
                vals = df[f].values[idx_r[:n_base]]
                baseline[(regime, f)] = float(np.mean(vals))
            else:
                # Fallback: global feature mean
                baseline[(regime, f)] = float(df[f].mean())
    return baseline


def self_fdr(feat_mat: np.ndarray, cond: np.ndarray, baseline: dict,
             eps: float = 1e-8) -> np.ndarray:
    """FDR using bearing's own per-regime baseline."""
    ratios = np.zeros_like(feat_mat, dtype=float)
    for regime in [0, 1]:
        idx_r = np.where(cond == regime)[0]
        if len(idx_r) == 0:
            continue
        bvec = np.array([baseline.get((regime, f), 1.0) for f in ALL_FEATS])
        bvec = np.where(np.abs(bvec) < eps, eps, bvec)
        ratios[idx_r] = (feat_mat[idx_r] - bvec) / (np.abs(bvec) + eps)
    return ratios


# ══════════════════════════════════════════════════════════════════
# Utils
# ══════════════════════════════════════════════════════════════════
def monotonicity(s):
    if len(s) <= 1: return 0.0
    d = np.diff(s)
    return abs(np.sum(d > 0) - np.sum(d < 0)) / len(d)

def trendability(s):
    if len(s) <= 1: return 0.0
    rho, _ = spearmanr(np.arange(len(s)), s)
    return abs(rho) if not np.isnan(rho) else 0.0

def moving_average(x, w=SMOOTH_WIN):
    if w <= 1: return x.copy()
    pad   = w // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, np.ones(w) / w, mode="valid")[:len(x)]

def ema_smooth(x, alpha=EMA_ALPHA):
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


# ══════════════════════════════════════════════════════════════════
# LOO stats: feat_q + per-group direction
# ══════════════════════════════════════════════════════════════════
def compute_loo_stats(dfs: dict, self_baselines: dict, exclude_bid=None) -> dict:
    """
    Compute feat_q and group directions from LOO training bearings.
    No p5/p95 anchoring — only directions and quality scores.
    """
    train_bids = [b for b in BEARINGS if b != exclude_bid]

    # feat_q: Spearman of |self_fdr| vs time, averaged over LOO bearings
    feat_q_votes = {f: [] for f in ALL_FEATS}
    for bid in train_bids:
        df    = dfs[bid]
        cond  = df["cond"].values
        ratios = self_fdr(df[ALL_FEATS].values, cond, self_baselines[bid])
        for fi, f in enumerate(ALL_FEATS):
            rho, _ = spearmanr(np.arange(len(ratios)), np.abs(ratios[:, fi]))
            if not np.isnan(rho):
                feat_q_votes[f].append(abs(rho))
    feat_q = {f: float(np.mean(v)) if v else FEATURE_Q_GLOBAL[f]
              for f, v in feat_q_votes.items()}

    # Per-group direction (vote over LOO bearings)
    group_stats = {}
    for gname, feats in FEATURE_GROUPS.items():
        fidx    = [ALL_FEATS.index(f) for f in feats]
        weights = np.array([feat_q[f] for f in feats])
        weights /= weights.sum() + 1e-12

        dir_votes      = []
        raw_scores_all = {}  # bid → score array (undirected)

        for bid in train_bids:
            df    = dfs[bid]
            cond  = df["cond"].values
            ratios = self_fdr(df[ALL_FEATS].values, cond, self_baselines[bid])
            score  = (ratios[:, fidx] * weights).sum(axis=1)
            raw_scores_all[bid] = score
            rho, _ = spearmanr(np.arange(len(score)), score)
            dir_votes.append(+1 if (not np.isnan(rho) and rho >= 0) else -1)

        direction = +1 if sum(dir_votes) >= 0 else -1
        group_stats[gname] = {
            "direction": direction,
            "feat_q":    feat_q,
            "weights":   weights,
            "fidx":      fidx,
        }

    return {"feat_q": feat_q, "group_stats": group_stats}


# ══════════════════════════════════════════════════════════════════
# Apply HI (raw, unclipped)
# ══════════════════════════════════════════════════════════════════
def apply_hi_raw(feat_mat: np.ndarray, cond: np.ndarray,
                 self_baseline: dict, loo_stats: dict) -> np.ndarray:
    """
    Compute weighted-group HI without failure anchoring.
    Returns a raw (possibly > 1) signal that increases toward failure.
    """
    feat_q     = loo_stats["feat_q"]
    group_stats = loo_stats["group_stats"]
    ratios      = self_fdr(feat_mat, cond, self_baseline)

    group_scores = {}
    for gname, feats in FEATURE_GROUPS.items():
        gs      = group_stats[gname]
        fidx    = gs["fidx"]
        weights = gs["weights"]
        score   = (ratios[:, fidx] * weights).sum(axis=1)
        deg     = ema_smooth(score * gs["direction"])  # positive = degraded
        group_scores[gname] = deg

    # Weight groups by their mean feat_q
    gw = np.array([np.mean([feat_q[f] for f in FEATURE_GROUPS[g]])
                   for g in FEATURE_GROUPS])
    gw /= gw.sum() + 1e-12
    sub_mat  = np.column_stack([group_scores[g] for g in FEATURE_GROUPS])
    hi_raw   = (sub_mat * gw).sum(axis=1)

    return moving_average(hi_raw, SMOOTH_WIN)


# ══════════════════════════════════════════════════════════════════
# Failure anchor computation (key change B)
# ══════════════════════════════════════════════════════════════════
def compute_eol_score(hi_raw: np.ndarray, eol_ratio: float = EOL_RATIO) -> float:
    """Mean raw HI over the last eol_ratio of the signal."""
    n_eol = max(3, int(len(hi_raw) * eol_ratio))
    return float(np.mean(hi_raw[-n_eol:]))


def compute_failure_anchor(dfs: dict, self_baselines: dict,
                           loo_stats: dict, bids=None) -> float:
    """
    Median of EOL raw HI values across training bearings.
    This is the reference: a 'typical bearing' raw HI at death.
    """
    if bids is None:
        bids = BEARINGS
    eol_vals = []
    for bid in bids:
        df      = dfs[bid]
        hi_raw  = apply_hi_raw(df[ALL_FEATS].values, df["cond"].values,
                               self_baselines[bid], loo_stats)
        eol_vals.append(compute_eol_score(hi_raw))
    anchor = float(np.median(eol_vals))
    return max(anchor, 1e-6)


# ══════════════════════════════════════════════════════════════════
# Final calibrated HI
# ══════════════════════════════════════════════════════════════════
def calibrate_hi(hi_raw: np.ndarray, anchor: float) -> np.ndarray:
    """Normalize so that anchor-level raw HI maps to 1.0."""
    return np.clip(hi_raw / anchor, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════
# Fleet-normalized HI for test (preserves absolute lifecycle position)
# ══════════════════════════════════════════════════════════════════
def compute_fleet_baseline(self_baselines: dict, bids=None) -> dict:
    """
    Per-regime MEDIAN of training bearing self-baselines.
    Median is robust to B4-type outliers (high intrinsic vibration).
    A test bearing with normal healthy features will appear near HI=0.
    """
    if bids is None:
        bids = BEARINGS
    fleet_bl = {}
    for regime in [0, 1]:
        for f in ALL_FEATS:
            vals = [self_baselines[b][(regime, f)] for b in bids
                    if (regime, f) in self_baselines[b]]
            fleet_bl[(regime, f)] = float(np.median(vals)) if vals else 1.0
    return fleet_bl


def compute_fleet_failure_anchor(dfs: dict, self_baselines: dict,
                                  fleet_baseline: dict,
                                  loo_stats: dict,
                                  bids=None) -> float:
    """
    For each training bearing, compute fleet-FDR at EOL (using fleet baseline).
    Failure anchor = median of these EOL fleet-FDR values.
    This anchors the test HI so that a bearing at training-EOL feature level → HI ≈ 1.
    """
    if bids is None:
        bids = BEARINGS
    eol_vals = []
    for bid in bids:
        df   = dfs[bid]
        cond = df["cond"].values
        # Compute fleet-FDR for this bearing (using fleet baseline, not self baseline)
        hi_raw = apply_hi_raw(df[ALL_FEATS].values, cond, fleet_baseline, loo_stats)
        eol_vals.append(compute_eol_score(hi_raw))
    return max(float(np.median(eol_vals)), 1e-6)


# ══════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════
def _plot_hi(ax, t, hi, cond, title, q):
    for lbl, col, lname in [(0, "#4C72B0", "Low"), (1, "#DD8452", "High")]:
        idx = cond == lbl
        ax.scatter(t[idx], hi[idx], s=12, color=col, alpha=0.6,
                   label=lname, zorder=3)
    ax.plot(t, hi, color="gray", lw=0.8, alpha=0.4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{title}  Q={q:.3f}", fontsize=10)
    ax.set_xlabel("Time [hr]")
    ax.set_ylabel("HI")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)

def save_hi_png(t, hi, cond, title, q, fpath):
    fig, ax = plt.subplots(figsize=(9, 4))
    _plot_hi(ax, t, hi, cond, title, q)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  HI Self-Normalized + Failure-Anchored v1")
    print("=" * 70)
    print(f"  Features: {ALL_FEATS}")
    print(f"  Self-baseline ratio: {BASELINE_RATIO*100:.0f}%")
    print(f"  EOL anchor ratio:    {EOL_RATIO*100:.0f}% (last N steps)")

    # ── Load training data ────────────────────────────────────────
    print("\n[1] Loading training features...")
    dfs = {}
    for bid in BEARINGS:
        dfs[bid] = load_bearing(bid)
        n = len(dfs[bid])
        n_low  = int((dfs[bid]["cond"] == 0).sum())
        n_high = int((dfs[bid]["cond"] == 1).sum())
        print(f"  Bearing{bid}: n={n}  low={n_low}  high={n_high}")

    # ── Self-baselines (per bearing, per regime) ──────────────────
    print("\n[2] Computing self-baselines...")
    self_baselines = {}
    for bid in BEARINGS:
        self_baselines[bid] = compute_self_baseline(dfs[bid])
        for regime in [0, 1]:
            sample_feat = "ch3_rms"
            b_val = self_baselines[bid][(regime, sample_feat)]
            print(f"  B{bid} regime={regime}  ch3_rms_baseline={b_val:.6f}")

    # ── All-train stats + fleet baseline/anchor (needed for FLOOR_RATIO) ────
    print("\n[3a] Computing all-train stats and fleet failure anchor...")
    all_train_stats = compute_loo_stats(dfs, self_baselines, exclude_bid=None)
    fleet_baseline  = compute_fleet_baseline(self_baselines, BEARINGS)
    fleet_failure_anchor = compute_fleet_failure_anchor(
        dfs, self_baselines, fleet_baseline, all_train_stats, BEARINGS)
    print(f"  Fleet ch3_rms baseline: "
          f"r0={fleet_baseline[(0,'ch3_rms')]:.5f}  r1={fleet_baseline[(1,'ch3_rms')]:.5f}")
    print(f"  Fleet failure anchor: {fleet_failure_anchor:.6f}")
    eol_floor = EOL_FLOOR_RATIO * fleet_failure_anchor
    print(f"  EOL floor ({EOL_FLOOR_RATIO:.0%} of anchor): {eol_floor:.6f}")

    # ── Training LOO HI ───────────────────────────────────────────
    print("\n[3b] Train LOO HI (self-norm + FLOOR_RATIO calibration)...")
    train_rows = []
    fig_tr, axes_tr = plt.subplots(2, 2, figsize=(14, 10))
    fig_tr.suptitle("Train LOO HI — Self-Normalized + Failure-Anchored (v1)",
                    fontsize=13, fontweight="bold")
    axes_tr = axes_tr.flatten()

    for i, bid in enumerate(BEARINGS):
        print(f"\n  Bearing{bid}")
        loo_stats = compute_loo_stats(dfs, self_baselines, exclude_bid=bid)

        df   = dfs[bid]
        cond = df["cond"].values

        # Raw HI (self-FDR based, no clipping)
        hi_raw = apply_hi_raw(df[ALL_FEATS].values, cond,
                               self_baselines[bid], loo_stats)

        # Failure anchor: own EOL raw score, with FLOOR_RATIO to prevent over-amplification
        eol_score  = compute_eol_score(hi_raw)
        eol_anchor = max(eol_score, eol_floor)
        print(f"    EOL raw HI: {eol_score:.4f}  anchor(floored): {eol_anchor:.4f}")

        # Calibrate: each training bearing ends at HI = eol_score/eol_anchor ≤ 1.0
        hi = calibrate_hi(hi_raw, eol_anchor)

        pd.DataFrame({"HI": hi, "regime": cond}).to_csv(
            os.path.join(OUT_TRAIN, f"Bearing{bid}_HI.csv"), index=False)

        mon  = monotonicity(hi)
        tred = trendability(hi)
        q    = (mon + tred) / 2
        print(f"    hi_start={hi[0]:.4f}  hi_end={hi[-1]:.4f}")
        print(f"    Mon={mon:.4f}  Tred={tred:.4f}  Q={q:.4f}")

        # Comparison with old FDR
        old_q = {"1": 0.5728, "2": 0.6387, "3": 0.8038, "4": 0.5011}
        print(f"    OLD Q={old_q[str(bid)]:.4f}  Δ={q-old_q[str(bid)]:+.4f}")

        train_rows.append({
            "bearing": bid, "n_total": len(cond),
            "hi_start": round(float(hi[0]), 4),
            "hi_end":   round(float(hi[-1]), 4),
            "eol_raw":  round(eol_score, 6),
            "mon":      round(mon, 4),
            "tred":     round(tred, 4),
            "q_score":  round(q, 4),
        })

        t = np.arange(len(hi)) * INTERVAL_SEC / 3600
        save_hi_png(t, hi, cond, f"Bearing{bid}", q,
                    os.path.join(OUT_TRAIN, f"Bearing{bid}_HI.png"))
        _plot_hi(axes_tr[i], t, hi, cond, f"Bearing{bid}", q)

    df_train_sum = pd.DataFrame(train_rows)
    df_train_sum.to_csv(os.path.join(OUT_TRAIN, "summary.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_TRAIN, "Train_HI_all.png"), dpi=150)
    plt.close()

    print(f"\n  Train LOO avg Q: {df_train_sum['q_score'].mean():.4f}")
    print(f"  OLD avg Q:       0.6291")

    # ── Fleet failure anchor diagnostic ──────────────────────────
    print("\n[4] Fleet failure anchor diagnostic (already computed in [3a]):")
    for bid in BEARINGS:
        df   = dfs[bid]
        hi_r = apply_hi_raw(df[ALL_FEATS].values, df["cond"].values,
                             fleet_baseline, all_train_stats)
        ev   = compute_eol_score(hi_r)
        hi_cal = calibrate_hi(hi_r, fleet_failure_anchor)
        print(f"  B{bid}: fleet EOL raw={ev:.4f}  "
              f"fleet HI start={hi_cal[0]:.4f}  end={hi_cal[-1]:.4f}")

    # ── Test HI (fleet-normalized) ────────────────────────────────
    # Test features → fleet-FDR → / fleet_failure_anchor
    # This preserves absolute lifecycle position across test bearings.
    print("\n[5] Test HI (fleet-baseline + fleet failure anchor)...")
    test_rows = []
    fig_te, axes_te = plt.subplots(2, 3, figsize=(18, 10))
    fig_te.suptitle("Test HI — Fleet-Normalized (v1)\n"
                    "(fleet baseline, EOL-anchored)",
                    fontsize=13, fontweight="bold")
    axes_te = axes_te.flatten()

    for i, tid in enumerate(TEST_IDS):
        print(f"\n  Test{tid}")
        feat_df = extract_test_features(tid)[ALL_FEATS].copy().reset_index(drop=True)
        cond    = classify_regime_fft(tid)
        n       = min(len(feat_df), len(cond))
        feat_df = feat_df.iloc[:n].copy()
        cond    = cond[:n]
        feat_df["cond"] = cond

        # Use fleet baseline (absolute lifecycle position preserved)
        hi_raw = apply_hi_raw(feat_df[ALL_FEATS].values, cond,
                               fleet_baseline, all_train_stats)
        hi = calibrate_hi(hi_raw, fleet_failure_anchor)

        pd.DataFrame({"HI": hi, "regime": cond}).to_csv(
            os.path.join(OUT_TEST, f"Test{tid}_HI.csv"), index=False)

        mon  = monotonicity(hi)
        tred = trendability(hi)
        q    = (mon + tred) / 2
        print(f"    n={n}  hi_start={hi[0]:.4f}  hi_end={hi[-1]:.4f}  Q={q:.4f}")

        test_rows.append({
            "test_id": tid, "n_total": n,
            "hi_start": round(float(hi[0]), 4),
            "hi_end":   round(float(hi[-1]), 4),
            "mon":      round(mon, 4),
            "tred":     round(tred, 4),
            "q_score":  round(q, 4),
        })

        t = np.arange(len(hi)) * INTERVAL_SEC / 3600
        save_hi_png(t, hi, cond, f"Test{tid}", q,
                    os.path.join(OUT_TEST, f"Test{tid}_HI.png"))
        _plot_hi(axes_te[i], t, hi, cond, f"Test{tid}", q)

    df_test_sum = pd.DataFrame(test_rows)
    df_test_sum.to_csv(os.path.join(OUT_TEST, "summary.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_TEST, "Test_HI_all.png"), dpi=150)
    plt.close()

    # ── Final summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print("\nTrain LOO  (hi_start should be ~0, hi_end should be ~1):")
    print(df_train_sum[["bearing","hi_start","hi_end","eol_raw","q_score"]].to_string(index=False))
    print(f"\n  Mean Q: {df_train_sum['q_score'].mean():.4f}  (OLD: 0.6291)")

    print("\nTest  (hi_start≈life-fraction-consumed, relative to training fleet):")
    print(df_test_sum[["test_id","hi_start","hi_end","q_score"]].to_string(index=False))
    print(f"\n  Mean Q: {df_test_sum['q_score'].mean():.4f}")

    print(f"\n[Done] {OUT_TRAIN}")
    print(f"        {OUT_TEST}")


if __name__ == "__main__":
    main()
