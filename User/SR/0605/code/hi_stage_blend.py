"""
HI Stage-Blended v1 (composite_q)
==================================
Extends hi_self_norm_v1.py with a two-track stage-blend:

  hi_smooth_raw  = SMOOTH_FEATS (no kurtosis/crest) — overall degradation
  hi_late_raw    = LATE_FEATS (kurtosis, crest only) — EOL impulse signal
  beta[t]        = sigmoid((t/T - 0.70) * 10)
  hi_blend[t]    = (1 - beta) * hi_smooth + beta * hi_late

For B3, kurtosis explodes at t~82 (of 89 cycles).
With T_actual=89: beta@t82 ≈ 0.90  → late track dominates → hi_end ≈ 1.0
With T_estimated=125.7 (LOO sim): beta@t82 ≈ 0.38  → partial improvement

Two HI sets are output:
  output_cq/train/  — full-lifecycle LOO HI (self-baseline + T_actual), for LSTM training
  output_cq/loocv/  — test-sim LOO HI (fleet-baseline + T_estimated), for honest LOOCV
  output_cq/test/   — test HI (fleet-baseline + T_estimated=116.5)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.signal import welch
import nptdms
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
BASE        = "/data/home/ksphm/2026-challenge-KSPHM"
SR_BASE     = f"{BASE}/User/SR/0605"
DATASET_DIR = os.path.join(BASE, "dataset")
TEST_DIR    = os.path.join(DATASET_DIR, "Test")

V4_TRAIN_CACHE = f"{BASE}/User/SR/0604/output/train"
V4_TEST_CACHE  = f"{BASE}/User/SR/0604/output/test"

OUT_TRAIN = os.path.join(SR_BASE, "output_cq/train")
OUT_LOOCV = os.path.join(SR_BASE, "output_cq/loocv")
OUT_TEST  = os.path.join(SR_BASE, "output_cq/test")
for d in [OUT_TRAIN, OUT_LOOCV, OUT_TEST]:
    os.makedirs(d, exist_ok=True)

BEARINGS       = [1, 2, 3, 4]
TEST_IDS       = [1, 2, 3, 4, 5, 6]
FS             = 25600
INTERVAL_SEC   = 600
MEAS_WIN_SEC   = 60
RPM_BOUNDARY   = 850
BASELINE_RATIO = 0.10
EOL_RATIO      = 0.05
EOL_FLOOR_RATIO = 0.25
EMA_ALPHA      = 0.10
SMOOTH_WIN     = 7

EOL_CYCLES = {1: 126, 2: 114, 3: 89, 4: 137}

# Feature split: smooth (no impulse) vs late (impulse only)
SMOOTH_FEATS = [
    "ch3_high_band", "ch4_high_band",
    "ch3_total_power", "ch3_energy", "ch3_rms",
    "ch3_std", "ch3_p2p",
]
LATE_FEATS = ["ch1_kurt_log", "ch2_kurt_log", "ch1_crest", "ch2_crest"]
ALL_FEATS  = SMOOTH_FEATS + LATE_FEATS

SMOOTH_GROUPS = {
    "highfreq":  ["ch3_high_band", "ch4_high_band"],
    "energy":    ["ch3_total_power", "ch3_energy", "ch3_rms"],
    "variation": ["ch3_std", "ch3_p2p"],
}

LOG_TRANSFORM_FEATS = ["ch3_total_power", "ch3_energy", "ch3_rms"]

# Stage-blend parameters
LATE_FRAC    = 0.33   # fraction of lifecycle used for late_Q computation
BETA_PIVOT   = 0.70   # sigmoid inflection (t/T fraction)
BETA_STEEP   = 10.0
LATE_Q_FLOOR = 0.05

MEAN_TRAIN_LIFE = float(np.mean(list(EOL_CYCLES.values())))  # 116.5


# ══════════════════════════════════════════════════════════════════
# Feature loading (identical to hi_self_norm_v1.py)
# ══════════════════════════════════════════════════════════════════
def _apply_log_transforms(df):
    df = df.copy()
    for f in LOG_TRANSFORM_FEATS:
        if f in df.columns:
            df[f] = np.log1p(df[f])
    return df


def extract_train_features(bid):
    cache = os.path.join(V4_TRAIN_CACHE, f"Bearing{bid}_features_raw.csv")
    if not os.path.exists(cache):
        raise FileNotFoundError(f"Feature cache missing: {cache}")
    return _apply_log_transforms(pd.read_csv(cache))


def extract_test_features(tid):
    cache = os.path.join(V4_TEST_CACHE, f"Test{tid}_features_raw.csv")
    if not os.path.exists(cache):
        raise FileNotFoundError(f"Feature cache missing: {cache}")
    return _apply_log_transforms(pd.read_csv(cache))


def get_train_cond(bid, n):
    op = pd.read_csv(
        os.path.join(DATASET_DIR, f"Train{bid}_Operation.csv"), encoding="cp949"
    )
    op.columns = [c.strip() for c in op.columns]
    op = op.rename(columns={"Time[sec]": "time_sec", "Motor speed[rpm]": "rpm"})
    rpm = np.zeros(n)
    for k in range(n):
        t0, t1 = k * INTERVAL_SEC, k * INTERVAL_SEC + MEAS_WIN_SEC
        mask = (op["time_sec"] >= t0) & (op["time_sec"] < t1)
        vals = op.loc[mask, "rpm"].values
        rpm[k] = vals.mean() if len(vals) > 0 else np.nan
    nans = np.isnan(rpm)
    if nans.any():
        idx = np.arange(n)
        rpm[nans] = np.interp(idx[nans], idx[~nans], rpm[~nans])
    return (rpm >= RPM_BOUNDARY).astype(int)


def classify_regime_fft(test_id):
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


def load_bearing(bid):
    feat = extract_train_features(bid)
    n    = len(feat)
    cond = get_train_cond(bid, n)
    df   = feat[ALL_FEATS].copy().reset_index(drop=True)
    df["cond"] = cond
    return df


# ══════════════════════════════════════════════════════════════════
# Self-baseline
# ══════════════════════════════════════════════════════════════════
def compute_self_baseline(df, baseline_ratio=BASELINE_RATIO, min_obs=5):
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
                baseline[(regime, f)] = float(df[f].mean())
    return baseline


def compute_fleet_baseline(self_baselines, bids=None):
    if bids is None:
        bids = BEARINGS
    fleet_bl = {}
    for regime in [0, 1]:
        for f in ALL_FEATS:
            vals = [self_baselines[b][(regime, f)] for b in bids
                    if (regime, f) in self_baselines[b]]
            fleet_bl[(regime, f)] = float(np.median(vals)) if vals else 1.0
    return fleet_bl


def self_fdr(feat_mat, cond, baseline, feat_list=None, eps=1e-8):
    """FDR using per-regime baseline. feat_list defaults to ALL_FEATS."""
    if feat_list is None:
        feat_list = ALL_FEATS
    ratios = np.zeros_like(feat_mat, dtype=float)
    for regime in [0, 1]:
        idx_r = np.where(cond == regime)[0]
        if len(idx_r) == 0:
            continue
        bvec = np.array([baseline.get((regime, f), 1.0) for f in feat_list])
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

def compute_eol_score(hi_raw, eol_ratio=EOL_RATIO):
    n_eol = max(3, int(len(hi_raw) * eol_ratio))
    return float(np.mean(hi_raw[-n_eol:]))

def calibrate_hi(hi_raw, anchor):
    return np.clip(hi_raw / anchor, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════
# Stage-blend core
# ══════════════════════════════════════════════════════════════════
def beta_blend_weights(n, T_total, pivot=BETA_PIVOT, steepness=BETA_STEEP):
    """Sigmoid ramp: 0 at t/T << pivot, 1 at t/T >> pivot."""
    x = (np.arange(n) / max(T_total, 1.0) - pivot) * steepness
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def blend_hi(hi_smooth_raw, hi_late_raw, T_total):
    b = beta_blend_weights(len(hi_smooth_raw), T_total)
    return (1.0 - b) * hi_smooth_raw + b * hi_late_raw


# ══════════════════════════════════════════════════════════════════
# LOO stats: smooth track
# ══════════════════════════════════════════════════════════════════
def compute_smooth_loo_stats(dfs, self_baselines, exclude_bid=None):
    """Overall Q + group directions for SMOOTH_FEATS only."""
    train_bids = [b for b in BEARINGS if b != exclude_bid]

    feat_q_votes = {f: [] for f in SMOOTH_FEATS}
    for bid in train_bids:
        df   = dfs[bid]
        cond = df["cond"].values
        fidx = [ALL_FEATS.index(f) for f in SMOOTH_FEATS]
        ratios = self_fdr(df[ALL_FEATS].values[:, fidx], cond,
                          self_baselines[bid], feat_list=SMOOTH_FEATS)
        for fi, f in enumerate(SMOOTH_FEATS):
            rho, _ = spearmanr(np.arange(len(ratios)), np.abs(ratios[:, fi]))
            if not np.isnan(rho):
                feat_q_votes[f].append(abs(rho))

    feat_q = {f: float(np.mean(v)) if v else 0.3
              for f, v in feat_q_votes.items()}

    group_stats = {}
    for gname, feats in SMOOTH_GROUPS.items():
        fidx_in_all    = [ALL_FEATS.index(f) for f in feats]
        fidx_in_smooth = [SMOOTH_FEATS.index(f) for f in feats]
        weights = np.array([feat_q[f] for f in feats])
        weights /= weights.sum() + 1e-12

        dir_votes = []
        for bid in train_bids:
            df   = dfs[bid]
            cond = df["cond"].values
            smat = df[ALL_FEATS].values[:, fidx_in_all]
            ratios = self_fdr(smat, cond, self_baselines[bid], feat_list=feats)
            score  = (ratios * weights).sum(axis=1)
            rho, _ = spearmanr(np.arange(len(score)), score)
            dir_votes.append(+1 if (not np.isnan(rho) and rho >= 0) else -1)

        direction = +1 if sum(dir_votes) >= 0 else -1
        group_stats[gname] = {
            "direction": direction,
            "weights":   weights,
            "fidx_in_all": fidx_in_all,
            "feats":     feats,
        }

    return {"smooth_feat_q": feat_q, "smooth_group_stats": group_stats}


# ══════════════════════════════════════════════════════════════════
# LOO stats: late track (last LATE_FRAC of each bearing)
# ══════════════════════════════════════════════════════════════════
def compute_late_loo_stats(dfs, self_baselines, exclude_bid=None,
                            late_frac=LATE_FRAC):
    """late_Q per LATE_FEAT: Q computed on last late_frac of each LOO bearing."""
    train_bids = [b for b in BEARINGS if b != exclude_bid]
    late_q_votes = {f: [] for f in LATE_FEATS}
    overall_q_votes = {f: [] for f in LATE_FEATS}

    for bid in train_bids:
        df   = dfs[bid]
        cond = df["cond"].values
        n    = len(df)
        n_late = max(3, int(n * late_frac))

        fidx_in_all = [ALL_FEATS.index(f) for f in LATE_FEATS]
        late_mat    = df[ALL_FEATS].values[:, fidx_in_all]
        ratios      = self_fdr(late_mat, cond, self_baselines[bid],
                               feat_list=LATE_FEATS)
        abs_ratios  = np.abs(ratios)

        for fi, f in enumerate(LATE_FEATS):
            # overall Q
            rho_all, _ = spearmanr(np.arange(n), abs_ratios[:, fi])
            if not np.isnan(rho_all):
                overall_q_votes[f].append(abs(rho_all))

            # late Q (last late_frac only)
            v_late = abs_ratios[-n_late:, fi]
            q_late = (monotonicity(v_late) + trendability(v_late)) / 2
            late_q_votes[f].append(q_late)

    late_feat_q    = {f: float(np.mean(v)) if v else LATE_Q_FLOOR
                      for f, v in late_q_votes.items()}
    overall_feat_q = {f: float(np.mean(v)) if v else LATE_Q_FLOOR
                      for f, v in overall_q_votes.items()}

    return {"late_feat_q": late_feat_q, "overall_late_feat_q": overall_feat_q}


# ══════════════════════════════════════════════════════════════════
# Apply smooth HI (SMOOTH_FEATS only)
# ══════════════════════════════════════════════════════════════════
def apply_hi_smooth_raw(feat_mat_all, cond, baseline, smooth_loo_stats):
    """Compute weighted-group HI from SMOOTH_FEATS only."""
    feat_q    = smooth_loo_stats["smooth_feat_q"]
    grp_stats = smooth_loo_stats["smooth_group_stats"]

    group_scores = {}
    for gname in SMOOTH_GROUPS:
        gs     = grp_stats[gname]
        fidx   = gs["fidx_in_all"]
        feats  = gs["feats"]
        weights = gs["weights"]
        smat   = feat_mat_all[:, fidx]
        ratios = self_fdr(smat, cond, baseline, feat_list=feats)
        score  = (ratios * weights).sum(axis=1)
        group_scores[gname] = ema_smooth(score * gs["direction"])

    gw = np.array([np.mean([feat_q[f] for f in SMOOTH_GROUPS[g]])
                   for g in SMOOTH_GROUPS])
    gw /= gw.sum() + 1e-12
    sub_mat = np.column_stack([group_scores[g] for g in SMOOTH_GROUPS])
    return moving_average((sub_mat * gw).sum(axis=1), SMOOTH_WIN)


# ══════════════════════════════════════════════════════════════════
# Apply late HI (LATE_FEATS, absolute FDR)
# ══════════════════════════════════════════════════════════════════
def apply_hi_late_raw(feat_mat_all, cond, baseline, late_loo_stats):
    """Compute HI from LATE_FEATS using |FDR|, weighted by late_Q."""
    late_feat_q = late_loo_stats["late_feat_q"]
    fidx_in_all = [ALL_FEATS.index(f) for f in LATE_FEATS]
    late_mat    = feat_mat_all[:, fidx_in_all]
    abs_ratios  = np.abs(self_fdr(late_mat, cond, baseline, feat_list=LATE_FEATS))

    weights = np.array([max(late_feat_q[f], LATE_Q_FLOOR) for f in LATE_FEATS])
    weights /= weights.sum() + 1e-12
    score = (abs_ratios * weights).sum(axis=1)
    return moving_average(ema_smooth(score), SMOOTH_WIN)


# ══════════════════════════════════════════════════════════════════
# Blended EOL helper (for fleet EOL floor computation)
# ══════════════════════════════════════════════════════════════════
def compute_bearing_blended_eol(bid, dfs, self_baselines,
                                 smooth_loo, late_loo, T=None):
    """Compute EOL raw value of blended HI for bearing bid."""
    df   = dfs[bid]
    cond = df["cond"].values
    fmat = df[ALL_FEATS].values
    if T is None:
        T = EOL_CYCLES[bid]
    hi_s = apply_hi_smooth_raw(fmat, cond, self_baselines[bid], smooth_loo)
    hi_l = apply_hi_late_raw(fmat, cond, self_baselines[bid], late_loo)
    hi_b = blend_hi(hi_s, hi_l, T)
    return compute_eol_score(hi_b)


# ══════════════════════════════════════════════════════════════════
# Composite Q diagnostic (not used for weighting)
# ══════════════════════════════════════════════════════════════════
def compute_composite_q_diag(smooth_feat_q, overall_late_feat_q, late_feat_q,
                              dfs, self_baselines, exclude_bid=None):
    """Compute composite_Q per feature for diagnostic printing."""
    train_bids = [b for b in BEARINGS if b != exclude_bid]

    # Prognosability: consistency of EOL |FDR| values across bearings
    prog = {}
    for f in LATE_FEATS:
        fi = ALL_FEATS.index(f)
        eol_vals = []
        for bid in train_bids:
            df   = dfs[bid]
            cond = df["cond"].values
            fmat = df[ALL_FEATS].values[:, [fi]]
            ratio = np.abs(self_fdr(fmat, cond, self_baselines[bid],
                                    feat_list=[f]))
            eol_vals.append(compute_eol_score(ratio[:, 0]))
        mu    = float(np.mean(eol_vals))
        sigma = float(np.std(eol_vals))
        prog[f] = float(np.clip(1 - sigma / (abs(mu) + 1e-8), 0, 1))

    composite = {}
    for f in ALL_FEATS:
        oq = smooth_feat_q.get(f, overall_late_feat_q.get(f, 0.0))
        lq = late_feat_q.get(f, 0.0) if f in LATE_FEATS else oq
        pr = prog.get(f, 0.5)
        composite[f] = 0.4 * oq + 0.4 * lq + 0.2 * pr
    return composite, prog


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


def save_blend_png(t, hi_final, hi_smooth_cal, hi_late_cal, beta, cond,
                   title, q, fpath):
    """Two-panel plot: HI + blend components, and beta trajectory."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.plot(t, hi_final, color="black", lw=1.5, label="HI blend (final)", zorder=4)
    ax.plot(t, hi_smooth_cal, color="#4C72B0", lw=1.0, ls="--",
            alpha=0.7, label="HI smooth", zorder=3)
    ax.plot(t, hi_late_cal, color="#C44E52", lw=1.0, ls=":",
            alpha=0.7, label="HI late", zorder=3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{title}  Q={q:.3f}", fontsize=10)
    ax.set_xlabel("Time [hr]")
    ax.set_ylabel("HI")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)

    ax2 = axes[1]
    ax2.plot(t, beta, color="#2ca02c", lw=1.5)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Beta gate (late-track weight)", fontsize=10)
    ax2.set_xlabel("Time [hr]")
    ax2.set_ylabel("beta")
    ax2.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax2.grid(True, ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  HI Stage-Blended v1 (composite_q)")
    print("=" * 70)
    print(f"  SMOOTH_FEATS ({len(SMOOTH_FEATS)}): {SMOOTH_FEATS}")
    print(f"  LATE_FEATS   ({len(LATE_FEATS)}):  {LATE_FEATS}")
    print(f"  beta pivot={BETA_PIVOT}, steep={BETA_STEEP}, late_frac={LATE_FRAC}")

    # ── Load training data ────────────────────────────────────────
    print("\n[1] Loading training features...")
    dfs = {}
    for bid in BEARINGS:
        dfs[bid] = load_bearing(bid)
        n = len(dfs[bid])
        print(f"  Bearing{bid}: n={n}")

    # ── Self-baselines ────────────────────────────────────────────
    print("\n[2] Computing self-baselines...")
    self_baselines = {}
    for bid in BEARINGS:
        self_baselines[bid] = compute_self_baseline(dfs[bid])

    # ── All-train stats (for test + floor computation) ────────────
    print("\n[3] All-train smooth + late stats...")
    all_smooth = compute_smooth_loo_stats(dfs, self_baselines, exclude_bid=None)
    all_late   = compute_late_loo_stats(dfs, self_baselines, exclude_bid=None)

    print("  late_Q per LATE_FEAT:")
    for f, q in all_late["late_feat_q"].items():
        oq = all_late["overall_late_feat_q"][f]
        print(f"    {f:20s}  overall_Q={oq:.3f}  late_Q={q:.3f}")

    # All-train fleet EOL (for LOOCV test-fold anchor)
    fleet_baseline_all = compute_fleet_baseline(self_baselines, BEARINGS)
    fleet_eol_vals = []
    for bid in BEARINGS:
        df   = dfs[bid]
        cond = df["cond"].values
        fmat = df[ALL_FEATS].values
        hi_s = apply_hi_smooth_raw(fmat, cond, fleet_baseline_all, all_smooth)
        hi_l = apply_hi_late_raw(fmat, cond, fleet_baseline_all, all_late)
        hi_b = blend_hi(hi_s, hi_l, MEAN_TRAIN_LIFE)
        fleet_eol_vals.append(compute_eol_score(hi_b))
    fleet_anchor_all = max(float(np.median(fleet_eol_vals)), 1e-6)
    print(f"\n  Fleet anchor (all-train, T_est): {fleet_anchor_all:.6f}")
    eol_floor_all = EOL_FLOOR_RATIO * fleet_anchor_all

    # ── Training LOO HI (self-baseline + T_actual) ────────────────
    print("\n[4] Train LOO HI (self-baseline + T_actual)...")
    train_rows = []
    fig_tr, axes_tr = plt.subplots(2, 2, figsize=(14, 10))
    fig_tr.suptitle("Train LOO HI — Stage-Blended (v1)", fontsize=13, fontweight="bold")
    axes_tr = axes_tr.flatten()

    for i, bid in enumerate(BEARINGS):
        print(f"\n  Bearing{bid}")
        loo_bids   = [b for b in BEARINGS if b != bid]
        smooth_loo = compute_smooth_loo_stats(dfs, self_baselines, exclude_bid=bid)
        late_loo   = compute_late_loo_stats(dfs, self_baselines, exclude_bid=bid)

        df   = dfs[bid]
        cond = df["cond"].values
        fmat = df[ALL_FEATS].values
        T    = EOL_CYCLES[bid]

        hi_s = apply_hi_smooth_raw(fmat, cond, self_baselines[bid], smooth_loo)
        hi_l = apply_hi_late_raw(fmat, cond, self_baselines[bid], late_loo)
        hi_b = blend_hi(hi_s, hi_l, T)

        # LOO fleet EOL floor
        loo_eol_vals = [compute_bearing_blended_eol(b, dfs, self_baselines,
                                                     smooth_loo, late_loo)
                        for b in loo_bids]
        fleet_eol_loo = max(float(np.median(loo_eol_vals)), 1e-6)
        eol_floor_loo = EOL_FLOOR_RATIO * fleet_eol_loo

        eol_score  = compute_eol_score(hi_b)
        eol_anchor = max(eol_score, eol_floor_loo)
        hi = calibrate_hi(hi_b, eol_anchor)

        print(f"    T_actual={T}  EOL raw={eol_score:.4f}  anchor={eol_anchor:.4f}")

        pd.DataFrame({"HI": hi, "regime": cond}).to_csv(
            os.path.join(OUT_TRAIN, f"Bearing{bid}_HI.csv"), index=False)

        mon  = monotonicity(hi)
        tred = trendability(hi)
        q    = (mon + tred) / 2
        print(f"    hi_start={hi[0]:.4f}  hi_end={hi[-1]:.4f}  Q={q:.4f}")

        train_rows.append({
            "bearing": bid, "n_total": len(cond),
            "hi_start": round(float(hi[0]), 4),
            "hi_end":   round(float(hi[-1]), 4),
            "eol_raw":  round(eol_score, 6),
            "eol_anchor": round(eol_anchor, 6),
            "mon":      round(mon, 4),
            "tred":     round(tred, 4),
            "q_score":  round(q, 4),
        })

        beta = beta_blend_weights(len(hi), T)
        hi_s_cal = calibrate_hi(hi_s, eol_anchor)
        hi_l_cal = calibrate_hi(hi_l, eol_anchor)
        t_hr = np.arange(len(hi)) * INTERVAL_SEC / 3600
        save_blend_png(t_hr, hi, hi_s_cal, hi_l_cal, beta, cond,
                       f"Bearing{bid} (train, T_actual={T})", q,
                       os.path.join(OUT_TRAIN, f"Bearing{bid}_HI.png"))
        _plot_hi(axes_tr[i], t_hr, hi, cond, f"Bearing{bid}", q)

    df_train_sum = pd.DataFrame(train_rows)
    df_train_sum.to_csv(os.path.join(OUT_TRAIN, "summary.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_TRAIN, "Train_HI_all.png"), dpi=150)
    plt.close()
    print(f"\n  Train LOO avg Q: {df_train_sum['q_score'].mean():.4f}")

    # ── LOOCV test-fold HI (fleet-baseline + T_estimated) ─────────
    print("\n[5] LOOCV test-fold HI (fleet-baseline + T_estimated)...")
    loocv_rows = []
    for bid in BEARINGS:
        loo_bids   = [b for b in BEARINGS if b != bid]
        smooth_loo = compute_smooth_loo_stats(dfs, self_baselines, exclude_bid=bid)
        late_loo   = compute_late_loo_stats(dfs, self_baselines, exclude_bid=bid)

        T_est = float(np.mean([EOL_CYCLES[b] for b in loo_bids]))

        fleet_bl_loo = {
            (regime, f): float(np.median([self_baselines[b][(regime, f)]
                                          for b in loo_bids]))
            for regime in [0, 1] for f in ALL_FEATS
        }

        df   = dfs[bid]
        cond = df["cond"].values
        fmat = df[ALL_FEATS].values

        hi_s = apply_hi_smooth_raw(fmat, cond, fleet_bl_loo, smooth_loo)
        hi_l = apply_hi_late_raw(fmat, cond, fleet_bl_loo, late_loo)
        hi_b = blend_hi(hi_s, hi_l, T_est)

        # Fleet anchor from the LOO training bearings (T_actual for them)
        loo_eol_vals = [compute_bearing_blended_eol(b, dfs, self_baselines,
                                                     smooth_loo, late_loo)
                        for b in loo_bids]
        fleet_eol_loo = max(float(np.median(loo_eol_vals)), 1e-6)
        eol_floor_loo = EOL_FLOOR_RATIO * fleet_eol_loo
        eol_raw = compute_eol_score(hi_b)
        anchor  = max(eol_raw, eol_floor_loo)
        hi = calibrate_hi(hi_b, anchor)

        print(f"  Bearing{bid}: T_est={T_est:.1f}  EOL raw={eol_raw:.4f}  "
              f"anchor={anchor:.4f}  hi_end={hi[-1]:.4f}")

        pd.DataFrame({"HI": hi, "regime": cond}).to_csv(
            os.path.join(OUT_LOOCV, f"Bearing{bid}_HI.csv"), index=False)

        mon  = monotonicity(hi)
        tred = trendability(hi)
        q    = (mon + tred) / 2
        loocv_rows.append({
            "bearing": bid, "n_total": len(cond), "T_est": round(T_est, 1),
            "hi_start": round(float(hi[0]), 4), "hi_end": round(float(hi[-1]), 4),
            "eol_raw": round(eol_raw, 6), "mon": round(mon, 4),
            "tred": round(tred, 4), "q_score": round(q, 4),
        })

    df_loocv_sum = pd.DataFrame(loocv_rows)
    df_loocv_sum.to_csv(os.path.join(OUT_LOOCV, "summary.csv"), index=False)
    print(f"  LOOCV test-sim avg Q: {df_loocv_sum['q_score'].mean():.4f}")

    # ── Test HI (fleet-baseline + T_estimated=MEAN_TRAIN_LIFE) ────
    print("\n[6] Test HI (fleet-baseline + T_estimated)...")
    test_rows = []
    fig_te, axes_te = plt.subplots(2, 3, figsize=(18, 10))
    fig_te.suptitle("Test HI — Stage-Blended (fleet-norm, T_est)", fontsize=13)
    axes_te = axes_te.flatten()

    for i, tid in enumerate(TEST_IDS):
        feat_df = extract_test_features(tid)[ALL_FEATS].copy().reset_index(drop=True)
        cond    = classify_regime_fft(tid)
        n       = min(len(feat_df), len(cond))
        feat_df = feat_df.iloc[:n].copy()
        cond    = cond[:n]
        fmat    = feat_df.values

        hi_s = apply_hi_smooth_raw(fmat, cond, fleet_baseline_all, all_smooth)
        hi_l = apply_hi_late_raw(fmat, cond, fleet_baseline_all, all_late)
        hi_b = blend_hi(hi_s, hi_l, MEAN_TRAIN_LIFE)
        hi   = calibrate_hi(hi_b, fleet_anchor_all)

        pd.DataFrame({"HI": hi, "regime": cond}).to_csv(
            os.path.join(OUT_TEST, f"Test{tid}_HI.csv"), index=False)

        mon  = monotonicity(hi)
        tred = trendability(hi)
        q    = (mon + tred) / 2
        print(f"  Test{tid}: n={n}  hi_start={hi[0]:.4f}  hi_end={hi[-1]:.4f}  Q={q:.4f}")

        test_rows.append({
            "test_id": tid, "n_total": n,
            "hi_start": round(float(hi[0]), 4), "hi_end": round(float(hi[-1]), 4),
            "mon": round(mon, 4), "tred": round(tred, 4), "q_score": round(q, 4),
        })

        t_hr = np.arange(n) * INTERVAL_SEC / 3600
        _plot_hi(axes_te[i], t_hr, hi, cond, f"Test{tid}", q)

    df_test_sum = pd.DataFrame(test_rows)
    df_test_sum.to_csv(os.path.join(OUT_TEST, "summary.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_TEST, "Test_HI_all.png"), dpi=150)
    plt.close()

    # ── Composite Q diagnostics ───────────────────────────────────
    print("\n[7] Composite Q diagnostics (all-train)...")
    composite_q, prog = compute_composite_q_diag(
        all_smooth["smooth_feat_q"],
        all_late["overall_late_feat_q"],
        all_late["late_feat_q"],
        dfs, self_baselines, exclude_bid=None,
    )
    print(f"  {'Feature':22s}  {'overall_Q':>9}  {'late_Q':>7}  {'prog':>6}  {'composite_Q':>11}")
    for f in ALL_FEATS:
        oq = all_smooth["smooth_feat_q"].get(f, all_late["overall_late_feat_q"].get(f, 0.0))
        lq = all_late["late_feat_q"].get(f, 0.0) if f in LATE_FEATS else oq
        pr = prog.get(f, 0.0)
        cq = composite_q[f]
        print(f"  {f:22s}  {oq:9.3f}  {lq:7.3f}  {pr:6.3f}  {cq:11.3f}")

    # ── Final summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print("\nTrain LOO HI (self-baseline + T_actual):")
    print(df_train_sum[["bearing","hi_start","hi_end","eol_raw","q_score"]].to_string(index=False))
    print(f"  Mean Q: {df_train_sum['q_score'].mean():.4f}  (baseline: 0.6670)")

    print("\nLOOCV test-sim HI (fleet-baseline + T_estimated):")
    print(df_loocv_sum[["bearing","hi_start","hi_end","T_est","q_score"]].to_string(index=False))
    print(f"  Mean Q: {df_loocv_sum['q_score'].mean():.4f}")

    print("\nTest HI (fleet + T_estimated=116.5):")
    print(df_test_sum[["test_id","hi_start","hi_end","q_score"]].to_string(index=False))
    print(f"  Mean Q: {df_test_sum['q_score'].mean():.4f}")

    print(f"\n[Done]  {OUT_TRAIN}")
    print(f"         {OUT_LOOCV}")
    print(f"         {OUT_TEST}")


if __name__ == "__main__":
    main()
