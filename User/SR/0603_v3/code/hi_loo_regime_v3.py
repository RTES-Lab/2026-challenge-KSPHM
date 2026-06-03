"""
HI LOO Regime v3 — F2S2 on both Train and Test
===============================================
v2 대비 변경:
  - raw TDMS 캐시 재사용 (feature extraction 동일)
  - F2S2 변환 추가: high-speed → low-speed 스케일 매핑
  - LOO-aware: 각 fold의 3개 train bearing으로 F2S2 파라미터 추정
  - Test: 전체 4개 train bearing으로 F2S2 파라미터 추정
  - 출력: output/train_f2s2/, output/test_f2s2/
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
from scipy.interpolate import interp1d
import nptdms
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
BASE          = "/data/home/ksphm/2026-challenge-KSPHM"
SR_BASE       = f"{BASE}/User/SR/0603_v3"
DATASET_DIR   = os.path.join(BASE, "dataset")
TEST_DIR      = os.path.join(DATASET_DIR, "Test")

# 캐시는 v2와 공유
CACHE_TRAIN = os.path.join(SR_BASE, "output/train")
CACHE_TEST  = os.path.join(SR_BASE, "output/test")

OUT_TRAIN = os.path.join(SR_BASE, "output/train_f2s2")
OUT_TEST  = os.path.join(SR_BASE, "output/test_f2s2")
os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_TEST,  exist_ok=True)

BEARINGS       = [1, 2, 3, 4]
TEST_IDS       = [1, 2, 3, 4, 5, 6]
FS             = 25600
INTERVAL_SEC   = 600
MEAS_WIN_SEC   = 60
RPM_BOUNDARY   = 850
BASELINE_RATIO = 0.10
EMA_ALPHA      = 0.10
MIN_REGIME_WIN = 5
F2S2_EMA_ALPHA = 0.30   # EMA for F2S2 param estimation
BASELINE_REGIME = 0     # low-speed as baseline
TARGET_REGIME   = 1     # high-speed → map to low-speed scale

FEATURE_GROUPS = {
    "highfreq":  ["ch3_high_band", "ch4_high_band"],
    "energy":    ["ch3_total_power", "ch3_energy", "ch3_rms"],
    "variation": ["ch3_std", "ch3_p2p"],
}
ALL_FEATS = [f for feats in FEATURE_GROUPS.values() for f in feats]

FEATURE_Q_GLOBAL = {
    "ch3_high_band":   0.4316,
    "ch4_high_band":   0.4193,
    "ch3_std":         0.4144,
    "ch3_total_power": 0.4121,
    "ch3_energy":      0.4111,
    "ch3_rms":         0.4111,
    "ch3_p2p":         0.3665,
}


# ══════════════════════════════════════════════════════════════════
# Feature loading (reuse v2 cache)
# ══════════════════════════════════════════════════════════════════
def _extract_one(fpath: str) -> dict:
    f   = nptdms.TdmsFile(fpath)
    grp = f["Vibration"]
    feat = {}
    for ch_name in ["CH3", "CH4"]:
        x  = grp[ch_name][:]
        ch = ch_name.lower()
        feat[f"{ch}_rms"]         = float(np.sqrt(np.mean(x ** 2)))
        feat[f"{ch}_p2p"]         = float(np.max(x) - np.min(x))
        feat[f"{ch}_std"]         = float(np.std(x))
        feat[f"{ch}_energy"]      = float(np.sum(x ** 2))
        nperseg = min(4096, len(x) // 8)
        fv, psd = welch(x, fs=FS, nperseg=nperseg)
        tp = float(np.sum(psd))
        feat[f"{ch}_total_power"] = tp
        idx_hb = (fv >= 3000) & (fv < 8000)
        feat[f"{ch}_high_band"]   = float(np.sum(psd[idx_hb]) / (tp + 1e-12))
    return feat


def extract_train_features(bid: int) -> pd.DataFrame:
    cache = os.path.join(CACHE_TRAIN, f"Bearing{bid}_features_raw.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    tdms_dir = os.path.join(DATASET_DIR, f"Train{bid}_Vibration")
    files    = sorted(glob.glob(os.path.join(tdms_dir, "*.tdms")))
    print(f"    TDMS extracting ({len(files)} files)...")
    rows = []
    for i, fp in enumerate(files):
        row = _extract_one(fp)
        row["file_idx"] = i + 1
        rows.append(row)
        if (i + 1) % 30 == 0:
            print(f"      [{i+1}/{len(files)}]")
    df = pd.DataFrame(rows)[["file_idx"] + ALL_FEATS]
    df.to_csv(cache, index=False)
    return df


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


def load_bearing(bid: int) -> pd.DataFrame:
    feat = extract_train_features(bid)
    n    = len(feat)
    cond = get_train_cond(bid, n)
    df   = feat[ALL_FEATS].copy().reset_index(drop=True)
    df["cond"] = cond
    return df


def extract_test_features(tid: int) -> pd.DataFrame:
    cache = os.path.join(CACHE_TEST, f"Test{tid}_features_raw.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    tdms_dir = os.path.join(TEST_DIR, f"Test{tid}")
    files    = sorted(glob.glob(os.path.join(tdms_dir, "*.tdms")))
    print(f"    TDMS extracting ({len(files)} files)...")
    rows = []
    for i, fp in enumerate(files):
        row = _extract_one(fp)
        row["file_idx"] = i + 1
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"      [{i+1}/{len(files)}]")
    df = pd.DataFrame(rows)[["file_idx"] + ALL_FEATS]
    df.to_csv(cache, index=False)
    return df


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


# ══════════════════════════════════════════════════════════════════
# F2S2 Transform
# ══════════════════════════════════════════════════════════════════
def _ema(x, alpha=F2S2_EMA_ALPHA):
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def estimate_f2s2_params(dfs: dict, train_bids: list, feature: str) -> tuple:
    """
    Estimate linear F2S2 params (a, b):
      target_regime_signal ≈ a * raw + b  (maps to baseline_regime scale)
    Uses pooled regression across train_bids.
    """
    y_p_all, y_bar_p_all = [], []
    for bid in train_bids:
        df   = dfs[bid]
        cond = df["cond"].values
        y    = _ema(df[feature].values.astype(float))
        t    = np.arange(len(y))
        mask_B = (cond == BASELINE_REGIME)
        mask_p = (cond == TARGET_REGIME)
        if not np.any(mask_B) or not np.any(mask_p):
            continue
        t_B, y_B = t[mask_B], y[mask_B]
        t_p, y_p = t[mask_p], y[mask_p]
        f_interp = interp1d(t_B, y_B, kind="linear",
                            bounds_error=False, fill_value="extrapolate")
        y_bar_p = f_interp(t_p)
        y_p_all.extend(y_p.tolist())
        y_bar_p_all.extend(y_bar_p.tolist())
    if len(y_p_all) < 2:
        return 1.0, 0.0
    a, b = np.polyfit(y_p_all, y_bar_p_all, 1)
    return float(a), float(b)


def apply_f2s2_to_df(df: pd.DataFrame, f2s2_params: dict) -> pd.DataFrame:
    """Apply F2S2 transform in-place on a copy. df must have 'cond' column."""
    df_t = df.copy()
    cond = df["cond"].values
    mask = (cond == TARGET_REGIME)
    for feat, (a, b) in f2s2_params.items():
        y = df[feat].values.astype(float)
        y_t = y.copy()
        y_t[mask] = a * y[mask] + b
        df_t[feat] = y_t
    return df_t


def compute_all_f2s2_params(dfs: dict, train_bids: list) -> dict:
    """Estimate F2S2 params for all features from the given train bearings."""
    params = {}
    for feat in ALL_FEATS:
        params[feat] = estimate_f2s2_params(dfs, train_bids, feat)
    return params


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

def moving_average(x, w=7):
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

def train_anchored_scale(x, p5, p95):
    denom = p95 - p5
    if abs(denom) < 1e-12: return np.zeros_like(x, dtype=float)
    return np.clip((x - p5) / denom, 0.0, 1.0)

def fdr_ratios(feat_mat: np.ndarray, baseline: dict, eps: float = 1e-8) -> np.ndarray:
    bvec = np.array([baseline[f] for f in ALL_FEATS])
    bvec = np.where(np.abs(bvec) < eps, eps, bvec)
    return (feat_mat - bvec) / (np.abs(bvec) + eps)


# ══════════════════════════════════════════════════════════════════
# LOO Regime Stats  (identical logic to v2)
# ══════════════════════════════════════════════════════════════════
def compute_regime_stats(dfs: dict, exclude_bid=None) -> dict:
    train_bids = [b for b in BEARINGS if b != exclude_bid]
    result = {}

    for regime in [0, 1]:
        baseline_acc = {f: [] for f in ALL_FEATS}
        for bid in train_bids:
            df    = dfs[bid]
            idx_r = np.where(df["cond"].values == regime)[0]
            n_base = max(3, int(len(idx_r) * BASELINE_RATIO))
            for f in ALL_FEATS:
                baseline_acc[f].extend(df[f].values[idx_r[:n_base]].tolist())
        baseline = {f: float(np.mean(v)) if v else 1.0
                    for f, v in baseline_acc.items()}

        feat_q_votes = {f: [] for f in ALL_FEATS}
        for bid in train_bids:
            df    = dfs[bid]
            idx_r = np.where(df["cond"].values == regime)[0]
            if len(idx_r) < MIN_REGIME_WIN:
                continue
            feat_mat = df[ALL_FEATS].values[idx_r]
            ratios   = fdr_ratios(feat_mat, baseline)
            time_idx = np.arange(len(idx_r))
            for fi, f in enumerate(ALL_FEATS):
                rho, _ = spearmanr(time_idx, np.abs(ratios[:, fi]))
                if not np.isnan(rho):
                    feat_q_votes[f].append(abs(rho))

        feat_q = {f: float(np.mean(v)) if v else FEATURE_Q_GLOBAL[f]
                  for f, v in feat_q_votes.items()}

        group_scores_all = {g: [] for g in FEATURE_GROUPS}
        dir_votes        = {g: [] for g in FEATURE_GROUPS}
        for bid in train_bids:
            df    = dfs[bid]
            idx_r = np.where(df["cond"].values == regime)[0]
            if len(idx_r) < MIN_REGIME_WIN:
                continue
            feat_mat = df[ALL_FEATS].values[idx_r]
            ratios   = fdr_ratios(feat_mat, baseline)
            time_idx = np.arange(len(idx_r))
            for gname, feats in FEATURE_GROUPS.items():
                fidx    = [ALL_FEATS.index(f) for f in feats]
                weights = np.array([feat_q[f] for f in feats])
                weights /= weights.sum() + 1e-12
                score   = (ratios[:, fidx] * weights).sum(axis=1)
                rho, _  = spearmanr(time_idx, score)
                dir_votes[gname].append(+1 if (not np.isnan(rho) and rho >= 0) else -1)
                group_scores_all[gname].extend(score.tolist())

        group_stats = {}
        for gname in FEATURE_GROUPS:
            direction = +1 if sum(dir_votes[gname]) >= 0 else -1
            arr = (np.array(group_scores_all[gname]) * direction
                   if group_scores_all[gname] else np.array([0.0]))
            group_stats[gname] = {
                "direction": direction,
                "p5":  float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
            }

        result[regime] = {
            "baseline":    baseline,
            "feat_q":      feat_q,
            "group_stats": group_stats,
        }

    return result


# ══════════════════════════════════════════════════════════════════
# Apply HI  (identical logic to v2)
# ══════════════════════════════════════════════════════════════════
def apply_regime_hi(feat_mat: np.ndarray, cond: np.ndarray,
                    regime_stats: dict) -> np.ndarray:
    hi = np.zeros(len(feat_mat), dtype=float)
    for regime in [0, 1]:
        idx_r = np.where(cond == regime)[0]
        if len(idx_r) == 0:
            continue
        rs       = regime_stats[regime]
        baseline = rs["baseline"]
        feat_q   = rs["feat_q"]
        gs       = rs["group_stats"]
        ratios   = fdr_ratios(feat_mat[idx_r], baseline)

        sub_his    = {}
        grp_weights = []
        for gname, feats in FEATURE_GROUPS.items():
            fidx    = [ALL_FEATS.index(f) for f in feats]
            weights = np.array([feat_q[f] for f in feats])
            weights /= weights.sum() + 1e-12
            score   = (ratios[:, fidx] * weights).sum(axis=1)
            score   = score * gs[gname]["direction"]
            score   = train_anchored_scale(score, gs[gname]["p5"], gs[gname]["p95"])
            score   = ema_smooth(score)
            sub_his[gname] = np.clip(score, 0.0, 1.0)
            grp_weights.append(np.mean([feat_q[f] for f in feats]))

        gw = np.array(grp_weights)
        gw /= gw.sum() + 1e-12
        sub_mat  = np.column_stack([sub_his[g] for g in FEATURE_GROUPS])
        hi[idx_r] = np.clip((sub_mat * gw).sum(axis=1), 0.0, 1.0)

    return np.clip(moving_average(hi, 7), 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════
def _plot_hi(ax, t, hi, cond, title, q):
    for lbl, col, lname in [(0, "#4C72B0", "Low"), (1, "#DD8452", "High")]:
        idx = cond == lbl
        ax.scatter(t[idx], hi[idx], s=12, color=col, alpha=0.6, label=lname, zorder=3)
    ax.plot(t, hi, color="gray", lw=0.8, alpha=0.4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{title}  Q={q:.3f}", fontsize=10)
    ax.set_xlabel("Time [hr]")
    ax.set_ylabel("HI")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)


def save_individual_hi(t, hi, cond, title, q, fpath):
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
    print("  HI LOO Regime v3 — F2S2 on both Train and Test")
    print("=" * 70)

    print("\n[Loading raw features]")
    dfs = {}
    for bid in BEARINGS:
        print(f"  Bearing{bid}...")
        dfs[bid] = load_bearing(bid)
        n_low  = int((dfs[bid]["cond"] == 0).sum())
        n_high = int((dfs[bid]["cond"] == 1).sum())
        print(f"    n={len(dfs[bid])}, low={n_low}, high={n_high}")

    # ── Train LOO HI with F2S2 ────────────────────────────────────
    print("\n[Train LOO HI — F2S2 transformed]")
    train_rows = []
    fig_tr, axes_tr = plt.subplots(2, 2, figsize=(14, 10))
    fig_tr.suptitle("Train LOO Regime HI (v3 — F2S2 transformed)", fontsize=13, fontweight="bold")
    axes_tr = axes_tr.flatten()

    for i, bid in enumerate(BEARINGS):
        print(f"\n  Bearing{bid}")
        train_bids = [b for b in BEARINGS if b != bid]

        # Estimate F2S2 params from LOO train bearings
        print(f"    Estimating F2S2 params from bearings {train_bids}...")
        f2s2_params = compute_all_f2s2_params(dfs, train_bids)
        for feat in ALL_FEATS[:3]:
            a, b = f2s2_params[feat]
            print(f"      {feat}: a={a:.4f}, b={b:.6f}")

        # Apply F2S2 to all 4 bearings (LOO-aware params)
        dfs_t = {b: apply_f2s2_to_df(dfs[b], f2s2_params) for b in BEARINGS}

        # Compute regime stats from transformed features (LOO)
        rs  = compute_regime_stats(dfs_t, exclude_bid=bid)
        df_t = dfs_t[bid]
        cond = df_t["cond"].values
        hi   = apply_regime_hi(df_t[ALL_FEATS].values, cond, rs)

        pd.DataFrame({"HI": hi, "regime": cond}).to_csv(
            os.path.join(OUT_TRAIN, f"Bearing{bid}_HI.csv"), index=False)

        mon  = monotonicity(hi)
        tred = trendability(hi)
        q    = (mon + tred) / 2

        for regime in [0, 1]:
            idx_r = np.where(cond == regime)[0]
            if len(idx_r) > 1:
                hi_r = hi[idx_r]
                lbl  = "low" if regime == 0 else "high"
                print(f"    [{lbl}] Mon={monotonicity(hi_r):.4f}  "
                      f"Tred={trendability(hi_r):.4f}  "
                      f"Q={(monotonicity(hi_r)+trendability(hi_r))/2:.4f}")
        print(f"    [overall] Mon={mon:.4f}  Tred={tred:.4f}  Q={q:.4f}")

        train_rows.append({
            "bearing": bid,
            "n_total": len(cond),
            "n_low":   int((cond == 0).sum()),
            "n_high":  int((cond == 1).sum()),
            "hi_start": round(float(hi[0]), 4),
            "hi_end":   round(float(hi[-1]), 4),
            "mon":      round(mon, 4),
            "tred":     round(tred, 4),
            "q_score":  round(q, 4),
        })

        t = np.arange(len(hi)) * INTERVAL_SEC / 3600
        save_individual_hi(t, hi, cond, f"Bearing{bid}", q,
                           os.path.join(OUT_TRAIN, f"Bearing{bid}_HI.png"))
        _plot_hi(axes_tr[i], t, hi, cond, f"Bearing{bid}", q)

    df_train_sum = pd.DataFrame(train_rows)
    df_train_sum.to_csv(os.path.join(OUT_TRAIN, "summary.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_TRAIN, "Bearing_LOO_Regime_HI.png"), dpi=150)
    plt.close()
    print(f"\n  Train avg Q: {df_train_sum['q_score'].mean():.4f}")

    # ── Test HI with F2S2 ─────────────────────────────────────────
    print("\n[Test HI — F2S2 params from all 4 train bearings]")

    # F2S2 params from all 4 train bearings
    f2s2_params_all = compute_all_f2s2_params(dfs, BEARINGS)
    print("  F2S2 params (all 4 bearings):")
    for feat in ALL_FEATS[:3]:
        a, b = f2s2_params_all[feat]
        print(f"    {feat}: a={a:.4f}, b={b:.6f}")

    # Transformed train dfs for test regime stats
    dfs_t_all = {b: apply_f2s2_to_df(dfs[b], f2s2_params_all) for b in BEARINGS}
    all_rs = compute_regime_stats(dfs_t_all, exclude_bid=None)

    test_rows = []
    fig_te, axes_te = plt.subplots(2, 3, figsize=(18, 10))
    fig_te.suptitle("Test Regime HI (v3 — F2S2 on both Train and Test)",
                    fontsize=13, fontweight="bold")
    axes_te = axes_te.flatten()

    for i, tid in enumerate(TEST_IDS):
        print(f"\n  Test{tid}")
        feat_df = extract_test_features(tid)[ALL_FEATS].copy().reset_index(drop=True)
        print(f"    FFT regime classification...")
        cond    = classify_regime_fft(tid)
        n       = min(len(feat_df), len(cond))
        feat_df = feat_df.iloc[:n].reset_index(drop=True)
        cond    = cond[:n]

        # Apply F2S2 to test features
        feat_df["cond"] = cond
        feat_df_t = apply_f2s2_to_df(feat_df, f2s2_params_all)
        feat_mat  = feat_df_t[ALL_FEATS].values

        hi = apply_regime_hi(feat_mat, cond, all_rs)
        pd.DataFrame({"HI": hi, "regime": cond}).to_csv(
            os.path.join(OUT_TEST, f"Test{tid}_HI.csv"), index=False)

        mon  = monotonicity(hi)
        tred = trendability(hi)
        q    = (mon + tred) / 2
        print(f"    n={n}, low={(cond==0).sum()}, high={(cond==1).sum()}"
              f"  start={hi[0]:.3f}  end={hi[-1]:.3f}  Q={q:.4f}")

        test_rows.append({
            "test_id": tid, "n_total": n,
            "n_low":   int((cond == 0).sum()),
            "n_high":  int((cond == 1).sum()),
            "hi_start": round(float(hi[0]), 4),
            "hi_end":   round(float(hi[-1]), 4),
            "mon":      round(mon, 4),
            "tred":     round(tred, 4),
            "q_score":  round(q, 4),
        })

        t = np.arange(len(hi)) * INTERVAL_SEC / 3600
        save_individual_hi(t, hi, cond, f"Test{tid}", q,
                           os.path.join(OUT_TEST, f"Test{tid}_HI.png"))
        _plot_hi(axes_te[i], t, hi, cond, f"Test{tid}", q)

    df_test_sum = pd.DataFrame(test_rows)
    df_test_sum.to_csv(os.path.join(OUT_TEST, "summary.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_TEST, "Test_Regime_HI.png"), dpi=150)
    plt.close()

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print("\nTrain LOO (F2S2):")
    print(df_train_sum[["bearing", "n_low", "n_high",
                         "hi_start", "hi_end", "q_score"]].to_string(index=False))
    print(f"\n  Mean Q: {df_train_sum['q_score'].mean():.4f}")
    print("\nTest (F2S2):")
    print(df_test_sum[["test_id", "n_low", "n_high",
                        "hi_start", "hi_end", "q_score"]].to_string(index=False))
    print(f"\n  Mean Q: {df_test_sum['q_score'].mean():.4f}")
    print(f"\n[Done] {OUT_TRAIN}, {OUT_TEST}")


if __name__ == "__main__":
    main()
