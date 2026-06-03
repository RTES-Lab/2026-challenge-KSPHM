"""
HI LOO Regime v2 — Leakage-free Per-regime LOO Health Index
============================================================
핵심 변경 (v1 대비):
  1. Train raw features를 TDMS에서 직접 추출 (F2S2 변환 없음)
     → Train/Test 동일 피처 스케일
  2. LOO baseline 유지: 다른 3개 베어링의 레짐별 첫 10% 평균
     → Test가 이미 열화 상태에서 시작해도 기준점이 올바름
     (self-anchored는 test 시작점이 열화 상태일 때 잘못된 기준점 가능성)
  3. 개별 PNG 생성: Bearing{b}_HI.png, Test{tid}_HI.png
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
BASE          = "/data/home/ksphm/2026-challenge-KSPHM"
SR_BASE       = f"{BASE}/User/SR/0603_v3"
DATASET_DIR   = os.path.join(BASE, "dataset")
TEST_DIR      = os.path.join(DATASET_DIR, "Test")
TEST_FEAT_DIR = os.path.join(BASE, "User/SC/HI/05072245_signal_transform_v5_test/output")

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
EMA_ALPHA      = 0.10
MIN_REGIME_WIN = 5

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
# Feature extraction from TDMS
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
    cache = os.path.join(OUT_TRAIN, f"Bearing{bid}_features_raw.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    tdms_dir = os.path.join(DATASET_DIR, f"Train{bid}_Vibration")
    files    = sorted(glob.glob(os.path.join(tdms_dir, "*.tdms")))
    print(f"    TDMS 추출 중 ({len(files)}개 파일)...")
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


def load_test(tid: int) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(TEST_FEAT_DIR, f"Test{tid}_features.csv")
    )[ALL_FEATS].copy().reset_index(drop=True)


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
# LOO Regime Stats
# ══════════════════════════════════════════════════════════════════
def compute_regime_stats(dfs: dict, exclude_bid=None) -> dict:
    """
    exclude_bid 제외 베어링들의 레짐별 첫 10%로 baseline 계산 (pooled mean).
    Q-score, direction, p5/p95도 동일 베어링들에서 계산.
    exclude_bid=None → Test용 (전체 4개 사용).
    """
    train_bids = [b for b in BEARINGS if b != exclude_bid]
    result = {}

    for regime in [0, 1]:
        # 1. Pooled LOO baseline
        baseline_acc = {f: [] for f in ALL_FEATS}
        for bid in train_bids:
            df    = dfs[bid]
            idx_r = np.where(df["cond"].values == regime)[0]
            n_base = max(3, int(len(idx_r) * BASELINE_RATIO))
            for f in ALL_FEATS:
                baseline_acc[f].extend(df[f].values[idx_r[:n_base]].tolist())
        baseline = {f: float(np.mean(v)) if v else 1.0
                    for f, v in baseline_acc.items()}

        # 2. Per-feature Q-score (LOO)
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

        # 3. Group direction + p5/p95 (LOO)
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
# Apply HI
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
    print("  HI LOO Regime v2 — Leakage-free (raw TDMS + LOO baseline)")
    print("=" * 70)

    print("\n[Train loading / TDMS extraction]")
    dfs = {}
    for bid in BEARINGS:
        print(f"  Bearing{bid}...")
        dfs[bid] = load_bearing(bid)
        n_low  = int((dfs[bid]["cond"] == 0).sum())
        n_high = int((dfs[bid]["cond"] == 1).sum())
        print(f"    n={len(dfs[bid])}, low={n_low}, high={n_high}")

    # ── Train LOO HI ──────────────────────────────────────────────
    print("\n[Train LOO HI]")
    train_rows = []
    fig_tr, axes_tr = plt.subplots(2, 2, figsize=(14, 10))
    fig_tr.suptitle("Train LOO Regime HI (v2 — raw features)", fontsize=13, fontweight="bold")
    axes_tr = axes_tr.flatten()

    for i, bid in enumerate(BEARINGS):
        print(f"\n  Bearing{bid}")
        rs   = compute_regime_stats(dfs, exclude_bid=bid)
        df   = dfs[bid]
        cond = df["cond"].values
        hi   = apply_regime_hi(df[ALL_FEATS].values, cond, rs)

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

    # ── Test HI ───────────────────────────────────────────────────
    print("\n[Test HI — all 4 train bearings for baseline/stats]")
    all_rs = compute_regime_stats(dfs, exclude_bid=None)

    test_rows = []
    fig_te, axes_te = plt.subplots(2, 3, figsize=(18, 10))
    fig_te.suptitle("Test Regime HI (v2 — raw features + LOO train baseline)",
                    fontsize=13, fontweight="bold")
    axes_te = axes_te.flatten()

    for i, tid in enumerate(TEST_IDS):
        print(f"\n  Test{tid}")
        feat_df  = load_test(tid)
        print(f"    FFT regime 분류 중...")
        cond     = classify_regime_fft(tid)
        n        = min(len(feat_df), len(cond))
        feat_mat = feat_df[ALL_FEATS].values[:n]
        cond     = cond[:n]

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
    print("\nTrain LOO:")
    print(df_train_sum[["bearing", "n_low", "n_high",
                         "hi_start", "hi_end", "q_score"]].to_string(index=False))
    print(f"\n  Mean Q: {df_train_sum['q_score'].mean():.4f}")
    print("\nTest:")
    print(df_test_sum[["test_id", "n_low", "n_high",
                        "hi_start", "hi_end", "q_score"]].to_string(index=False))
    print(f"\n  Mean Q: {df_test_sum['q_score'].mean():.4f}")
    print(f"\n[Done] {SR_BASE}/output/")


if __name__ == "__main__":
    main()
