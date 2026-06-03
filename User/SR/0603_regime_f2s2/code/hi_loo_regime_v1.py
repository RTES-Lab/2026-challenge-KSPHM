"""
HI LOO Regime v1 — Self-contained
==================================
TDMS에서 직접 특징 추출 → 레짐 분류 → Per-regime LOO HI 계산.
외부 디렉토리(SC/) 의존 없음. F2S2 변환 없음(regime-specific FDR로 대체).

Train:
  - 특징: dataset/Train{N}_Vibration/*.tdms
  - 레짐: dataset/Train{N}_Operation.csv → RPM 850 기준 이진 분류
  - HI:   LOO (Bearing b 계산 시 b 제외한 3개로 통계 계산)

Test:
  - 특징: dataset/Test/Test{N}/*.tdms
  - 레짐: CH2 FFT 피크 → RPM 추정 → 850 기준 이진 분류
  - HI:   Train 전체 4개로 통계 계산
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

# ── 경로 ───────────────────────────────────────────────────────────────
BASE     = "/data/home/ksphm/2026-challenge-KSPHM"
DATA_DIR = f"{BASE}/dataset"
HERE     = f"{BASE}/User/SR/0603_regime_f2s2"

OUT_TRAIN  = f"{HERE}/output/train"
OUT_TEST   = f"{HERE}/output/test"
CACHE_DIR  = f"{HERE}/output/cache"
for d in [OUT_TRAIN, OUT_TEST, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

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
    "ch1":       ["ch1_rms", "ch1_energy", "ch1_p2p"],
}
ALL_FEATS = [f for feats in FEATURE_GROUPS.values() for f in feats]


# ══════════════════════════════════════════════════════════════════
# 특징 추출 (TDMS)
# ══════════════════════════════════════════════════════════════════
def extract_features_from_file(tdms_path: str) -> dict:
    f   = nptdms.TdmsFile(tdms_path)
    grp = f["Vibration"]
    feat = {}
    for ch_name in ["CH1", "CH2", "CH3", "CH4"]:
        x  = grp[ch_name][:]
        ch = ch_name.lower()
        feat[f"{ch}_rms"]         = float(np.sqrt(np.mean(x ** 2)))
        feat[f"{ch}_std"]         = float(np.std(x))
        feat[f"{ch}_p2p"]         = float(np.max(x) - np.min(x))
        feat[f"{ch}_energy"]      = float(np.sum(x ** 2))
        nperseg      = min(4096, len(x) // 8)
        fv, psd      = welch(x, fs=FS, nperseg=nperseg)
        total_power  = float(np.sum(psd))
        feat[f"{ch}_total_power"] = total_power
        idx_high = (fv >= 3000) & (fv < 8000)
        feat[f"{ch}_high_band"]   = float(np.sum(psd[idx_high]) / (total_power + 1e-12))
    return feat


def extract_features(tdms_files: list) -> pd.DataFrame:
    rows = []
    for i, fpath in enumerate(tdms_files):
        file_idx = int(os.path.splitext(os.path.basename(fpath))[0])
        row = extract_features_from_file(fpath)
        row["file_idx"] = file_idx
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(tdms_files)} 완료")
    return pd.DataFrame(rows).sort_values("file_idx").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# 레짐 분류
# ══════════════════════════════════════════════════════════════════
def classify_regime_train(bid: int, n_files: int) -> np.ndarray:
    op = pd.read_csv(os.path.join(DATA_DIR, f"Train{bid}_Operation.csv"),
                     encoding="cp949")
    op.columns = [c.strip() for c in op.columns]
    op = op.rename(columns={"Time[sec]": "time_sec", "Motor speed[rpm]": "rpm"})
    rpm = np.zeros(n_files)
    for k in range(n_files):
        t0, t1 = k * INTERVAL_SEC, k * INTERVAL_SEC + MEAS_WIN_SEC
        mask = (op["time_sec"] >= t0) & (op["time_sec"] < t1)
        vals = op.loc[mask, "rpm"].values
        rpm[k] = vals.mean() if len(vals) > 0 else np.nan
    nans = np.isnan(rpm)
    if nans.any():
        idx = np.arange(n_files)
        rpm[nans] = np.interp(idx[nans], idx[~nans], rpm[~nans])
    return (rpm >= RPM_BOUNDARY).astype(int)


def classify_regime_test(tdms_files: list) -> np.ndarray:
    cond = np.zeros(len(tdms_files), dtype=int)
    for k, fpath in enumerate(tdms_files):
        f = nptdms.TdmsFile(fpath)
        x = f["Vibration"]["CH2"][:]
        nperseg = min(65536, len(x) // 4)
        freqs, psd = welch(x, fs=FS, nperseg=nperseg)
        mask = (freqs >= 8) & (freqs <= 20)
        rpm  = float(freqs[mask][np.argmax(psd[mask])] * 60)
        cond[k] = 1 if rpm >= RPM_BOUNDARY else 0
    return cond


# ══════════════════════════════════════════════════════════════════
# 캐시 로드 / 저장
# ══════════════════════════════════════════════════════════════════
def load_bearing(bid: int) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f"Bearing{bid}_features.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
    else:
        print(f"  Bearing{bid} 특징 추출 중...")
        tdms_files = sorted(glob.glob(
            os.path.join(DATA_DIR, f"Train{bid}_Vibration", "*.tdms")))
        df = extract_features(tdms_files)
        df.to_csv(cache, index=False)
    cond = classify_regime_train(bid, len(df))
    df   = df[ALL_FEATS].copy().reset_index(drop=True)
    df["cond"] = cond
    return df


def load_test(tid: int) -> tuple:
    """(feat_df: DataFrame, cond: ndarray, tdms_files: list)"""
    cache = os.path.join(CACHE_DIR, f"Test{tid}_features.csv")
    tdms_dir   = os.path.join(DATA_DIR, "Test", f"Test{tid}")
    tdms_files = sorted(glob.glob(os.path.join(tdms_dir, "*.tdms")))
    if os.path.exists(cache):
        df = pd.read_csv(cache)
    else:
        print(f"  Test{tid} 특징 추출 중...")
        df = extract_features(tdms_files)
        df.to_csv(cache, index=False)
    feat_df = df[ALL_FEATS].copy().reset_index(drop=True)
    print(f"  Test{tid} 레짐 분류 중 (FFT)...")
    cond = classify_regime_test(tdms_files)
    n    = min(len(feat_df), len(cond))
    return feat_df.iloc[:n], cond[:n]


# ══════════════════════════════════════════════════════════════════
# 유틸
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


def compute_own_baseline(feat_mat: np.ndarray, cond: np.ndarray) -> dict:
    """
    각 레짐별 자기 자신의 첫 BASELINE_RATIO 윈도우 평균.
    레짐이 없으면 전체 첫 BASELINE_RATIO 사용.
    반환: {0: {feat: val}, 1: {feat: val}}
    """
    own = {}
    for regime in [0, 1]:
        idx_r  = np.where(cond == regime)[0]
        n_base = max(3, int(len(idx_r) * BASELINE_RATIO))
        if len(idx_r) == 0:
            n_base = max(3, int(len(feat_mat) * BASELINE_RATIO))
            base_mat = feat_mat[:n_base]
        else:
            base_mat = feat_mat[idx_r[:n_base]]
        own[regime] = {
            f: float(v) if abs(float(v)) > 1e-8 else 1.0
            for f, v in zip(ALL_FEATS, base_mat.mean(axis=0))
        }
    return own


# ══════════════════════════════════════════════════════════════════
# LOO Per-regime 통계 계산
# ══════════════════════════════════════════════════════════════════
def compute_regime_stats(dfs: dict, exclude_bid=None) -> dict:
    """
    exclude_bid를 제외한 베어링으로 레짐별 통계 계산.
    feat_q / direction / p5 / p95 는 각 베어링의 own baseline 기반 FDR로 계산.
    exclude_bid=None → 전체 사용 (Test용).
    """
    train_bids = [b for b in BEARINGS if b != exclude_bid]
    result = {}

    for regime in [0, 1]:
        # 1. per-feature Q-score: 각 베어링의 own baseline 기준 FDR 사용
        feat_q_votes = {f: [] for f in ALL_FEATS}
        for bid in train_bids:
            df    = dfs[bid]
            cond  = df["cond"].values
            idx_r = np.where(cond == regime)[0]
            if len(idx_r) < MIN_REGIME_WIN:
                continue
            own_bl   = compute_own_baseline(df[ALL_FEATS].values, cond)
            feat_mat = df[ALL_FEATS].values[idx_r]
            ratios   = fdr_ratios(feat_mat, own_bl[regime])
            time_idx = np.arange(len(idx_r))
            for fi, f in enumerate(ALL_FEATS):
                rho, _ = spearmanr(time_idx, np.abs(ratios[:, fi]))
                if not np.isnan(rho):
                    feat_q_votes[f].append(abs(rho))

        feat_q = {}
        for f in ALL_FEATS:
            if feat_q_votes[f]:
                feat_q[f] = float(np.mean(feat_q_votes[f]))
            else:
                votes_all = []
                for bid in train_bids:
                    df  = dfs[bid]
                    arr = df[f].values.astype(float)
                    rho, _ = spearmanr(np.arange(len(arr)), arr)
                    if not np.isnan(rho):
                        votes_all.append(abs(rho))
                feat_q[f] = float(np.mean(votes_all)) if votes_all else 0.4

        # 2. direction / p5 / p95: 각 베어링 own baseline 기준
        group_scores_all = {g: [] for g in FEATURE_GROUPS}
        dir_votes        = {g: [] for g in FEATURE_GROUPS}

        for bid in train_bids:
            df    = dfs[bid]
            cond  = df["cond"].values
            idx_r = np.where(cond == regime)[0]
            if len(idx_r) < MIN_REGIME_WIN:
                continue
            own_bl   = compute_own_baseline(df[ALL_FEATS].values, cond)
            feat_mat = df[ALL_FEATS].values[idx_r]
            ratios   = fdr_ratios(feat_mat, own_bl[regime])
            time_idx = np.arange(len(idx_r))

            for gname, feats in FEATURE_GROUPS.items():
                fidx    = [ALL_FEATS.index(f) for f in feats]
                weights = np.array([feat_q[f] for f in feats])
                weights /= weights.sum() + 1e-12
                score   = (ratios[:, fidx] * weights).sum(axis=1)
                rho, _  = spearmanr(time_idx, score)
                dir_votes[gname].append(
                    +1 if (not np.isnan(rho) and rho >= 0) else -1)
                group_scores_all[gname].extend(score.tolist())

        group_stats = {}
        for gname in FEATURE_GROUPS:
            direction = +1 if sum(dir_votes[gname]) >= 0 else -1
            arr = (np.array(group_scores_all[gname]) * direction
                   if group_scores_all[gname] else np.array([0.0]))
            arr = np.sign(arr) * np.log1p(np.abs(arr))
            group_stats[gname] = {
                "direction": direction,
                "p5":  float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
            }

        result[regime] = {
            "feat_q":      feat_q,
            "group_stats": group_stats,
        }

    return result


# ══════════════════════════════════════════════════════════════════
# Per-regime HI 적용
# ══════════════════════════════════════════════════════════════════
def apply_regime_hi(feat_mat: np.ndarray, cond: np.ndarray,
                    regime_stats: dict) -> np.ndarray:
    """own_baseline은 feat_mat/cond에서 직접 계산 (자기 자신 기준)."""
    own_bl = compute_own_baseline(feat_mat, cond)
    hi = np.zeros(len(feat_mat), dtype=float)

    for regime in [0, 1]:
        idx_r = np.where(cond == regime)[0]
        if len(idx_r) == 0:
            continue
        rs     = regime_stats[regime]
        feat_q = rs["feat_q"]
        gs     = rs["group_stats"]
        ratios = fdr_ratios(feat_mat[idx_r], own_bl[regime])

        sub_his    = {}
        grp_weights = []
        for gname, feats in FEATURE_GROUPS.items():
            fidx    = [ALL_FEATS.index(f) for f in feats]
            weights = np.array([feat_q[f] for f in feats])
            weights /= weights.sum() + 1e-12
            score   = (ratios[:, fidx] * weights).sum(axis=1)
            score   = score * gs[gname]["direction"]
            score   = np.sign(score) * np.log1p(np.abs(score))
            score   = train_anchored_scale(score, gs[gname]["p5"],
                                           gs[gname]["p95"])
            score   = ema_smooth(score)
            sub_his[gname] = np.clip(score, 0.0, 1.0)
            grp_weights.append(np.mean([feat_q[f] for f in feats]))

        gw      = np.array(grp_weights)
        gw     /= gw.sum() + 1e-12
        sub_mat = np.column_stack([sub_his[g] for g in FEATURE_GROUPS])
        hi[idx_r] = np.clip((sub_mat * gw).sum(axis=1), 0.0, 1.0)

    return np.clip(moving_average(hi, 7), 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  HI LOO Regime v1 — Self-contained (TDMS → feature → HI)")
    print("=" * 70)

    print("\n[데이터 로드]")
    dfs = {bid: load_bearing(bid) for bid in BEARINGS}

    # ── Train LOO HI ──────────────────────────────────────────────
    print("\n[Train LOO HI]")
    train_rows = []
    fig_tr, axes_tr = plt.subplots(2, 2, figsize=(14, 10))
    fig_tr.suptitle("Train LOO Regime HI", fontsize=13, fontweight="bold")
    axes_tr = axes_tr.flatten()

    for i, bid in enumerate(BEARINGS):
        print(f"\n  Bearing{bid} (LOO: exclude {bid})")
        regime_stats = compute_regime_stats(dfs, exclude_bid=bid)

        df       = dfs[bid]
        cond     = df["cond"].values
        feat_mat = df[ALL_FEATS].values
        n_low    = int((cond == 0).sum())
        n_high   = int((cond == 1).sum())
        print(f"    windows: total={len(cond)}, low={n_low}, high={n_high}")

        hi  = apply_regime_hi(feat_mat, cond, regime_stats)
        mon = monotonicity(hi)
        trd = trendability(hi)
        q   = (mon + trd) / 2
        print(f"    Mon={mon:.4f}  Tred={trd:.4f}  Q={q:.4f}")

        pd.DataFrame({"HI": hi, "regime": cond}).to_csv(
            os.path.join(OUT_TRAIN, f"Bearing{bid}_HI.csv"), index=False)

        train_rows.append({
            "bearing": bid, "n_total": len(cond),
            "n_low": n_low, "n_high": n_high,
            "hi_start": round(float(hi[0]), 4), "hi_end": round(float(hi[-1]), 4),
            "mon": round(mon, 4), "tred": round(trd, 4), "q_score": round(q, 4),
        })

        ax = axes_tr[i]
        t  = np.arange(len(hi)) * INTERVAL_SEC / 3600
        for lbl, col, lname in [(0, "#4C72B0", "Low"), (1, "#DD8452", "High")]:
            idx = cond == lbl
            ax.scatter(t[idx], hi[idx], s=12, color=col, alpha=0.6,
                       label=lname, zorder=3)
        ax.plot(t, hi, color="gray", lw=0.8, alpha=0.4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Bearing{bid}  Q={q:.3f}", fontsize=10)
        ax.set_xlabel("Time [hr]")
        ax.set_ylabel("HI")
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.3)

    df_tr = pd.DataFrame(train_rows)
    df_tr.to_csv(os.path.join(OUT_TRAIN, "summary.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_TRAIN, "Bearing_LOO_Regime_HI.png"), dpi=150)
    plt.close()
    print(f"\n  Train avg Q: {df_tr['q_score'].mean():.4f}")

    # ── Test HI ───────────────────────────────────────────────────
    print("\n[Test HI — Train 전체 4개 기반 통계]")
    all_regime_stats = compute_regime_stats(dfs, exclude_bid=None)

    test_rows = []
    fig_te, axes_te = plt.subplots(2, 3, figsize=(18, 10))
    fig_te.suptitle("Test Regime HI", fontsize=13, fontweight="bold")
    axes_te = axes_te.flatten()

    for i, tid in enumerate(TEST_IDS):
        print(f"\n  Test{tid}")
        feat_df, cond = load_test(tid)
        feat_mat = feat_df[ALL_FEATS].values
        n_low    = int((cond == 0).sum())
        n_high   = int((cond == 1).sum())
        print(f"    windows: total={len(cond)}, low={n_low}, high={n_high}")

        hi  = apply_regime_hi(feat_mat, cond, all_regime_stats)
        mon = monotonicity(hi)
        trd = trendability(hi)
        q   = (mon + trd) / 2
        print(f"    start={hi[0]:.3f}  end={hi[-1]:.3f}  "
              f"Mon={mon:.4f}  Tred={trd:.4f}  Q={q:.4f}")

        pd.DataFrame({"HI": hi, "regime": cond}).to_csv(
            os.path.join(OUT_TEST, f"Test{tid}_HI.csv"), index=False)

        test_rows.append({
            "test_id": tid, "n_total": len(cond),
            "n_low": n_low, "n_high": n_high,
            "hi_start": round(float(hi[0]), 4), "hi_end": round(float(hi[-1]), 4),
            "mon": round(mon, 4), "tred": round(trd, 4), "q_score": round(q, 4),
        })

        ax = axes_te[i]
        t  = np.arange(len(hi)) * INTERVAL_SEC / 3600
        for lbl, col, lname in [(0, "#4C72B0", "Low"), (1, "#DD8452", "High")]:
            idx = cond == lbl
            ax.scatter(t[idx], hi[idx], s=12, color=col, alpha=0.6,
                       label=lname, zorder=3)
        ax.plot(t, hi, color="gray", lw=0.8, alpha=0.4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Test{tid}  Q={q:.3f}", fontsize=10)
        ax.set_xlabel("Time [hr]")
        ax.set_ylabel("HI")
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.3)

    df_te = pd.DataFrame(test_rows)
    df_te.to_csv(os.path.join(OUT_TEST, "summary.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_TEST, "Test_Regime_HI.png"), dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print("\nTrain LOO:")
    print(df_tr[["bearing", "n_low", "n_high", "hi_start", "hi_end",
                  "q_score"]].to_string(index=False))
    print(f"\n  Mean Q: {df_tr['q_score'].mean():.4f}")
    print("\nTest:")
    print(df_te[["test_id", "n_low", "n_high", "hi_start", "hi_end",
                  "q_score"]].to_string(index=False))
    print(f"\n  Mean Q: {df_te['q_score'].mean():.4f}")
    print(f"\n[완료] {HERE}/output/")


if __name__ == "__main__":
    main()
