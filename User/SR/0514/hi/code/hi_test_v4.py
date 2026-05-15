"""
Test HI v4: Train-Anchored Absolute Scaling
============================================
변경점 (v3 대비):
  [Fix] minmax_scale (윈도우 내부 상대 정규화) 제거
    - v3 문제: minmax가 50개 파일 내부 min=0/max=1로 압축
      → Test5/6처럼 이미 열화된 베어링도 시작점이 0으로 찍힘
      → Test2처럼 교대 패턴 신호도 윈도우 내 분포에 따라 방향이 뒤집힘
    - v4 수정: Train 전체 수명 데이터의 FDR score 분포 (p5/p95) 로 절대 스케일링
      → 이미 열화된 베어링은 시작 HI가 0이 아닌 의미 있는 절대값으로 나옴
  [Fix] postprocess_score (test 윈도우 correlation 기반 sign flip) 제거
    - v3 문제: test 윈도우의 국소적 패턴(교대 RPM, spike)에 flip이 오작동
    - v4 수정: Train 데이터 기반 direction (+1/-1) 을 미리 결정해 고정
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
BASE    = "/data/home/ksphm/2026-challenge-KSPHM"
SC_BASE = f"{BASE}/User/SC"
SR_BASE = f"{BASE}/User/SR/0514"

TEST_DIR = os.path.join(BASE, "dataset/Test")

TRAIN_V2_DIR  = os.path.join(SC_BASE, "HI/04142304_signal_transform_v2/output")
TRAIN_SSM_DIR = os.path.join(SC_BASE, "HI/04142304_signal_transform_v2/output")
TEST_FEAT_DIR = os.path.join(SC_BASE, "HI/05072245_signal_transform_v5_test/output")

OUT_DIR   = os.path.join(SR_BASE, "hi/output/test_v4")
OUT_DIR_B = os.path.join(SR_BASE, "hi/output/test_hib_v4")
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(OUT_DIR_B, exist_ok=True)

TEST_IDS     = [1, 2, 3, 4, 5, 6]
FS           = 25600
INTERVAL_SEC = 600
RPM_BOUNDARY = 850
BASELINE_RATIO = 0.10
EMA_ALPHA      = 0.1

# ── HI-A 피처 ─────────────────────────────────────────────────────────
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

# ── HI-B 피처 ─────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════
# RPM 레짐 분류
# ══════════════════════════════════════════════════════════════════
def estimate_rpm_from_vibration(tdms_path: str) -> float:
    f = nptdms.TdmsFile(tdms_path)
    x = f["Vibration"]["CH2"][:]
    nperseg = min(65536, len(x) // 4)
    freqs, psd = welch(x, fs=FS, nperseg=nperseg)
    mask = (freqs >= 8) & (freqs <= 20)
    return float(freqs[mask][np.argmax(psd[mask])] * 60)


def classify_regime_fft(test_id: int) -> np.ndarray:
    tdms_dir   = os.path.join(TEST_DIR, f"Test{test_id}")
    tdms_files = sorted(glob.glob(os.path.join(tdms_dir, "*.tdms")))
    cond = np.zeros(len(tdms_files), dtype=int)
    for k, fpath in enumerate(tdms_files):
        cond[k] = 1 if estimate_rpm_from_vibration(fpath) >= RPM_BOUNDARY else 0
    return cond


# ══════════════════════════════════════════════════════════════════
# Train FDR Baseline (v3와 동일)
# ══════════════════════════════════════════════════════════════════
def compute_train_fdr_baseline() -> dict:
    feat_vals = {(r, f): [] for r in [0, 1] for f in ALL_FEATS}
    for bid in [1, 2, 3, 4]:
        feat_df = pd.read_csv(os.path.join(TRAIN_V2_DIR, f"Bearing{bid}_features_transformed.csv"))
        ssm_df  = pd.read_csv(os.path.join(TRAIN_SSM_DIR, f"Bearing{bid}_SSM_result.csv"))
        n = min(len(feat_df), len(ssm_df))
        feat_df, ssm_df = feat_df.iloc[:n], ssm_df.iloc[:n]
        cond   = (ssm_df["rpm"] >= RPM_BOUNDARY).astype(int).values
        n_base = max(3, int(n * BASELINE_RATIO))
        for regime in [0, 1]:
            base_idx = np.where(cond == regime)[0][:n_base]
            for f in ALL_FEATS:
                if f in feat_df.columns:
                    feat_vals[(regime, f)].extend(feat_df[f].values[base_idx].tolist())
    return {
        (r, f): float(np.mean(feat_vals[(r, f)])) if feat_vals[(r, f)] else 1.0
        for r in [0, 1] for f in ALL_FEATS
    }


def compute_train_fdr_baseline_b() -> dict:
    feat_vals = {(r, f): [] for r in [0, 1] for f in ALL_FEATS_B}
    for bid in [1, 2, 3, 4]:
        feat_df = pd.read_csv(os.path.join(TRAIN_V2_DIR, f"Bearing{bid}_features_transformed.csv"))
        ssm_df  = pd.read_csv(os.path.join(TRAIN_SSM_DIR, f"Bearing{bid}_SSM_result.csv"))
        n = min(len(feat_df), len(ssm_df))
        feat_df, ssm_df = feat_df.iloc[:n], ssm_df.iloc[:n]
        cond   = (ssm_df["rpm"] >= RPM_BOUNDARY).astype(int).values
        n_base = max(3, int(n * BASELINE_RATIO))
        for regime in [0, 1]:
            base_idx = np.where(cond == regime)[0][:n_base]
            for f in ALL_FEATS_B:
                if f in feat_df.columns:
                    feat_vals[(regime, f)].extend(feat_df[f].values[base_idx].tolist())
    return {
        (r, f): float(np.mean(feat_vals[(r, f)])) if feat_vals[(r, f)] else 1.0
        for r in [0, 1] for f in ALL_FEATS_B
    }


# ══════════════════════════════════════════════════════════════════
# [v4 신규] Train 전체 수명 기반 그룹 score 통계
# ══════════════════════════════════════════════════════════════════
def compute_train_group_score_stats(train_baseline: dict) -> dict:
    """
    Train 전체 수명 FDR score로 그룹별 direction / p5 / p95 계산.
    - direction: Train에서 score가 시간에 따라 증가(+1) or 감소(-1)
    - p5, p95  : direction 보정 후 score 분포 (스케일링 기준점)
    Returns: {group_name: {"p5": float, "p95": float, "direction": int}}
    """
    all_scores = {g: [] for g in FEATURE_GROUPS}
    dir_votes  = {g: [] for g in FEATURE_GROUPS}

    for bid in [1, 2, 3, 4]:
        feat_df = pd.read_csv(os.path.join(TRAIN_V2_DIR, f"Bearing{bid}_features_transformed.csv"))
        ssm_df  = pd.read_csv(os.path.join(TRAIN_SSM_DIR, f"Bearing{bid}_SSM_result.csv"))
        n = min(len(feat_df), len(ssm_df))
        feat_df = feat_df.iloc[:n].reset_index(drop=True)
        ssm_df  = ssm_df.iloc[:n].reset_index(drop=True)
        cond = (ssm_df["rpm"] >= RPM_BOUNDARY).astype(int).values

        for gname, feats in FEATURE_GROUPS.items():
            available = [f for f in feats if f in feat_df.columns]
            if not available:
                continue
            mat     = feat_df[available].values
            ratios  = build_feature_ratios_from_train(mat, available, cond, train_baseline)
            weights = np.array([FEATURE_Q[f] for f in available], dtype=float)
            weights /= weights.sum() + 1e-12
            score   = (ratios * weights.reshape(1, -1)).sum(axis=1)

            rho, _ = spearmanr(np.arange(len(score)), score)
            dir_votes[gname].append(+1 if (not np.isnan(rho) and rho >= 0) else -1)
            all_scores[gname].extend(score.tolist())

    stats = {}
    for gname in FEATURE_GROUPS:
        direction = +1 if sum(dir_votes[gname]) >= 0 else -1
        arr = np.array(all_scores[gname]) * direction
        stats[gname] = {
            "p5":        float(np.percentile(arr, 5)),
            "p95":       float(np.percentile(arr, 95)),
            "direction": direction,
        }
        print(f"  HI-A [{gname}]  direction={direction:+d}  "
              f"p5={stats[gname]['p5']:.4f}  p95={stats[gname]['p95']:.4f}")
    return stats


def compute_train_group_score_stats_b(train_baseline_b: dict) -> dict:
    """HI-B용 그룹 score 통계 계산."""
    all_scores = {g: [] for g in FEATURE_GROUPS_B}
    dir_votes  = {g: [] for g in FEATURE_GROUPS_B}

    for bid in [1, 2, 3, 4]:
        feat_df = pd.read_csv(os.path.join(TRAIN_V2_DIR, f"Bearing{bid}_features_transformed.csv"))
        ssm_df  = pd.read_csv(os.path.join(TRAIN_SSM_DIR, f"Bearing{bid}_SSM_result.csv"))
        n = min(len(feat_df), len(ssm_df))
        feat_df = feat_df.iloc[:n].reset_index(drop=True)
        ssm_df  = ssm_df.iloc[:n].reset_index(drop=True)
        cond = (ssm_df["rpm"] >= RPM_BOUNDARY).astype(int).values

        for gname, feats in FEATURE_GROUPS_B.items():
            available = [f for f in feats if f in feat_df.columns]
            if not available:
                continue
            mat     = feat_df[available].values
            ratios  = build_feature_ratios_b(mat, available, cond, train_baseline_b)
            weights = np.array([FEATURE_Q_B[f] for f in available], dtype=float)
            weights /= weights.sum() + 1e-12
            score   = (ratios * weights.reshape(1, -1)).sum(axis=1)

            rho, _ = spearmanr(np.arange(len(score)), score)
            dir_votes[gname].append(+1 if (not np.isnan(rho) and rho >= 0) else -1)
            all_scores[gname].extend(score.tolist())

    stats = {}
    for gname in FEATURE_GROUPS_B:
        direction = +1 if sum(dir_votes[gname]) >= 0 else -1
        arr = np.array(all_scores[gname]) * direction
        stats[gname] = {
            "p5":        float(np.percentile(arr, 5)),
            "p95":       float(np.percentile(arr, 95)),
            "direction": direction,
        }
        print(f"  HI-B [{gname}]  direction={direction:+d}  "
              f"p5={stats[gname]['p5']:.4f}  p95={stats[gname]['p95']:.4f}")
    return stats


# ══════════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════════
def monotonicity(series):
    if len(series) <= 1:
        return 0.0
    diff = np.diff(series)
    return abs(np.sum(diff > 0) - np.sum(diff < 0)) / len(diff)


def trendability(series):
    if len(series) <= 1:
        return 0.0
    rho, _ = spearmanr(np.arange(len(series)), series)
    return abs(rho) if not np.isnan(rho) else 0.0


def moving_average(x, window=7):
    if window <= 1:
        return x.copy()
    pad   = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, np.ones(window) / window, mode="valid")[:len(x)]


def ema_smooth(x, alpha=EMA_ALPHA):
    y    = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def train_anchored_scale(x: np.ndarray, p5: float, p95: float) -> np.ndarray:
    """
    [v4 핵심] Train p5/p95 기준으로 절대 스케일링.
    윈도우 내부 min/max를 쓰지 않으므로 절대 레벨 보존.
    """
    denom = p95 - p5
    if abs(denom) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return np.clip((x - p5) / denom, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════
# FDR ratio 계산 (v3와 동일)
# ══════════════════════════════════════════════════════════════════
def build_feature_ratios_from_train(feat_matrix: np.ndarray,
                                     feature_names: list,
                                     cond: np.ndarray,
                                     train_baseline: dict,
                                     eps: float = 1e-8) -> np.ndarray:
    ratios = np.zeros_like(feat_matrix, dtype=float)
    for regime in [0, 1]:
        idx = np.where(cond == regime)[0]
        if len(idx) == 0:
            continue
        baseline_vec = np.array([train_baseline[(regime, f)] for f in feature_names])
        baseline_vec = np.where(np.abs(baseline_vec) < eps, eps, baseline_vec)
        ratios[idx]  = (feat_matrix[idx] - baseline_vec) / (np.abs(baseline_vec) + eps)
    return ratios


def build_feature_ratios_b(feat_matrix, feature_names, cond, train_baseline_b, eps=1e-8):
    ratios = np.zeros_like(feat_matrix, dtype=float)
    for regime in [0, 1]:
        idx = np.where(cond == regime)[0]
        if len(idx) == 0:
            continue
        baseline_vec = np.array([train_baseline_b[(regime, f)] for f in feature_names])
        baseline_vec = np.where(np.abs(baseline_vec) < eps, eps, baseline_vec)
        ratios[idx]  = (feat_matrix[idx] - baseline_vec) / (np.abs(baseline_vec) + eps)
    return ratios


# ══════════════════════════════════════════════════════════════════
# [v4] 그룹별 HI 계산 (절대 스케일링 + Train-고정 direction)
# ══════════════════════════════════════════════════════════════════
def make_group_hi_fdr(feat_matrix: np.ndarray,
                       feature_names: list,
                       cond: np.ndarray,
                       train_baseline: dict,
                       group_stats: dict) -> np.ndarray:
    """
    [v4 변경]
    - postprocess_score 제거: test 윈도우 correlation 기반 flip 대신
      Train 데이터에서 미리 결정한 direction 적용
    - minmax_scale 제거: train_anchored_scale 로 절대 레벨 보존
    """
    ratios  = build_feature_ratios_from_train(feat_matrix, feature_names, cond, train_baseline)
    weights = np.array([FEATURE_Q[f] for f in feature_names], dtype=float)
    weights /= weights.sum() + 1e-12
    score   = (ratios * weights.reshape(1, -1)).sum(axis=1)

    score = score * group_stats["direction"]
    score = train_anchored_scale(score, group_stats["p5"], group_stats["p95"])
    score = ema_smooth(score)
    return np.clip(score, 0.0, 1.0)


def make_group_hi_fdr_b(feat_matrix: np.ndarray,
                         feature_names: list,
                         cond: np.ndarray,
                         train_baseline_b: dict,
                         group_stats_b: dict) -> np.ndarray:
    ratios  = build_feature_ratios_b(feat_matrix, feature_names, cond, train_baseline_b)
    weights = np.array([FEATURE_Q_B[f] for f in feature_names], dtype=float)
    weights /= weights.sum() + 1e-12
    score   = (ratios * weights.reshape(1, -1)).sum(axis=1)

    score = score * group_stats_b["direction"]
    score = train_anchored_scale(score, group_stats_b["p5"], group_stats_b["p95"])
    score = ema_smooth(score)
    return np.clip(score, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════
# [v4] 파이프라인 (최종 minmax_scale 제거)
# ══════════════════════════════════════════════════════════════════
def v3_pipeline(df: pd.DataFrame, cond: np.ndarray,
                train_baseline: dict, group_score_stats: dict) -> np.ndarray:
    sub_his       = {}
    group_weights = {}
    for gname, feats in FEATURE_GROUPS.items():
        sub_his[gname]       = make_group_hi_fdr(df[feats].values, feats, cond,
                                                  train_baseline, group_score_stats[gname])
        group_weights[gname] = np.mean([FEATURE_Q[f] for f in feats])

    sub_mat  = np.column_stack([sub_his[g] for g in FEATURE_GROUPS])
    w        = np.array([group_weights[g] for g in FEATURE_GROUPS], dtype=float)
    w       /= w.sum() + 1e-12
    final_hi = (sub_mat * w.reshape(1, -1)).sum(axis=1)
    # [v4] minmax_scale 제거 → 절대 레벨 보존
    return np.clip(moving_average(final_hi, 7), 0.0, 1.0)


def hib_pipeline(df: pd.DataFrame, cond: np.ndarray,
                  train_baseline_b: dict, group_score_stats_b: dict) -> np.ndarray:
    sub_his       = {}
    group_weights = {}
    for gname, feats in FEATURE_GROUPS_B.items():
        available = [f for f in feats if f in df.columns]
        if not available:
            continue
        sub_his[gname]       = make_group_hi_fdr_b(df[available].values, available, cond,
                                                    train_baseline_b, group_score_stats_b[gname])
        group_weights[gname] = np.mean([FEATURE_Q_B[f] for f in available])

    if not sub_his:
        return np.zeros(len(df))

    sub_mat  = np.column_stack([sub_his[g] for g in sub_his])
    w        = np.array([group_weights[g] for g in sub_his], dtype=float)
    w       /= w.sum() + 1e-12
    final_hi = (sub_mat * w.reshape(1, -1)).sum(axis=1)
    return np.clip(moving_average(final_hi, 7), 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  Test HI v4: Train-Anchored Absolute Scaling")
    print("=" * 70)

    print("\n[Step 0-A] HI-A Train FDR baseline 계산...")
    train_baseline = compute_train_fdr_baseline()

    print("[Step 0-B] HI-B Train FDR baseline 계산...")
    train_baseline_b = compute_train_fdr_baseline_b()

    print("\n[Step 0-C] HI-A Train group score stats 계산...")
    group_score_stats = compute_train_group_score_stats(train_baseline)

    print("\n[Step 0-D] HI-B Train group score stats 계산...")
    group_score_stats_b = compute_train_group_score_stats_b(train_baseline_b)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Test HI v4 (Train-Anchored Absolute Scaling)", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    summary_rows = []

    for i, tid in enumerate(TEST_IDS):
        print(f"\n[Test{tid}]")

        cache_path = os.path.join(TEST_FEAT_DIR, f"Test{tid}_features.csv")
        df_feat    = pd.read_csv(cache_path)
        print(f"  피처 로드: {len(df_feat)} 슬롯")

        print("  FFT RPM 레짐 분류 중...")
        cond   = classify_regime_fft(tid)
        n_low  = int((cond == 0).sum())
        n_high = int((cond == 1).sum())
        print(f"  레짐: 저속={n_low}, 고속={n_high}")

        n       = min(len(df_feat), len(cond))
        df_feat = df_feat.iloc[:n].reset_index(drop=True)
        cond    = cond[:n]

        hi   = v3_pipeline(df_feat, cond, train_baseline, group_score_stats)
        hi_b = hib_pipeline(df_feat, cond, train_baseline_b, group_score_stats_b)

        pd.DataFrame({"HI": hi}).to_csv(
            os.path.join(OUT_DIR, f"Test{tid}_best.csv"), index=False)
        pd.DataFrame({"HI": hi_b}).to_csv(
            os.path.join(OUT_DIR_B, f"Test{tid}_best.csv"), index=False)

        mon    = monotonicity(hi)
        tred   = trendability(hi)
        q      = (mon + tred) / 2
        mon_b  = monotonicity(hi_b)
        tred_b = trendability(hi_b)
        q_b    = (mon_b + tred_b) / 2
        print(f"  HI-A: start={hi[0]:.3f}  end={hi[-1]:.3f}  "
              f"Mon={mon:.4f}  Tred={tred:.4f}  Q={q:.4f}")
        print(f"  HI-B: start={hi_b[0]:.3f}  end={hi_b[-1]:.3f}  "
              f"Mon={mon_b:.4f}  Tred={tred_b:.4f}  Q={q_b:.4f}")
        summary_rows.append({
            "test_id": tid, "n_files": n,
            "hi_start": round(float(hi[0]), 4), "hi_end": round(float(hi[-1]), 4),
            "mon":      round(mon,  4), "tred":    round(tred, 4), "q_score": round(q, 4),
            "q_score_b": round(q_b, 4),
            "n_low": n_low, "n_high": n_high,
        })

        ax = axes[i]
        t  = np.arange(len(hi)) * INTERVAL_SEC / 3600
        for lbl, col, lname in [(0, "#4C72B0", "Low"), (1, "#DD8452", "High")]:
            idx = cond == lbl
            ax.scatter(t[idx], hi[idx], s=10, color=col, alpha=0.6, label=lname, zorder=3)
        ax.plot(t, hi, color="gray", lw=0.8, alpha=0.4)
        ax.axhline(hi[0], color="red", lw=0.8, ls="--", alpha=0.5, label=f"start={hi[0]:.2f}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Test{tid}  start={hi[0]:.2f}  Q={q:.3f}", fontsize=10)
        ax.set_xlabel("Time [hr]")
        ax.set_ylabel("HI (absolute)")
        ax.legend(fontsize=7)
        ax.grid(True, ls="--", alpha=0.3)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(OUT_DIR, "summary_test_v4.csv"), index=False)

    print("\n" + "=" * 70)
    print("  v3 vs v4 비교 (HI-A 시작점 / Q-score)")
    print("=" * 70)
    v3_path = os.path.join(SR_BASE, "hi/output/test/summary_test_v2.csv")
    if os.path.exists(v3_path):
        v3_sum = pd.read_csv(v3_path)
        print(f"  {'Test':<6} {'v3_start':>10} {'v4_start':>10} "
              f"{'v3_Q':>8} {'v4_Q':>8} {'ΔQ':>8}")
        print("-" * 55)
        for _, r4 in df_summary.iterrows():
            tid = int(r4["test_id"])
            r3  = v3_sum[v3_sum["test_id"] == tid].iloc[0]
            v3_start = pd.read_csv(
                os.path.join(SR_BASE, f"hi/output/test/Test{tid}_best.csv"))["HI"].iloc[0]
            delta = r4["q_score"] - r3["q_score"]
            sign  = "+" if delta >= 0 else ""
            print(f"  Test{tid}  {v3_start:>10.3f}  {r4['hi_start']:>10.3f}  "
                  f"{r3['q_score']:>8.4f}  {r4['q_score']:>8.4f}  {sign}{delta:.4f}")
        v3_avg = v3_sum["q_score"].mean()
        v4_avg = df_summary["q_score"].mean()
        print(f"\n  Overall:  v3={v3_avg:.4f}  →  v4={v4_avg:.4f}  "
              f"(Δ={v4_avg - v3_avg:+.4f})")
    else:
        print(df_summary.to_string(index=False))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Test_HI_v4_all.png"), dpi=150)
    plt.close()
    print(f"\n[완료] {OUT_DIR}")


if __name__ == "__main__":
    main()
