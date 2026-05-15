"""
Test HI v3: FFT RPM 레짐 분류 + Train FDR Baseline Transfer
============================================================
변경점 (v2 대비):
  [Bug Fix 1] Signal Transformation(z-score) 제거 → FDR만 사용
    - 이중 정규화 문제 해결: z-score 후 FDR 비율 계산 시 baseline≈0이 되어
      ratio가 1e+8 배로 폭발하던 버그 수정
  [Bug Fix 2] Train baseline 소스 통일
    - 기존: 04140103_initial_pca_result (구형 PCA 피처, 다른 feature 공간)
    - 수정: 04142304_signal_transform_v2 (v5_test와 동일한 raw feature 컬럼 공간)
  [변경] FDR baseline 계산 방식
    - 기존: test 자체 첫 10% → already-degraded test(Test2,6)에서 baseline 오염
    - 수정: Train 4개 베어링의 레짐별 건강 구간 평균 μ → 안정적인 외부 baseline
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

# Train v2 features: v5_test와 동일한 raw feature 컬럼(ch3_rms, ch3_p2p 등) 사용
TRAIN_V2_DIR  = os.path.join(SC_BASE, "HI/04142304_signal_transform_v2/output")
TRAIN_SSM_DIR = os.path.join(SC_BASE, "HI/04142304_signal_transform_v2/output")

# Test features: v5_test 캐시 재사용 (읽기 전용)
TEST_FEAT_DIR = os.path.join(SC_BASE, "HI/05072245_signal_transform_v5_test/output")

# 출력: SR 전용
OUT_DIR = os.path.join(SR_BASE, "hi/output/test")
os.makedirs(OUT_DIR, exist_ok=True)

TEST_IDS     = [1, 2, 3, 4, 5, 6]
FS           = 25600
INTERVAL_SEC = 600
RPM_BOUNDARY = 850

# v3_best 파라미터 (br=0.10: 타이트한 건강구간 기준, LOO Q=0.7311)
BASELINE_RATIO = 0.10
EMA_ALPHA      = 0.1

# ── HI-A: 주파수도메인 위주 ────────────────────────────────────────
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

# ── HI-B: 시간도메인 위주 ──────────────────────────────────────────
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
# Solution 1: FFT RPM 추정 (유지)
# ══════════════════════════════════════════════════════════════════
def estimate_rpm_from_vibration(tdms_path: str) -> float:
    """CH2 PSD의 8~20Hz 피크 → RPM 역산"""
    f = nptdms.TdmsFile(tdms_path)
    x = f["Vibration"]["CH2"][:]
    nperseg = min(65536, len(x) // 4)
    freqs, psd = welch(x, fs=FS, nperseg=nperseg)
    mask = (freqs >= 8) & (freqs <= 20)
    sub_f = freqs[mask]
    sub_psd = psd[mask]
    return float(sub_f[np.argmax(sub_psd)] * 60)


def classify_regime_fft(test_id: int) -> np.ndarray:
    """Test 데이터의 TDMS 파일들에서 FFT RPM 추정 후 850 기준 레짐 분류"""
    tdms_dir = os.path.join(TEST_DIR, f"Test{test_id}")
    tdms_files = sorted(glob.glob(os.path.join(tdms_dir, "*.tdms")))
    cond = np.zeros(len(tdms_files), dtype=int)
    for k, fpath in enumerate(tdms_files):
        rpm = estimate_rpm_from_vibration(fpath)
        cond[k] = 1 if rpm >= RPM_BOUNDARY else 0
    return cond


# ══════════════════════════════════════════════════════════════════
# Train FDR Baseline 계산 (v2 features 사용, v5_test와 동일 feature 공간)
# ══════════════════════════════════════════════════════════════════
def compute_train_fdr_baseline() -> dict:
    """
    Train 1~4 베어링의 레짐별 건강 구간(초기 10%) 평균 μ를 HI-A FDR baseline으로 계산.
    Returns: {(regime, feat): mu}
    """
    feat_vals = {(r, f): [] for r in [0, 1] for f in ALL_FEATS}

    for bid in [1, 2, 3, 4]:
        feat_df = pd.read_csv(
            os.path.join(TRAIN_V2_DIR, f"Bearing{bid}_features_transformed.csv")
        )
        ssm_df = pd.read_csv(
            os.path.join(TRAIN_SSM_DIR, f"Bearing{bid}_SSM_result.csv")
        )
        n = min(len(feat_df), len(ssm_df))
        feat_df = feat_df.iloc[:n]
        ssm_df  = ssm_df.iloc[:n]

        cond   = (ssm_df["rpm"] >= RPM_BOUNDARY).astype(int).values
        n_base = max(3, int(n * BASELINE_RATIO))

        for regime in [0, 1]:
            idx_regime = np.where(cond == regime)[0]
            base_idx   = idx_regime[:n_base]
            if len(base_idx) == 0:
                continue
            for f in ALL_FEATS:
                if f in feat_df.columns:
                    vals = feat_df[f].values[base_idx]
                    feat_vals[(regime, f)].extend(vals.tolist())

    baseline = {}
    for regime in [0, 1]:
        for f in ALL_FEATS:
            vals = feat_vals[(regime, f)]
            baseline[(regime, f)] = float(np.mean(vals)) if vals else 1.0

    return baseline


def compute_train_fdr_baseline_b() -> dict:
    """
    HI-B용 Train FDR baseline (시간도메인 피처: Kurtosis/Crest/RMS/P2P).
    Returns: {(regime, feat): mu}
    """
    feat_vals = {(r, f): [] for r in [0, 1] for f in ALL_FEATS_B}

    for bid in [1, 2, 3, 4]:
        feat_df = pd.read_csv(
            os.path.join(TRAIN_V2_DIR, f"Bearing{bid}_features_transformed.csv")
        )
        ssm_df = pd.read_csv(
            os.path.join(TRAIN_SSM_DIR, f"Bearing{bid}_SSM_result.csv")
        )
        n = min(len(feat_df), len(ssm_df))
        feat_df = feat_df.iloc[:n]
        ssm_df  = ssm_df.iloc[:n]

        cond   = (ssm_df["rpm"] >= RPM_BOUNDARY).astype(int).values
        n_base = max(3, int(n * BASELINE_RATIO))

        for regime in [0, 1]:
            idx_regime = np.where(cond == regime)[0]
            base_idx   = idx_regime[:n_base]
            if len(base_idx) == 0:
                continue
            for f in ALL_FEATS_B:
                if f in feat_df.columns:
                    vals = feat_df[f].values[base_idx]
                    feat_vals[(regime, f)].extend(vals.tolist())

    baseline = {}
    for regime in [0, 1]:
        for f in ALL_FEATS_B:
            vals = feat_vals[(regime, f)]
            baseline[(regime, f)] = float(np.mean(vals)) if vals else 1.0

    return baseline


# ══════════════════════════════════════════════════════════════════
# FDR 파이프라인 (Train baseline 외부 주입 방식)
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
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, np.ones(window) / window, mode="valid")[:len(x)]


def minmax_scale(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def robust_clip(x, low_q=0.01, high_q=0.99):
    return np.clip(x, np.quantile(x, low_q), np.quantile(x, high_q))


def ema_smooth(x, alpha=EMA_ALPHA):
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def build_feature_ratios_from_train(feat_matrix: np.ndarray,
                                     feature_names: list,
                                     cond: np.ndarray,
                                     train_baseline: dict,
                                     eps: float = 1e-8) -> np.ndarray:
    """
    레짐별 Train baseline μ를 사용하여 FDR 비율 계산.
    ratio = (x - mu_train) / |mu_train|
    test 자체 baseline 사용 없음 → already-degraded test(Test2,6) 문제 해결.
    """
    ratios = np.zeros_like(feat_matrix, dtype=float)
    for regime in [0, 1]:
        idx = np.where(cond == regime)[0]
        if len(idx) == 0:
            continue
        baseline_vec = np.array([train_baseline[(regime, f)] for f in feature_names])
        baseline_vec = np.where(np.abs(baseline_vec) < eps, eps, baseline_vec)
        ratios[idx] = (feat_matrix[idx] - baseline_vec) / (np.abs(baseline_vec) + eps)
    return ratios


def postprocess_score(score):
    score = robust_clip(score, 0.01, 0.99)
    corr = np.corrcoef(np.arange(len(score)), score)[0, 1]
    if not np.isnan(corr) and corr < 0:
        score = -score
    return score


def make_group_hi_fdr(feat_matrix: np.ndarray,
                       feature_names: list,
                       cond: np.ndarray,
                       train_baseline: dict) -> np.ndarray:
    ratios  = build_feature_ratios_from_train(feat_matrix, feature_names, cond, train_baseline)
    weights = np.array([FEATURE_Q[f] for f in feature_names], dtype=float)
    weights = weights / (weights.sum() + 1e-12)
    score   = (ratios * weights.reshape(1, -1)).sum(axis=1)
    score   = postprocess_score(score)
    score   = minmax_scale(score)
    score   = ema_smooth(score)
    return minmax_scale(score)


def v3_pipeline(df: pd.DataFrame, cond: np.ndarray, train_baseline: dict) -> np.ndarray:
    """
    HI-A FDR Group Weight 파이프라인 (주파수도메인 위주).
    cond: 슬롯별 레짐 라벨 (0=저속, 1=고속)
    train_baseline: compute_train_fdr_baseline() 반환값
    """
    sub_his       = {}
    group_weights = {}
    for gname, feats in FEATURE_GROUPS.items():
        mat    = df[feats].values
        sub_hi = make_group_hi_fdr(mat, feats, cond, train_baseline)
        sub_his[gname]       = sub_hi
        group_weights[gname] = np.mean([FEATURE_Q[f] for f in feats])

    sub_mat = np.column_stack([sub_his[g] for g in FEATURE_GROUPS])
    w = np.array([group_weights[g] for g in FEATURE_GROUPS], dtype=float)
    w = w / (w.sum() + 1e-12)
    final_hi = (sub_mat * w.reshape(1, -1)).sum(axis=1)
    return moving_average(minmax_scale(final_hi), 7)


def build_feature_ratios_b(feat_matrix, feature_names, cond, train_baseline_b, eps=1e-8):
    """HI-B용 FDR 비율 계산."""
    ratios = np.zeros_like(feat_matrix, dtype=float)
    for regime in [0, 1]:
        idx = np.where(cond == regime)[0]
        if len(idx) == 0:
            continue
        baseline_vec = np.array([train_baseline_b[(regime, f)] for f in feature_names])
        baseline_vec = np.where(np.abs(baseline_vec) < eps, eps, baseline_vec)
        ratios[idx] = (feat_matrix[idx] - baseline_vec) / (np.abs(baseline_vec) + eps)
    return ratios


def hib_pipeline(df: pd.DataFrame, cond: np.ndarray, train_baseline_b: dict) -> np.ndarray:
    """
    HI-B FDR Group Weight 파이프라인 (시간도메인: Kurtosis/Crest/RMS/P2P 위주).
    train_baseline_b: compute_train_fdr_baseline_b() 반환값
    """
    sub_his       = {}
    group_weights = {}
    for gname, feats in FEATURE_GROUPS_B.items():
        available = [f for f in feats if f in df.columns]
        if not available:
            continue
        mat = df[available].values
        ratios = build_feature_ratios_b(mat, available, cond, train_baseline_b)
        weights = np.array([FEATURE_Q_B[f] for f in available], dtype=float)
        weights = weights / (weights.sum() + 1e-12)
        score = (ratios * weights.reshape(1, -1)).sum(axis=1)
        score = postprocess_score(score)
        score = minmax_scale(score)
        score = ema_smooth(score)
        sub_his[gname]       = minmax_scale(score)
        group_weights[gname] = np.mean([FEATURE_Q_B[f] for f in available])

    if not sub_his:
        return np.zeros(len(df))

    sub_mat = np.column_stack([sub_his[g] for g in sub_his])
    w = np.array([group_weights[g] for g in sub_his], dtype=float)
    w = w / (w.sum() + 1e-12)
    final_hi = (sub_mat * w.reshape(1, -1)).sum(axis=1)
    return moving_average(minmax_scale(final_hi), 7)


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  Test HI v3: FFT RPM 레짐분류 + Train FDR Baseline Transfer")
    print("  (HI-A: 주파수도메인, HI-B: 시간도메인)")
    print("=" * 70)

    # Step 0-A: HI-A Train FDR baseline 계산
    print("\n[Step 0-A] HI-A Train FDR baseline 계산...")
    train_baseline = compute_train_fdr_baseline()

    # Step 0-B: HI-B Train FDR baseline 계산
    print("[Step 0-B] HI-B Train FDR baseline 계산...")
    train_baseline_b = compute_train_fdr_baseline_b()

    # HI-B 출력 디렉토리
    out_dir_b = os.path.join(SR_BASE, "hi/output/test_hib")
    os.makedirs(out_dir_b, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Test HI v3 (FFT RPM + Train FDR Baseline Transfer)", fontsize=14,
                 fontweight="bold")
    axes = axes.flatten()

    summary_rows = []

    for i, tid in enumerate(TEST_IDS):
        print(f"\n[Test{tid}]")

        # 1. 피처 로드 (v5_test 캐시)
        cache_path = os.path.join(TEST_FEAT_DIR, f"Test{tid}_features.csv")
        df_feat    = pd.read_csv(cache_path)
        print(f"  피처 로드: {cache_path} ({len(df_feat)} 슬롯)")

        # 2. FFT RPM 기반 레짐 분류
        print("  FFT RPM 레짐 분류 중...")
        cond   = classify_regime_fft(tid)
        n_low  = int((cond == 0).sum())
        n_high = int((cond == 1).sum())
        print(f"  레짐: 저속={n_low}, 고속={n_high}")

        # 길이 불일치 보정 (TDMS 수 vs 피처 행 수)
        n = min(len(df_feat), len(cond))
        df_feat = df_feat.iloc[:n].reset_index(drop=True)
        cond    = cond[:n]

        # 3-A. HI-A FDR 파이프라인
        hi = v3_pipeline(df_feat, cond, train_baseline)

        # 3-B. HI-B FDR 파이프라인
        hi_b = hib_pipeline(df_feat, cond, train_baseline_b)

        # 4. 저장
        out_path = os.path.join(OUT_DIR, f"Test{tid}_best.csv")
        pd.DataFrame({"HI": hi}).to_csv(out_path, index=False)
        print(f"  HI-A 저장: {out_path}")

        out_path_b = os.path.join(out_dir_b, f"Test{tid}_best.csv")
        pd.DataFrame({"HI": hi_b}).to_csv(out_path_b, index=False)
        print(f"  HI-B 저장: {out_path_b}")

        # 5. 품질 평가
        mon  = monotonicity(hi)
        tred = trendability(hi)
        q    = (mon + tred) / 2
        mon_b  = monotonicity(hi_b)
        tred_b = trendability(hi_b)
        q_b    = (mon_b + tred_b) / 2
        print(f"  HI-A: Mon={mon:.4f}, Tred={tred:.4f}, Q={q:.4f}")
        print(f"  HI-B: Mon={mon_b:.4f}, Tred={tred_b:.4f}, Q={q_b:.4f}")
        summary_rows.append({
            "test_id": tid, "n_files": len(df_feat),
            "mon":     round(mon,  4),
            "tred":    round(tred, 4),
            "q_score": round(q,    4),
            "q_score_b": round(q_b, 4),
            "n_low":   n_low,
            "n_high":  n_high,
        })

        # 6. 시각화
        ax = axes[i]
        t  = np.arange(len(hi)) * INTERVAL_SEC / 3600
        for lbl, col, lname in [(0, "#4C72B0", "Low(저속)"),
                                  (1, "#DD8452", "High(고속)")]:
            idx = cond == lbl
            ax.scatter(t[idx], hi[idx], s=10, color=col, alpha=0.6,
                       label=lname, zorder=3)
        ax.plot(t, hi, color="gray", lw=0.8, alpha=0.4)
        ax.set_title(f"Test{tid}  Mon={mon:.3f} Tred={tred:.3f} Q={q:.3f}", fontsize=10)
        ax.set_xlabel("Time [hr]")
        ax.set_ylabel("HI")
        ax.legend(fontsize=7)
        ax.grid(True, ls="--", alpha=0.3)

    # 요약 저장
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(OUT_DIR, "summary_test_v2.csv"), index=False)

    # 이전 버전(v2)과 비교
    old_summary_path = os.path.join(
        SC_BASE, "HI/05072245_signal_transform_v5_test/output/summary_test.csv"
    )
    print("\n" + "=" * 70)
    if os.path.exists(old_summary_path):
        old_summary = pd.read_csv(old_summary_path)
        print("  v1(K-Means+SignalTransform) vs v3(FFT+FDR Only) 비교")
        print("=" * 70)
        print(f"  {'Test':<6} {'v1_Q':>8} {'v3_Q':>8} {'Delta':>8} "
              f"{'Regime(v1)':>12} {'Regime(v3)':>12}")
        print("-" * 65)
        for _, r1 in old_summary.iterrows():
            tid = int(r1["test_id"])
            r3  = df_summary[df_summary["test_id"] == tid].iloc[0]
            delta = r3["q_score"] - r1["q_score"]
            sign  = "+" if delta >= 0 else ""
            print(f"  Test{tid} {r1['q_score']:>8.4f} {r3['q_score']:>8.4f} "
                  f"{sign}{delta:>7.4f}   {r1['n_low']}/{r1['n_high']:>3}"
                  f"         {r3['n_low']}/{r3['n_high']:>3}")
        v1_avg = old_summary["q_score"].mean()
        v3_avg = df_summary["q_score"].mean()
        print(f"\n  Overall:  v1={v1_avg:.4f}  →  v3={v3_avg:.4f}  "
              f"(Δ={v3_avg - v1_avg:+.4f})")
    else:
        print("  이전 요약 파일 없음, 현재 결과만 출력")
        print(df_summary.to_string(index=False))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Test_HI_v2_all.png"), dpi=150)
    plt.close()
    print(f"\n[완료] {OUT_DIR}")


if __name__ == "__main__":
    main()
