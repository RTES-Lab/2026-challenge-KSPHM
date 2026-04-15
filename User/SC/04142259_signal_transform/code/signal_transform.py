"""
Signal Transformation (F2S2 논문 Section 3)
=============================================
Reference: Li et al. (2019), RESS 186, 88-100.

목적:
  RPM(운전조건) 변화에 의한 특징의 Signal Jump를 제거하여
  순수한 열화 추세만 남기는 전처리.

원리:
  원본 특징 y_k^p (조건 p에서 관측):
    y_k = a_p * (b_p + x_k^c + v_k)
    → a_p, b_p가 RPM마다 다르므로 조건 변환 시 "점프" 발생

  Signal Transformation:
    y_k^B = a'_p * y_k^p + b'_p
    → 모든 신호를 baseline 조건 B 아래의 신호로 변환
    → a'_p, b'_p: 오프라인 학습(least squares)

  변환 후: 조건 변화에 의한 점프 제거, 열화 추세만 잔류

변환 파라미터 추정 방법 (논문 방식):
  1. Baseline 조건 선택 (가장 빈번한 RPM 구간)
  2. 각 비baseline 조건 p에 대해:
     a) 조건 p가 활성인 시간 인덱스 추출
     b) 해당 시간대의 baseline 신호를 선형 보간으로 추정
     c) min_{a'_p, b'_p} Σ(y_interp - a'_p * y_p - b'_p)² → LSQ
  3. 모든 시간 스텝에서 조건별 변환 적용

데이터 특이사항:
  - Train1~4: 각각 다른 미세 고장 인가 후 운전 (고장 유형 미공개)
  - RPM: 700~950, 1시간 단위 변동 (6파일마다 RPM 레벨 변화)
  - Operation CSV: 10초 샘플링
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = "/data/home/ksphm/2026-challenge-KSPHM"
DATA_DIR   = os.path.join(BASE_DIR, "dataset")
FEAT_DIR   = os.path.join(BASE_DIR, "User", "SC",
                          "04140103_initial_pca_result", "output")
OUTPUT_DIR = os.path.join(BASE_DIR, "User", "SC",
                          "04142259_signal_transform", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEARING_IDS  = [1, 2, 3, 4]
INTERVAL_SEC = 600
MEAS_WIN_SEC = 60
N_COND       = 3         # RPM 이산화 레벨 수

# ─────────────────────────────────────────────
# Step 0: RPM 로드 및 정렬
# ─────────────────────────────────────────────

def load_operation(bear_id: int) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"Train{bear_id}_Operation.csv")
    df = pd.read_csv(path, encoding="cp949")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Time[sec]":        "time_sec",
        "Motor speed[rpm]": "rpm",
    })
    return df


def align_rpm(op_df: pd.DataFrame, n_files: int) -> np.ndarray:
    """진동 파일 인덱스별 평균 RPM (1분 측정 구간 기준)"""
    rpm = np.zeros(n_files)
    for k in range(n_files):
        t0 = k * INTERVAL_SEC
        t1 = t0 + MEAS_WIN_SEC
        mask = (op_df["time_sec"] >= t0) & (op_df["time_sec"] < t1)
        vals = op_df.loc[mask, "rpm"].values
        rpm[k] = vals.mean() if len(vals) > 0 else np.nan
    nans = np.isnan(rpm)
    if nans.any():
        idx = np.arange(n_files)
        rpm[nans] = np.interp(idx[nans], idx[~nans], rpm[~nans])
    return rpm


# ─────────────────────────────────────────────
# Step 1: RPM 이산화 & Baseline 선택
# ─────────────────────────────────────────────

def discretize_rpm_global(all_rpm_arrays: list, n_cond: int = N_COND):
    """
    전체 베어링의 RPM을 합쳐서 분위수 기반으로 n_cond개 구간을 결정.
    Returns: thresholds (n_cond+1,), baseline_cond (int)
    """
    all_rpm = np.concatenate(all_rpm_arrays)
    quantiles = np.linspace(0, 100, n_cond + 1)
    thresholds = np.percentile(all_rpm, quantiles)
    # 빈도 가장 높은 구간 = baseline
    cond_all = np.zeros_like(all_rpm, dtype=int)
    for j in range(1, n_cond):
        cond_all[all_rpm >= thresholds[j]] = j
    counts = np.bincount(cond_all, minlength=n_cond)
    baseline = int(np.argmax(counts))
    return thresholds, baseline, counts


def apply_discretize(rpm_arr: np.ndarray, thresholds: np.ndarray,
                     n_cond: int) -> np.ndarray:
    cond = np.zeros(len(rpm_arr), dtype=int)
    for j in range(1, n_cond):
        cond[rpm_arr >= thresholds[j]] = j
    return cond


# ─────────────────────────────────────────────
# Step 2: Signal Transformation (F2S2 핵심)
# ─────────────────────────────────────────────

def estimate_transform_params(feature_series: np.ndarray,
                              cond_array: np.ndarray,
                              baseline_cond: int,
                              n_cond: int) -> dict:
    """
    논문 방식으로 Signal Transformation 파라미터 (a'_p, b'_p) 추정.

    알고리즘:
      1. baseline 조건 구간의 (시간인덱스, 특징값) 쌍 수집
      2. 비baseline 조건 p의 시간 인덱스에서 baseline 신호를 선형 보간
      3. min_{a'_p, b'_p} Σ(y_baseline_interp - a'_p * y_p - b'_p)²
         → closed-form LSQ

    Parameters
    ----------
    feature_series : shape (N,) - 단일 특징의 시계열
    cond_array     : shape (N,) - 운전 조건 레이블
    baseline_cond  : int - baseline 조건 인덱스
    n_cond         : int - 조건 수

    Returns
    -------
    params : dict {cond_idx: (a_prime, b_prime)}
             baseline 조건은 (1.0, 0.0)
    """
    N      = len(feature_series)
    t_idx  = np.arange(N)
    params = {}

    # Baseline 구간의 (인덱스, 값) 추출
    bl_mask  = (cond_array == baseline_cond)
    bl_times = t_idx[bl_mask]
    bl_vals  = feature_series[bl_mask]

    if len(bl_times) < 2:
        # Baseline 데이터 부족 → 변환 불가, 항등 변환
        for c in range(n_cond):
            params[c] = (1.0, 0.0)
        return params

    # Baseline 조건은 항등 변환
    params[baseline_cond] = (1.0, 0.0)

    for c in range(n_cond):
        if c == baseline_cond:
            continue

        c_mask  = (cond_array == c)
        c_times = t_idx[c_mask]
        c_vals  = feature_series[c_mask]

        if len(c_times) < 2:
            params[c] = (1.0, 0.0)
            continue

        # baseline 신호를 조건 c의 시간 인덱스에서 보간
        y_bl_interp = np.interp(c_times, bl_times, bl_vals)

        # LSQ: y_bl_interp = a' * c_vals + b'
        # 일반 선형 최소제곱
        A = np.column_stack([c_vals, np.ones(len(c_vals))])
        result, _, _, _ = np.linalg.lstsq(A, y_bl_interp, rcond=None)
        a_prime = result[0]
        b_prime = result[1]

        params[c] = (float(a_prime), float(b_prime))

    return params


def transform_signal(feature_series: np.ndarray,
                     cond_array: np.ndarray,
                     params: dict) -> np.ndarray:
    """
    추정된 파라미터로 Signal Transformation 적용.
    y_k^B = a'_{c_k} * y_k + b'_{c_k}
    """
    y_transformed = np.zeros_like(feature_series)
    for k in range(len(feature_series)):
        c = cond_array[k]
        a_p, b_p = params[c]
        y_transformed[k] = a_p * feature_series[k] + b_p
    return y_transformed


# ─────────────────────────────────────────────
# Step 3: 품질 지표 (단조성, 추세성)
# ─────────────────────────────────────────────

def monotonicity(x):
    dx    = np.diff(x)
    n_pos = np.sum(dx > 0)
    n_neg = np.sum(dx < 0)
    return abs(n_pos - n_neg) / (len(dx) + 1e-12)

def trendability(x):
    rho, _ = spearmanr(x, np.arange(len(x)))
    return abs(rho) if not np.isnan(rho) else 0.0


# ─────────────────────────────────────────────
# Step 4: Main Pipeline
# ─────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Signal Transformation (F2S2 논문)")
    print("  RPM 변화에 의한 Signal Jump 제거")
    print("=" * 70)

    # ── 데이터 로드
    feat_cols = None
    bearing_data = {}

    for bid in BEARING_IDS:
        df = pd.read_csv(os.path.join(FEAT_DIR,
                                      f"Bearing{bid}_features.csv"))
        if feat_cols is None:
            feat_cols = [c for c in df.columns
                         if c not in ["file_idx","time_sec","bearing_id"]]
        op = load_operation(bid)
        n  = len(df)
        rpm = align_rpm(op, n)
        bearing_data[bid] = {"df": df, "rpm": rpm, "n": n}

    print(f"\n특징 수: {len(feat_cols)}")

    # ── RPM 이산화
    all_rpms = [bearing_data[b]["rpm"] for b in BEARING_IDS]
    thresholds, baseline, counts = discretize_rpm_global(all_rpms, N_COND)
    print(f"RPM 구간 경계: {np.round(thresholds, 0)}")
    print(f"구간별 빈도: {counts}")
    print(f"Baseline 조건: {baseline} (빈도 최대)")

    for bid in BEARING_IDS:
        bearing_data[bid]["cond"] = apply_discretize(
            bearing_data[bid]["rpm"], thresholds, N_COND)

    # ── 각 베어링별 Signal Transformation 적용
    print("\n[Signal Transformation 파라미터 추정 & 적용]")

    quality_before = []  # (feature, Mon_before, Tre_before) 평균 over bearings
    quality_after  = []

    # 특징별 변환 결과 저장
    transformed_dfs = {}

    for bid in BEARING_IDS:
        df   = bearing_data[bid]["df"]
        cond = bearing_data[bid]["cond"]
        n    = bearing_data[bid]["n"]

        df_trans = df[["file_idx", "time_sec", "bearing_id"]].copy()

        for feat in feat_cols:
            y_raw = df[feat].values.astype(np.float64)

            # 파라미터 추정 (이 베어링 자체 데이터 사용)
            params = estimate_transform_params(y_raw, cond, baseline, N_COND)

            # 변환 적용
            y_trans = transform_signal(y_raw, cond, params)

            df_trans[feat] = y_trans

        transformed_dfs[bid] = df_trans
        print(f"  Bearing {bid}: 변환 완료 ({len(feat_cols)} features)")

    # ── 품질 지표 비교 (Before vs After)
    print("\n[품질 지표 비교: Before vs After Signal Transformation]")

    before_scores = {f: {"mon": [], "tre": []} for f in feat_cols}
    after_scores  = {f: {"mon": [], "tre": []} for f in feat_cols}

    for bid in BEARING_IDS:
        df_raw   = bearing_data[bid]["df"]
        df_trans = transformed_dfs[bid]
        for feat in feat_cols:
            y_raw   = df_raw[feat].values.astype(np.float64)
            y_trans = df_trans[feat].values.astype(np.float64)

            before_scores[feat]["mon"].append(monotonicity(y_raw))
            before_scores[feat]["tre"].append(trendability(y_raw))
            after_scores[feat]["mon"].append(monotonicity(y_trans))
            after_scores[feat]["tre"].append(trendability(y_trans))

    # 평균 계산 및 비교 테이블
    comparison = []
    for feat in feat_cols:
        mon_b = np.mean(before_scores[feat]["mon"])
        tre_b = np.mean(before_scores[feat]["tre"])
        mon_a = np.mean(after_scores[feat]["mon"])
        tre_a = np.mean(after_scores[feat]["tre"])
        q_b   = (mon_b + tre_b) / 2
        q_a   = (mon_a + tre_a) / 2
        comparison.append({
            "feature":    feat,
            "Mon_before": round(mon_b, 4),
            "Tre_before": round(tre_b, 4),
            "Q_before":   round(q_b, 4),
            "Mon_after":  round(mon_a, 4),
            "Tre_after":  round(tre_a, 4),
            "Q_after":    round(q_a, 4),
            "Mon_delta":  round(mon_a - mon_b, 4),
            "Tre_delta":  round(tre_a - tre_b, 4),
            "Q_delta":    round(q_a - q_b, 4),
        })

    df_comp = pd.DataFrame(comparison).sort_values("Q_after", ascending=False)
    df_comp = df_comp.reset_index(drop=True)
    df_comp["rank_after"] = range(1, len(df_comp) + 1)

    # 출력
    print(f"\n{'Rank':>4}  {'Feature':<30} {'Q_before':>9} {'Q_after':>9}"
          f" {'ΔQ':>7} {'ΔMon':>7} {'ΔTre':>7}")
    print("-" * 80)
    for _, r in df_comp.iterrows():
        sign = "+" if r["Q_delta"] > 0 else ""
        print(f"{int(r['rank_after']):>4}  {r['feature']:<30}"
              f" {r['Q_before']:>9.4f} {r['Q_after']:>9.4f}"
              f" {sign}{r['Q_delta']:>6.4f}"
              f" {'+' if r['Mon_delta']>0 else ''}{r['Mon_delta']:>6.4f}"
              f" {'+' if r['Tre_delta']>0 else ''}{r['Tre_delta']:>6.4f}")

    # ── 저장
    csv_path = os.path.join(OUTPUT_DIR, "quality_comparison.csv")
    df_comp.to_csv(csv_path, index=False)
    print(f"\n[저장] 비교 표 → {csv_path}")

    # 변환된 특징 캐시 저장
    for bid in BEARING_IDS:
        trans_path = os.path.join(OUTPUT_DIR,
                                  f"Bearing{bid}_features_transformed.csv")
        transformed_dfs[bid].to_csv(trans_path, index=False)
    print(f"[저장] 변환 특징 CSV → output/Bearing*_features_transformed.csv")

    # ── 전체 평균 개선도 요약
    avg_q_before = df_comp["Q_before"].mean()
    avg_q_after  = df_comp["Q_after"].mean()
    n_improved   = (df_comp["Q_delta"] > 0).sum()
    n_degraded   = (df_comp["Q_delta"] < 0).sum()
    print(f"\n[요약]")
    print(f"  평균 Quality: {avg_q_before:.4f} → {avg_q_after:.4f}"
          f"  (Δ={avg_q_after-avg_q_before:+.4f})")
    print(f"  개선된 특징: {n_improved}/{len(feat_cols)}")
    print(f"  악화된 특징: {n_degraded}/{len(feat_cols)}")

    # ── 시각화: 대표 특징 Before/After 비교
    top_feats = df_comp.head(6)["feature"].tolist()
    _plot_before_after(bearing_data, transformed_dfs, top_feats,
                       feat_cols, OUTPUT_DIR)

    # ── 시각화: Q_before vs Q_after 산점도
    _plot_quality_scatter(df_comp, OUTPUT_DIR)

    print("\n[완료]")


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def _plot_before_after(bearing_data, transformed_dfs, top_feats,
                       feat_cols, save_dir):
    """상위 6개 특징의 Before/After 비교 (Bearing 1 기준)"""
    for bid in [1, 2]:  # Bearing 1, 2만 플로팅
        fig, axes = plt.subplots(len(top_feats), 2,
                                 figsize=(16, 3 * len(top_feats)),
                                 sharex=True)
        fig.suptitle(f"Bearing {bid} — Signal Transformation Before vs After\n"
                     f"(RPM-induced Signal Jump 제거)",
                     fontsize=13, fontweight="bold")

        df_raw  = bearing_data[bid]["df"]
        df_trans = transformed_dfs[bid]
        cond    = bearing_data[bid]["cond"]
        t_hr    = np.arange(bearing_data[bid]["n"]) * INTERVAL_SEC / 3600

        for i, feat in enumerate(top_feats):
            y_raw   = df_raw[feat].values.astype(np.float64)
            y_trans = df_trans[feat].values

            # Raw 정규화 (시각 비교용)
            y_raw_n  = (y_raw - y_raw.min()) / (y_raw.ptp() + 1e-12)
            y_trans_n = (y_trans - y_trans.min()) / (y_trans.ptp() + 1e-12)

            # Before
            ax = axes[i, 0]
            for c_val in range(N_COND):
                mask = (cond == c_val)
                ax.scatter(t_hr[mask], y_raw_n[mask], s=8, alpha=0.7,
                           label=f"Cond {c_val}")
            ax.plot(t_hr, y_raw_n, color="gray", lw=0.5, alpha=0.4)
            mon_b = monotonicity(y_raw)
            tre_b = trendability(y_raw)
            ax.set_ylabel(feat, fontsize=7, rotation=0, ha="right", labelpad=90)
            ax.set_title(f"BEFORE  Mon={mon_b:.3f} Tre={tre_b:.3f}",
                         fontsize=9, loc="left")
            ax.set_ylim(-0.1, 1.2)
            ax.grid(True, ls="--", alpha=0.3)
            if i == 0:
                ax.legend(fontsize=6, ncol=3, loc="upper right")

            # After
            ax = axes[i, 1]
            ax.plot(t_hr, y_trans_n, color=COLORS[bid-1], lw=1.2, alpha=0.85)
            mon_a = monotonicity(y_trans)
            tre_a = trendability(y_trans)
            ax.set_title(f"AFTER   Mon={mon_a:.3f} Tre={tre_a:.3f}",
                         fontsize=9, loc="left")
            ax.set_ylim(-0.1, 1.2)
            ax.grid(True, ls="--", alpha=0.3)

        axes[-1, 0].set_xlabel("Time [hr]", fontsize=10)
        axes[-1, 1].set_xlabel("Time [hr]", fontsize=10)
        plt.tight_layout()
        path = os.path.join(save_dir, f"Bearing{bid}_before_after.png")
        plt.savefig(path, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"[저장] Before/After 플롯 → {path}")


def _plot_quality_scatter(df_comp, save_dir):
    """Q_before vs Q_after 산점도"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df_comp["Q_before"], df_comp["Q_after"],
               c=df_comp["Q_delta"], cmap="RdYlGn", s=50, edgecolors="gray",
               linewidths=0.5, alpha=0.8)
    # 대각선 (변화 없음)
    lim = [0, max(df_comp["Q_before"].max(), df_comp["Q_after"].max()) + 0.05]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5, label="No change")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Quality BEFORE (Mon+Tre)/2", fontsize=11)
    ax.set_ylabel("Quality AFTER signal transform", fontsize=11)
    ax.set_title("Feature Quality: Before vs After Signal Transformation\n"
                 "(위=개선, 아래=악화)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)

    # 상위 10개 레이블
    for _, r in df_comp.head(10).iterrows():
        ax.annotate(r["feature"], (r["Q_before"], r["Q_after"]),
                    fontsize=6, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

    plt.tight_layout()
    path = os.path.join(save_dir, "quality_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] Q scatter → {path}")


if __name__ == "__main__":
    main()
