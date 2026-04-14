"""
SPE(Squared Prediction Error) 기반 Health Index (HI) 구성
============================================================
대회: 2026 KSPHM-KIMM 기계 데이터 챌린지
실험: 04140132_spe_hi

SPE란?
  - PCA 모델로 데이터를 재구성한 후, 원본과의 차이(오차)의 제곱합
  - SPE = ||x - x_hat||^2  where  x_hat = PCA 재구성값
  - 정상 데이터: PCA 모델이 잘 설명 → SPE 낮음
  - 이상 데이터: PCA 모델로 설명 불가 → SPE 높음
  - PC1 Score와 달리 "정상 공간으로부터의 이탈 거리"를 직접 측정
    → 열화 진행에 따라 단조 증가하는 경향이 강함

PC1 Score vs SPE:
  PC1 Score: "정상 분산 방향으로 얼마나 이동했는가" (방향 의존)
  SPE:       "정상 공간에서 얼마나 벗어났는가"      (방향 무관, 크기만)

데이터 주의사항:
  - Train1~4 베어링은 각각 다른 종류의 미세 고장이 인가된 후 운전됨
  - 고장 종류(내륜/외륜/전동체 등)는 미공개, 베어링마다 다를 수 있음
  → 각 베어링이 서로 다른 열화 패턴을 가질 수 있음
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import nptdms
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR    = "/data/home/ksphm/2026-challenge-KSPHM"
DATA_DIR    = os.path.join(BASE_DIR, "dataset")
# 특징 캐시는 이전 실험의 것을 재사용 (동일 특징 추출 로직)
CACHE_DIR   = os.path.join(BASE_DIR, "User", "SC",
                           "04140103_initial_pca_result", "output")
OUTPUT_DIR  = os.path.join(BASE_DIR, "User", "SC",
                           "04140132_spe_hi", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS           = 25600
INTERVAL_SEC = 600
NORMAL_RATIO = 0.15
N_COMPONENTS = 10          # SPE에서는 더 많은 PC를 사용 (잔차 공간 확보)

# ─────────────────────────────────────────────
# 특징 추출 (캐시 재사용)
# ─────────────────────────────────────────────
LOG_TRANSFORM_KEYWORDS = ["energy", "kurt_rms"]

def log_transform_features(X: np.ndarray, feat_cols: list) -> np.ndarray:
    X = X.copy()
    for i, col in enumerate(feat_cols):
        if any(kw in col for kw in LOG_TRANSFORM_KEYWORDS):
            X[:, i] = np.log1p(np.abs(X[:, i]))
    return X

def load_features(bear_id: int) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f"Bearing{bear_id}_features.csv")
    if os.path.exists(cache):
        print(f"  [Bearing {bear_id}] 특징 캐시 로드: {cache}")
        return pd.read_csv(cache)
    else:
        raise FileNotFoundError(f"특징 캐시 없음: {cache}\n"
                                f"먼저 04140103 실험을 실행하세요.")

# ─────────────────────────────────────────────
# SPE 기반 HI 계산
# ─────────────────────────────────────────────

def build_hi_spe(df: pd.DataFrame,
                 normal_ratio: float = NORMAL_RATIO,
                 n_components: int = N_COMPONENTS) -> tuple:
    """
    SPE(Q-statistic) 기반 HI 구성

    알고리즘:
      1. 정상 구간으로 StandardScaler + PCA(n_components) 학습
      2. 전체 데이터를 PCA 공간으로 투영 후 재구성: x_hat = scores @ components + mean
      3. SPE = mean((x - x_hat)^2) per sample  → 스칼라
      4. SPE를 [0,1] 정규화: (SPE - SPE_normal_mean) / (SPE_max - SPE_normal_mean)

    Parameters
    ----------
    df           : 특징 DataFrame
    normal_ratio : 정상 구간 비율
    n_components : PCA 성분 수 (유지할 PC 수; 클수록 재구성 오차 ↓)

    Returns
    -------
    df_result, pca, scaler, n_components_used
    """
    feat_cols = [c for c in df.columns
                 if c not in ["file_idx", "time_sec", "bearing_id"]]

    X_raw = df[feat_cols].values.astype(np.float64)
    X = log_transform_features(X_raw, feat_cols)          # 스케일 압축

    n_normal = max(int(len(df) * normal_ratio), 5)
    X_normal = X[:n_normal]

    # 스케일링 (정상 구간 기준)
    scaler = StandardScaler()
    scaler.fit(X_normal)
    X_scaled = scaler.transform(X)
    X_normal_scaled = X_scaled[:n_normal]

    # PCA 학습 (n_components개 주성분 유지)
    n_components = min(n_components, len(feat_cols), n_normal - 1)
    pca = PCA(n_components=n_components)
    pca.fit(X_normal_scaled)

    explained = pca.explained_variance_ratio_
    cum_explained = explained.cumsum()
    print(f"  PCA {n_components}성분 누적 분산 설명: {cum_explained[-1]:.3f}")
    print(f"  → 잔차공간(SPE가 측정하는 영역): {1-cum_explained[-1]:.3f}")

    # 재구성 및 SPE 계산
    scores     = pca.transform(X_scaled)          # (N, n_components)
    X_recon    = pca.inverse_transform(scores)    # (N, n_features)
    residual   = X_scaled - X_recon              # 재구성 오차
    spe        = np.mean(residual ** 2, axis=1)   # SPE per sample (MSE)

    # 정규화
    spe_normal_mean = np.mean(spe[:n_normal])
    spe_normal_std  = np.std(spe[:n_normal])
    spe_max         = np.max(spe)

    print(f"  SPE 정상 구간: mean={spe_normal_mean:.6f}, std={spe_normal_std:.6f}")
    print(f"  SPE 전체 최대: {spe_max:.6f}")

    # HI = (SPE - normal_mean) / (max - normal_mean), clip [0,1]
    hi = (spe - spe_normal_mean) / (spe_max - spe_normal_mean + 1e-12)
    hi = np.clip(hi, 0, 1).astype(np.float32)

    # 통계 임계값 (μ + 3σ 기준) → FPT 후보
    threshold = (spe_normal_mean + 3 * spe_normal_std - spe_normal_mean) / \
                (spe_max - spe_normal_mean + 1e-12)
    threshold = np.clip(threshold, 0, 1)
    print(f"  FPT 임계값(μ+3σ 기준 HI): {threshold:.4f}")

    df_result = df.copy()
    df_result["SPE_raw"]   = spe.astype(np.float32)
    df_result["HI_SPE"]    = hi
    df_result["HI_SPE_cm"] = np.maximum.accumulate(hi)   # cumulative max (단조 강제)

    # FPT 탐지
    fpt_idx = np.where(hi > threshold)[0]
    if len(fpt_idx) > 0:
        fpt_time = df_result["time_sec"].iloc[fpt_idx[0]]
        df_result["FPT_sec"] = fpt_time
        print(f"  FPT 탐지: {fpt_time/3600:.2f} hr "
              f"(파일 #{df_result['file_idx'].iloc[fpt_idx[0]]})")
    else:
        df_result["FPT_sec"] = np.nan
        print("  FPT 탐지: 임계값 초과 없음")

    return df_result, pca, scaler, n_components, threshold

# ─────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

def plot_hi_comparison(bear_id: int, df: pd.DataFrame,
                       threshold: float, save_dir: str):
    """PC1 HI vs SPE HI vs SPE cummax 비교 플롯"""
    time_hr   = df["time_sec"] / 3600
    n_normal  = max(int(len(df) * NORMAL_RATIO), 5)
    t_norm_end = time_hr.iloc[n_normal - 1]
    color     = COLORS[bear_id - 1]

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(f"Bearing {bear_id} — PC1 Score vs SPE-based HI",
                 fontsize=13, fontweight="bold")

    # ── subplot 1: SPE raw (log scale)
    ax = axes[0]
    ax.semilogy(time_hr, df["SPE_raw"], color=color, linewidth=1.0, alpha=0.9)
    ax.axvspan(0, t_norm_end, alpha=0.1, color="green")
    ax.axvline(t_norm_end, color="green", linestyle="--", linewidth=1.0)
    ax.set_ylabel("SPE (log scale)", fontsize=10)
    ax.set_title("SPE Raw (Squared Prediction Error)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── subplot 2: HI_SPE vs HI_PC1(이전 실험)
    ax = axes[1]
    ax.plot(time_hr, df["HI_SPE"], color=color,
            linewidth=1.3, label="HI (SPE)", alpha=0.9)
    ax.axhline(threshold, color="red", linestyle="--",
               linewidth=0.9, label=f"FPT threshold ({threshold:.3f})")
    # FPT 마커
    if not np.isnan(df["FPT_sec"].iloc[0]):
        fpt_hr = df["FPT_sec"].iloc[0] / 3600
        ax.axvline(fpt_hr, color="orange", linestyle=":", linewidth=1.5,
                   label=f"FPT ({fpt_hr:.1f} hr)")
    ax.axvspan(0, t_norm_end, alpha=0.1, color="green")
    ax.axvline(t_norm_end, color="green", linestyle="--", linewidth=1.0)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("HI (SPE norm.)", fontsize=10)
    ax.set_title("Health Index via SPE", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── subplot 3: SPE cummax (단조 강제)
    ax = axes[2]
    ax.plot(time_hr, df["HI_SPE_cm"], color=color,
            linewidth=1.3, label="HI (SPE cummax)", alpha=0.9)
    ax.axhline(threshold, color="red", linestyle="--",
               linewidth=0.9, label=f"FPT threshold ({threshold:.3f})")
    if not np.isnan(df["FPT_sec"].iloc[0]):
        ax.axvline(fpt_hr, color="orange", linestyle=":", linewidth=1.5,
                   label=f"FPT ({fpt_hr:.1f} hr)")
    ax.axvspan(0, t_norm_end, alpha=0.1, color="green")
    ax.axvline(t_norm_end, color="green", linestyle="--", linewidth=1.0)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("HI (cummax)", fontsize=10)
    ax.set_xlabel("Time [hr]", fontsize=10)
    ax.set_title("HI (SPE + Cumulative Max → 단조 강제)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"Bearing{bear_id}_HI_SPE.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [저장] {save_path}")


def plot_all_comparison(hi_dfs: dict, save_dir: str):
    """4개 베어링 SPE HI + cummax 비교"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()
    fig.suptitle("SPE-based Health Index — All Bearings\n"
                 "Solid: HI_SPE  |  Dashed: HI_SPE (cummax)",
                 fontsize=13, fontweight="bold")

    for i, (bear_id, info) in enumerate(hi_dfs.items()):
        df, threshold = info["df"], info["threshold"]
        ax = axes[i]
        time_hr  = df["time_sec"] / 3600
        n_normal = max(int(len(df) * NORMAL_RATIO), 5)
        t_ne     = time_hr.iloc[n_normal - 1]
        color    = COLORS[i]

        ax.plot(time_hr, df["HI_SPE"],    color=color, linewidth=1.2,
                alpha=0.7, label="HI_SPE")
        ax.plot(time_hr, df["HI_SPE_cm"], color=color, linewidth=1.6,
                linestyle="--", alpha=0.95, label="HI_SPE (cummax)")
        ax.axhline(threshold, color="red", linestyle=":", linewidth=0.9,
                   label=f"FPT thr.({threshold:.3f})")
        ax.axvspan(0, t_ne, alpha=0.08, color="green")
        ax.axvline(t_ne,    color="green", linestyle="--", linewidth=0.8)

        if not np.isnan(df["FPT_sec"].iloc[0]):
            fpt_hr = df["FPT_sec"].iloc[0] / 3600
            ax.axvline(fpt_hr, color="orange", linestyle=":",
                       linewidth=1.5, label=f"FPT ({fpt_hr:.1f}hr)")

        ax.set_xlim(left=0)
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(f"Bearing {bear_id} (N={len(df)}, {time_hr.max():.1f}hr)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Time [hr]", fontsize=9)
        ax.set_ylabel("HI", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(save_dir, "All_HI_SPE_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[저장] 전체 비교 플롯 → {path}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SPE 기반 Health Index 구성")
    print("=" * 65)

    bearing_ids = [1, 2, 3, 4]
    hi_dfs      = {}
    summary     = []

    for bear_id in bearing_ids:
        print(f"\n{'='*20} Bearing {bear_id} {'='*20}")
        df_feat = load_features(bear_id)
        df_hi, pca, scaler, n_comp, thr = build_hi_spe(
            df_feat, normal_ratio=NORMAL_RATIO, n_components=N_COMPONENTS)

        # 저장
        out_cols = ["bearing_id", "file_idx", "time_sec",
                    "SPE_raw", "HI_SPE", "HI_SPE_cm", "FPT_sec"]
        csv_path = os.path.join(OUTPUT_DIR, f"Bearing{bear_id}_HI_SPE.csv")
        df_hi[out_cols].to_csv(csv_path, index=False)
        print(f"  [저장] {csv_path}")

        # 개별 플롯
        plot_hi_comparison(bear_id, df_hi, thr, OUTPUT_DIR)

        hi_dfs[bear_id] = {"df": df_hi, "threshold": thr}

        n_normal = max(int(len(df_hi) * NORMAL_RATIO), 5)
        summary.append({
            "Bearing":        bear_id,
            "N_segments":     len(df_hi),
            "Duration_hr":    round(df_hi["time_sec"].max() / 3600, 2),
            "FPT_hr":         round(df_hi["FPT_sec"].iloc[0] / 3600, 3)
                              if not np.isnan(df_hi["FPT_sec"].iloc[0]) else "N/A",
            "HI_SPE_final":   round(float(df_hi["HI_SPE"].iloc[-1]), 4),
            "HI_cm_final":    round(float(df_hi["HI_SPE_cm"].iloc[-1]), 4),
            "FPT_threshold":  round(float(thr), 4),
            "n_components":   n_comp,
        })

    # 전체 비교 플롯
    plot_all_comparison(hi_dfs, OUTPUT_DIR)

    # 요약 통계
    print("\n" + "=" * 65)
    print("  HI 요약 통계")
    print("=" * 65)
    df_sum = pd.DataFrame(summary)
    print(df_sum.to_string(index=False))
    sum_path = os.path.join(OUTPUT_DIR, "HI_SPE_summary.csv")
    df_sum.to_csv(sum_path, index=False)
    print(f"\n[저장] 요약 → {sum_path}")
    print("\n[완료]")


if __name__ == "__main__":
    main()
