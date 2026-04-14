"""
KPCA(Kernel PCA) 기반 Health Index (HI) 구성
==============================================
대회: 2026 KSPHM-KIMM 기계 데이터 챌린지
실험: 04140135_kpca_hi

KPCA란?
  - 데이터를 커널 함수로 고차원 공간에 암묵적으로 매핑한 뒤 PCA 적용
  - 비선형 구조를 선형 분리 가능하게 변환 → 가변속/노이즈 조건에 강건
  - RBF(Gaussian) 커널: K(x,y) = exp(-γ||x-y||²)
    → 데이터 간 유사도를 거리 기반으로 측정

HI 구성 전략 (Three-in-one 비교):
  (A) KPCA PC1 Score → PC1 투영값
  (B) KPCA SPE       → 커널 공간에서의 재구성 오차 (= pre-image 오차)
      ※ KPCA SPE는 직접 계산하기 어려우므로, 입력 공간에서의 근사치 사용
         SPE_approx = ||Kx - K_recon||² (커널 행벡터 기반)
  (C) Mahalanobis Distance (MD from normal data center in kernel space)

데이터 특이사항:
  - Train1~4 베어링은 각각 다른 종류의 미세 고장이 실험 전 인가됨
  - 고장 유형(내륜/외륜/전동체 등) 미공개, 베어링마다 다를 수 있음
  - 가변속 조건(700~950 RPM, 1시간 단위 변동)
  - 다량의 환경 노이즈 포함
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import mahalanobis
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR   = "/data/home/ksphm/2026-challenge-KSPHM"
CACHE_DIR  = os.path.join(BASE_DIR, "User", "SC",
                          "04140103_initial_pca_result", "output")
OUTPUT_DIR = os.path.join(BASE_DIR, "User", "SC",
                          "04140135_kpca_hi", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NORMAL_RATIO = 0.15
N_COMPONENTS = 10    # 유지할 커널 주성분 수
GAMMA        = None  # None → 1/n_features (sklearn default)

LOG_TRANSFORM_KEYWORDS = ["energy", "kurt_rms"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ─────────────────────────────────────────────
# 전처리
# ─────────────────────────────────────────────

def log_transform_features(X: np.ndarray, feat_cols: list) -> np.ndarray:
    X = X.copy()
    for i, col in enumerate(feat_cols):
        if any(kw in col for kw in LOG_TRANSFORM_KEYWORDS):
            X[:, i] = np.log1p(np.abs(X[:, i]))
    return X


def load_features(bear_id: int) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f"Bearing{bear_id}_features.csv")
    print(f"  [Bearing {bear_id}] 특징 로드: {path}")
    return pd.read_csv(path)


# ─────────────────────────────────────────────
# KPCA 기반 HI 계산
# ─────────────────────────────────────────────

def build_hi_kpca(df: pd.DataFrame,
                  normal_ratio: float = NORMAL_RATIO,
                  n_components: int = N_COMPONENTS,
                  gamma=GAMMA) -> tuple:
    """
    KPCA 기반 HI 구성

    (A) KPCA PC1 Score
    (B) KPCA SPE (근사 재구성 오차: 커널 공간 내 잔차)
        sklearn KernelPCA fit_inverse_transform=True 활용
    (C) Mahalanobis Distance from normal cluster centroid (커널 공간)

    Returns: df_result, kpca, scaler
    """
    feat_cols = [c for c in df.columns
                 if c not in ["file_idx", "time_sec", "bearing_id"]]

    X_raw = df[feat_cols].values.astype(np.float64)
    X     = log_transform_features(X_raw, feat_cols)

    n_normal = max(int(len(df) * normal_ratio), 5)

    # 스케일링
    scaler = StandardScaler()
    scaler.fit(X[:n_normal])
    X_scaled = scaler.transform(X)

    # ── KPCA 학습 (정상 구간, fit_inverse_transform for SPE)
    actual_n_comp = min(n_components, n_normal - 1)
    gamma_val     = gamma if gamma else 1.0 / X_scaled.shape[1]

    kpca = KernelPCA(
        n_components=actual_n_comp,
        kernel="rbf",
        gamma=gamma_val,
        fit_inverse_transform=True,   # SPE 계산을 위한 역변환 활성화
        random_state=42
    )
    kpca.fit(X_scaled[:n_normal])
    print(f"  KPCA 학습 완료 (gamma={gamma_val:.5f}, n_comp={actual_n_comp})")

    # ── (A) KPCA PC1 Score
    X_kpca   = kpca.transform(X_scaled)     # (N, n_components) 커널 공간 좌표
    pc1_raw  = X_kpca[:, 0]
    # 방향 확인 (말기값이 더 커야 함)
    if pc1_raw[-1] < pc1_raw[0]:
        pc1_raw = -pc1_raw
    pc1_normal_mean = np.mean(pc1_raw[:n_normal])
    pc1_max         = np.max(pc1_raw)
    hi_pc1 = np.clip((pc1_raw - pc1_normal_mean) /
                     (pc1_max - pc1_normal_mean + 1e-12), 0, 1)

    # ── (B) KPCA SPE (입력 공간 재구성 오차)
    X_recon  = kpca.inverse_transform(X_kpca)   # 역변환 (입력 공간 근사)
    residual = X_scaled - X_recon
    spe_raw  = np.mean(residual ** 2, axis=1)   # MSE per sample

    spe_n_mean = np.mean(spe_raw[:n_normal])
    spe_n_std  = np.std(spe_raw[:n_normal])
    spe_max    = np.max(spe_raw)
    hi_spe     = np.clip((spe_raw - spe_n_mean) /
                         (spe_max - spe_n_mean + 1e-12), 0, 1)

    print(f"  SPE 정상 구간: mean={spe_n_mean:.6f}, std={spe_n_std:.6f}")
    print(f"  SPE 전체 최대: {spe_max:.4f}")

    # FPT 임계값 (μ + 3σ)
    thr_spe = np.clip(3 * spe_n_std / (spe_max - spe_n_mean + 1e-12), 0, 1)
    print(f"  FPT 임계값(SPE μ+3σ 기준 HI): {thr_spe:.4f}")

    # ── (C) Mahalanobis Distance (커널 공간 내)
    normal_kpca = X_kpca[:n_normal]
    mu_n        = np.mean(normal_kpca, axis=0)
    cov_n       = np.cov(normal_kpca.T)

    # 공분산 행렬 안정화 (작은 값 대각 추가)
    cov_n += np.eye(cov_n.shape[0]) * 1e-6
    try:
        cov_inv = np.linalg.inv(cov_n)
        md_vals = np.array([
            np.sqrt(max((x - mu_n) @ cov_inv @ (x - mu_n), 0))
            for x in X_kpca
        ])
    except np.linalg.LinAlgError:
        print("  [경고] 공분산 역행렬 계산 불가 → 단위행렬 사용")
        cov_inv = np.eye(cov_n.shape[0])
        md_vals = np.array([np.linalg.norm(x - mu_n) for x in X_kpca])

    md_n_mean = np.mean(md_vals[:n_normal])
    md_n_std  = np.std(md_vals[:n_normal])
    md_max    = np.max(md_vals)
    hi_md     = np.clip((md_vals - md_n_mean) /
                        (md_max - md_n_mean + 1e-12), 0, 1)

    thr_md = np.clip(3 * md_n_std / (md_max - md_n_mean + 1e-12), 0, 1)
    print(f"  MD 전체 최대: {md_max:.4f}, FPT 임계값: {thr_md:.4f}")

    # ── FPT 탐지 (SPE 기준)
    fpt_idx = np.where(hi_spe > thr_spe)[0]
    if len(fpt_idx) > 0:
        fpt_time = df["time_sec"].iloc[fpt_idx[0]]
        print(f"  FPT(SPE) 탐지: {fpt_time/3600:.2f} hr (파일 #{df['file_idx'].iloc[fpt_idx[0]]})")
    else:
        fpt_time = np.nan
        print("  FPT(SPE) 탐지: 임계값 초과 없음")

    df_r = df.copy()
    df_r["hi_kpca_pc1"]    = hi_pc1.astype(np.float32)
    df_r["hi_kpca_spe"]    = hi_spe.astype(np.float32)
    df_r["hi_kpca_spe_cm"] = np.maximum.accumulate(hi_spe).astype(np.float32)
    df_r["hi_kpca_md"]     = hi_md.astype(np.float32)
    df_r["hi_kpca_md_cm"]  = np.maximum.accumulate(hi_md).astype(np.float32)
    df_r["spe_raw"]        = spe_raw.astype(np.float32)
    df_r["md_raw"]         = md_vals.astype(np.float32)
    df_r["fpt_sec"]        = fpt_time

    return df_r, kpca, scaler, thr_spe, thr_md


# ─────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────

def plot_bearing(bear_id: int, df: pd.DataFrame,
                 thr_spe: float, thr_md: float, save_dir: str):
    """3종 HI 비교: PC1 / SPE / MD"""
    time_hr  = df["time_sec"] / 3600
    n_normal = max(int(len(df) * NORMAL_RATIO), 5)
    t_ne     = time_hr.iloc[n_normal - 1]
    color    = COLORS[bear_id - 1]

    fpt_hr = df["fpt_sec"].iloc[0] / 3600 if not np.isnan(df["fpt_sec"].iloc[0]) else None

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    fig.suptitle(f"Bearing {bear_id} — KPCA-based HI (PC1 / SPE / Mahalanobis)",
                 fontsize=13, fontweight="bold")

    specs = [
        ("hi_kpca_pc1",    "hi_kpca_pc1",    None,    "KPCA PC1 Score",          None),
        ("hi_kpca_spe",    "hi_kpca_spe_cm", thr_spe, "KPCA SPE",                "KPCA SPE (cummax)"),
        ("hi_kpca_md",     "hi_kpca_md_cm",  thr_md,  "KPCA Mahalanobis Dist.",  "MD (cummax)"),
    ]

    for ax, (col_main, col_cm, thr, title, cm_label) in zip(axes, specs):
        ax.plot(time_hr, df[col_main], color=color,
                linewidth=1.3, alpha=0.75, label=title)
        if col_cm != col_main:
            ax.plot(time_hr, df[col_cm], color=color,
                    linewidth=1.6, linestyle="--", alpha=0.95, label=cm_label)
        if thr is not None:
            ax.axhline(thr, color="red", linestyle=":", linewidth=0.9,
                       label=f"FPT thr. ({thr:.3f})")
        if fpt_hr:
            ax.axvline(fpt_hr, color="orange", linestyle=":", linewidth=1.5,
                       label=f"FPT ({fpt_hr:.1f}hr)")
        ax.axvspan(0, t_ne, alpha=0.08, color="green")
        ax.axvline(t_ne, color="green", linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("HI", fontsize=9)
        ax.legend(fontsize=7, loc="upper left", ncol=3)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Time [hr]", fontsize=10)
    plt.tight_layout()
    path = os.path.join(save_dir, f"Bearing{bear_id}_HI_KPCA.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [저장] {path}")


def plot_all(hi_dfs: dict, save_dir: str):
    """전체 베어링 KPCA SPE HI 비교"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()
    fig.suptitle("KPCA-based Health Index — All Bearings\n"
                 "Solid: SPE HI  |  Dashed: SPE HI (cummax)  |  Dotted: MD HI (cummax)",
                 fontsize=12, fontweight="bold")

    for i, (bear_id, info) in enumerate(hi_dfs.items()):
        df, thr = info["df"], info["thr_spe"]
        ax = axes[i]
        t  = df["time_sec"] / 3600
        n_normal = max(int(len(df) * NORMAL_RATIO), 5)
        t_ne = t.iloc[n_normal - 1]
        c  = COLORS[i]

        ax.plot(t, df["hi_kpca_spe"],    color=c, lw=1.2, alpha=0.65, label="SPE")
        ax.plot(t, df["hi_kpca_spe_cm"], color=c, lw=1.7, ls="--",    label="SPE (cummax)")
        ax.plot(t, df["hi_kpca_md_cm"],  color="gray", lw=1.2, ls=":",
                alpha=0.8, label="MD (cummax)")
        ax.axhline(thr, color="red", ls=":", lw=0.9)
        ax.axvspan(0, t_ne, alpha=0.08, color="green")
        ax.axvline(t_ne, color="green", ls="--", lw=0.8)

        if not np.isnan(df["fpt_sec"].iloc[0]):
            ax.axvline(df["fpt_sec"].iloc[0] / 3600,
                       color="orange", ls=":", lw=1.5)

        ax.set_xlim(left=0); ax.set_ylim(-0.05, 1.15)
        ax.set_title(f"Bearing {bear_id} ({t.max():.1f}hr)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time [hr]", fontsize=9)
        ax.set_ylabel("HI", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(save_dir, "All_HI_KPCA_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[저장] 전체 비교 → {path}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  KPCA 기반 Health Index 구성 (PC1 / SPE / Mahalanobis)")
    print("=" * 65)

    bearing_ids = [1, 2, 3, 4]
    hi_dfs      = {}
    summary     = []

    for bear_id in bearing_ids:
        print(f"\n{'='*20} Bearing {bear_id} {'='*20}")
        df_feat = load_features(bear_id)

        df_hi, kpca, scaler, thr_spe, thr_md = build_hi_kpca(
            df_feat, normal_ratio=NORMAL_RATIO,
            n_components=N_COMPONENTS, gamma=GAMMA)

        # CSV 저장
        out_cols = ["bearing_id", "file_idx", "time_sec",
                    "hi_kpca_pc1", "hi_kpca_spe", "hi_kpca_spe_cm",
                    "hi_kpca_md", "hi_kpca_md_cm",
                    "spe_raw", "md_raw", "fpt_sec"]
        csv_path = os.path.join(OUTPUT_DIR, f"Bearing{bear_id}_HI_KPCA.csv")
        df_hi[out_cols].to_csv(csv_path, index=False)
        print(f"  [저장] {csv_path}")

        # 개별 플롯
        plot_bearing(bear_id, df_hi, thr_spe, thr_md, OUTPUT_DIR)

        hi_dfs[bear_id] = {"df": df_hi, "thr_spe": thr_spe, "thr_md": thr_md}

        summary.append({
            "Bearing":       bear_id,
            "N_segments":    len(df_hi),
            "Duration_hr":   round(df_hi["time_sec"].max() / 3600, 2),
            "FPT_hr(SPE)":   round(df_hi["fpt_sec"].iloc[0] / 3600, 3)
                             if not np.isnan(df_hi["fpt_sec"].iloc[0]) else "N/A",
            "HI_SPE_final":  round(float(df_hi["hi_kpca_spe"].iloc[-1]), 4),
            "HI_SPEcm_final":round(float(df_hi["hi_kpca_spe_cm"].iloc[-1]), 4),
            "HI_MD_final":   round(float(df_hi["hi_kpca_md"].iloc[-1]), 4),
            "thr_SPE":       round(float(thr_spe), 4),
            "thr_MD":        round(float(thr_md), 4),
        })

    # 전체 비교 플롯
    plot_all(hi_dfs, OUTPUT_DIR)

    # 요약
    print("\n" + "=" * 65)
    print("  HI 요약 통계")
    print("=" * 65)
    df_sum = pd.DataFrame(summary)
    print(df_sum.to_string(index=False))
    sum_path = os.path.join(OUTPUT_DIR, "HI_KPCA_summary.csv")
    df_sum.to_csv(sum_path, index=False)
    print(f"\n[저장] 요약 → {sum_path}")
    print("\n[완료]")


if __name__ == "__main__":
    main()
