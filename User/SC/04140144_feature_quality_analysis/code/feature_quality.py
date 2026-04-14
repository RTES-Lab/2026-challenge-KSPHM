"""
특징 품질 분석: 단조성(Monotonicity) & 추세성(Trendability)
=============================================================
대회: 2026 KSPHM-KIMM 기계 데이터 챌린지
실험: 04140144_feature_quality_analysis

목적:
  76개 특징 각각에 대해 열화 추적 능력을 정량 평가
  → 단조성, 추세성 기준으로 HI 구성에 유효한 특징 선별

평가 지표 정의:
  1. Monotonicity (단조성): 특징이 시간에 따라 일관되게 증가하거나 감소하는 경향
     Mon(f) = (1/(N-1)) * |#{Δf > 0} - #{Δf < 0}|
     범위: [0, 1], 1에 가까울수록 단조 증가/감소가 강함

  2. Trendability (추세성): 특징값과 시간 사이의 선형/순위 상관 강도
     Tre(f) = |Spearman's ρ(f, t)|
     범위: [0, 1], 1에 가까울수록 시간에 따른 일관된 추세 존재

  두 지표를 각 베어링별 계산 후 평균값으로 최종 순위 결정

정규화 관련 주의사항:
  - features.csv는 raw(비정규화) 값 포함
  - 단조성/추세성은 특징의 상대적 순위/트렌드를 보므로 raw 값으로 계산
  - 단, 시각화 시에는 MinMaxScaler([0,1])로 정규화하여 비교

데이터 특이사항:
  - Train1~4 베어링은 각각 다른 종류의 미세 고장이 실험 전 인가됨
  - 가변속 조건(700~950 RPM) → 특징이 RPM 변화에 의해 영향받을 수 있음
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 경로
# ─────────────────────────────────────────────
BASE_DIR   = "/data/home/ksphm/2026-challenge-KSPHM"
CACHE_DIR  = os.path.join(BASE_DIR, "User", "SC",
                          "04140103_initial_pca_result", "output")
OUTPUT_DIR = os.path.join(BASE_DIR, "User", "SC",
                          "04140144_feature_quality_analysis", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEARING_IDS  = [1, 2, 3, 4]
NORMAL_RATIO = 0.15

# ─────────────────────────────────────────────
# 단조성 / 추세성 계산
# ─────────────────────────────────────────────

def monotonicity(x: np.ndarray) -> float:
    """
    Monotonicity (단조성)
    ─────────────────────────────────────────
    식: Mon(f) = (1 / (N-1)) * |#{Δx > 0} - #{Δx < 0}|
    
    - Δx = x[i+1] - x[i]  (연속 차분)
    - #{Δx > 0}: 증가 횟수,  #{Δx < 0}: 감소 횟수
    - |증가 - 감소| / (N-1): 얼마나 한쪽 방향으로만 변하는가
    - 범위: [0, 1]
    - 완전 단조 증가/감소 → 1.0
    - 랜덤 노이즈 → ~0.0
    """
    dx      = np.diff(x)
    n_pos   = np.sum(dx > 0)
    n_neg   = np.sum(dx < 0)
    return abs(n_pos - n_neg) / (len(dx) + 1e-12)


def trendability(x: np.ndarray) -> float:
    """
    Trendability (추세성)
    ─────────────────────────────────────────
    식: Tre(f) = |Spearman's ρ(f, t)|

    - Spearman 순위 상관계수: 단조 관계의 강도를 측정
    - Pearson과 달리 선형성 가정 없음 → 비선형 추세도 포착
    - |ρ|: 증가/감소 방향 무관하게 추세 강도만 측정
    - 범위: [0, 1]
    - 시간에 따라 일관되게 변하는 특징 → 1.0
    - 무작위 변동 → ~0.0
    """
    t   = np.arange(len(x))
    rho, _ = spearmanr(x, t)
    return abs(rho) if not np.isnan(rho) else 0.0


def compute_feature_quality(df: pd.DataFrame) -> pd.DataFrame:
    """단일 베어링의 모든 특징에 대해 Mon/Tre 계산"""
    feat_cols = [c for c in df.columns
                 if c not in ["file_idx", "time_sec", "bearing_id"]]
    rows = []
    for col in feat_cols:
        x   = df[col].values.astype(np.float64)
        mon = monotonicity(x)
        tre = trendability(x)
        rows.append({"feature": col, "monotonicity": mon, "trendability": tre})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 데이터 로드 및 지표 계산
# ─────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  특징 품질 분석: Monotonicity & Trendability")
    print("=" * 70)

    all_quality = {}   # {bear_id: DataFrame}

    for bear_id in BEARING_IDS:
        path = os.path.join(CACHE_DIR, f"Bearing{bear_id}_features.csv")
        print(f"\n[Bearing {bear_id}] 로드: {path}")
        df = pd.read_csv(path)
        q  = compute_feature_quality(df)
        q["bearing_id"] = bear_id
        all_quality[bear_id] = q

    # ── 베어링별 평균 지표
    feat_cols = all_quality[1]["feature"].tolist()
    summary   = pd.DataFrame({"feature": feat_cols})

    for bear_id in BEARING_IDS:
        q = all_quality[bear_id].set_index("feature")
        summary[f"Mon_B{bear_id}"] = summary["feature"].map(q["monotonicity"])
        summary[f"Tre_B{bear_id}"] = summary["feature"].map(q["trendability"])

    # 전체 평균
    mon_cols = [f"Mon_B{b}" for b in BEARING_IDS]
    tre_cols = [f"Tre_B{b}" for b in BEARING_IDS]
    summary["Mon_mean"] = summary[mon_cols].mean(axis=1)
    summary["Tre_mean"] = summary[tre_cols].mean(axis=1)
    summary["Quality"]  = (summary["Mon_mean"] + summary["Tre_mean"]) / 2

    summary = summary.sort_values("Quality", ascending=False).reset_index(drop=True)
    summary["rank"] = range(1, len(summary) + 1)

    # ── 출력
    print("\n" + "=" * 70)
    print("  전체 특징 품질 지표 (상위 → 하위 정렬)")
    print("=" * 70)
    print(f"{'Rank':>4}  {'Feature':<30} {'Mon_mean':>9} {'Tre_mean':>9} {'Quality':>9}")
    print("-" * 70)
    for _, row in summary.iterrows():
        print(f"{int(row['rank']):>4}  {row['feature']:<30}"
              f" {row['Mon_mean']:>9.4f} {row['Tre_mean']:>9.4f}"
              f" {row['Quality']:>9.4f}")

    # ── 저장
    csv_path = os.path.join(OUTPUT_DIR, "feature_quality_all.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\n[저장] 전체 특징 품질 CSV → {csv_path}")

    # ── 임계값 기반 선별
    MON_THR = 0.3
    TRE_THR = 0.3
    selected = summary[(summary["Mon_mean"] >= MON_THR) |
                       (summary["Tre_mean"] >= TRE_THR)]
    print(f"\n선별 기준: Mon ≥ {MON_THR} OR Tre ≥ {TRE_THR}")
    print(f"선별된 특징 수: {len(selected)} / {len(summary)}")
    sel_list = selected["feature"].tolist()
    print(f"선별 특징 목록:\n  {sel_list}")

    sel_path = os.path.join(OUTPUT_DIR, "selected_features.txt")
    with open(sel_path, "w") as f:
        for feat in sel_list:
            f.write(feat + "\n")
    print(f"[저장] 선별 특징 목록 → {sel_path}")

    # ── 시각화 1: 버블 차트 (Mon vs Tre, 크기=Quality)
    _plot_bubble(summary, OUTPUT_DIR)

    # ── 시각화 2: 상위 20개 특징 막대 그래프
    _plot_top20_bar(summary, OUTPUT_DIR)

    # ── 시각화 3: 베어링별 Mon/Tre 히트맵
    _plot_heatmap(summary, BEARING_IDS, OUTPUT_DIR)

    # ── 시각화 4: 선별 특징의 정규화된 시계열 (베어링별)
    _plot_selected_timeseries(BEARING_IDS, sel_list[:10], OUTPUT_DIR)

    print("\n[완료]")


# ─────────────────────────────────────────────
# 시각화 함수들
# ─────────────────────────────────────────────

def _plot_bubble(summary: pd.DataFrame, save_dir: str):
    """Monotonicity vs Trendability 버블 차트"""
    fig, ax = plt.subplots(figsize=(12, 8))

    sc = ax.scatter(
        summary["Mon_mean"], summary["Tre_mean"],
        c=summary["Quality"], cmap="RdYlGn",
        s=summary["Quality"] * 400 + 30,
        alpha=0.7, edgecolors="gray", linewidths=0.5
    )
    plt.colorbar(sc, ax=ax, label="Quality (Mon+Tre)/2")

    # 상위 15개 레이블 표시
    for _, row in summary.head(15).iterrows():
        ax.annotate(row["feature"], (row["Mon_mean"], row["Tre_mean"]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

    ax.axvline(0.3, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Mon threshold (0.3)")
    ax.axhline(0.3, color="blue", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Tre threshold (0.3)")
    ax.set_xlabel("Monotonicity (mean over 4 bearings)", fontsize=11)
    ax.set_ylabel("Trendability (mean over 4 bearings)", fontsize=11)
    ax.set_title("Feature Quality: Monotonicity vs Trendability\n"
                 "(size & color = Quality score)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    path = os.path.join(save_dir, "feature_quality_bubble.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] 버블 차트 → {path}")


def _plot_top20_bar(summary: pd.DataFrame, save_dir: str):
    """상위 20개 특징 Mon/Tre 막대 그래프"""
    top20 = summary.head(20)
    x     = np.arange(len(top20))
    w     = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w/2, top20["Mon_mean"], width=w, label="Monotonicity",
           color="#4C72B0", alpha=0.85)
    ax.bar(x + w/2, top20["Tre_mean"], width=w, label="Trendability",
           color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(top20["feature"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Top-20 Features by Quality Score (Mon & Tre)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(0.3, color="red", linestyle="--", linewidth=0.8,
               alpha=0.6, label="threshold=0.3")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    path = os.path.join(save_dir, "feature_quality_top20_bar.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] 상위 20개 막대 그래프 → {path}")


def _plot_heatmap(summary: pd.DataFrame, bearing_ids: list, save_dir: str):
    """베어링별 Mon/Tre 히트맵 (상위 30개 특징)"""
    top30 = summary.head(30)["feature"].tolist()

    mon_data = summary[summary["feature"].isin(top30)].set_index("feature")
    tre_data = mon_data.copy()
    mon_mat  = mon_data[[f"Mon_B{b}" for b in bearing_ids]].loc[top30].T.values
    tre_mat  = tre_data[[f"Tre_B{b}" for b in bearing_ids]].loc[top30].T.values

    fig, axes = plt.subplots(1, 2, figsize=(22, 5))

    for ax, mat, title, cmap in zip(
        axes, [mon_mat, tre_mat],
        ["Monotonicity per Bearing", "Trendability per Bearing"],
        ["Blues", "Oranges"]
    ):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(top30)))
        ax.set_xticklabels(top30, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(bearing_ids)))
        ax.set_yticklabels([f"B{b}" for b in bearing_ids], fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax)

        # 셀 값 표시
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.2f}",
                        ha="center", va="center", fontsize=5.5,
                        color="black" if mat[i,j] < 0.7 else "white")

    fig.suptitle("Top-30 Features: Mon/Tre Heatmap by Bearing",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "feature_quality_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] 히트맵 → {path}")


def _plot_selected_timeseries(bearing_ids: list, feat_list: list, save_dir: str):
    """선별 상위 특징의 정규화 시계열 (베어링별 subplot)"""
    n_feat = len(feat_list)
    if n_feat == 0:
        return

    for bear_id in bearing_ids:
        path = os.path.join(CACHE_DIR, f"Bearing{bear_id}_features.csv")
        df   = pd.read_csv(path)
        t    = df["time_sec"].values / 3600  # hr

        fig, axes = plt.subplots(n_feat, 1, figsize=(12, 2.2 * n_feat), sharex=True)
        if n_feat == 1:
            axes = [axes]
        fig.suptitle(f"Bearing {bear_id} — Top-{n_feat} 선별 특징 (MinMax 정규화)",
                     fontsize=12, fontweight="bold")

        scaler = MinMaxScaler()
        for ax, feat in zip(axes, feat_list):
            if feat not in df.columns:
                continue
            x_norm = scaler.fit_transform(df[[feat]].values).flatten()
            ax.plot(t, x_norm, linewidth=1.0, color="#4C72B0", alpha=0.85)
            ax.set_ylabel(feat, fontsize=7, rotation=0, ha="right", labelpad=80)
            ax.set_ylim(-0.05, 1.15)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.tick_params(labelsize=7)

        axes[-1].set_xlabel("Time [hr]", fontsize=10)
        plt.tight_layout()
        ts_path = os.path.join(save_dir, f"Bearing{bear_id}_top_features_timeseries.png")
        plt.savefig(ts_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"[저장] Bearing {bear_id} 시계열 → {ts_path}")


if __name__ == "__main__":
    main()
