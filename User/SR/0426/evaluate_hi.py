"""
HI (Health Index) 품질 평가 및 시각화.

각 Train run의 추출된 특성(features CSV)에서 대표 HI 후보들을 평가한다.
평가 지표:
  - Monotonicity (Mon): HI가 시간에 따라 단조적으로 변하는 정도
  - Trendability (Tre): HI의 상관관계 (시간과의 Spearman 상관)
  - Prognosability (Pro): HI의 초기/말기 분포 분리도

출력:
  - plots/hi_metrics_comparison.png   — 지표별 비교 히트맵
  - plots/hi_trajectories_top10.png   — 상위 10개 HI의 시간 추이
  - results/hi_metrics.txt            — 전체 평가 결과 텍스트
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
FEAT_DIR = ROOT / "features"
PLOT_DIR = ROOT / "plots"; PLOT_DIR.mkdir(exist_ok=True)
RESULT_DIR = ROOT / "results"; RESULT_DIR.mkdir(exist_ok=True)

# 4채널 평균으로 pool하여 대표 HI 후보 생성
CHANNELS = ["CH1", "CH2", "CH3", "CH4"]


def pool_channels(df: pd.DataFrame, key: str) -> pd.Series:
    cols = [c for c in df.columns
            if c.endswith(f"_{key}") and c.split("_")[0] in CHANNELS]
    if not cols:
        return None
    return df[cols].mean(axis=1)


# HI 후보 목록
HI_KEYS = [
    "rms", "peak", "std", "kurt", "skew", "crest", "impulse", "shape", "margin",
    "BPFI_h1", "BPFI_h2", "BPFI_h3",
    "BPFO_h1", "BPFO_h2", "BPFO_h3",
    "BSF_h1", "BSF_h2", "BSF_h3",
    "FTF_h1", "FTF_h2", "FTF_h3",
    "BPFI_sb_p", "BPFI_sb_m", "BPFO_sb_p", "BPFO_sb_m",
    "BSF_sb_p", "BSF_sb_m",
    "E_low", "E_mid", "E_high", "E_vhigh",
]


# ─────────────────── 평가 지표 ───────────────────

def monotonicity(x: np.ndarray) -> float:
    """
    Monotonicity: 연속된 샘플 간 증감 방향의 일관성.
    Mon = |sum(sign(Δx))| / (N-1), ∈ [0, 1]. 1이면 완전 단조.
    """
    d = np.diff(x)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    return abs(float(np.sum(np.sign(d)))) / len(d)


def trendability(x: np.ndarray) -> float:
    """
    Trendability: 시간 인덱스와의 Spearman 상관 계수의 절댓값.
    |ρ| ∈ [0, 1]. 1이면 완벽한 단조 추세.
    """
    if x.std() < 1e-12:
        return 0.0
    rho, _ = spearmanr(np.arange(len(x)), x)
    return abs(float(rho))


def prognosability(x: np.ndarray, early_frac: float = 0.2, late_frac: float = 0.2) -> float:
    """
    Prognosability: 초기 구간과 말기 구간의 분포가 얼마나 분리되는지.
    Pro = 1 - exp(-|μ_late - μ_early| / max(σ_early + σ_late, ε))
    ∈ [0, 1]. 1이면 완전히 분리됨.
    """
    n = len(x)
    n_early = max(int(n * early_frac), 1)
    n_late = max(int(n * late_frac), 1)
    early = x[:n_early]
    late = x[-n_late:]
    mu_diff = abs(late.mean() - early.mean())
    sigma_sum = early.std() + late.std() + 1e-12
    return float(1.0 - np.exp(-mu_diff / sigma_sum))


def composite_score(mon: float, tre: float, pro: float) -> float:
    """Composite = (Mon + Tre + Pro) / 3"""
    return (mon + tre + pro) / 3.0


# ─────────────────── 메인 ───────────────────

def main():
    csvs = sorted(FEAT_DIR.glob("Train*_features.csv"))
    if not csvs:
        print("No feature CSVs found. Run extract_all.py first.")
        return

    # 전체 결과 저장용
    all_results = []
    run_results = {}   # run_name -> DataFrame of metrics

    for fp in csvs:
        run_name = fp.stem.split("_")[0]
        df = pd.read_csv(fp).sort_values("idx").reset_index(drop=True)
        rows = []
        for key in HI_KEYS:
            series = pool_channels(df, key)
            if series is None:
                continue
            x = series.values.astype(np.float64)
            mon = monotonicity(x)
            tre = trendability(x)
            pro = prognosability(x)
            comp = composite_score(mon, tre, pro)
            rows.append({
                "run": run_name, "HI": key,
                "Monotonicity": mon, "Trendability": tre,
                "Prognosability": pro, "Composite": comp,
            })
        run_df = pd.DataFrame(rows)
        run_results[run_name] = run_df
        all_results.append(run_df)

    combined = pd.concat(all_results, ignore_index=True)

    # ─── 평균 지표 (전 run 평균) ───
    avg = combined.groupby("HI")[["Monotonicity", "Trendability", "Prognosability", "Composite"]].mean()
    avg = avg.sort_values("Composite", ascending=False)

    # ─── 텍스트 결과 저장 ───
    lines = []
    lines.append("=" * 80)
    lines.append("HI (Health Index) 품질 평가 결과")
    lines.append("=" * 80)
    lines.append("")
    lines.append("평가 지표 설명:")
    lines.append("  Monotonicity (Mon): 시간에 따른 단조 변화 정도 [0-1], 1이 최고")
    lines.append("  Trendability (Tre): 시간과의 Spearman 상관 (절대값) [0-1], 1이 최고")
    lines.append("  Prognosability (Pro): 초기/말기 분포 분리도 [0-1], 1이 최고")
    lines.append("  Composite: 세 지표의 평균 [0-1]")
    lines.append("")

    # 전체 평균
    lines.append("-" * 80)
    lines.append("■ 전체 Run 평균 (Composite 내림차순)")
    lines.append("-" * 80)
    lines.append(f"{'HI':<16} {'Mon':>8} {'Tre':>8} {'Pro':>8} {'Comp':>8}")
    lines.append("-" * 48)
    for hi, row in avg.iterrows():
        lines.append(f"{hi:<16} {row['Monotonicity']:>8.4f} {row['Trendability']:>8.4f} "
                     f"{row['Prognosability']:>8.4f} {row['Composite']:>8.4f}")

    # Run별 결과
    for run_name, run_df in run_results.items():
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"■ {run_name}")
        lines.append("-" * 80)
        run_sorted = run_df.sort_values("Composite", ascending=False)
        lines.append(f"{'HI':<16} {'Mon':>8} {'Tre':>8} {'Pro':>8} {'Comp':>8}")
        lines.append("-" * 48)
        for _, row in run_sorted.iterrows():
            lines.append(f"{row['HI']:<16} {row['Monotonicity']:>8.4f} {row['Trendability']:>8.4f} "
                         f"{row['Prognosability']:>8.4f} {row['Composite']:>8.4f}")

    # Top 10
    top10 = avg.head(10)
    lines.append("")
    lines.append("=" * 80)
    lines.append("★ 최고 HI 후보 Top 10 (전체 Run 평균 Composite 기준)")
    lines.append("=" * 80)
    for i, (hi, row) in enumerate(top10.iterrows(), 1):
        lines.append(f"  {i:2d}. {hi:<16}  Composite={row['Composite']:.4f}  "
                     f"(Mon={row['Monotonicity']:.3f}, Tre={row['Trendability']:.3f}, "
                     f"Pro={row['Prognosability']:.3f})")

    result_text = "\n".join(lines) + "\n"
    result_path = RESULT_DIR / "hi_metrics.txt"
    result_path.write_text(result_text, encoding="utf-8")
    print(f"[save] {result_path}")
    print(result_text)

    # ─── 히트맵: 전체 평균 지표 비교 ───
    fig, ax = plt.subplots(figsize=(8, max(6, len(avg) * 0.35)))
    data = avg[["Monotonicity", "Trendability", "Prognosability"]].values
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(avg)))
    ax.set_yticklabels(avg.index, fontsize=8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Monotonicity", "Trendability", "Prognosability"])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if data[i, j] > 0.6 else "black"
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=7, color=color)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Score [0-1]")
    ax.set_title("HI Quality Metrics (All Runs Average)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "hi_metrics_comparison.png", dpi=130)
    plt.close(fig)
    print(f"[save] {PLOT_DIR / 'hi_metrics_comparison.png'}")

    # ─── 상위 10개 HI 시간 추이 ───
    top10_keys = list(top10.index)
    n_runs = len(run_results)
    fig, axes = plt.subplots(len(top10_keys), n_runs,
                             figsize=(4 * n_runs, 1.8 * len(top10_keys)),
                             sharex="col")
    if n_runs == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, (run_name, run_df) in enumerate(sorted(run_results.items())):
        csv_path = FEAT_DIR / f"{run_name}_features.csv"
        df = pd.read_csv(csv_path).sort_values("idx").reset_index(drop=True)
        t_hours = df["t_min"].values / 60.0
        for row_idx, key in enumerate(top10_keys):
            ax = axes[row_idx, col_idx]
            series = pool_channels(df, key)
            if series is not None:
                y = series.values
                ax.plot(t_hours, y, ".", ms=1.5, alpha=0.3, color="steelblue")
                # rolling median
                rm = pd.Series(y).rolling(11, center=True, min_periods=1).median()
                ax.plot(t_hours, rm, "-", lw=1.2, color="crimson")
            if row_idx == 0:
                ax.set_title(run_name, fontsize=9, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(key, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.2)
            if row_idx == len(top10_keys) - 1:
                ax.set_xlabel("time [hours]", fontsize=7)

    fig.suptitle("Top 10 HI Candidates — Trajectories per Run", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "hi_trajectories_top10.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {PLOT_DIR / 'hi_trajectories_top10.png'}")

    # ─── Composite 바 차트 ───
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(avg["Composite"].values)
    bars = ax.barh(range(len(avg)), avg["Composite"].values, color=colors)
    ax.set_yticks(range(len(avg)))
    ax.set_yticklabels(avg.index, fontsize=7)
    ax.set_xlabel("Composite Score", fontsize=10)
    ax.set_title("HI Quality Ranking (Composite = avg of Mon + Tre + Pro)", fontsize=11)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="gray", ls="--", alpha=0.5, label="threshold=0.5")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "hi_composite_ranking.png", dpi=130)
    plt.close(fig)
    print(f"[save] {PLOT_DIR / 'hi_composite_ranking.png'}")


if __name__ == "__main__":
    main()
