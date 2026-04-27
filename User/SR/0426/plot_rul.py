"""
RUL 예측 결과 시각화 및 점수 정리.

sanity_Train*.csv 파일을 읽어서:
  1) Run별 RUL 예측 vs 실측 그래프 (pred vs true 곡선)
  2) Run별 Competition Score 추이 그래프
  3) 전체 결과 요약 텍스트

출력:
  - plots/rul_pred_vs_true.png    — 전체 Run RUL 예측 곡선
  - plots/rul_score_trajectory.png — Competition Score 추이
  - results/rul_scores.txt         — 점수 요약 텍스트
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
PLOT_DIR = ROOT / "plots"; PLOT_DIR.mkdir(exist_ok=True)
RESULT_DIR = ROOT / "results"; RESULT_DIR.mkdir(exist_ok=True)

COLORS = {"Train1": "#1f77b4", "Train2": "#ff7f0e",
          "Train3": "#2ca02c", "Train4": "#d62728"}


def main():
    csvs = sorted(ROOT.glob("sanity_Train*.csv"))
    if not csvs:
        print("No sanity_Train*.csv files found. Run sanity_check.py first.")
        return

    runs = {}
    for fp in csvs:
        name = fp.stem.replace("sanity_", "")
        df = pd.read_csv(fp).sort_values("idx").reset_index(drop=True)
        runs[name] = df

    # ─── 1. RUL Predicted vs True (all runs) ───
    n_runs = len(runs)
    fig, axes = plt.subplots(n_runs, 1, figsize=(12, 3.5 * n_runs), sharex=False)
    if n_runs == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, sorted(runs.items())):
        t_hours = df["t_min"].values / 60.0
        true_sec = df["rul_true_sec"].values
        pred_sec = df["rul_pred_sec"].values

        # convert to minutes for readability
        true_min = true_sec / 60.0
        pred_min = pred_sec / 60.0

        color = COLORS.get(name, "steelblue")
        ax.plot(t_hours, true_min, "-", lw=2, color="black", label="True RUL", alpha=0.8)
        ax.plot(t_hours, pred_min, "-", lw=1.8, color=color, label="Predicted RUL", alpha=0.8)
        ax.fill_between(t_hours, pred_min, true_min, alpha=0.15, color=color)

        # final frame annotation
        last_true = true_min[-1]
        last_pred = pred_min[-1]
        last_score = df["score"].iloc[-1]
        ax.annotate(f"pred={last_pred:.1f}min\ntrue={last_true:.1f}min\nscore={last_score:.3f}",
                    xy=(t_hours[-1], pred_min[-1]),
                    xytext=(-120, 40), textcoords="offset points",
                    fontsize=8, ha="center",
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

        ax.set_ylabel("RUL [min]", fontsize=10)
        ax.set_title(f"{name} — RUL Prediction vs Ground Truth", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_xlabel("time [hours]", fontsize=10)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rul_pred_vs_true.png", dpi=130)
    plt.close(fig)
    print(f"[save] {PLOT_DIR / 'rul_pred_vs_true.png'}")

    # ─── 2. Competition Score trajectory ───
    fig, axes = plt.subplots(n_runs, 1, figsize=(12, 3 * n_runs), sharex=False)
    if n_runs == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, sorted(runs.items())):
        t_hours = df["t_min"].values / 60.0
        scores = df["score"].values
        color = COLORS.get(name, "steelblue")

        ax.plot(t_hours, scores, "-o", ms=3, lw=1.5, color=color, alpha=0.7)
        ax.axhline(0.5, color="gray", ls="--", alpha=0.4, label="score=0.5")
        ax.axhline(scores.mean(), color="crimson", ls=":", alpha=0.6,
                   label=f"mean={scores.mean():.3f}")

        # highlight last 25%
        n_late = max(1, int(len(scores) * 0.25))
        late_mean = scores[-n_late:].mean()
        ax.axvspan(t_hours[-n_late], t_hours[-1], alpha=0.08, color="red",
                   label=f"late-25% mean={late_mean:.3f}")

        ax.set_ylabel("Competition Score", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{name} — Competition Score Trajectory", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)
        ax.set_xlabel("time [hours]", fontsize=10)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rul_score_trajectory.png", dpi=130)
    plt.close(fig)
    print(f"[save] {PLOT_DIR / 'rul_score_trajectory.png'}")

    # ─── 3. Combined comparison (all runs in one plot) ───
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    for name, df in sorted(runs.items()):
        # Normalize time to 0-1 (fraction of lifetime)
        t_frac = df["t_min"].values / df["t_min"].values.max()
        color = COLORS.get(name, "steelblue")

        true_norm = df["rul_true_sec"].values / df["rul_true_sec"].values.max() if df["rul_true_sec"].values.max() > 0 else df["rul_true_sec"].values
        pred_norm = df["rul_norm_pred"].values

        ax1.plot(t_frac, pred_norm, "-", lw=1.5, color=color, label=f"{name} (pred)", alpha=0.8)
        ax2.plot(t_frac, df["score"].values, "-", lw=1.5, color=color, label=name, alpha=0.8)

    # ideal line on ax1
    ax1.plot([0, 1], [1, 0], "--", color="black", lw=2, alpha=0.5, label="Ideal (linear)")
    ax1.set_xlabel("Normalized Lifetime", fontsize=10)
    ax1.set_ylabel("Normalized RUL", fontsize=10)
    ax1.set_title("All Runs — Normalized RUL Prediction", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1); ax1.set_ylim(-0.05, 1.05)

    ax2.axhline(0.5, color="gray", ls="--", alpha=0.4)
    ax2.set_xlabel("Normalized Lifetime", fontsize=10)
    ax2.set_ylabel("Competition Score", fontsize=10)
    ax2.set_title("All Runs — Score Comparison", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1); ax2.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rul_all_runs_comparison.png", dpi=130)
    plt.close(fig)
    print(f"[save] {PLOT_DIR / 'rul_all_runs_comparison.png'}")

    # ─── 4. 텍스트 결과 저장 ───
    lines = []
    lines.append("=" * 80)
    lines.append("RUL 예측 결과 요약")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"평가 대상: {', '.join(sorted(runs.keys()))}")
    lines.append("")

    lines.append("-" * 80)
    lines.append(f"{'Run':<10} {'N files':>8} {'Lifetime':>12} {'Last RUL pred':>14} "
                 f"{'Last RUL true':>14} {'Last Score':>12} {'Mean Score':>12} {'Late-25% Score':>14}")
    lines.append("-" * 80)

    summary_rows = []
    for name, df in sorted(runs.items()):
        n = len(df)
        lifetime_min = df["rul_true_sec"].iloc[0] / 60.0 + df["t_min"].iloc[0]
        last_pred_min = df["rul_pred_sec"].iloc[-1] / 60.0
        last_true_min = df["rul_true_sec"].iloc[-1] / 60.0
        last_score = df["score"].iloc[-1]
        mean_score = df["score"].mean()
        n_late = max(1, int(n * 0.25))
        late_score = df["score"].iloc[-n_late:].mean()

        lines.append(f"{name:<10} {n:>8} {lifetime_min:>10.1f}min {last_pred_min:>12.1f}min "
                     f"{last_true_min:>12.1f}min {last_score:>12.4f} {mean_score:>12.4f} {late_score:>14.4f}")
        summary_rows.append({
            "run": name, "n_files": n,
            "lifetime_min": lifetime_min,
            "last_pred_min": last_pred_min,
            "last_true_min": last_true_min,
            "last_score": last_score,
            "mean_score": mean_score,
            "late_25pct_score": late_score,
        })

    # 전체 평균
    mean_last = np.mean([r["last_score"] for r in summary_rows])
    mean_all = np.mean([r["mean_score"] for r in summary_rows])
    mean_late = np.mean([r["late_25pct_score"] for r in summary_rows])
    lines.append("-" * 80)
    lines.append(f"{'AVERAGE':<10} {'':>8} {'':>12} {'':>14} {'':>14} "
                 f"{mean_last:>12.4f} {mean_all:>12.4f} {mean_late:>14.4f}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("점수 해석:")
    lines.append("  - Competition Score ∈ (0, 1]: 1이 완벽한 예측")
    lines.append("  - Last Score: 수명 종료 시점에서의 마지막 예측 점수 (대회 제출 기준)")
    lines.append("  - Mean Score: 전체 시간에 걸친 평균 점수")
    lines.append("  - Late-25% Score: 수명 후반 25% 구간의 평균 점수")
    lines.append("")
    lines.append("주의사항:")
    lines.append("  - rul_true_sec=0 시점의 score=0은 분모가 0이 되는 경계 케이스")
    lines.append("  - 과대예측(predicted > true)은 τ=20으로 가혹한 페널티")
    lines.append("  - 과소예측(predicted < true)은 τ=50으로 완화된 페널티")
    lines.append("=" * 80)

    result_text = "\n".join(lines) + "\n"
    result_path = RESULT_DIR / "rul_scores.txt"
    result_path.write_text(result_text, encoding="utf-8")
    print(f"[save] {result_path}")
    print(result_text)


if __name__ == "__main__":
    main()
