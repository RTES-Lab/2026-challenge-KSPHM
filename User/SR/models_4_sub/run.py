"""
RUL Model Benchmark  —  run.py
================================
Evaluate multiple RUL models with LOO cross-validation, then predict test RUL.

HI 입력 형식 (단일 CSV, 누가 만들든 이 포맷만 맞추면 됨)
──────────────────────────────────────────────────────
  train_hi.csv          test_hi.csv
  ────────────          ───────────
  id, HI [,regime]      id, HI [,regime]
  1, 0.001 [,0]         1, 0.012 [,0]
  1, 0.014 [,0]         1, 0.025 [,0]
  ...                   ...
  2, 0.003 [,1]         2, 0.008 [,1]
  ...                   ...

  - id    : bearing 번호 (train 1~4, test 1~6)
  - HI    : 건강 지수값 (어떤 스케일이든 무관)
  - regime: 선택 컬럼 (없어도 동작)
  - 같은 id 내 행은 반드시 시간순 정렬

Usage
-----
# 기본 (내장 기본값 사용)
  conda run -n ksphm_env python run.py

# 다른 사람 HI 넣기 — 경로만 지정
  conda run -n ksphm_env python run.py \\
      --hi_train /path/to/train_hi.csv \\
      --hi_test  /path/to/test_hi.csv

# 모델 선택 + bias 보정
  conda run -n ksphm_env python run.py \\
      --hi_train /path/to/train_hi.csv \\
      --models KNN,SVR,RF --bias_search

Output
------
<out_dir>/
  loocv_summary.csv          model x bearing score table
  loocv_comparison.png       ranked bar chart of mean LOO scores
  <ModelName>/
    loocv_all.png            2×2 LOO prediction panels
    B{i}_loo.png             individual bearing LOO plots
  test/
    test_predictions.csv     long-form: model, test_id, rul_hours
    test_pivot.csv           wide-form: model × test bearing
    test_comparison.png      grouped bar chart
"""

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── path setup so utils/models import without install
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from utils  import (load_hi_csv, rul_labels, avg_score,
                     comp_score, INTERVAL_SEC, COMP_EOL, COMP_NU)
from models import BASE_MODELS

# ── competition defaults (기본 HI CSV 경로)
_DEFAULT_TRAIN = BASE / "hi_data/train_hi.csv"
_DEFAULT_TEST  = BASE / "hi_data/test_hi.csv"

TRAIN_IDS = [1, 2, 3, 4]
TEST_IDS  = [1, 2, 3, 4, 5, 6]
WIN_SIZE  = 50
STRIDE    = 1

# Persistent result log (all runs accumulate here)
RESULT_CSV  = BASE / "result.csv"
_RESULT_COLS = ["hi_name", "model",
                "B1", "B2", "B3", "B4", "mean", "bias", "elapsed_s"]


# ═══════════════════════════════════════════════════════════════════════════════
# Core evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def run_loocv(model, data, bearings, eol_dict, nu_dict,
              win_size=WIN_SIZE, stride=STRIDE):
    """
    LOO cross-validation. Leakage guarantee:
      - model.fit() is called with train_bids = bearings \\ {test_bid}
      - test bearing data is never passed to fit()
      - per-fold normalisation uses training folds only

    Returns
    -------
    dict {bearing_id: (t_ends, true_ruls_cycles, pred_ruls_cycles, score)}
    """
    fold_results = {}
    for test_bid in bearings:
        train_bids = [b for b in bearings if b != test_bid]
        model.fit(data, train_bids, eol_dict, nu_dict, win_size)

        hi     = data[test_bid]["hi"]
        regime = data[test_bid]["regime"]
        rul    = rul_labels(len(hi), eol_dict[test_bid], nu_dict[test_bid])
        N      = len(hi)

        t_ends, trues, preds = [], [], []
        for t_s in range(0, N - win_size + 1, stride):
            t_e     = t_s + win_size
            hi_win  = hi[t_s:t_e]
            reg_win = regime[t_s:t_e] if regime is not None else None
            pred    = model.predict(hi_win, reg_win)
            t_ends.append(int(t_e - 1))
            trues.append(float(rul[t_e - 1]))
            preds.append(float(pred))

        sc = avg_score(trues, preds)
        fold_results[test_bid] = (t_ends, trues, preds, sc)
    return fold_results


def run_test_predict(model, train_data, test_data, train_bids, test_ids,
                     eol_dict, nu_dict, win_size=WIN_SIZE):
    """
    Train on all training bearings, predict final RUL for each test bearing.
    Uses the last WIN_SIZE cycles of each test bearing (current observation window).

    Returns {test_id: rul_hours}
    """
    model.fit(train_data, train_bids, eol_dict, nu_dict, win_size)
    results = {}
    for tid in test_ids:
        hi     = test_data[tid]["hi"]
        regime = test_data[tid]["regime"]
        if len(hi) < win_size:
            pad     = win_size - len(hi)
            hi_win  = np.pad(hi,     (pad, 0), "edge")
            reg_win = (np.pad(regime, (pad, 0), "edge")
                       if regime is not None else None)
        else:
            hi_win  = hi[-win_size:]
            reg_win = regime[-win_size:] if regime is not None else None
        cycles = model.predict(hi_win, reg_win)
        results[tid] = cycles * INTERVAL_SEC / 3600  # → hours
    return results


def append_result_csv(row: dict) -> None:
    """Append one LOO result row to the shared persistent result.csv."""
    exists = RESULT_CSV.exists()
    pd.DataFrame([{c: row.get(c, "") for c in _RESULT_COLS}]).to_csv(
        RESULT_CSV, mode="a", header=not exists, index=False
    )
    print(f"     → result.csv  ({RESULT_CSV})")


def bias_search(fold_results, bearings, lo=0.5, hi_val=1.2, step=0.025):
    """
    Grid-search a global multiplicative bias that maximises pooled LOO score.
    Biasing toward under-prediction exploits the asymmetric scoring metric.
    Does NOT use test data → valid post-processing.
    """
    all_true = [r for b in bearings for r in fold_results[b][1]]
    all_pred = [p for b in bearings for p in fold_results[b][2]]
    best_bias, best_sc = 1.0, avg_score(all_true, all_pred)
    for bias in np.arange(lo, hi_val + 1e-9, step):
        sc = avg_score(all_true, [p * bias for p in all_pred])
        if sc > best_sc:
            best_sc, best_bias = sc, float(bias)
    return best_bias, best_sc


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════
_COLORS = {"true": "#2C3E50", "pred": "#E74C3C", "bias": "#27AE60"}


def plot_loocv_per_model(fold_results, out_dir, model_name, bearings,
                          best_bias=1.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = [fold_results[b][3] for b in bearings]
    mean_sc = float(np.nanmean(scores))

    # 2×2 combined panel
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{model_name} — LOO Cross-Validation\n"
        f"Mean: {mean_sc:.4f}  "
        + "  ".join(f"B{b}={fold_results[b][3]:.3f}" for b in bearings),
        fontsize=12, fontweight="bold"
    )
    axes = axes.flatten()

    for ax, bid in zip(axes, bearings):
        t_ends, trues, preds, sc = fold_results[bid]
        t_hr    = np.array(t_ends)  * INTERVAL_SEC / 3600
        true_hr = np.array(trues)   * INTERVAL_SEC / 3600
        pred_hr = np.array(preds)   * INTERVAL_SEC / 3600
        bias_hr = pred_hr * best_bias

        ax.plot(t_hr, true_hr, color=_COLORS["true"], lw=1.5, label="True RUL")
        ax.plot(t_hr, pred_hr, color=_COLORS["pred"],  lw=1.5,
                ls="--", alpha=0.85, label=f"Predicted (score={sc:.3f})")
        if abs(best_bias - 1.0) > 0.01:
            bias_sc = avg_score(trues, [p * best_bias for p in preds])
            ax.plot(t_hr, bias_hr, color=_COLORS["bias"], lw=1.2,
                    ls=":", alpha=0.8, label=f"Biased×{best_bias:.2f} ({bias_sc:.3f})")
        ax.set_title(f"Bearing{bid}", fontsize=10)
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("RUL (hr)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "loocv_all.png", dpi=150)
    plt.close()

    # Individual bearing plots
    for bid in bearings:
        t_ends, trues, preds, sc = fold_results[bid]
        t_hr    = np.array(t_ends)  * INTERVAL_SEC / 3600
        true_hr = np.array(trues)   * INTERVAL_SEC / 3600
        pred_hr = np.array(preds)   * INTERVAL_SEC / 3600

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_hr, true_hr, color=_COLORS["true"], lw=1.5, label="True RUL")
        ax.plot(t_hr, pred_hr, color=_COLORS["pred"],  lw=1.5,
                ls="--", alpha=0.9, label="Predicted")
        if abs(best_bias - 1.0) > 0.01:
            ax.plot(t_hr, pred_hr * best_bias, color=_COLORS["bias"],
                    lw=1.2, ls=":", alpha=0.8,
                    label=f"Biased ×{best_bias:.2f}")
        ax.set_title(
            f"{model_name} — Bearing{bid} LOO  score={sc:.4f}", fontsize=11)
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("RUL (hr)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"B{bid}_loo.png", dpi=150)
        plt.close()


def plot_loocv_comparison(summary_df, out_dir):
    """Ranked horizontal bar chart comparing models on LOO mean score."""
    df = summary_df.sort_values("mean").reset_index(drop=True)
    models = df["model"].tolist()
    means  = df["mean"].tolist()
    n = len(df)

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.7 + 1.5)))
    cmap = plt.cm.RdYlGn
    colors = [cmap(0.15 + 0.7 * i / max(n - 1, 1)) for i in range(n)]

    bars = ax.barh(range(n), means, color=colors, edgecolor="gray", alpha=0.85)
    ax.set_yticks(range(n))
    ax.set_yticklabels(models)
    ax.set_xlabel("Mean LOO Score")
    ax.set_title("Model Comparison — LOO Cross-Validation\n"
                 "(score closer to 1.0 = better)", fontsize=12, fontweight="bold")

    for i, (bar, sc) in enumerate(zip(bars, means)):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{sc:.4f}", va="center", fontsize=9, fontweight="bold")

    # Per-bearing dots
    bearing_cols = [f"B{i}" for i in range(1, 5)]
    cmap2 = plt.cm.Set1
    for j, col in enumerate(bearing_cols):
        if col in df.columns:
            vals = df[col].tolist()
            ax.scatter(vals, range(n), marker="D",
                       color=cmap2(j / 4), s=35, zorder=5,
                       label=f"Bearing{j+1}", alpha=0.8)

    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, min(ax.get_xlim()[1] * 1.12, 1.15))
    ax.axvline(1.0, color="gray", ls=":", lw=1, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loocv_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_dir}/loocv_comparison.png")


def plot_test_comparison(test_rows, out_dir):
    """Grouped bar chart of test RUL predictions across all models."""
    if not test_rows:
        return
    df       = pd.DataFrame(test_rows)
    models   = df["model"].unique().tolist()
    test_ids = sorted(df["test_id"].unique())
    n_m, n_t = len(models), len(test_ids)
    x = np.arange(n_t)
    w = min(0.8 / n_m, 0.15)
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(max(10, n_t * 2.2), 6))
    for i, mn in enumerate(models):
        sub  = df[df["model"] == mn]
        ruls = [float(sub[sub["test_id"] == tid]["rul_hours"].iloc[0])
                for tid in test_ids]
        off  = (i - n_m / 2 + 0.5) * w
        bars = ax.bar(x + off, ruls, w, label=mn,
                      color=cmap(i / max(n_m - 1, 1)), alpha=0.82,
                      edgecolor="gray")
        for bar, v in zip(bars, ruls):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.04,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Test{tid}" for tid in test_ids])
    ax.set_ylabel("Predicted RUL (hours)")
    ax.set_title("Test RUL Predictions — All Models", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "test_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_dir}/test_comparison.png")


def plot_loocv_detail(fold_results, out_dir, model_name, bearings,
                       best_bias=1.0):
    """Per-bearing: true vs predicted scatter and residual histogram."""
    out_dir = Path(out_dir)
    all_true, all_pred = [], []
    for b in bearings:
        all_true.extend(fold_results[b][1])
        all_pred.extend(fold_results[b][2])
    all_true = np.array(all_true) * INTERVAL_SEC / 3600
    all_pred = np.array(all_pred) * best_bias * INTERVAL_SEC / 3600

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter
    ax = axes[0]
    ax.scatter(all_true, all_pred, s=8, alpha=0.4, color="#3498DB")
    lim = max(all_true.max(), all_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="Perfect")
    ax.set_xlabel("True RUL (hr)")
    ax.set_ylabel("Predicted RUL (hr)")
    ax.set_title(f"{model_name} — True vs Predicted (LOO)", fontsize=10)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Relative error histogram
    ax2 = axes[1]
    mask = all_true > 0
    er = 100.0 * (all_true[mask] - all_pred[mask]) / all_true[mask]
    ax2.hist(er, bins=40, color="#E67E22", alpha=0.7, edgecolor="gray")
    ax2.axvline(0, color="black", lw=1.5, ls="--")
    ax2.axvline(float(np.median(er)), color="red", lw=1.5,
                label=f"Median Er={np.median(er):.1f}%")
    ax2.set_xlabel("Relative Error Er = 100*(true-pred)/true  (%)")
    ax2.set_ylabel("Count")
    ax2.set_title("Relative Error Distribution", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.suptitle(f"{model_name}  (bias={best_bias:.2f})", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "loocv_detail.png", dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="RUL Model Benchmark with LOO Cross-Validation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--hi_train", type=str, default=str(_DEFAULT_TRAIN),
        help="Train HI CSV  (columns: id, HI [,regime])\n"
             f"default: {_DEFAULT_TRAIN}")
    p.add_argument(
        "--hi_test", type=str, default=str(_DEFAULT_TEST),
        help="Test HI CSV  (columns: id, HI [,regime])\n"
             f"default: {_DEFAULT_TEST}")
    p.add_argument(
        "--out", type=str, default=None,
        help="Output directory  (default: output/<MMDD_HHMMSS>)")
    p.add_argument(
        "--win_size", type=int, default=WIN_SIZE,
        help=f"Sliding window size in cycles  (default: {WIN_SIZE})")
    p.add_argument(
        "--stride", type=int, default=STRIDE,
        help=f"Stride for LOO evaluation  (default: {STRIDE})")
    p.add_argument(
        "--models", type=str, default=None,
        help=("Comma-separated model names to run.\n"
              f"Available: {', '.join(BASE_MODELS.keys())}\n"
              "Default: all base models (no LSTM)"))
    p.add_argument(
        "--bias_search", action="store_true",
        help="Search optimal global multiplicative bias on LOO pool")
    p.add_argument(
        "--hi_name", type=str, default=None,
        help="Short name for the HI source stored in result.csv.\n"
             "Default: parent directory name of --hi_train")
    p.add_argument(
        "--eol", type=str, default=None,
        help="EOL override: '1:126,2:114,3:89,4:137'  (default: built-in)")
    p.add_argument(
        "--nu", type=str, default=None,
        help="Normal-until override: '1:89,2:92,3:62,4:78'  (default: built-in)")
    return p.parse_args()


def _parse_id_dict(s):
    return {int(k): int(v) for k, v in (pair.split(":") for pair in s.split(","))}


def main():
    args = parse_args()

    hi_train_path = Path(args.hi_train)
    hi_test_path  = Path(args.hi_test)

    ts      = datetime.now().strftime("%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else BASE / f"output/{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive HI source label for result.csv
    if args.hi_name:
        hi_label = args.hi_name
    else:
        parent = hi_train_path.parent.name
        hi_label = parent if parent not in (".", "") else hi_train_path.stem

    eol_dict = _parse_id_dict(args.eol) if args.eol else dict(COMP_EOL)
    nu_dict  = _parse_id_dict(args.nu)  if args.nu  else dict(COMP_NU)

    # Select models
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = list(BASE_MODELS.keys())
    model_names = [m for m in model_names if m in BASE_MODELS]

    print("=" * 72)
    print("  RUL Model Benchmark — LOO Cross-Validation")
    print(f"  HI train : {hi_train_path}")
    print(f"  HI test  : {hi_test_path}")
    print(f"  Output   : {out_dir}")
    print(f"  Win size : {args.win_size}  |  Stride: {args.stride}")
    print(f"  Models   : {model_names}")
    print(f"  EOL      : {eol_dict}")
    print(f"  NU       : {nu_dict}")
    print("=" * 72)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    train_data = load_hi_csv(hi_train_path)
    train_ids  = sorted(train_data.keys())

    has_test  = hi_test_path.exists()
    test_data = load_hi_csv(hi_test_path) if has_test else None
    test_ids  = sorted(test_data.keys()) if test_data else []

    # EOL/NU: keep only entries that exist in train_ids
    eol_dict = {k: v for k, v in eol_dict.items() if k in train_ids}
    nu_dict  = {k: v for k, v in nu_dict.items()  if k in train_ids}

    for bid in train_ids:
        n = train_data[bid]["n"]
        has_reg = train_data[bid]["regime"] is not None
        print(f"  Bearing{bid}: {n} cycles  regime={'yes' if has_reg else 'no'}")
    if test_data:
        print(f"  Test: ids={test_ids}  "
              f"(each {test_data[test_ids[0]]['n']} cycles)")

    # ── LOO Cross-Validation ───────────────────────────────────────────────────
    print("\n[2] LOO Cross-Validation (no leakage)...")
    summary_rows = []
    all_fold_results = {}

    for model_name in model_names:
        cls   = BASE_MODELS[model_name]
        model = cls()
        print(f"\n  ┌─ {model_name} ─────────────")
        t0 = time.time()

        try:
            fold_results = run_loocv(
                model, train_data, train_ids, eol_dict, nu_dict,
                win_size=args.win_size, stride=args.stride,
            )
        except Exception as e:
            print(f"  └─ ERROR: {e}")
            continue

        scores   = {b: fold_results[b][3] for b in train_ids}
        mean_sc  = float(np.nanmean(list(scores.values())))
        elapsed  = time.time() - t0

        for b in train_ids:
            print(f"  │  Bearing{b}: {scores[b]:.4f}", end="")
        print(f"\n  └─ Mean: {mean_sc:.4f}  ({elapsed:.1f}s)")

        # Bias search
        opt_bias = 1.0
        if args.bias_search:
            opt_bias, biased_sc = bias_search(fold_results, train_ids)
            print(f"     Bias search: best={opt_bias:.3f}  biased_mean={biased_sc:.4f}")

        # Real-time append to persistent result.csv
        append_result_csv({
            "hi_name":   hi_label,
            "model":     model_name,
            **{f"B{b}": round(scores[b], 4) for b in train_ids},
            "mean":      round(mean_sc, 4),
            "bias":      round(opt_bias, 4),
            "elapsed_s": round(elapsed, 1),
        })

        # Per-model plots
        mdir = out_dir / model_name
        plot_loocv_per_model(fold_results, mdir, model_name, train_ids, opt_bias)
        plot_loocv_detail(fold_results, mdir, model_name, train_ids, opt_bias)

        row = {
            "model": model_name,
            "mean":  round(mean_sc, 4),
            "bias":  opt_bias,
            "elapsed_s": round(elapsed, 1),
        }
        for b in train_ids:
            row[f"B{b}"] = round(scores[b], 4)
        summary_rows.append(row)
        all_fold_results[model_name] = (fold_results, opt_bias)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n[3] Results Summary")
    summary_df = (pd.DataFrame(summary_rows)
                  .sort_values("mean", ascending=False)
                  .reset_index(drop=True))
    summary_df.to_csv(out_dir / "loocv_summary.csv", index=False)

    bid_cols = [f"B{b}" for b in train_ids]
    header   = f"{'Rank':<5} {'Model':<20}" + "".join(f"{'B'+str(b):>8}" for b in train_ids) + f"{'Mean':>9}"
    print(f"\n  {header}")
    print("  " + "-" * (5 + 20 + 8 * len(train_ids) + 9))
    for rank, (_, r) in enumerate(summary_df.iterrows(), 1):
        line = f"  {rank:<5} {r['model']:<20}"
        line += "".join(f"{r.get(f'B{b}', float('nan')):>8.4f}" for b in train_ids)
        line += f"{r['mean']:>9.4f}"
        print(line)
    print(f"\n  Saved: {out_dir}/loocv_summary.csv")

    if len(summary_rows) > 1:
        plot_loocv_comparison(summary_df, out_dir)

    # ── Test prediction ────────────────────────────────────────────────────────
    if test_data is not None:
        print(f"\n[4] Test RUL Prediction (full training set)...")
        test_rows = []
        header2 = f"  {'Model':<20}" + "".join(f"{'T'+str(t):>8}" for t in test_ids)
        print(header2)
        print("  " + "-" * (20 + 8 * len(test_ids)))

        for model_name in model_names:
            if model_name not in all_fold_results:
                continue
            _, opt_bias = all_fold_results[model_name]
            cls   = BASE_MODELS[model_name]
            model = cls()
            try:
                preds = run_test_predict(
                    model, train_data, test_data, train_ids, test_ids,
                    eol_dict, nu_dict, win_size=args.win_size,
                )
            except Exception as e:
                print(f"  {model_name:<20} ERROR: {e}")
                continue

            line = f"  {model_name:<20}"
            for tid in test_ids:
                raw_hr    = preds[tid]
                biased_hr = raw_hr * opt_bias
                test_rows.append({
                    "model":      model_name,
                    "test_id":    tid,
                    "rul_hours":  round(biased_hr, 3),
                    "rul_raw_hr": round(raw_hr,    3),
                    "bias":       opt_bias,
                })
                line += f"{biased_hr:>8.2f}"
            print(line)

        test_dir = out_dir / "test"
        test_dir.mkdir(exist_ok=True)
        if test_rows:
            test_df = pd.DataFrame(test_rows)
            test_df.to_csv(test_dir / "test_predictions.csv", index=False)
            pivot = test_df.pivot(index="model", columns="test_id",
                                   values="rul_hours")
            pivot.to_csv(test_dir / "test_pivot.csv")
            plot_test_comparison(test_rows, test_dir)
            print(f"\n  Saved: {test_dir}/test_predictions.csv")

    print(f"\n{'=' * 72}")
    print(f"  Done.  Output: {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
