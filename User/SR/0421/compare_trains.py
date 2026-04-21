"""
compare_trains.py
=================
Batch trajectory analysis + cross-train comparison.
Imports run() from bearing_trajectory.py, so keep both files in the same folder.

What it does
------------
1. Finds all subdirectories under <parent_dir> that contain .tdms files
   (e.g. train1/, train2/, train3/, ...).
2. Runs the full kurtogram + trajectory pipeline on each train in sequence.
3. Generates cross-train comparison plots and a summary table.

Usage
-----
    python compare_trains.py /path/to/parent_dir --channel CH2 --out ./compare_out

Expected directory layout:
    parent_dir/
      train1/  000001.tdms, 000002.tdms, ...
      train2/  000001.tdms, ...
      ...

Runtime note: each train takes 5-15 min. For N trains allow 30 min to 2 hrs total.
"""
from __future__ import annotations
import argparse, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse everything from bearing_trajectory.py
from bearing_trajectory import run as run_train_analysis


# ---------- Per-train summary ----------
def analyze_train(df: pd.DataFrame) -> dict:
    """Extract key stats from one train's feature trajectory."""
    hi = df["HI_all"].values
    n = len(hi)
    baseline = hi[: max(n // 4, 1)].mean()     # first 25% as baseline
    threshold = 3 * baseline                   # naive onset: 3x baseline
    over = hi > threshold
    onset_idx = int(np.argmax(over)) if over.any() else -1
    onset_hours = float(df["time_min"].iloc[onset_idx] / 60) if onset_idx > 0 else np.nan
    total_hours = float(df["time_min"].iloc[-1] / 60)

    # Dominant fault = which single-component final amplitude is largest
    finals = {name: float(df[f"{name}_h1"].iloc[-1]) for name in ("BPFI", "BPFO", "BSF")}
    dominant = max(finals, key=finals.get)

    return {
        "n_segments": n,
        "total_hours": total_hours,
        "onset_hours": onset_hours,
        "degrade_hours": total_hours - onset_hours if not np.isnan(onset_hours) else np.nan,
        "dominant_fault": dominant,
        "BPFI_final": finals["BPFI"],
        "BPFO_final": finals["BPFO"],
        "BSF_final":  finals["BSF"],
        "HI_all_final": float(hi[-1]),
        "baseline_HI_all": float(baseline),
        "final_over_baseline": float(hi[-1] / (baseline + 1e-12)),
    }


# ---------- Comparison plots ----------
_HI_PLOT_COLS = ["BPFI_h1", "BPFO_h1", "BSF_h1", "HI_IR", "HI_all", "rms"]


def _rolling_med(series, win=11):
    return series.rolling(win, center=True, min_periods=1).median().values


def plot_comparison(all_dfs: dict, summary: pd.DataFrame, out_dir: Path):
    names = list(all_dfs.keys())
    cmap = plt.cm.tab10 if len(names) <= 10 else plt.cm.tab20
    colors = {n: cmap(i % cmap.N) for i, n in enumerate(names)}

    # ---- Plot 1. Raw time axis (shows lifetime differences) ----
    fig, axes = plt.subplots(len(_HI_PLOT_COLS), 1,
                              figsize=(13, 1.9 * len(_HI_PLOT_COLS)))
    if len(_HI_PLOT_COLS) == 1: axes = [axes]
    for ax, col in zip(axes, _HI_PLOT_COLS):
        for name, df in all_dfs.items():
            if col not in df.columns: continue
            x = df["time_min"].values / 60.0
            y = _rolling_med(df[col])
            ax.plot(x, y, "-", lw=1.1, alpha=0.85, color=colors[name], label=name)
        ax.set_ylabel(col); ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9, ncol=min(len(names), 5))
    axes[-1].set_xlabel("time [hours]")
    fig.suptitle("HI trajectories — raw time axis", y=1.0005)
    fig.tight_layout(); fig.savefig(out_dir / "compare_hi_raw.png", dpi=120); plt.close(fig)

    # ---- Plot 2. Normalized time axis 0..1 (shows pattern, hides lifetime) ----
    fig, axes = plt.subplots(len(_HI_PLOT_COLS), 1,
                              figsize=(13, 1.9 * len(_HI_PLOT_COLS)), sharex=True)
    if len(_HI_PLOT_COLS) == 1: axes = [axes]
    for ax, col in zip(axes, _HI_PLOT_COLS):
        for name, df in all_dfs.items():
            if col not in df.columns: continue
            x = np.linspace(0, 1, len(df))
            y = _rolling_med(df[col])
            ax.plot(x, y, "-", lw=1.1, alpha=0.85, color=colors[name], label=name)
        ax.set_ylabel(col); ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9, ncol=min(len(names), 5))
    axes[-1].set_xlabel("normalized life (0 = start, 1 = failure)")
    fig.suptitle("HI trajectories — normalized time axis", y=1.0005)
    fig.tight_layout(); fig.savefig(out_dir / "compare_hi_norm_time.png", dpi=120); plt.close(fig)

    # ---- Plot 3. Shape: both axes normalized (y = HI / HI_final) ----
    fig, axes = plt.subplots(len(_HI_PLOT_COLS), 1,
                              figsize=(13, 1.9 * len(_HI_PLOT_COLS)), sharex=True)
    if len(_HI_PLOT_COLS) == 1: axes = [axes]
    for ax, col in zip(axes, _HI_PLOT_COLS):
        for name, df in all_dfs.items():
            if col not in df.columns: continue
            x = np.linspace(0, 1, len(df))
            y = _rolling_med(df[col])
            if y[-1] > 0: y = y / y[-1]
            ax.plot(x, y, "-", lw=1.1, alpha=0.85, color=colors[name], label=name)
        ax.set_ylabel(col); ax.grid(alpha=0.3); ax.set_ylim(-0.1, 1.5)
    axes[0].legend(loc="upper left", fontsize=9, ncol=min(len(names), 5))
    axes[-1].set_xlabel("normalized life (0-1)")
    fig.suptitle("HI trajectories — shape only  (y normalized by each train's final)", y=1.0005)
    fig.tight_layout(); fig.savefig(out_dir / "compare_hi_shape.png", dpi=120); plt.close(fig)

    # ---- Plot 4. RPM trajectories side by side ----
    fig, axes = plt.subplots(len(names), 1, figsize=(13, 1.6 * len(names)), sharex=False)
    if len(names) == 1: axes = [axes]
    for ax, (name, df) in zip(axes, all_dfs.items()):
        ax.plot(df["time_min"] / 60.0, df["rpm"], ".", ms=2, color=colors[name])
        ax.axhline(700, color="gray", ls="--", alpha=0.4)
        ax.axhline(950, color="gray", ls="--", alpha=0.4)
        ax.set_ylabel(f"{name}\nRPM"); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [hours]")
    fig.suptitle("RPM trajectories (spec 700-950 dashed)", y=1.0005)
    fig.tight_layout(); fig.savefig(out_dir / "compare_rpm.png", dpi=120); plt.close(fig)

    # ---- Plot 5. Lifetime / onset bar chart ----
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(names)), 4))
    xs = np.arange(len(summary))
    ax.bar(xs, summary["total_hours"], color="lightgray", edgecolor="black",
           label="total life", width=0.7)
    ax.bar(xs, summary["onset_hours"], color="steelblue", edgecolor="black",
           label="healthy (before onset)", width=0.7)
    for i, row in summary.reset_index(drop=True).iterrows():
        ax.text(i, row["total_hours"] + 0.2, f"[{row['dominant_fault']}]",
                ha="center", fontsize=9, color="crimson")
    ax.set_xticks(xs); ax.set_xticklabels(summary["train"], rotation=30)
    ax.set_ylabel("hours")
    ax.set_title("Per-train lifetime, onset time, dominant fault")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out_dir / "compare_lifetime.png", dpi=120); plt.close(fig)


# ---------- Main ----------
def main(parent_dir, channel="CH2", out_dir="./compare_out", kmax=5):
    parent_dir = Path(parent_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # auto-detect train folders (direct children containing .tdms)
    train_dirs = sorted([d for d in parent_dir.iterdir()
                          if d.is_dir() and any(d.glob("*.tdms"))])
    print(f"[batch] {len(train_dirs)} train directories under {parent_dir}")
    for d in train_dirs:
        print(f"   - {d.name}: {len(list(d.glob('*.tdms')))} segments")
    if not train_dirs:
        raise RuntimeError(f"No train subdirectories with .tdms files found under {parent_dir}")

    all_dfs, summary = {}, []
    for td in train_dirs:
        name = td.name
        print(f"\n{'='*60}\n[{name}]\n{'='*60}")
        try:
            df, mdf = run_train_analysis(
                train_dir=td, channel=channel,
                out_dir=out_dir / name, nlevel=kmax,
            )
            all_dfs[name] = df
            stats = {"train": name, **analyze_train(df),
                     "best_HI_by_mono": mdf.iloc[0]["HI"],
                     "best_HI_mono":    float(mdf.iloc[0]["mono"])}
            summary.append(stats)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            traceback.print_exc()

    if not summary:
        print("[abort] no trains were processed successfully"); return

    sdf = pd.DataFrame(summary)
    col_order = ["train", "n_segments", "total_hours", "onset_hours", "degrade_hours",
                 "dominant_fault", "BPFI_final", "BPFO_final", "BSF_final",
                 "HI_all_final", "final_over_baseline",
                 "best_HI_by_mono", "best_HI_mono"]
    sdf = sdf[[c for c in col_order if c in sdf.columns]]
    sdf.to_csv(out_dir / "summary.csv", index=False)

    print("\n" + "=" * 60)
    print("[summary]")
    print("=" * 60)
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", lambda x: f"{x:.3f}"):
        print(sdf.to_string(index=False))

    if len(all_dfs) >= 2:
        plot_comparison(all_dfs, sdf, out_dir)
        print(f"\n[done] comparison outputs -> {out_dir.resolve()}")
    else:
        print(f"\n[warn] only {len(all_dfs)} train processed — comparison plots skipped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parent_dir", help="Parent directory containing train subdirs")
    ap.add_argument("--channel", default="CH2")
    ap.add_argument("--out", default="./compare_out")
    ap.add_argument("--kmax", type=int, default=5)
    args = ap.parse_args()
    main(args.parent_dir, channel=args.channel, out_dir=args.out, kmax=args.kmax)