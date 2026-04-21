"""
channel_sweep.py
================
Diagnose which vibration channel (CH1-CH4) best reveals bearing degradation
for a single train run. Runs the full kurtogram+trajectory pipeline on each
channel and produces overlay comparison plots.

Use case: when `compare_trains.py` summary shows one train's HI is
degenerate (e.g. final_over_baseline < 1, mono < 0.15), run this sweep
on that train to check if another channel has a cleaner signature.

Requires bearing_trajectory.py in the same folder.

Usage:
    python channel_sweep.py /path/to/Train4_Vibration --out ./sweep_train4

    # Subset of channels:
    python channel_sweep.py /path/to/Train4_Vibration --channels CH1 CH3 CH4
"""
from __future__ import annotations
import argparse, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bearing_trajectory import run as run_train_analysis


def main(train_dir, channels=("CH1", "CH2", "CH3", "CH4"),
         out_dir="./channel_sweep_out", kmax=5):
    train_dir = Path(train_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs, summary = {}, []
    for ch in channels:
        print(f"\n{'=' * 60}\n[{ch}]\n{'=' * 60}")
        try:
            df, mdf = run_train_analysis(
                train_dir=train_dir, channel=ch,
                out_dir=out_dir / ch, nlevel=kmax,
            )
            all_dfs[ch] = df
            best = mdf.iloc[0]
            # Ratios for a few specific HIs (end/start) to see absolute growth
            def _ratio(col):
                v0 = df[col].iloc[:max(len(df) // 4, 1)].mean()
                v1 = df[col].iloc[-max(len(df) // 10, 1):].mean()
                return float(v1 / (v0 + 1e-12))

            summary.append({
                "channel": ch,
                "best_HI_by_mono": best["HI"],
                "best_mono": float(best["mono"]),
                "best_trend": float(best["trend"]),
                "BPFI_h1_ratio": _ratio("BPFI_h1"),
                "BPFO_h1_ratio": _ratio("BPFO_h1"),
                "BSF_h1_ratio":  _ratio("BSF_h1"),
                "HI_all_ratio":  _ratio("HI_all"),
            })
        except Exception as e:
            print(f"[ERROR] {ch}: {e}")
            traceback.print_exc()

    if not summary:
        print("[abort] no channels processed successfully"); return

    sdf = pd.DataFrame(summary)
    sdf.to_csv(out_dir / "channel_summary.csv", index=False)
    print("\n" + "=" * 60)
    print(f"[summary] {train_dir.name}")
    print("=" * 60)
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", lambda x: f"{x:.3f}"):
        print(sdf.to_string(index=False))

    # Comparison overlay plots
    if len(all_dfs) >= 2:
        _plot_channel_comparison(all_dfs, train_dir.name, out_dir)

    # Simple recommendation
    print("\n[recommend]")
    best_ch = sdf.loc[sdf["best_mono"].idxmax()]
    print(f"  Best channel by monotonicity : {best_ch['channel']}  "
          f"(mono={best_ch['best_mono']:.3f}, HI={best_ch['best_HI_by_mono']})")
    best_ratio = sdf.loc[sdf["BSF_h1_ratio"].idxmax()]
    print(f"  Biggest BSF growth channel   : {best_ratio['channel']}  "
          f"(ratio={best_ratio['BSF_h1_ratio']:.2f}x)")
    print(f"\n[done] outputs -> {out_dir.resolve()}")


def _plot_channel_comparison(all_dfs, run_name, out_dir):
    hi_cols = ["BPFI_h1", "BPFO_h1", "BSF_h1", "HI_IR", "HI_all", "rms"]
    chs = list(all_dfs.keys())
    colors = {ch: c for ch, c in zip(chs, plt.cm.tab10(np.linspace(0, 1, max(len(chs), 3))))}

    # Raw time axis
    fig, axes = plt.subplots(len(hi_cols), 1, figsize=(13, 1.9 * len(hi_cols)), sharex=True)
    for ax, col in zip(axes, hi_cols):
        for ch, df in all_dfs.items():
            x = df["time_min"].values / 60.0
            y = df[col].rolling(11, center=True, min_periods=1).median().values
            ax.plot(x, y, "-", lw=1.3, alpha=0.85, color=colors[ch], label=ch)
        ax.set_ylabel(col); ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", ncol=len(chs), fontsize=9)
    axes[-1].set_xlabel("time [hours]")
    fig.suptitle(f"Channel sweep — {run_name}  (raw)", y=1.0005)
    fig.tight_layout(); fig.savefig(out_dir / "channels_raw.png", dpi=120); plt.close(fig)

    # Shape (normalized)
    fig, axes = plt.subplots(len(hi_cols), 1, figsize=(13, 1.9 * len(hi_cols)), sharex=True)
    for ax, col in zip(axes, hi_cols):
        for ch, df in all_dfs.items():
            x = np.linspace(0, 1, len(df))
            y = df[col].rolling(11, center=True, min_periods=1).median().values
            if y[-1] > 0: y = y / y[-1]
            ax.plot(x, y, "-", lw=1.3, alpha=0.85, color=colors[ch], label=ch)
        ax.set_ylabel(col); ax.grid(alpha=0.3); ax.set_ylim(-0.1, 1.5)
    axes[0].legend(loc="upper left", ncol=len(chs), fontsize=9)
    axes[-1].set_xlabel("normalized life (0-1)")
    fig.suptitle(f"Channel sweep — {run_name}  (shape)", y=1.0005)
    fig.tight_layout(); fig.savefig(out_dir / "channels_shape.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dir", help="Single train directory (e.g. /.../Train4_Vibration)")
    ap.add_argument("--channels", nargs="+", default=["CH1", "CH2", "CH3", "CH4"])
    ap.add_argument("--out", default="./channel_sweep_out")
    ap.add_argument("--kmax", type=int, default=5)
    args = ap.parse_args()
    main(args.train_dir, channels=tuple(args.channels),
         out_dir=args.out, kmax=args.kmax)