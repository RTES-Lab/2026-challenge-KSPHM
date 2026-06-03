"""
SR/0603/code/rms_trend.py
=========================
Per-file RMS of CH3 plotted over file order (= time order).
Scatter colored by RPM (low/high), fault point marked with dashed line.
"""

import os, sys, warnings
import numpy as np
import nptdms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

FA_CODE = "/data/home/ksphm/2026-challenge-KSPHM/User/SP/06-02/feature_analysis/code"
sys.path.insert(0, FA_CODE)
from feature_snr_analysis import _load_operation, _rpm_class

BASE     = "/data/home/ksphm/2026-challenge-KSPHM"
VIB_DIR  = os.path.join(BASE, "dataset")
OUT_DIR  = os.path.join(BASE, "User", "SR", "0603", "output")
os.makedirs(OUT_DIR, exist_ok=True)

BEARINGS     = [1, 2, 3, 4]
FAULT_POINTS = {1: 65, 2: 85, 3: 65, 4: 75}
RPM_COLOR    = {"low": "#1f77b4", "high": "#d62728", "skip": "#888888"}


def build_rms(bid):
    oper  = _load_operation(bid)
    vib   = os.path.join(VIB_DIR, f"Train{bid}_Vibration")
    files = sorted(f for f in os.listdir(vib) if f.endswith(".tdms"))
    fidxs, rms_vals, rpms = [], [], []
    for k, fn in enumerate(files):
        fidx = int(os.path.splitext(fn)[0])
        rpm  = _rpm_class(oper, fidx)
        try:
            t = nptdms.TdmsFile(os.path.join(vib, fn))
            x = t["Vibration"]["CH3"][:].astype(np.float64)
        except Exception as ex:
            print(f"  skip B{bid} {fidx}: {ex}")
            continue
        x -= x.mean()
        fidxs.append(fidx)
        rms_vals.append(np.sqrt(np.mean(x ** 2)))
        rpms.append(rpm)
        if (k + 1) % 30 == 0:
            print(f"  [B{bid}] {k+1}/{len(files)}")
    return np.array(fidxs), np.array(rms_vals), np.array(rpms)


def main():
    print("=== Per-file RMS trend (CH3) ===")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)

    for i, bid in enumerate(BEARINGS):
        print(f"Building B{bid}...")
        fidxs, rms_vals, rpms = build_rms(bid)
        ax = axes[i]

        for label, color in RPM_COLOR.items():
            mask = rpms == label
            if mask.any():
                lbl = "transition" if label == "skip" else label
                ax.scatter(fidxs[mask], rms_vals[mask],
                           color=color, s=18, alpha=0.85, label=lbl, zorder=3)

        fp = FAULT_POINTS[bid]
        ax.axvline(fp, color="black", lw=1.3, ls="--", label=f"fault={fp}")

        ax.set_title(f"Bearing{bid}  ({len(fidxs)} files)", fontsize=10)
        ax.set_ylabel("RMS (g)")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left", ncol=3)

    axes[-1].set_xlabel("File index (time order)")
    fig.suptitle("Per-file RMS over bearing life — CH3", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = os.path.join(OUT_DIR, "rms_trend_ch3.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
