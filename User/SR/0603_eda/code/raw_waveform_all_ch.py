"""
SR/0603/code/raw_waveform_all_ch.py
====================================
Raw waveform envelope over bearing life — all 4 channels.
Output: one PNG per bearing (4 subplots = 4 channels).
Same approach as SP/06-03/code/raw_waveform_life.py:
  - per-file min/max envelope (1500 bins/file) to keep spike shape
  - RPM colored (low=blue, high=red, transition=gray)
  - fault point dashed line
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
CHANNELS     = ["CH1", "CH2", "CH3", "CH4"]
FAULT_POINTS = {1: 65, 2: 85, 3: 65, 4: 75}
BINS_PER_FILE = 1500
RPM_COLOR    = {"low": "#1f77b4", "high": "#d62728", "skip": "#888888"}


def envelope_of_file(x, n_bins):
    x = x - x.mean()
    N = len(x)
    edges = np.linspace(0, N, n_bins + 1).astype(int)
    mn = np.empty(n_bins); mx = np.empty(n_bins)
    for i in range(n_bins):
        seg = x[edges[i]:edges[i+1]]
        mn[i], mx[i] = (seg.min(), seg.max()) if len(seg) else (0.0, 0.0)
    return mn, mx


def build_bearing(bid):
    """Returns dict: ch -> (x, mn, mx, rpm), plus boundaries/fidxs."""
    oper  = _load_operation(bid)
    vib   = os.path.join(VIB_DIR, f"Train{bid}_Vibration")
    files = sorted(f for f in os.listdir(vib) if f.endswith(".tdms"))

    ch_mn  = {ch: [] for ch in CHANNELS}
    ch_mx  = {ch: [] for ch in CHANNELS}
    all_rpm = []
    boundaries = []
    fidxs = []
    pos = 0

    for k, fn in enumerate(files):
        fidx = int(os.path.splitext(fn)[0])
        rpm  = _rpm_class(oper, fidx)
        try:
            t = nptdms.TdmsFile(os.path.join(vib, fn))
            data = {ch: t["Vibration"][ch][:].astype(np.float64) for ch in CHANNELS}
        except Exception as ex:
            print(f"  skip B{bid} {fidx}: {ex}")
            continue

        for ch in CHANNELS:
            mn, mx = envelope_of_file(data[ch], BINS_PER_FILE)
            ch_mn[ch].append(mn)
            ch_mx[ch].append(mx)

        all_rpm.append(np.array([rpm] * BINS_PER_FILE))
        boundaries.append(pos)
        fidxs.append(fidx)
        pos += BINS_PER_FILE

        if (k + 1) % 30 == 0:
            print(f"  [B{bid}] {k+1}/{len(files)}")

    rpm_arr = np.concatenate(all_rpm)
    x = np.arange(len(rpm_arr))
    result = {}
    for ch in CHANNELS:
        result[ch] = dict(
            x=x,
            mn=np.concatenate(ch_mn[ch]),
            mx=np.concatenate(ch_mx[ch]),
            rpm=rpm_arr,
        )
    return result, np.array(boundaries), np.array(fidxs)


def plot_bearing(bid, ch_data, bnds, fidxs):
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(16, 10), sharex=True)

    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        d = ch_data[ch]
        x, mn, mx, rpm = d["x"], d["mn"], d["mx"], d["rpm"]

        for label, color in RPM_COLOR.items():
            m = rpm == label
            if m.any():
                lbl = "transition" if label == "skip" else label
                ax.fill_between(x, mn, mx, where=m, color=color,
                                lw=0, alpha=0.85, label=lbl)

        # fault point
        fp = FAULT_POINTS[bid]
        j = np.searchsorted(fidxs, fp)
        if j < len(bnds):
            ax.axvline(bnds[j], color="black", lw=1.3, ls="--",
                       label=f"fault={fp}")

        yl = max(np.nanpercentile(np.abs(mx), 99.8),
                 np.nanpercentile(np.abs(mn), 99.8))
        ax.set_ylim(-yl, yl)
        ax.set_ylabel(f"{ch}\namp")
        ax.grid(alpha=0.15)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left", ncol=3)

    axes[-1].set_xlabel(f"File order  (1 file = {BINS_PER_FILE} bins)")
    fig.suptitle(f"Bearing{bid} — Raw waveform over life, all channels  ({len(fidxs)} files)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = os.path.join(OUT_DIR, f"raw_life_B{bid}_allch.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  -> {out}")


def main():
    print("=== Raw waveform over life — all channels ===")
    for bid in BEARINGS:
        print(f"Building B{bid}...")
        ch_data, bnds, fidxs = build_bearing(bid)
        plot_bearing(bid, ch_data, bnds, fidxs)
    print("Done.")


if __name__ == "__main__":
    main()
