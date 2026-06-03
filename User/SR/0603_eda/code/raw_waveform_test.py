"""
SR/0603/code/raw_waveform_test.py
==================================
Raw waveform envelope over life — validation (Test) bearings, all 4 channels.
No RPM info available, so all bins colored blue.
Output: one PNG per test bearing.
"""

import os, sys, warnings
import numpy as np
import nptdms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE     = "/data/home/ksphm/2026-challenge-KSPHM"
VIB_DIR  = os.path.join(BASE, "dataset", "Test")
OUT_DIR  = os.path.join(BASE, "User", "SR", "0603", "output")
os.makedirs(OUT_DIR, exist_ok=True)

TESTS        = [1, 2, 3, 4, 5, 6]
CHANNELS     = ["CH1", "CH2", "CH3", "CH4"]
BINS_PER_FILE = 1500
COLOR        = "#1f77b4"


def envelope_of_file(x, n_bins):
    x = x - x.mean()
    N = len(x)
    edges = np.linspace(0, N, n_bins + 1).astype(int)
    mn = np.empty(n_bins); mx = np.empty(n_bins)
    for i in range(n_bins):
        seg = x[edges[i]:edges[i+1]]
        mn[i], mx[i] = (seg.min(), seg.max()) if len(seg) else (0.0, 0.0)
    return mn, mx


def build_test(tid):
    vib   = os.path.join(VIB_DIR, f"Test{tid}")
    files = sorted(f for f in os.listdir(vib) if f.endswith(".tdms"))
    ch_mn = {ch: [] for ch in CHANNELS}
    ch_mx = {ch: [] for ch in CHANNELS}
    fidxs = []

    for k, fn in enumerate(files):
        fidx = int(os.path.splitext(fn)[0])
        try:
            t = nptdms.TdmsFile(os.path.join(vib, fn))
            data = {ch: t["Vibration"][ch][:].astype(np.float64) for ch in CHANNELS}
        except Exception as ex:
            print(f"  skip T{tid} {fidx}: {ex}")
            continue
        for ch in CHANNELS:
            mn, mx = envelope_of_file(data[ch], BINS_PER_FILE)
            ch_mn[ch].append(mn)
            ch_mx[ch].append(mx)
        fidxs.append(fidx)

    x = np.arange(len(fidxs) * BINS_PER_FILE)
    result = {}
    for ch in CHANNELS:
        result[ch] = dict(
            x=x,
            mn=np.concatenate(ch_mn[ch]),
            mx=np.concatenate(ch_mx[ch]),
        )
    return result, np.array(fidxs)


def plot_test(tid, ch_data, fidxs):
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(14, 10), sharex=True)

    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        d = ch_data[ch]
        x, mn, mx = d["x"], d["mn"], d["mx"]

        ax.fill_between(x, mn, mx, color=COLOR, lw=0, alpha=0.85)

        yl = max(np.nanpercentile(np.abs(mx), 99.8),
                 np.nanpercentile(np.abs(mn), 99.8))
        ax.set_ylim(-yl, yl)
        ax.set_ylabel(f"{ch}\namp")
        ax.grid(alpha=0.15)

    # x축 눈금을 파일 번호로
    n_files = len(fidxs)
    tick_pos = np.arange(0, n_files) * BINS_PER_FILE + BINS_PER_FILE // 2
    tick_lbl = fidxs
    step = max(1, n_files // 10)
    axes[-1].set_xticks(tick_pos[::step])
    axes[-1].set_xticklabels(tick_lbl[::step])
    axes[-1].set_xlabel("File index")

    fig.suptitle(f"Test{tid} — Raw waveform over life, all channels  ({n_files} files)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = os.path.join(OUT_DIR, f"raw_life_T{tid}_allch.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  -> {out}")


def main():
    print("=== Raw waveform over life — Test bearings, all channels ===")
    for tid in TESTS:
        print(f"Building T{tid}...")
        ch_data, fidxs = build_test(tid)
        plot_test(tid, ch_data, fidxs)
    print("Done.")


if __name__ == "__main__":
    main()
