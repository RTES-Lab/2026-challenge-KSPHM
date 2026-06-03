"""
SR/0603/code/channel_compare.py
================================
Compare all 4 channels for degradation signal quality.

Channel layout:
  CH1 = Front Vertical  (radial, front)
  CH2 = Front Axial     (axial,  front)
  CH3 = Rear Vertical   (radial, rear)
  CH4 = Rear Axial      (axial,  rear)

Per bearing:
  - Plot 1: RMS trend for all 4 channels overlaid (low/high RPM separated)
  - Plot 2: Summary table — degradation SNR per channel per bearing
            SNR = mean(RMS after fault) / mean(RMS before fault)

Output:
  output/channel_compare_B{N}.png   per-bearing overlay plot
  output/channel_snr_summary.png    heatmap of degradation ratio
"""

import os, sys, warnings
import numpy as np
import nptdms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
CH_LABEL     = {
    "CH1": "CH1 Front Vertical",
    "CH2": "CH2 Front Axial",
    "CH3": "CH3 Rear Vertical",
    "CH4": "CH4 Rear Axial",
}
FAULT_POINTS = {1: 65, 2: 85, 3: 65, 4: 75}
CH_COLOR     = {"CH1": "#1f77b4", "CH2": "#ff7f0e", "CH3": "#2ca02c", "CH4": "#d62728"}
RPM_ALPHA    = {"low": 0.6, "high": 1.0, "skip": 0.3}
RPM_MARKER   = {"low": "o", "high": "s", "skip": "x"}


def build_rms_all_ch(bid):
    oper  = _load_operation(bid)
    vib   = os.path.join(VIB_DIR, f"Train{bid}_Vibration")
    files = sorted(f for f in os.listdir(vib) if f.endswith(".tdms"))

    fidxs = []
    rms   = {ch: [] for ch in CHANNELS}
    rpms  = []

    for k, fn in enumerate(files):
        fidx = int(os.path.splitext(fn)[0])
        rpm  = _rpm_class(oper, fidx)
        try:
            t = nptdms.TdmsFile(os.path.join(vib, fn))
            data = {ch: t["Vibration"][ch][:].astype(np.float64) for ch in CHANNELS}
        except Exception as ex:
            print(f"  skip B{bid} {fidx}: {ex}")
            continue

        fidxs.append(fidx)
        rpms.append(rpm)
        for ch in CHANNELS:
            x = data[ch]; x -= x.mean()
            rms[ch].append(np.sqrt(np.mean(x ** 2)))

        if (k + 1) % 30 == 0:
            print(f"  [B{bid}] {k+1}/{len(files)}")

    fidxs = np.array(fidxs)
    rpms  = np.array(rpms)
    rms   = {ch: np.array(v) for ch, v in rms.items()}
    return fidxs, rms, rpms


def plot_bearing(bid, fidxs, rms, rpms):
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fp = FAULT_POINTS[bid]

    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        for label in ("low", "high", "skip"):
            m = rpms == label
            if not m.any():
                continue
            lbl = "transition" if label == "skip" else label
            ax.scatter(fidxs[m], rms[ch][m],
                       color=CH_COLOR[ch], s=20,
                       alpha=RPM_ALPHA[label],
                       marker=RPM_MARKER[label],
                       label=lbl if i == 0 else None)

        ax.axvline(fp, color="black", lw=1.3, ls="--",
                   label=f"fault={fp}" if i == 0 else None)

        # shade pre/post fault
        ax.axvspan(0, fp, alpha=0.04, color="green")
        ax.axvspan(fp, fidxs[-1]+1, alpha=0.04, color="red")

        ax.set_ylabel(f"{CH_LABEL[ch]}\nRMS (g)", fontsize=8)
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=8, loc="upper left", ncol=3)
    axes[-1].set_xlabel("File index (time order)")
    fig.suptitle(f"Bearing{bid} — RMS per file, all channels  ({len(fidxs)} files)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT_DIR, f"channel_compare_B{bid}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  -> {out}")


def degradation_ratio(fidxs, rms_arr, fp):
    """mean RMS after fault / mean RMS before fault (per RPM combined)."""
    pre  = rms_arr[fidxs <  fp]
    post = rms_arr[fidxs >= fp]
    if len(pre) == 0 or len(post) == 0:
        return np.nan
    return post.mean() / (pre.mean() + 1e-12)


def plot_summary(all_data):
    """Heatmap: rows=bearing, cols=channel, value=degradation ratio."""
    ratios = np.zeros((len(BEARINGS), len(CHANNELS)))
    for i, bid in enumerate(BEARINGS):
        fidxs, rms, _ = all_data[bid]
        fp = FAULT_POINTS[bid]
        for j, ch in enumerate(CHANNELS):
            ratios[i, j] = degradation_ratio(fidxs, rms[ch], fp)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(ratios, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="RMS after / RMS before fault")

    ax.set_xticks(range(len(CHANNELS)))
    ax.set_xticklabels([CH_LABEL[c] for c in CHANNELS], fontsize=9)
    ax.set_yticks(range(len(BEARINGS)))
    ax.set_yticklabels([f"Bearing{b}" for b in BEARINGS])

    for i in range(len(BEARINGS)):
        for j in range(len(CHANNELS)):
            ax.text(j, i, f"{ratios[i,j]:.1f}x",
                    ha="center", va="center", fontsize=11,
                    color="white" if ratios[i,j] > ratios.max()*0.6 else "black",
                    fontweight="bold")

    ax.set_title("Degradation ratio (post-fault RMS / pre-fault RMS) per channel",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "channel_snr_summary.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  -> {out}")

    # print table
    print("\n=== Degradation Ratio (post/pre fault RMS) ===")
    header = f"{'':10s}" + "".join(f"{CH_LABEL[c]:22s}" for c in CHANNELS)
    print(header)
    for i, bid in enumerate(BEARINGS):
        row = f"Bearing{bid}  " + "".join(f"{ratios[i,j]:>18.2f}x    " for j in range(len(CHANNELS)))
        print(row)


def main():
    print("=== Channel comparison (all 4 channels) ===")
    all_data = {}
    for bid in BEARINGS:
        print(f"Building B{bid}...")
        fidxs, rms, rpms = build_rms_all_ch(bid)
        all_data[bid] = (fidxs, rms, rpms)
        plot_bearing(bid, fidxs, rms, rpms)

    plot_summary(all_data)
    print("\nDone.")


if __name__ == "__main__":
    main()
