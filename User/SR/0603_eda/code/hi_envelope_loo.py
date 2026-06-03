"""
Envelope Spectrum Energy Health Index (LOO)
==========================================

Pipeline per file:
  1. Per-channel shaft frequency estimation (HPS on 10-17 Hz PSD)
  2. High-pass filter (>= HP_CUTOFF Hz) -> Hilbert envelope
  3. Envelope PSD (Welch)
  4. Sum PSD in BPFO / BPFI bands using estimated shaft_hz
  5. Average across 4 channels -> (E_BPFO, E_BPFI, E_total)

LOO HI:
  HI = (E_total - mu_ref) / (max_E_total - mu_ref), clipped [0, 1]
  mu_ref = mean of other bearings' early NORMAL_RATIO data

Fault orders (RPM-invariant, derived at 1000 RPM reference):
  BPFI = 8.40  (98.0 ~ 133.0 Hz @ 700-950 RPM)
  BPFO = 5.58  (65.1 ~  88.4 Hz @ 700-950 RPM)
  BSF  = 4.68  (54.6 ~  74.1 Hz @ 700-950 RPM)
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert, welch
import nptdms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = "/data/home/ksphm/2026-challenge-KSPHM"
DATA_DIR   = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "User", "SR", "0603", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FS           = 25_600          # Hz
INTERVAL_SEC = 600             # 10 min between files
NORMAL_RATIO = 0.15            # first 15% = healthy reference
N_SEC_USE    = 30              # seconds to use per file (first half)
N_USE        = N_SEC_USE * FS  # = 768 000 samples

# Shaft frequency search window
F_SHAFT_MIN  = 700  / 60.0    # 11.67 Hz
F_SHAFT_MAX  = 950  / 60.0    # 15.83 Hz
N_HARM_HPS   = 4               # harmonics for Harmonic Product Spectrum

# Fault orders (shaft_hz × order = fault_hz)
FAULT_ORDERS = {"BPFI": 8.40, "BPFO": 5.58, "BSF": 4.68}

# Envelope pre-filter and band integration
HP_CUTOFF     = 1000.0         # high-pass cutoff [Hz]
BAND_HALF_PCT = 0.10           # ± 10 % of fault frequency for band integration
WELCH_NPERSEG = 1 << 15        # 32 768 samples -> freq_res ≈ 0.78 Hz

BEARING_IDS = [1, 2, 3, 4]
COLORS      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
EPS         = 1e-20


# ── Signal processing ─────────────────────────────────────────────────────────

def estimate_shaft_hz(signal: np.ndarray) -> float:
    """Harmonic Product Spectrum estimate of shaft frequency."""
    f, P = welch(signal, fs=FS, nperseg=min(len(signal), 1 << 16))
    mask  = (f >= F_SHAFT_MIN) & (f <= F_SHAFT_MAX)
    f_cand = f[mask]
    if len(f_cand) == 0:
        return (F_SHAFT_MIN + F_SHAFT_MAX) / 2.0

    logP   = np.log(P + EPS)
    scores = np.zeros(len(f_cand))
    for i, f0 in enumerate(f_cand):
        for h in range(1, N_HARM_HPS + 1):
            idx = np.argmin(np.abs(f - h * f0))
            scores[i] += logP[idx]
    return float(f_cand[np.argmax(scores)])


def envelope_psd(signal: np.ndarray):
    """High-pass -> Hilbert envelope -> Welch PSD. Returns (f, P)."""
    sos   = butter(6, HP_CUTOFF, btype="high", fs=FS, output="sos")
    sig_f = sosfiltfilt(sos, signal)
    env   = np.abs(hilbert(sig_f))
    env  -= env.mean()
    nperseg = min(len(env), WELCH_NPERSEG)
    f, P = welch(env, fs=FS, nperseg=nperseg, noverlap=nperseg // 2)
    return f, P


def band_energy(f: np.ndarray, P: np.ndarray, fc: float) -> float:
    """Sum PSD within ± BAND_HALF_PCT * fc around centre frequency fc."""
    bw   = fc * BAND_HALF_PCT
    mask = (f >= fc - bw) & (f <= fc + bw)
    return float(P[mask].sum()) if mask.any() else 0.0


# ── Per-file feature extraction ───────────────────────────────────────────────

def load_channels(path: str) -> list:
    with nptdms.TdmsFile.open(path) as tdms:
        grp = tdms["Vibration"]
        return [grp[ch][:N_USE] for ch in ["CH1", "CH2", "CH3", "CH4"]]


def extract_file_features(path: str) -> dict:
    channels = load_channels(path)

    shaft_hzs = [estimate_shaft_hz(ch) for ch in channels]
    shaft_hz  = float(np.mean(shaft_hzs))

    e_bpfo_list, e_bpfi_list, e_bsf_list = [], [], []
    for ch in channels:
        f, P = envelope_psd(ch)
        e_bpfo_list.append(band_energy(f, P, shaft_hz * FAULT_ORDERS["BPFO"]))
        e_bpfi_list.append(band_energy(f, P, shaft_hz * FAULT_ORDERS["BPFI"]))
        e_bsf_list.append( band_energy(f, P, shaft_hz * FAULT_ORDERS["BSF"]))

    e_bpfo = float(np.mean(e_bpfo_list))
    e_bpfi = float(np.mean(e_bpfi_list))
    e_bsf  = float(np.mean(e_bsf_list))

    return {
        "shaft_hz": shaft_hz,
        "rpm_est":  shaft_hz * 60.0,
        "E_BPFO":   e_bpfo,
        "E_BPFI":   e_bpfi,
        "E_BSF":    e_bsf,
        "E_total":  e_bpfo + e_bpfi,
    }


def process_bearing(bid: int) -> pd.DataFrame:
    vib_dir = os.path.join(DATA_DIR, f"Train{bid}_Vibration")
    files   = sorted(glob.glob(os.path.join(vib_dir, "*.tdms")))
    print(f"  Bearing {bid}: {len(files)} files", flush=True)

    rows = []
    for i, path in enumerate(files):
        file_idx = int(os.path.splitext(os.path.basename(path))[0])
        feat = extract_file_features(path)
        rows.append({"bearing_id": bid,
                     "file_idx":   file_idx,
                     "time_hr":    (file_idx - 1) * INTERVAL_SEC / 3600.0,
                     **feat})
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(files)}]", flush=True)

    return pd.DataFrame(rows).sort_values("file_idx").reset_index(drop=True)


# ── LOO Health Index ──────────────────────────────────────────────────────────

def build_hi_loo(all_dfs: list, target_idx: int,
                 feature: str = "E_total") -> pd.DataFrame:
    """
    LOO baseline: other bearings' first NORMAL_RATIO fraction of E_total.
    HI = (E - mu_ref) / (max_E - mu_ref), clipped [0, 1]
    """
    ref_vals = []
    for i, df in enumerate(all_dfs):
        if i == target_idx:
            continue
        n_norm = max(int(len(df) * NORMAL_RATIO), 5)
        ref_vals.extend(df[feature].iloc[:n_norm].tolist())
    ref_vals = np.array(ref_vals)

    mu_ref   = ref_vals.mean()
    sig_ref  = ref_vals.std()

    tgt      = all_dfs[target_idx][feature].values
    max_val  = tgt.max()
    denom    = max(max_val - mu_ref, 1e-20)

    hi = np.clip((tgt - mu_ref) / denom, 0.0, 1.0)

    df_out = all_dfs[target_idx].copy()
    df_out["HI"]      = hi
    df_out["mu_ref"]  = mu_ref
    df_out["sig_ref"] = sig_ref
    return df_out


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_bearing(df: pd.DataFrame, bid: int):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(df["time_hr"], df["HI"], color=COLORS[bid - 1], lw=1.5)
    axes[0].axhline(0, color="k", lw=0.5, ls="--")
    axes[0].axhline(1, color="r", lw=0.5, ls="--")
    axes[0].set_ylabel("HI")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title(f"Bearing {bid} — Envelope Spectrum Energy HI (LOO)")

    axes[1].plot(df["time_hr"], df["E_BPFO"], color="steelblue",
                 lw=1.2, label="BPFO")
    axes[1].plot(df["time_hr"], df["E_BPFI"], color="darkorange",
                 lw=1.2, label="BPFI")
    axes[1].plot(df["time_hr"], df["E_BSF"],  color="seagreen",
                 lw=1.0, ls="--", label="BSF", alpha=0.7)
    axes[1].set_ylabel("Envelope Band Energy")
    axes[1].legend(fontsize=8)

    axes[2].plot(df["time_hr"], df["rpm_est"], color="gray", lw=1.0)
    axes[2].set_ylabel("Estimated RPM")
    axes[2].set_xlabel("Time [hr]")
    axes[2].set_ylim(600, 1000)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"Bearing{bid}_HI_env.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"    -> {out}")


def plot_all(hi_dfs: list):
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, df in enumerate(hi_dfs):
        ax.plot(df["time_hr"], df["HI"],
                color=COLORS[i], lw=1.5, label=f"Bearing {i+1}")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.axhline(1, color="r", lw=0.5, ls="--")
    ax.set_xlabel("Time [hr]")
    ax.set_ylabel("HI (Envelope Energy, LOO)")
    ax.set_title("All Bearings — Envelope Spectrum Energy HI (LOO)")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "All_HI_env_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"    -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Envelope Spectrum Energy HI (LOO) ===\n")

    print("[1/3] Feature extraction ...")
    all_dfs = []
    for bid in BEARING_IDS:
        df = process_bearing(bid)
        df.to_csv(os.path.join(OUTPUT_DIR, f"Bearing{bid}_env_feats.csv"),
                  index=False)
        all_dfs.append(df)

    print("\n[2/3] LOO HI construction ...")
    hi_dfs = []
    for i, bid in enumerate(BEARING_IDS):
        hi_df = build_hi_loo(all_dfs, i)
        hi_df.to_csv(os.path.join(OUTPUT_DIR, f"Bearing{bid}_HI_env.csv"),
                     index=False)
        plot_bearing(hi_df, bid)
        hi_dfs.append(hi_df)

    print("\n[3/3] Summary ...")
    plot_all(hi_dfs)

    rows = []
    for i, df in enumerate(hi_dfs):
        rows.append({
            "bearing": i + 1,
            "n_files": len(df),
            "HI_final": round(df["HI"].iloc[-1], 4),
            "HI_max":   round(df["HI"].max(), 4),
            "rpm_mean": round(df["rpm_est"].mean(), 1),
            "rpm_std":  round(df["rpm_est"].std(), 1),
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "HI_env_summary.csv"),
                               index=False)

    print("\nDone!")
