"""
Stage 0: Tacholess RPM Estimation and Validation
=================================================
Three methods are tried and compared against the operation CSV (actual RPM):
  1. BPFO envelope  — envelope spectrum peak in [BPFO_min, BPFO_max] Hz
  2. Shaft 1X       — raw FFT peak in [rpm_min/60, rpm_max/60] Hz (high resolution)
  3. BPFI envelope  — envelope spectrum peak in [BPFI_min, BPFI_max] Hz

Decision gate:
  best RMSE < RPM_THRESH (30 rpm) → proceed, return best method name
  best RMSE >= RPM_THRESH          → print warning, return fallback flag

Usage:
  python s0_rpm_estimation.py
  python s0_rpm_estimation.py --step 10   # evaluate every 10th file (fast check)
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy.signal import butter, sosfiltfilt, hilbert

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM")
DATA_DIR = BASE_DIR / "dataset"
OUT_DIR  = BASE_DIR / "User/SR/0605_ref/output/rpm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
FS           = 25600    # Hz
INTERVAL_SEC = 600      # s per measurement cycle
BPFO_1000    = 93.0     # Hz @ 1000 RPM
BPFI_1000    = 140.0    # Hz @ 1000 RPM
RPM_MIN      = 700
RPM_MAX      = 950
RPM_THRESH   = 30.0     # RMSE gate [rpm]

TRAIN_EOL = {1: 126, 2: 114, 3: 89, 4: 137}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_ch1(path: Path) -> np.ndarray:
    tf = TdmsFile.read(str(path))
    return tf["Vibration"]["CH1"][:].astype(np.float32)


def get_actual_rpm(op_csv: Path, file_idx: int) -> float:
    """Match TDMS file index (1-based) to RPM from operation CSV."""
    df = pd.read_csv(op_csv, encoding="cp949")
    df.columns = df.columns.str.strip()
    t_center = (file_idx - 1) * INTERVAL_SEC + 30   # center of 1-min acquisition
    idx = (df["Time[sec]"] - t_center).abs().idxmin()
    return float(df.loc[idx, "Motor speed[rpm]"])


# ── RPM estimation methods ────────────────────────────────────────────────────

def _envelope_spectrum_peak(signal: np.ndarray, f_min: float, f_max: float,
                             carrier_lo: float = 1000.0,
                             carrier_hi: float = 8000.0) -> float:
    """
    Demodulate signal in [carrier_lo, carrier_hi] Hz, then find the dominant
    frequency inside [f_min, f_max] Hz in the envelope spectrum.
    Returns the peak frequency [Hz].
    """
    nyq = FS / 2
    sos = butter(4, [carrier_lo / nyq, carrier_hi / nyq], "bandpass", output="sos")
    filt = sosfiltfilt(sos, signal)

    env = np.abs(hilbert(filt))
    env -= env.mean()

    n      = len(env)
    fft_e  = np.abs(np.fft.rfft(env, n=n))
    freqs  = np.fft.rfftfreq(n, 1.0 / FS)

    mask = (freqs >= f_min) & (freqs <= f_max)
    if not mask.any():
        return np.nan
    return float(freqs[mask][np.argmax(fft_e[mask])])


def estimate_rpm_bpfo(signal: np.ndarray) -> float:
    """Envelope spectrum, BPFO-based RPM estimate."""
    f_min = BPFO_1000 * RPM_MIN / 1000
    f_max = BPFO_1000 * RPM_MAX / 1000
    peak  = _envelope_spectrum_peak(signal, f_min, f_max)
    return float(peak / BPFO_1000 * 1000) if not np.isnan(peak) else np.nan


def estimate_rpm_shaft(signal: np.ndarray) -> float:
    """High-resolution FFT in the shaft 1X frequency band (Δf ≈ 0.05 Hz)."""
    n     = min(len(signal), 512_000)   # 20 s → Δf = 0.05 Hz
    chunk = signal[:n] - signal[:n].mean()

    fft_raw = np.abs(np.fft.rfft(chunk, n=n))
    freqs   = np.fft.rfftfreq(n, 1.0 / FS)

    f_min = RPM_MIN / 60
    f_max = RPM_MAX / 60
    mask  = (freqs >= f_min) & (freqs <= f_max)
    if not mask.any():
        return np.nan
    return float(freqs[mask][np.argmax(fft_raw[mask])] * 60)


def estimate_rpm_bpfi(signal: np.ndarray) -> float:
    """Envelope spectrum, BPFI-based RPM estimate."""
    f_min = BPFI_1000 * RPM_MIN / 1000
    f_max = BPFI_1000 * RPM_MAX / 1000
    peak  = _envelope_spectrum_peak(signal, f_min, f_max)
    return float(peak / BPFI_1000 * 1000) if not np.isnan(peak) else np.nan


# ── main validation ───────────────────────────────────────────────────────────

def run_validation(step: int = 5) -> tuple:
    """
    Parameters
    ----------
    step : evaluate every `step`-th TDMS file per bearing (speed/accuracy tradeoff)

    Returns
    -------
    (best_method: str, best_rmse: float, proceed: bool)
    """
    rows = []

    for b in range(1, 5):
        vib_dir  = DATA_DIR / f"Train{b}_Vibration"
        op_csv   = DATA_DIR / f"Train{b}_Operation.csv"
        tdms_files = sorted(vib_dir.glob("*.tdms"))

        print(f"\n── Bearing {b}: {len(tdms_files)} files (sampling every {step}) ──")

        for tdms_path in tdms_files[::step]:
            file_idx   = int(tdms_path.stem)
            actual_rpm = get_actual_rpm(op_csv, file_idx)
            sig        = load_ch1(tdms_path)

            rows.append({
                "bearing"   : b,
                "file_idx"  : file_idx,
                "actual_rpm": actual_rpm,
                "bpfo"      : estimate_rpm_bpfo(sig),
                "shaft"     : estimate_rpm_shaft(sig),
                "bpfi"      : estimate_rpm_bpfi(sig),
            })
            print(f"  [{file_idx:03d}] actual={actual_rpm:.0f}  "
                  f"bpfo={rows[-1]['bpfo']:.0f}  "
                  f"shaft={rows[-1]['shaft']:.0f}  "
                  f"bpfi={rows[-1]['bpfi']:.0f}")

    df_all = pd.DataFrame(rows)
    df_all.to_csv(OUT_DIR / "rpm_raw.csv", index=False)

    # ── per-method metrics ────────────────────────────────────────────────────
    metrics = {}
    for method in ("bpfo", "shaft", "bpfi"):
        sub   = df_all[["actual_rpm", method]].dropna(subset=[method])
        err   = sub["actual_rpm"] - sub[method]
        rmse  = float(np.sqrt((err**2).mean()))
        mae   = float(err.abs().mean())
        bias  = float(err.mean())
        metrics[method] = {"rmse": rmse, "mae": mae, "bias": bias, "n": len(sub)}

    print("\n── Tacholess RPM estimation metrics ──────────────────")
    print(f"{'Method':6s}  {'RMSE':>8s}  {'MAE':>8s}  {'Bias':>8s}  {'N':>5s}")
    for method, m in metrics.items():
        print(f"{method:6s}  {m['rmse']:8.1f}  {m['mae']:8.1f}  {m['bias']:8.1f}  {m['n']:5d}")

    best_method = min(metrics, key=lambda k: metrics[k]["rmse"])
    best_rmse   = metrics[best_method]["rmse"]

    pd.DataFrame(
        [{"method": m, **v} for m, v in metrics.items()]
    ).to_csv(OUT_DIR / "rpm_metrics.csv", index=False)

    # ── scatter plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    lim = [RPM_MIN - 60, RPM_MAX + 60]
    for ax, method in zip(axes, ("bpfo", "shaft", "bpfi")):
        sub = df_all[["actual_rpm", method, "bearing"]].dropna(subset=[method])
        for b_id, color in zip(range(1, 5), ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
            s2 = sub[sub["bearing"] == b_id]
            ax.scatter(s2["actual_rpm"], s2[method], s=14, alpha=0.5,
                       color=color, label=f"B{b_id}")
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual RPM"); ax.set_ylabel("Estimated RPM")
        m = metrics[method]
        ax.set_title(f"{method.upper()}  RMSE={m['rmse']:.1f}  MAE={m['mae']:.1f} rpm")
        ax.legend(fontsize=7)
    plt.suptitle("Tacholess RPM Validation (Train Bearings)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rpm_validation.png", dpi=120)
    plt.close()
    print(f"Plot saved → {OUT_DIR / 'rpm_validation.png'}")

    # ── time-series overlay (bearing 1 only, for visual inspection) ──────────
    b1 = df_all[df_all["bearing"] == 1].sort_values("file_idx")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(b1["file_idx"], b1["actual_rpm"], "k-",  lw=1.5, label="Actual")
    ax.plot(b1["file_idx"], b1["bpfo"],       "b--", lw=1,   label="BPFO")
    ax.plot(b1["file_idx"], b1["shaft"],      "r-.", lw=1,   label="Shaft 1X")
    ax.plot(b1["file_idx"], b1["bpfi"],       "g:",  lw=1,   label="BPFI")
    ax.set_xlabel("File index"); ax.set_ylabel("RPM")
    ax.set_title("Bearing 1 – tacholess RPM tracking")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rpm_timeseries_B1.png", dpi=120)
    plt.close()

    # ── decision ──────────────────────────────────────────────────────────────
    proceed = best_rmse < RPM_THRESH
    status  = "✓ PROCEED" if proceed else "✗ FALLBACK (avg RPM will be used)"
    print(f"\nBest method: {best_method}  RMSE={best_rmse:.1f} rpm  → {status}")

    return best_method, best_rmse, proceed


# ── apply to test (no operation CSV) ─────────────────────────────────────────

def estimate_test_rpm(best_method: str) -> dict:
    """
    Estimate per-file RPM for all test bearings using the validated method.

    Returns
    -------
    dict  {test_id: ndarray of shape (n_files,)}
    """
    ESTIMATORS = {
        "bpfo" : estimate_rpm_bpfo,
        "shaft": estimate_rpm_shaft,
        "bpfi" : estimate_rpm_bpfi,
    }
    fn = ESTIMATORS[best_method]

    test_rpm = {}
    for t_id in range(1, 7):
        test_dir    = DATA_DIR / "Test" / f"Test{t_id}"
        tdms_files  = sorted(test_dir.glob("*.tdms"))
        rpms = []
        for tf_path in tdms_files:
            sig = load_ch1(tf_path)
            rpms.append(fn(sig))
        arr = np.array(rpms, dtype=np.float32)
        # fill NaN with median of non-NaN
        med = float(np.nanmedian(arr)) if not np.all(np.isnan(arr)) else 825.0
        arr[np.isnan(arr)] = med
        test_rpm[t_id] = arr
        print(f"Test{t_id}: median RPM={np.median(arr):.0f}  range=[{arr.min():.0f},{arr.max():.0f}]")

    np.save(OUT_DIR / "test_rpm_estimates.npy", test_rpm)
    return test_rpm


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=5,
                        help="Evaluate every Nth file (default 5)")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip test RPM estimation")
    args = parser.parse_args()

    best_method, best_rmse, proceed = run_validation(step=args.step)

    if not args.skip_test:
        avg_rpm = (RPM_MIN + RPM_MAX) / 2
        effective_method = best_method if proceed else None
        if effective_method is None:
            print(f"\nUsing constant average RPM={avg_rpm:.0f} for all test files.")
        else:
            print(f"\nEstimating test RPM using method: {best_method}")
            estimate_test_rpm(best_method)

    sys.exit(0)
