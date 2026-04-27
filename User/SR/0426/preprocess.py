"""
Per-file feature extraction.

For each .tdms minute-long capture (CH1..CH4 @ 25.6 kHz):
  1) Estimate shaft IRF with GLCT-GWO (paper #2)
  2) Angular resample raw signal => spectrum invariant to small RPM drift
  3) Compute time-domain stats + envelope-order amplitudes at fault frequencies

Bearing 30306 fault orders (per shaft revolution):
  BPFI = 140/16.67 = 8.40
  BPFO = 93 /16.67 = 5.58
  BSF  = 78 /16.67 = 4.68
  FTF  = 6.7/16.67 = 0.40
(reference shaft 1000 rpm = 16.67 Hz)

Test set has *no operation data* — never read RPM, only vibration.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert, welch
from scipy.interpolate import interp1d
from nptdms import TdmsFile

from glct import estimate_irf, downsample, bandpass

# ----------------------------- constants ------------------------------------
FS = 25_600.0
CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
# spec: 700-950 RPM, but observed late-stage transients reach ~1010 RPM, so widen.
FR_MIN = 650.0 / 60.0    # 10.83 Hz
FR_MAX = 1050.0 / 60.0   # 17.50 Hz
SHAFT_REF = 1000.0 / 60.0
FAULT_ORDERS = {
    "BPFI": 140.0 / SHAFT_REF,   # 8.40
    "BPFO": 93.0  / SHAFT_REF,   # 5.58
    "BSF":  78.0  / SHAFT_REF,   # 4.68
    "FTF":  6.7   / SHAFT_REF,   # 0.40
}


# ----------------------------- TDMS IO --------------------------------------
def load_tdms(file_path: str | Path) -> dict[str, np.ndarray]:
    tdms = TdmsFile.read(str(file_path))
    df = tdms.as_dataframe()
    df.columns = [c.split("/")[-1].strip("'") for c in df.columns]
    out = {}
    for ch in CHANNELS:
        if ch in df.columns:
            out[ch] = df[ch].dropna().to_numpy(dtype=np.float32)
    return out


# ----------------------------- order-tracking -------------------------------
def angular_resample(x: np.ndarray, fs: float, irf: np.ndarray, t_irf: np.ndarray,
                     samples_per_rev: int = 256) -> np.ndarray:
    """
    Resample x onto a uniform-angle grid using IRF(t).
    Total angle traversed = ∫ irf(t) dt (revolutions), so we sample at
    samples_per_rev points per revolution.
    """
    # extend irf onto x's full time grid
    t_x = np.arange(len(x)) / fs
    f_interp = interp1d(t_irf, irf, kind="linear",
                        bounds_error=False,
                        fill_value=(float(irf[0]), float(irf[-1])))
    irf_t = f_interp(t_x).astype(np.float32)
    # cumulative angle in revolutions
    theta = np.cumsum(irf_t) / fs   # rev
    total_rev = float(theta[-1])
    n_out = int(np.floor(total_rev * samples_per_rev))
    if n_out < 64:
        return np.zeros(0, dtype=np.float32)
    theta_uniform = np.arange(n_out, dtype=np.float64) / samples_per_rev
    # invert theta(t) by interpolation
    g = interp1d(theta, x.astype(np.float64), kind="linear",
                 bounds_error=False, fill_value=0.0)
    return g(theta_uniform).astype(np.float32)


# ----------------------------- features -------------------------------------
def time_domain_stats(x: np.ndarray) -> dict[str, float]:
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-20))
    peak = float(np.max(np.abs(x)))
    std = float(x.std() + 1e-20)
    kurt = float(np.mean(x ** 4) / (std ** 4 + 1e-20))
    skew = float(np.mean(x ** 3) / (std ** 3 + 1e-20))
    crest = peak / (rms + 1e-20)
    impulse = peak / (np.mean(np.abs(x)) + 1e-20)
    shape = rms / (np.mean(np.abs(x)) + 1e-20)
    margin = peak / (np.mean(np.sqrt(np.abs(x))) ** 2 + 1e-20)
    return {"rms": rms, "peak": peak, "std": std, "kurt": kurt, "skew": skew,
            "crest": crest, "impulse": impulse, "shape": shape, "margin": margin}


def envelope_order_spectrum(x_ang: np.ndarray, samples_per_rev: int,
                            band: tuple[float, float] = (10.0, 100.0)) -> tuple[np.ndarray, np.ndarray]:
    """
    Bandpass on the angular-resampled signal (band in *orders*, not Hz),
    then envelope -> Welch PSD => amplitude vs order.
    """
    fs_ang = float(samples_per_rev)  # samples per revolution = "orders" sampling rate
    lo, hi = max(band[0], 0.1), min(band[1], fs_ang / 2 - 0.5)
    if hi <= lo:
        return np.zeros(0), np.zeros(0)
    sos = butter(4, [lo, hi], btype="band", fs=fs_ang, output="sos")
    y = sosfiltfilt(sos, x_ang)
    env = np.abs(hilbert(y));  env -= env.mean()
    nperseg = min(len(env), 1 << 14)
    f, P = welch(env, fs=fs_ang, nperseg=nperseg)
    return f.astype(np.float32), np.sqrt(P).astype(np.float32)


def _peak_in_window(orders: np.ndarray, amp: np.ndarray,
                    target: float, win: float = 0.10) -> float:
    m = (orders > target - win) & (orders < target + win)
    return float(amp[m].max()) if m.any() else 0.0


def fault_order_features(orders: np.ndarray, amp: np.ndarray) -> dict[str, float]:
    out = {}
    for name, o_ref in FAULT_ORDERS.items():
        for h in (1, 2, 3):
            out[f"{name}_h{h}"] = _peak_in_window(orders, amp, h * o_ref)
    out["BPFI_sb_p"] = _peak_in_window(orders, amp, FAULT_ORDERS["BPFI"] + 1.0)
    out["BPFI_sb_m"] = _peak_in_window(orders, amp, FAULT_ORDERS["BPFI"] - 1.0)
    out["BPFO_sb_p"] = _peak_in_window(orders, amp, FAULT_ORDERS["BPFO"] + 1.0)
    out["BPFO_sb_m"] = _peak_in_window(orders, amp, FAULT_ORDERS["BPFO"] - 1.0)
    out["BSF_sb_p"]  = _peak_in_window(orders, amp, FAULT_ORDERS["BSF"]  + FAULT_ORDERS["FTF"])
    out["BSF_sb_m"]  = _peak_in_window(orders, amp, FAULT_ORDERS["BSF"]  - FAULT_ORDERS["FTF"])
    return out


def band_energies(orders: np.ndarray, amp: np.ndarray) -> dict[str, float]:
    """Energy in coarse order bands — robust catch-all features."""
    bands = {"low": (0.5, 3.0), "mid": (3.0, 7.0), "high": (7.0, 12.0),
             "vhigh": (12.0, 20.0)}
    P = amp ** 2
    out = {}
    for name, (lo, hi) in bands.items():
        m = (orders >= lo) & (orders < hi)
        out[f"E_{name}"] = float(P[m].sum()) if m.any() else 0.0
    return out


# ----------------------------- per-file pipeline ----------------------------
def features_one_file(file_path: str | Path,
                      samples_per_rev: int = 256) -> dict[str, float]:
    """
    Returns a flat dict of features for one .tdms file (4 channels).
    Keys: <ch>_<feature_name>, plus aggregated 'shaft_hz' and 'rpm_est'.
    """
    sigs = load_tdms(file_path)
    feats: dict[str, float] = {}

    # ---- shaft IRF estimated from CH1 (front vertical, usually cleanest) ----
    ref_ch = "CH1" if "CH1" in sigs else next(iter(sigs))
    x_ref = sigs[ref_ch]
    try:
        irf_info = estimate_irf(x_ref, FS, FR_MIN, FR_MAX)
        irf, t_irf = irf_info["irf"], irf_info["t"]
        shaft_hz = float(np.median(irf))
    except Exception:
        # fallback: HPS-style estimator
        f, P = welch(x_ref, fs=FS, nperseg=1 << 16)
        m = (f >= FR_MIN) & (f <= FR_MAX)
        shaft_hz = float(f[m][np.argmax(P[m])]) if m.any() else SHAFT_REF
        irf = np.array([shaft_hz, shaft_hz], dtype=np.float32)
        t_irf = np.array([0.0, len(x_ref) / FS], dtype=np.float32)

    feats["shaft_hz"] = shaft_hz
    feats["rpm_est"] = shaft_hz * 60.0
    feats["shaft_std"] = float(np.std(irf))

    # ---- per-channel features ---------------------------------------------
    for ch, x in sigs.items():
        # time-domain stats on raw and band-passed
        td = time_domain_stats(x)
        for k, v in td.items():
            feats[f"{ch}_{k}"] = v

        # angular resampling (using IRF) + envelope order spectrum
        x_ang = angular_resample(x, FS, irf, t_irf, samples_per_rev=samples_per_rev)
        if len(x_ang) >= 4096:
            orders, amp = envelope_order_spectrum(x_ang, samples_per_rev,
                                                  band=(10.0, 100.0))
            fo = fault_order_features(orders, amp)
            be = band_energies(orders, amp)
            for k, v in fo.items():
                feats[f"{ch}_{k}"] = v
            for k, v in be.items():
                feats[f"{ch}_{k}"] = v
        else:
            # propagate zeros so columns are stable
            for name in FAULT_ORDERS:
                for h in (1, 2, 3):
                    feats[f"{ch}_{name}_h{h}"] = 0.0
            for sb in ["BPFI_sb_p", "BPFI_sb_m", "BPFO_sb_p", "BPFO_sb_m",
                       "BSF_sb_p", "BSF_sb_m"]:
                feats[f"{ch}_{sb}"] = 0.0
            for be_name in ("low", "mid", "high", "vhigh"):
                feats[f"{ch}_E_{be_name}"] = 0.0
    return feats


def features_one_run(run_dir: str | Path, samples_per_rev: int = 256,
                     verbose: bool = True) -> pd.DataFrame:
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("*.tdms"))
    if not files:
        raise RuntimeError(f"No TDMS files under {run_dir}")
    rows = []
    for i, fp in enumerate(files):
        try:
            feats = features_one_file(fp, samples_per_rev=samples_per_rev)
        except Exception as e:
            if verbose:
                print(f"  [skip] {fp.name}: {e}")
            continue
        feats["file"] = fp.name
        feats["idx"] = i
        feats["t_min"] = i * 10.0  # 10-min cadence
        rows.append(feats)
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(files)} files done")
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # Smoke-test: extract features for one TDMS file and print summary.
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/home/ksphm/2026-challenge-KSPHM/dataset/Train1_Vibration/000050.tdms"
    feats = features_one_file(path)
    print(f"\nfeatures for {Path(path).name}: {len(feats)} keys")
    for k in list(feats)[:25]:
        print(f"  {k:<24}  {feats[k]:.4f}")
    print(f"  ...({len(feats)-25} more)")
