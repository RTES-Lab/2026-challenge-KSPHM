"""
User/SR/0424/tot_rul.py
=======================
KSPHM-KIMM 2026 Bearing Challenge
True Tacholess Order Tracking (TOT) + Random Forest RUL Prediction

Pipeline:
  1. HPS(Harmonic Product Spectrum) to estimate approximate 1X shaft freq.
  2. Bandpass filter around 1X, Hilbert transform to get Instantaneous Phase (IP).
  3. Resample Raw Signal at constant angle increments -> Order Domain Signal.
  4. Extract features (RMS, Kurtosis, Bandpass Envelope Energy) on Order Domain.
  5. Train Random Forest / RUL curve plotting.

Usage:
  python tot_rul.py /path/to/dataset/Train1_Vibration --channel CH2 --out ./tot_out
"""

from __future__ import annotations
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch, butter, sosfiltfilt, hilbert
from scipy.interpolate import interp1d
from nptdms import TdmsFile
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------- CONSTANTS ------------------------------------
FS = 25_600
FAULT_HZ_REF = {"BPFI": 140.0, "BPFO": 93.0, "BSF": 78.0, "FTF": 6.7}
SHAFT_HZ_REF = 1000.0 / 60.0
FAULT_ORDERS = {k: v / SHAFT_HZ_REF for k, v in FAULT_HZ_REF.items()}
SEG_PERIOD_MIN = 10.0


# ----------------------------- IO -------------------------------------------
def load_ch(file_path, channel):
    tdms = TdmsFile.read(str(file_path))
    df = tdms.as_dataframe()
    df.columns = [c.split("/")[-1].strip("'") for c in df.columns]
    return df[channel].dropna().to_numpy(dtype=np.float32)


# --------------------- Tacholess Order Tracking -----------------------------
def estimate_shaft_hz(x, fs=FS, f_range=(10.0, 18.0), n_harm=4):
    """Estimate shaft freq using Harmonic Product Spectrum."""
    f, P = welch(x, fs=fs, nperseg=min(len(x), 1 << 16))
    mask = (f >= f_range[0]) & (f <= f_range[1])
    f_cand = f[mask]
    logP = np.log(P + 1e-20)
    scores = np.zeros_like(f_cand)
    for i, f0 in enumerate(f_cand):
        s = 0.0
        for h in range(1, n_harm + 1):
            s += logP[np.argmin(np.abs(f - h * f0))]
        scores[i] = s
    return float(f_cand[np.argmax(scores)])


def extract_instantaneous_phase(x, fs, f_shaft_approx, bw=2.0):
    """
    Extract Instantaneous Phase (IP) by bandpass filtering around the estimated
    shaft frequency and taking the unwrapped angle of the Hilbert transform.
    """
    lo = max(f_shaft_approx - bw / 2, 1.0)
    hi = min(f_shaft_approx + bw / 2, fs / 2 - 1.0)

    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    x_bp = sosfiltfilt(sos, x)

    # Analytic signal
    x_analytic = hilbert(x_bp)
    
    # Unwrapped phase
    ip = np.unwrap(np.angle(x_analytic))
    
    # To prevent any negative differences (phase reversal due to noise)
    # We enforce monotonicity, though a narrow BP filter usually guarantees it.
    ip = np.maximum.accumulate(ip)
    
    return ip


def order_resample(x, ip, fs):
    """
    Resample time-domain signal `x` to order-domain signal `x_order`.
    `ip` is the instantaneous phase in radians.
    """
    t = np.arange(len(x)) / fs
    
    # Remove mean to center phase around 0 for initial
    ip = ip - ip[0]
    
    # We will resample such that we get a set number of samples per revolution.
    # To avoid aliasing, the new sampling rate in order domain (samples/rev)
    # should be at least 2 * max order of interest or matched to equivalent time domain fs.
    
    # Total radians / (2*pi) = total revolutions
    total_revs = ip[-1] / (2 * np.pi)
    
    # Average RPM to find equivalent samples per rev if we want to preserve total samples approx
    avg_rev_per_sec = total_revs / (t[-1] - t[0])
    
    # Samples per revolution -> (samples/sec) / (revs/sec)
    samples_per_rev = fs / avg_rev_per_sec
    
    # We round it to an integer or just use it as is
    n_samples_per_rev = int(np.ceil(samples_per_rev))
    
    # Target uniform angular spacing (in radians)
    target_d_theta = 2 * np.pi / n_samples_per_rev
    
    # Desired phase points
    theta_uniform = np.arange(0, ip[-1], target_d_theta)
    
    # Interpolate time points that correspond to theta_uniform
    # ip(t) is strictly increasing (monotonic), so we can interpolate t(ip)
    inv_interp = interp1d(ip, t, kind='linear', bounds_error=False, fill_value="extrapolate")
    t_uniform_angle = inv_interp(theta_uniform)
    
    # Resample original signal at these new times
    x_interp = interp1d(t, x, kind='cubic', bounds_error=False, fill_value=0.0)
    x_order = x_interp(t_uniform_angle)
    
    return x_order, n_samples_per_rev


# --------------------- Feature Extraction -----------------------------------
def envelope_spectrum_order(x_ord, spr, band_orders=None):
    """
    Envelope spectrum on order-domain signal.
    spr: samples per revolution (this is the "sampling rate" in order domain).
    """
    if band_orders is not None:
        lo, hi = band_orders
        # Bandpass filter in order domain. fs is `spr` (samples per rev), Nyquist is spr/2
        sos = butter(4, [lo, hi], btype="band", fs=spr, output="sos")
        x_filtered = sosfiltfilt(sos, x_ord)
    else:
        x_filtered = x_ord
        
    env = np.abs(hilbert(x_filtered))
    env -= env.mean()
    
    f_ord, P_ord = welch(env, fs=spr, nperseg=min(len(env), 1 << 14))
    return f_ord, np.sqrt(P_ord)


def _peak_in_order_window(orders, amp, target_order, win=0.1):
    mask = (orders > target_order - win) & (orders < target_order + win)
    return float(amp[mask].max()) if mask.any() else 0.0


def extract_features_tot(x, fs=FS):
    # 1. basic time domain
    x_ac = x - x.mean()
    rms = float(np.sqrt(np.mean(x_ac ** 2)))
    peak = float(np.max(np.abs(x_ac)))
    kurt = float(np.mean(x_ac ** 4) / (rms ** 4 + 1e-20))
    crest = peak / (rms + 1e-20)

    # 2. Extract Phase & Resample
    try:
        f_shaft_approx = estimate_shaft_hz(x, fs)
        ip = extract_instantaneous_phase(x, fs, f_shaft_approx, bw=2.0)
        x_ord, spr = order_resample(x, ip, fs)
    except Exception as e:
        print(f"TOT Failed, fallback to default. Error: {e}")
        f_shaft_approx = 15.0
        x_ord = x
        spr = fs / f_shaft_approx
        
    # 3. Envelope feature on resampled signal (broadband)
    orders, amp = envelope_spectrum_order(x_ord, spr, band_orders=None)
    
    out = {
        "shaft_hz_est": f_shaft_approx,
        "rpm_est": f_shaft_approx * 60,
        "rms": rms,
        "kurtosis": kurt,
        "crest": crest
    }
    
    for name, o_ref in FAULT_ORDERS.items():
        for h in (1, 2, 3):
            out[f"TOT_env_{name}_h{h}"] = _peak_in_order_window(orders, amp, h * o_ref)
            
    out["TOT_env_BPFI_sb_plus"] = _peak_in_order_window(orders, amp, FAULT_ORDERS["BPFI"] + 1.0)
    out["TOT_env_BPFI_sb_minus"] = _peak_in_order_window(orders, amp, FAULT_ORDERS["BPFI"] - 1.0)
    
    return out


# ----------------------------- RUL Pipeline ---------------------------------
def run_pipeline(train_dir, channel="CH2", out_dir="./tot_out", limit=None):
    train_dir = Path(train_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files = sorted(train_dir.glob("*.tdms"))
    if limit is not None:
        files = files[:limit]
        
    if not files:
        raise RuntimeError(f"No .tdms files under {train_dir}")
        
    print(f"[Pipeline] Processing {len(files)} TDMS segments with TOT...")
    rows = []
    
    t0 = time.time()
    for i, fp in enumerate(files):
        try:
            x = load_ch(fp, channel)
            feats = extract_features_tot(x, FS)
            feats["file"] = fp.name
            feats["idx"] = i
            feats["time_min"] = i * SEG_PERIOD_MIN
            # Basic RUL label generation:
            # We assume failure at the very last segment, so RUL is remaining time in minutes.
            feats["RUL_min"] = (len(files) - 1 - i) * SEG_PERIOD_MIN
            rows.append(feats)
        except Exception as e:
            print(f" [error] {fp.name}: {e}")
            
        if (i + 1) % 20 == 0:
            print(f"   {i+1}/{len(files)} processed ({time.time()-t0:.1f}s elapsed)")
            
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tot_features_target.csv", index=False)
    print(f"[Done] Features extracted and saved to tot_features_target.csv.")
    
    return df

def train_rul_model(df, out_dir):
    out_dir = Path(out_dir)
    # Drop non-feature columns
    drop_cols = ["file", "idx", "time_min", "RUL_min"]
    X = df.drop(columns=drop_cols)
    y = df["RUL_min"]
    
    # Train-test split (shuffled) just to evaluate model capacity,
    # though for true predictive maintenance, sequential validation is better.
    # For now, we fit a Random Forest and check its general fit.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    score = rf.score(X_test, y_test)
    y_pred = rf.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"[Model Eval] R2 Score (Test): {score:.4f}, Overall MAE: {mae:.2f} min, RMSE: {rmse:.2f} min")
    
    # Feature Importance
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    feat_imp.head(10).plot(kind='bar', ax=ax1)
    ax1.set_title("Top 10 Feature Importances (Random Forest)")
    fig1.tight_layout()
    fig1.savefig(out_dir / "feature_importance.png", dpi=120)
    plt.close(fig1)
    
    # Plot predicted vs actual RUL Trajectory
    df["Pred_RUL_min"] = y_pred
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(df["time_min"] / 60, df["RUL_min"] / 60, label="True RUL (hrs)", lw=2)
    ax2.plot(df["time_min"] / 60, df["Pred_RUL_min"] / 60, label="RF Predicted RUL (hrs)", alpha=0.8)
    ax2.set(xlabel="Operation Time [hours]", ylabel="RUL [hours]", title="RUL Prediction Trajectory")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / "rul_trajectory.png", dpi=120)
    plt.close(fig2)
    
    print(f"[Done] RUL Trajectory plot saved to {out_dir.resolve()}/rul_trajectory.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tacholess Order Tracking and RUL Prediction")
    parser.add_argument("train_dir", type=str, help="Path to TDMS directory (e.g. /dataset/Train1_Vibration)")
    parser.add_argument("--channel", type=str, default="CH2", help="Vibration channel")
    parser.add_argument("--out", type=str, default="./tot_out", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files")
    args = parser.parse_args()
    
    df_features = run_pipeline(args.train_dir, args.channel, args.out, limit=args.limit)
    train_rul_model(df_features, args.out)
