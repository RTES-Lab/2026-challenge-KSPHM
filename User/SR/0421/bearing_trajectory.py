"""
bearing_trajectory.py
=====================
KSPHM-KIMM 2026 bearing challenge — full-run HI trajectory analysis.

Pipeline:
  (1) Proper fast kurtogram on a reference (late) segment -> resonance band
      (DC and Nyquist edges excluded).
  (2) Iterate through ALL segments chronologically. Per segment:
      tacholess shaft freq -> envelope order spectrum in FIXED band
      -> fault-order amplitudes at BPFI/BPFO/BSF (+ harmonics & sidebands).
  (3) Plot HI trajectories, compute monotonicity / trendability.

Usage:
    python bearing_trajectory.py /path/to/train1 --channel CH2 --out ./traj_out
    python bearing_trajectory.py /path/to/train1 --ref-file 000125.tdms --kmax 5
"""
from __future__ import annotations
import argparse, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, sosfiltfilt, hilbert
from nptdms import TdmsFile

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------- 상수 -----------------------------------------
FS = 25_600
FAULT_HZ_REF = {"BPFI": 140.0, "BPFO": 93.0, "BSF": 78.0, "FTF": 6.7}
SHAFT_HZ_REF = 1000.0 / 60.0
FAULT_ORDERS = {k: v / SHAFT_HZ_REF for k, v in FAULT_HZ_REF.items()}
SEG_PERIOD_MIN = 10.0  # 10분 주기로 1분씩 취득


# ----------------------------- IO -------------------------------------------
def load_ch(file_path, channel):
    tdms = TdmsFile.read(str(file_path))
    df = tdms.as_dataframe()
    df.columns = [c.split("/")[-1].strip("'") for c in df.columns]
    return df[channel].dropna().to_numpy(dtype=np.float32)


# --------------------- Proper fast kurtogram (Antoni-style) -----------------
def _band_sk(x, fs, low, high):
    """Bandpass -> Hilbert -> kurtosis of |envelope|^2 (SK in Antoni sense)."""
    sos = butter(4, [low, high], btype="band", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    env2 = np.abs(hilbert(y)) ** 2
    m2 = env2.mean()
    m4 = (env2 * env2).mean()
    return m4 / (m2 * m2 + 1e-20) - 2.0


def fast_kurtogram(x, fs=FS, nlevel=5, fmin=200.0, fmax_ratio=0.9):
    """
    Explicit-grid fast kurtogram.
    Dyadic levels 1, 1.5, 2, 2.5, ..., nlevel -> bandwidths fs/2^(k+1).
    Returns (best_band_tuple, grid_array, best_kurt).
    grid_array columns: [level, center_hz, bw_hz, kurtosis]
    """
    fmax = fs / 2 * fmax_ratio
    levels = []
    for k in range(1, nlevel + 1):
        levels.append(float(k))
        if k < nlevel:
            levels.append(k + 0.5)

    rows = []
    for k in levels:
        bw = fs / (2 ** (k + 1))
        if bw < 50:  # too narrow, skip
            continue
        # centers uniformly spaced within [fmin, fmax]
        n = int(np.floor((fmax - fmin) / bw))
        if n < 1:
            continue
        for i in range(n):
            center = fmin + (i + 0.5) * bw
            low, high = center - bw / 2, center + bw / 2
            if low < fmin or high > fmax:
                continue
            try:
                kurt = _band_sk(x, fs, low, high)
            except Exception:
                continue
            rows.append([k, center, bw, kurt])
    grid = np.array(rows)
    best = grid[np.argmax(grid[:, 3])]
    band = (best[1] - best[2] / 2, best[1] + best[2] / 2)
    return band, grid, float(best[3])


def plot_kurtogram(grid, band, save_path):
    """Kurtogram scatter visualization (level × center, color = kurt)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    sc = ax.scatter(grid[:, 1], grid[:, 0], c=grid[:, 3], s=80,
                    cmap="hot_r", vmin=0, vmax=max(grid[:, 3].max(), 0.1))
    ax.axvspan(band[0], band[1], alpha=0.2, color="cyan",
               label=f"selected {band[0]:.0f}-{band[1]:.0f} Hz")
    ax.set(xlabel="center frequency [Hz]", ylabel="level k  (bw = fs / 2^(k+1))",
           title="Fast Kurtogram")
    fig.colorbar(sc, ax=ax, label="SK")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


# ------------------------- Per-segment features -----------------------------
def estimate_shaft_hz(x, fs=FS, f_range=(10.0, 18.0), n_harm=4):
    """Harmonic Product Spectrum (log-sum) tacholess shaft freq."""
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


def envelope_order_spectrum(x, fs, band):
    """Bandpass -> envelope -> welch PSD; returns (f, amplitude)."""
    sos = butter(6, band, btype="band", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    env = np.abs(hilbert(y));  env -= env.mean()
    f, P = welch(env, fs=fs, nperseg=min(len(env), 1 << 15))
    return f, np.sqrt(P)


def _peak_in_order_window(orders, amp, target_order, win=0.1):
    mask = (orders > target_order - win) & (orders < target_order + win)
    return float(amp[mask].max()) if mask.any() else 0.0


def extract_features(x, band, fs=FS):
    x_ac = x - x.mean()
    rms = float(np.sqrt(np.mean(x_ac ** 2)))
    peak = float(np.max(np.abs(x_ac)))
    kurt = float(np.mean(x_ac ** 4) / (rms ** 4 + 1e-20))
    crest = peak / (rms + 1e-20)

    f_shaft = estimate_shaft_hz(x, fs)
    f_env, amp = envelope_order_spectrum(x, fs, band)
    orders = f_env / f_shaft

    out = {"shaft_hz": f_shaft, "rpm": f_shaft * 60,
           "rms": rms, "kurtosis": kurt, "crest": crest}
    # fault orders + first 3 harmonics
    for name, o_ref in FAULT_ORDERS.items():
        for h in (1, 2, 3):
            out[f"{name}_h{h}"] = _peak_in_order_window(orders, amp, h * o_ref)
    # BPFI ± 1×shaft sidebands (IR signature)
    out["BPFI_sb_plus"]  = _peak_in_order_window(orders, amp, FAULT_ORDERS["BPFI"] + 1.0)
    out["BPFI_sb_minus"] = _peak_in_order_window(orders, amp, FAULT_ORDERS["BPFI"] - 1.0)
    return out


# ----------------------------- Metrics --------------------------------------
def monotonicity(y):
    """|#pos - #neg| / (N-1). Uses smoothed series."""
    y = pd.Series(y).rolling(11, center=True, min_periods=1).median().values
    d = np.diff(y)
    return float(abs((d > 0).sum() - (d < 0).sum()) / max(len(d), 1))


def trendability(y):
    """Spearman correlation with time (rank-based)."""
    t = np.arange(len(y))
    if np.std(y) == 0:
        return 0.0
    r_y = pd.Series(y).rank().values
    r_t = pd.Series(t).rank().values
    return float(np.corrcoef(r_t, r_y)[0, 1])


# ----------------------------- Main pipeline --------------------------------
def run(train_dir, ref_file=None, channel="CH2", out_dir="./traj_out", nlevel=5):
    train_dir = Path(train_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(train_dir.glob("*.tdms"))
    if not files:
        raise RuntimeError(f"No .tdms files under {train_dir}")
    print(f"[files] {len(files)} segments  |  dir: {train_dir}")

    # ---- (1) Kurtogram on reference ----
    ref = train_dir / ref_file if ref_file else files[-1]
    print(f"[kurtogram] reference = {ref.name}")
    x_ref = load_ch(ref, channel)
    t0 = time.time()
    band, grid, kmax = fast_kurtogram(x_ref, nlevel=nlevel)
    print(f"[kurtogram] best band = {band[0]:.0f}-{band[1]:.0f} Hz  "
          f"(SK={kmax:.2f})  took {time.time()-t0:.1f}s")
    plot_kurtogram(grid, band, out_dir / "kurtogram.png")

    # ---- (2) Sweep all segments ----
    print(f"[trajectory] extracting features from {len(files)} segments ...")
    rows = []
    t0 = time.time()
    for i, fp in enumerate(files):
        try:
            x = load_ch(fp, channel)
            feats = extract_features(x, band)
        except Exception as e:
            print(f"   [skip] {fp.name}: {e}"); continue
        feats["file"] = fp.name
        feats["idx"] = i
        feats["time_min"] = i * SEG_PERIOD_MIN
        rows.append(feats)
        if (i + 1) % 25 == 0:
            print(f"   {i+1}/{len(files)}  ({time.time()-t0:.0f}s elapsed)")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "features.csv", index=False)
    print(f"[save] features.csv  ({len(df)} rows)")

    # ---- (3) Composite HI + plots + metrics ----
    # IR-focused composite: BPFI harmonics + sidebands (this run is clearly IR)
    df["HI_IR"] = df[["BPFI_h1", "BPFI_h2", "BPFI_sb_plus", "BPFI_sb_minus"]].sum(axis=1)
    # generic composite: sum of all top-3 fault harmonics
    cols_all = [f"{n}_h{h}" for n in FAULT_ORDERS for h in (1, 2, 3)]
    df["HI_all"] = df[cols_all].sum(axis=1)

    hi_list = ["rms", "kurtosis", "crest",
               "BPFI_h1", "BPFI_h2", "BPFO_h1", "BSF_h1",
               "HI_IR", "HI_all"]

    # Trajectory plot (HI series)
    fig, axes = plt.subplots(len(hi_list), 1, figsize=(12, 1.8 * len(hi_list)),
                             sharex=True)
    for ax, col in zip(axes, hi_list):
        t_h = df["time_min"] / 60
        ax.plot(t_h, df[col], ".", ms=2, alpha=0.4, color="C0")
        ax.plot(t_h, df[col].rolling(11, center=True, min_periods=1).median(),
                "-", lw=1.3, color="crimson")
        m = monotonicity(df[col].values)
        tr = trendability(df[col].values)
        ax.set_ylabel(col); ax.grid(alpha=0.3)
        ax.set_title(f"{col}   mono={m:.2f}   trend={tr:.2f}", fontsize=9, loc="left")
    axes[-1].set_xlabel("time [hours]")
    fig.tight_layout(); fig.savefig(out_dir / "hi_trajectory.png", dpi=120); plt.close(fig)

    # RPM trajectory (sanity check)
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.plot(df["time_min"] / 60, df["rpm"], ".", ms=2)
    ax.axhline(700, color="gray", ls="--", alpha=0.4)
    ax.axhline(950, color="gray", ls="--", alpha=0.4)
    ax.set(xlabel="time [hours]", ylabel="estimated RPM",
           title="RPM trajectory  (spec range 700-950 dashed)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "rpm_trajectory.png", dpi=120); plt.close(fig)

    # Ranking table
    metrics = []
    for col in hi_list:
        y = df[col].values
        metrics.append({"HI": col,
                        "mono": monotonicity(y),
                        "trend": trendability(y),
                        "first": y[0], "last": y[-1],
                        "ratio": y[-1] / (abs(y[0]) + 1e-12)})
    mdf = (pd.DataFrame(metrics)
             .sort_values("mono", ascending=False))
    mdf.to_csv(out_dir / "hi_metrics.csv", index=False)
    print("\n[HI ranking by monotonicity]")
    print(mdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n[done] all outputs -> {out_dir.resolve()}")
    return df, mdf


# ----------------------------- CLI ------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dir", help="Train run directory with .tdms segments")
    ap.add_argument("--ref-file", default=None,
                    help="Reference tdms filename for kurtogram (default: last segment)")
    ap.add_argument("--channel", default="CH2")
    ap.add_argument("--out", default="./traj_out")
    ap.add_argument("--kmax", type=int, default=5, help="Max kurtogram level")
    args = ap.parse_args()
    run(args.train_dir, ref_file=args.ref_file, channel=args.channel,
        out_dir=args.out, nlevel=args.kmax)