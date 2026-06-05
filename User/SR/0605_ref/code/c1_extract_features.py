"""
c1_extract_features.py
======================
Feature extraction from raw TDMS vibration files.
Outputs Bearing{b}_features.csv for training bearings (1-4)
and Test{t}_features.csv for test bearings (1-6).

Original source : User/SC/HI/04140103_initial_pca_result/code/hi_pca_baseline.py
Copied + adapted : paths redirected to 0605_ref output directory.
No leakage issues in this stage — features are computed independently per file.
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.signal import welch
import nptdms

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM")
DATA_DIR = BASE_DIR / "dataset"
OUT_DIR  = BASE_DIR / "User/SR/0605_ref/output/hi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS           = 25600
INTERVAL_SEC = 600


# ── Feature extraction (copied verbatim from hi_pca_baseline.py) ──────────────

def extract_time_features(x: np.ndarray) -> dict:
    rms       = np.sqrt(np.mean(x ** 2))
    peak      = np.max(np.abs(x))
    p2p       = np.max(x) - np.min(x)
    mean_abs  = np.mean(np.abs(x))
    std       = np.std(x)
    skewness  = float(stats.skew(x))
    kurtosis  = float(stats.kurtosis(x))
    crest_f   = peak / (rms + 1e-12)
    shape_f   = rms / (mean_abs + 1e-12)
    impulse_f = peak / (mean_abs + 1e-12)
    energy    = np.sum(x ** 2)
    kurt_rms  = kurtosis * (rms ** 4)
    return {
        "rms": rms, "peak": peak, "p2p": p2p, "std": std,
        "skewness": skewness, "kurtosis": kurtosis,
        "crest_f": crest_f, "shape_f": shape_f, "impulse_f": impulse_f,
        "energy": energy, "kurt_rms": kurt_rms,
    }


def extract_freq_features(x: np.ndarray, fs: int = FS) -> dict:
    nperseg = min(4096, len(x) // 8)
    f, psd  = welch(x, fs=fs, nperseg=nperseg)
    total   = np.sum(psd)

    def band_ratio(flo, fhi):
        return float(np.sum(psd[(f >= flo) & (f < fhi)]) / (total + 1e-12))

    mean_freq = float(np.sum(f * psd) / (total + 1e-12))
    freq_std  = float(np.sqrt(np.sum(((f - mean_freq) ** 2) * psd) / (total + 1e-12)))
    spec_ent  = float(-np.sum((psd / (total + 1e-12)) *
                               np.log(psd / (total + 1e-12) + 1e-12)))
    return {
        "total_power":      float(total),
        "low_band":         band_ratio(0, 500),
        "mid_band":         band_ratio(500, 3000),
        "high_band":        band_ratio(3000, 8000),
        "bhigh_band":       band_ratio(8000, 12800),
        "mean_freq":        mean_freq,
        "freq_std":         freq_std,
        "spectral_entropy": spec_ent,
    }


def extract_all_features(x: np.ndarray, ch_name: str = "ch") -> dict:
    tf = extract_time_features(x)
    ff = extract_freq_features(x)
    return {f"{ch_name}_{k}": v for d in (tf, ff) for k, v in d.items()}


def load_tdms_channels(path: str) -> dict:
    tf    = nptdms.TdmsFile(path)
    group = tf["Vibration"]
    return {ch: group[ch][:] for ch in ["CH1", "CH2", "CH3", "CH4"]}


def extract_features_from_file(path: str) -> dict:
    channels = load_tdms_channels(path)
    feat = {}
    for ch_name, data in channels.items():
        feat.update(extract_all_features(data, ch_name=ch_name.lower()))
    return feat


def process_bearing_dir(vib_dir: Path, bearing_id: int, verbose: bool = True) -> pd.DataFrame:
    tdms_files = sorted(vib_dir.glob("*.tdms"))
    rows = []
    for i, fpath in enumerate(tdms_files):
        file_idx = int(fpath.stem)
        feat = extract_features_from_file(str(fpath))
        feat["file_idx"]   = file_idx
        feat["time_sec"]   = (file_idx - 1) * INTERVAL_SEC
        feat["bearing_id"] = bearing_id
        rows.append(feat)
        if verbose and (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(tdms_files)}] done")
    df = pd.DataFrame(rows).sort_values("file_idx").reset_index(drop=True)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run(force: bool = False):
    print("=" * 60)
    print("  c1 — Feature Extraction from TDMS")
    print("=" * 60)

    # Training bearings 1-4
    for b in range(1, 5):
        out_csv = OUT_DIR / f"Bearing{b}_features.csv"
        if out_csv.exists() and not force:
            print(f"  Train{b}: cache exists → skip ({out_csv.name})")
            continue
        print(f"\n  Train{b}: extracting features …")
        vib_dir = DATA_DIR / f"Train{b}_Vibration"
        df = process_bearing_dir(vib_dir, b)
        df.to_csv(out_csv, index=False)
        print(f"    saved → {out_csv.name}  ({len(df)} rows, {len(df.columns)-3} features)")

    # Test bearings 1-6
    for t in range(1, 7):
        out_csv = OUT_DIR / f"Test{t}_features.csv"
        if out_csv.exists() and not force:
            print(f"  Test{t}: cache exists → skip ({out_csv.name})")
            continue
        test_dir = DATA_DIR / "Test" / f"Test{t}"
        if not test_dir.exists():
            print(f"  Test{t}: directory not found, skipping")
            continue
        print(f"\n  Test{t}: extracting features …")
        df = process_bearing_dir(test_dir, t)
        df.to_csv(out_csv, index=False)
        print(f"    saved → {out_csv.name}  ({len(df)} rows)")

    print("\nc1 complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if cache exists")
    args = parser.parse_args()
    run(force=args.force)
