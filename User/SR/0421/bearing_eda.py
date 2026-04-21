"""
KSPHM-KIMM 2026 Bearing Challenge - EDA Script
================================================
한 개의 TDMS 파일을 로드해서 아래 3가지를 한 번에 확인합니다.

  1) 1x shaft harmonic이 PSD에서 보이는가        → tacholess RPM 추정 viable?
  2) Kurtogram에서 resonance band가 뽑히는가     → SK 기반 band selection 유효?
  3) Envelope spectrum에서 BPFI/BPFO/BSF 피크가  → HI 방향성 확인

사용법:
    python bearing_eda.py <tdms_file_path> [--channel CH2] [--out ./eda_out]

또는 Jupyter에서:
    from bearing_eda import explore
    result = explore("/path/to/segment.tdms")
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, stft, butter, sosfiltfilt, hilbert
from nptdms import TdmsFile

# ---------- 상수 ----------
FS = 25_600  # sampling rate [Hz], 과제 명세

# 1000 RPM (= 16.667 Hz shaft) 기준 고장 주파수
FAULT_HZ_REF = {"BPFI": 140.0, "BPFO": 93.0, "BSF": 78.0, "FTF": 6.7}
SHAFT_HZ_REF = 1000.0 / 60.0  # 16.667 Hz

# RPM-invariant한 order 값 (이 값들은 속도에 무관하게 고정)
FAULT_ORDERS = {k: v / SHAFT_HZ_REF for k, v in FAULT_HZ_REF.items()}
# -> BPFI 8.40, BPFO 5.58, BSF 4.68, FTF 0.40


# ---------- Core utilities ----------
def load_tdms(path):
    """TDMS -> DataFrame"""
    tdms = TdmsFile.read(path)
    df = tdms.as_dataframe()
    # 컬럼 이름이 "/'Group'/'CH1'" 식이므로 간단히 정리
    df.columns = [c.split("/")[-1].strip("'") for c in df.columns]
    return df


def estimate_shaft_hz(x, fs=FS, f_range=(10.0, 17.0), n_harm=4):
    """
    Harmonic Product Spectrum 기반 tacholess shaft frequency 추정.
    700-950 RPM 범위 = 11.67-15.83 Hz 이지만, 여유를 두고 (10, 17) 검색.
    """
    # 고해상도 PSD
    f, P = welch(x, fs=fs, nperseg=min(len(x), 1 << 16))
    # 후보 주파수 범위
    mask = (f >= f_range[0]) & (f <= f_range[1])
    f_cand = f[mask]
    # 각 후보 f0에 대해 log-PSD를 1x, 2x, 3x, ... 에서 합산 (HPS-log version)
    logP = np.log(P + 1e-20)
    scores = np.zeros_like(f_cand)
    for i, f0 in enumerate(f_cand):
        s = 0.0
        for h in range(1, n_harm + 1):
            idx = np.argmin(np.abs(f - h * f0))
            s += logP[idx]
        scores[i] = s
    return float(f_cand[np.argmax(scores)])


def spectral_kurtosis(x, fs=FS, nperseg=1024, noverlap=None):
    """
    STFT 기반 Spectral Kurtosis.
    SK(f) = E[|X(f,t)|^4] / E[|X(f,t)|^2]^2 - 2
    높을수록 impulsive (fault signal에 어울리는 band).
    """
    if noverlap is None:
        noverlap = 3 * nperseg // 4
    f, _, Z = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mag2 = np.abs(Z) ** 2
    num = np.mean(mag2 ** 2, axis=1)
    den = np.mean(mag2, axis=1) ** 2 + 1e-20
    sk = num / den - 2.0
    return f, sk


def envelope_spectrum(x, fs=FS, band=None, nperseg=None):
    """Bandpass -> Hilbert envelope -> PSD of envelope."""
    if band is not None:
        lo, hi = max(band[0], 1.0), min(band[1], fs / 2 - 1.0)
        sos = butter(6, [lo, hi], btype="band", fs=fs, output="sos")
        x = sosfiltfilt(sos, x)
    env = np.abs(hilbert(x))
    env = env - env.mean()
    nperseg = nperseg or min(len(env), 1 << 15)
    f, P = welch(env, fs=fs, nperseg=nperseg)
    return f, P


# ---------- HI 후보 ----------
def compute_hi_candidates(x, fs=FS):
    """이 segment 1개에 대한 HI 후보들을 딕셔너리로 반환."""
    # time-domain (RPM에 덜 민감한 shape feature들)
    x_ac = x - x.mean()
    rms = np.sqrt(np.mean(x_ac ** 2))
    peak = np.max(np.abs(x_ac))
    kurt = np.mean(x_ac ** 4) / (rms ** 4 + 1e-20)
    crest = peak / (rms + 1e-20)

    # shaft freq 추정 + SK band 선택
    f_shaft = estimate_shaft_hz(x, fs)
    f_sk, sk = spectral_kurtosis(x, fs)
    # SK는 저주파(<500Hz)는 피하고 고주파 resonance를 잡자
    mask = f_sk > 500.0
    if mask.any():
        f_best = f_sk[mask][np.argmax(sk[mask])]
    else:
        f_best = f_sk[np.argmax(sk)]
    band = (max(f_best - 500.0, 100.0), min(f_best + 500.0, fs / 2 - 100.0))

    # Envelope spectrum -> fault order 근방 amplitude
    f_env, P_env = envelope_spectrum(x, fs, band=band)
    fault_energy = {}
    for name, f_ref in FAULT_HZ_REF.items():
        f_est = f_ref * (f_shaft / SHAFT_HZ_REF)
        # f_est 주변 ±2Hz 구간의 peak amplitude
        idx = (f_env > f_est - 2.0) & (f_env < f_est + 2.0)
        fault_energy[name] = float(np.sqrt(P_env[idx].max())) if idx.any() else 0.0

    return {
        "shaft_hz": f_shaft,
        "rpm": f_shaft * 60,
        "sk_center_hz": float(f_best),
        "rms": float(rms),
        "kurtosis": float(kurt),
        "crest": float(crest),
        **{f"env_{k}": v for k, v in fault_energy.items()},
    }


# ---------- EDA plots ----------
def explore(file_path, channel=None, out_dir="./eda_out"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_tdms(file_path)
    print(f"[load] {file_path}")
    print(f"       columns: {df.columns.tolist()}")
    print(f"       rows:    {len(df)}")

    # 진동 채널 자동 탐색
    vib_cols = [c for c in df.columns if c.upper().startswith("CH")]
    if channel is None:
        # 30306은 tapered roller + 축방향 하중 지배 -> CH2(front axial) 우선
        channel = "CH2" if "CH2" in vib_cols else vib_cols[0]
    print(f"[pick] channel = {channel}")
    x = df[channel].dropna().to_numpy(dtype=np.float32)

    # HI 후보 계산 + 로그
    hi = compute_hi_candidates(x)
    print("[HI] " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in hi.items()))

    # ===== Plot 1. Time series =====
    t = np.arange(len(x)) / FS
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, x, lw=0.3)
    ax.set(xlabel="time [s]", ylabel=channel,
           title=f"Time series  ({Path(file_path).name})")
    fig.tight_layout(); fig.savefig(out_dir / "01_timeseries.png", dpi=120); plt.close(fig)

    # ===== Plot 2. PSD (0-500 Hz) + 추정 fault freq 표시 =====
    f, P = welch(x, fs=FS, nperseg=1 << 16)
    f_shaft = hi["shaft_hz"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.semilogy(f, P); ax.set_xlim(0, 500)
    ax.set(xlabel="Hz", ylabel="PSD",
           title=f"PSD  (est. shaft = {f_shaft:.2f} Hz = {f_shaft*60:.0f} RPM)")
    ax.axvline(f_shaft, color="g", lw=1, label=f"1x ({f_shaft:.1f} Hz)")
    for name, f_ref in FAULT_HZ_REF.items():
        f_e = f_ref * (f_shaft / SHAFT_HZ_REF)
        ax.axvline(f_e, color="r", ls="--", alpha=0.4)
        ax.text(f_e, ax.get_ylim()[1] * 0.3, name, rotation=90, fontsize=8)
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "02_psd.png", dpi=120); plt.close(fig)

    # ===== Plot 3. Spectral Kurtosis =====
    f_sk, sk = spectral_kurtosis(x)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(f_sk, sk)
    ax.axvline(hi["sk_center_hz"], color="r", ls="--",
               label=f"peak @ {hi['sk_center_hz']:.0f} Hz")
    ax.set(xlabel="Hz", ylabel="SK", title="Spectral Kurtosis")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "03_spectral_kurtosis.png", dpi=120); plt.close(fig)

    # ===== Plot 4. Envelope spectrum (order-scaled x축) =====
    band_c = hi["sk_center_hz"]
    band = (max(band_c - 500, 100), min(band_c + 500, FS / 2 - 100))
    f_env, P_env = envelope_spectrum(x, FS, band=band)
    orders = f_env / f_shaft  # order domain 변환
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(orders, np.sqrt(P_env)); ax.set_xlim(0, 15)
    ax.set(xlabel="order  (× shaft freq)", ylabel="envelope amplitude",
           title=f"Envelope order spectrum  (band {band[0]:.0f}-{band[1]:.0f} Hz)")
    for name, o in FAULT_ORDERS.items():
        ax.axvline(o, color="r", ls="--", alpha=0.5)
        ax.text(o, ax.get_ylim()[1] * 0.8, name, fontsize=9, color="r")
    fig.tight_layout()
    fig.savefig(out_dir / "04_envelope_order_spectrum.png", dpi=120); plt.close(fig)

    print(f"[save] plots -> {out_dir.resolve()}")
    return hi


# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="path to a TDMS segment file")
    ap.add_argument("--channel", default=None, help="channel name, e.g. CH2")
    ap.add_argument("--out", default="./eda_out")
    args = ap.parse_args()
    explore(args.file, channel=args.channel, out_dir=args.out)