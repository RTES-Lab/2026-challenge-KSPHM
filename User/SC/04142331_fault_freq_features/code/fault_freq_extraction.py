"""
결함 주파수 기반 특징 추출 (Fault Frequency Feature Extraction)
================================================================
대회: 2026 KSPHM-KIMM 기계 데이터 챌린지
실험: 04142331_fault_freq_features

베어링 스펙 (대회 공식 제공):
  모델: 30306
  1000 RPM 기준 고장 주파수:
    BPFI = 140 Hz  (내륜)
    BPFO =  93 Hz  (외륜)
    BSF  =  78 Hz  (전동체)
    FTF  =   6.7 Hz (케이지)

목적:
  1. 각 베어링의 결함 유형(BPFO/BPFI/BSF 우세 여부) 파악
  2. 결함 주파수 에너지를 HI로 사용 → Signal Jump 물리적 해소
  3. F2S2 SSM의 직접 관측값(y_k)으로 활용

방법: Envelope Analysis (포락선 분석)
  1. 진동 신호 → 고역통과 필터 (>500 Hz, 육진동 제거)
  2. Hilbert 변환 → 포락선(envelope) 추출
  3. 포락선 FFT → 포락선 스펙트럼 (BPFI/BPFO/BSF 성분 명확히 나타남)
  4. RPM-scaled 결함 주파수 대역 에너지 추출 (±10% 대역 + 고조파 2x, 3x)
  5. 전체 RMS로 나누어 RPM 효과 정규화 → Normalized Fault Energy (NFE)

RPM 정규화의 Signal Jump 해결 원리:
  - Signal Jump: RPM 높을수록 진동 진폭 상승 → 결함 주파수 에너지도 상승
  - 정규화: NFE = E_fault / E_total → RPM이 올라가면 분자·분모 모두 상승 → 비율 안정
  - 겹쳐 있는 결함 주파수 성분만 상대적으로 강조
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR   = "/data/home/ksphm/2026-challenge-KSPHM"
DATA_DIR   = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "User", "SC",
                          "04142331_fault_freq_features", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 상수 (대회 공식 제공값)
# ─────────────────────────────────────────────
FS           = 25600        # 샘플링 주파수 [Hz]
INTERVAL_SEC = 600          # 진동 파일 간격 [s]
MEAS_WIN_SEC = 60           # 1분 수집
RPM_REF      = 1000.0       # 기준 RPM
BPFI_1000    = 140.0        # BPFI @ 1000 RPM
BPFO_1000    =  93.0        # BPFO @ 1000 RPM
BSF_1000     =  78.0        # BSF  @ 1000 RPM
FTF_1000     =   6.7        # FTF  @ 1000 RPM

FAULT_FREQS_1000 = {
    "BPFI": BPFI_1000,
    "BPFO": BPFO_1000,
    "BSF":  BSF_1000,
    "FTF":  FTF_1000,
}

HARMONICS    = [1, 2, 3]    # 고조파 차수 (1x, 2x, 3x)
BW_RATIO     = 0.10         # 대역폭 = ±10% of fault frequency
HP_CUTOFF    = 500          # 고역통과 필터 컷오프 [Hz]
BEARING_IDS  = [1, 2, 3, 4]
CHANNELS     = [1, 2, 3, 4]
NORMAL_RATIO = 0.15         # 정상 구간 비율

# ─────────────────────────────────────────────
# TDMS 로드
# ─────────────────────────────────────────────

def load_tdms(bear_id: int, file_idx: int) -> dict:
    """
    파일 인덱스(1-based)에 해당하는 TDMS 파일에서
    4채널 진동 신호 반환.
    Returns: {1: array, 2: array, 3: array, 4: array} 또는 None
    """
    try:
        from nptdms import TdmsFile
        pattern = os.path.join(DATA_DIR, f"Train{bear_id}_Vibration", f"{file_idx:06d}.tdms")
        files   = glob.glob(pattern)
        if not files:
            return None
        with TdmsFile.open(files[0]) as tf:
            grp = tf["Vibration"]
            signals = {}
            for ch in CHANNELS:
                try:
                    signals[ch] = grp[f"CH{ch}"].read_data().astype(np.float64)
                except Exception:
                    signals[ch] = np.zeros(FS * MEAS_WIN_SEC)
        return signals
    except Exception as e:
        return None


def load_operation(bear_id: int) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"Train{bear_id}_Operation.csv")
    df = pd.read_csv(path, encoding="cp949")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Time[sec]": "time_sec",
                             "Motor speed[rpm]": "rpm"})
    return df


def align_rpm(op_df: pd.DataFrame, n_files: int) -> np.ndarray:
    rpm = np.zeros(n_files)
    for k in range(n_files):
        t0 = k * INTERVAL_SEC
        t1 = t0 + MEAS_WIN_SEC
        mask = (op_df["time_sec"] >= t0) & (op_df["time_sec"] < t1)
        vals = op_df.loc[mask, "rpm"].values
        rpm[k] = vals.mean() if len(vals) > 0 else np.nan
    nans = np.isnan(rpm)
    if nans.any():
        idx = np.arange(n_files)
        rpm[nans] = np.interp(idx[nans], idx[~nans], rpm[~nans])
    return rpm


# ─────────────────────────────────────────────
# 신호처리: Envelope Analysis
# ─────────────────────────────────────────────

def highpass_filter(x: np.ndarray, cutoff: float, fs: float,
                    order: int = 4) -> np.ndarray:
    """고역통과 필터 (Butterworth)"""
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype="high")
    return filtfilt(b, a, x)


def envelope_spectrum(x: np.ndarray, fs: float) -> tuple:
    """
    포락선 스펙트럼 계산.
    1. 고역통과 필터링
    2. Hilbert 변환으로 포락선 추출
    3. FFT → 포락선 스펙트럼

    Returns: freqs [Hz], amplitude spectrum
    """
    x_hp  = highpass_filter(x, HP_CUTOFF, fs)
    env   = np.abs(hilbert(x_hp))
    env   = env - env.mean()              # DC 제거
    N     = len(env)
    spec  = np.abs(np.fft.rfft(env, n=N)) / N
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    return freqs, spec


def extract_fault_energy(freqs: np.ndarray,
                         spec:  np.ndarray,
                         fault_freq_hz: float,
                         bw_ratio: float = BW_RATIO,
                         harmonics: list = None) -> float:
    """
    특정 결함 주파수 및 그 고조파에서의 포락선 스펙트럼 에너지 합산.

    fault_freq_hz: 해당 RPM에서의 결함 주파수 [Hz]
    bw_ratio: ±bw_ratio × fault_freq_hz 범위 내 에너지
    harmonics: 차수 리스트 (기본 [1,2,3])
    """
    if harmonics is None:
        harmonics = HARMONICS
    total_energy = 0.0
    for h in harmonics:
        fc  = fault_freq_hz * h       # 고조파 중심 주파수
        bw  = fc * bw_ratio           # 대역폭
        mask = (freqs >= fc - bw) & (freqs <= fc + bw)
        if mask.any():
            total_energy += np.sum(spec[mask] ** 2)
    return float(total_energy)


def extract_features_one_file(signals: dict,
                              rpm: float,
                              fs: float = FS) -> dict:
    """
    단일 진동 파일에서 결함 주파수 기반 특징 추출.

    반환 특징:
      - {ch}_{fault}_raw:  원시 결함 에너지
      - {ch}_{fault}_nfe:  정규화 결함 에너지 (÷ 전체 RMS²)
      - {ch}_rms:          전체 RMS
      - {ch}_kurtosis:     첨도
    """
    # 스케일된 결함 주파수
    fault_Hz = {name: f / RPM_REF * rpm
                for name, f in FAULT_FREQS_1000.items()}

    feats = {"rpm": rpm}
    for ch in CHANNELS:
        x = signals.get(ch, np.zeros(fs * MEAS_WIN_SEC))
        if len(x) == 0:
            x = np.zeros(fs * MEAS_WIN_SEC)

        # 전체 RMS (Signal Jump의 기준치)
        rms      = float(np.sqrt(np.mean(x**2)))
        rms_var  = rms**2 + 1e-20       # 나눗셈 안전

        # 첨도
        x_c = x - x.mean()
        kurt = float(np.mean(x_c**4) / (np.mean(x_c**2)**2 + 1e-20))

        feats[f"ch{ch}_rms"]      = rms
        feats[f"ch{ch}_kurtosis"] = kurt

        # Envelope spectrum
        freqs, spec = envelope_spectrum(x, fs)

        # 전체 포락선 에너지 (NFE 분모)
        env_total = float(np.sum(spec**2)) + 1e-20

        for fault_name, fc in fault_Hz.items():
            if fc < 1.0:
                feats[f"ch{ch}_{fault_name}_raw"] = 0.0
                feats[f"ch{ch}_{fault_name}_nfe"] = 0.0
                continue
            e_raw = extract_fault_energy(freqs, spec, fc)
            e_nfe = e_raw / env_total   # 전체 포락선 에너지로 정규화
            feats[f"ch{ch}_{fault_name}_raw"] = e_raw
            feats[f"ch{ch}_{fault_name}_nfe"] = e_nfe

    return feats


# ─────────────────────────────────────────────
# 품질 지표
# ─────────────────────────────────────────────

def monotonicity(x):
    dx = np.diff(x)
    return abs(np.sum(dx > 0) - np.sum(dx < 0)) / (len(dx) + 1e-12)

def trendability(x):
    rho, _ = spearmanr(x, np.arange(len(x)))
    return abs(rho) if not np.isnan(rho) else 0.0


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  결함 주파수 기반 특징 추출 (Envelope Analysis)")
    print("  BPFI=140Hz, BPFO=93Hz, BSF=78Hz @ 1000 RPM")
    print("=" * 70)

    all_dfs = {}

    for bid in BEARING_IDS:
        print(f"\n{'='*20} Bearing {bid} {'='*20}")

        # 파일 목록
        tdms_files = sorted(glob.glob(
            os.path.join(DATA_DIR, f"Train{bid}_Vibration", "*.tdms")))
        n_files = len(tdms_files)
        print(f"  TDMS 파일 수: {n_files}")

        # RPM 로드
        op_df    = load_operation(bid)
        rpm_arr  = align_rpm(op_df, n_files)

        # 파일별 특징 추출
        rows = []
        for k in range(n_files):
            file_idx = k + 1
            signals  = load_tdms(bid, file_idx)
            if signals is None:
                print(f"  [경고] Bearing{bid} 파일 {file_idx} 로드 실패 → 스킵")
                continue

            rpm = float(rpm_arr[k])
            feats = extract_features_one_file(signals, rpm)
            feats["bearing_id"] = bid
            feats["file_idx"]   = file_idx
            feats["time_sec"]   = k * INTERVAL_SEC
            rows.append(feats)

            if file_idx % 20 == 0 or file_idx == n_files:
                print(f"  [{file_idx}/{n_files}] RPM={rpm:.0f}")

        df = pd.DataFrame(rows)
        all_dfs[bid] = df

        # CSV 저장
        csv_path = os.path.join(OUTPUT_DIR,
                                f"Bearing{bid}_fault_freq_features.csv")
        df.to_csv(csv_path, index=False)
        print(f"  [저장] {csv_path}")

    # ── 품질 지표 계산 & 결함 유형 분석
    print("\n" + "=" * 70)
    print("  특징 품질 분석 & 결함 유형 추정")
    print("=" * 70)

    # NFE 특징 컬럼만 품질 평가
    nfe_cols = [c for c in all_dfs[1].columns
                if "_nfe" in c or (c.endswith("_rms") and c.startswith("ch"))
                or c.endswith("_kurtosis")]

    quality = []
    for feat in nfe_cols:
        mon_list, tre_list = [], []
        for bid in BEARING_IDS:
            if feat not in all_dfs[bid].columns:
                continue
            x = all_dfs[bid][feat].values.astype(np.float64)
            mon_list.append(monotonicity(x))
            tre_list.append(trendability(x))
        quality.append({
            "feature":  feat,
            "Mon_mean": round(np.mean(mon_list), 4),
            "Tre_mean": round(np.mean(tre_list), 4),
            "Q":        round((np.mean(mon_list)+np.mean(tre_list))/2, 4),
        })

    df_q = pd.DataFrame(quality).sort_values("Q", ascending=False)
    df_q["rank"] = range(1, len(df_q)+1)
    df_q.to_csv(os.path.join(OUTPUT_DIR, "fault_feature_quality.csv"),
                index=False)

    print(f"\n{'Rk':>3} {'Feature':<30} {'Mon':>7} {'Tre':>7} {'Q':>7}")
    print("-" * 60)
    for _, r in df_q.iterrows():
        print(f"{int(r['rank']):>3} {r['feature']:<30}"
              f" {r['Mon_mean']:>7.4f} {r['Tre_mean']:>7.4f} {r['Q']:>7.4f}")

    # ── 결함 유형 분석
    print("\n" + "=" * 70)
    print("  결함 유형 추정 (NFE 최대 성장 결함 주파수)")
    print("=" * 70)

    for bid in BEARING_IDS:
        df = all_dfs[bid]
        print(f"\n  Bearing {bid}:")
        for fault in ["BPFI", "BPFO", "BSF", "FTF"]:
            cols = [c for c in df.columns
                    if f"_{fault}_nfe" in c]
            if not cols:
                continue
            # 채널 평균 NFE
            avg_nfe = df[cols].mean(axis=1).values
            # 말기 30% 평균 vs 초기 15% 평균
            n = len(avg_nfe)
            n_init = max(int(n * 0.15), 3)
            n_late = max(int(n * 0.30), 3)
            growth = (avg_nfe[-n_late:].mean() /
                      (avg_nfe[:n_init].mean() + 1e-10))
            mon_v  = monotonicity(avg_nfe)
            tre_v  = trendability(avg_nfe)
            print(f"    {fault}: growth_ratio={growth:.2f}x,"
                  f" Mon={mon_v:.3f}, Tre={tre_v:.3f}")

    # ── 시각화: 결함 주파수 에너지 시계열
    _plot_fault_evolution(all_dfs, OUTPUT_DIR)

    # ── 시각화: 스펙트라 대표 파일 (초기 vs 말기)
    _plot_spectra_comparison(BEARING_IDS, OUTPUT_DIR)

    print("\n[완료]")


# ─────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
FAULT_COLORS = {"BPFI": "#4C72B0", "BPFO": "#DD8452",
                "BSF": "#55A868", "FTF": "#9467BD"}


def _plot_fault_evolution(all_dfs, save_dir):
    """각 베어링별 결함 주파수 NFE 시계열 (채널 평균)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle("Fault Frequency Normalized Energy (NFE) — "
                 "Envelope Analysis\n채널 평균, ± 10% 대역 에너지 / 전체 포락선 에너지",
                 fontsize=12, fontweight="bold")

    for i, bid in enumerate(BEARING_IDS):
        df  = all_dfs[bid]
        ax  = axes[i]
        t   = df["time_sec"].values / 3600

        for fault, fc in FAULT_COLORS.items():
            cols = [c for c in df.columns if f"_{fault}_nfe" in c]
            if not cols:
                continue
            avg = df[cols].mean(axis=1).values
            # MinMax 정규화 (시각 비교용)
            mn, mx = avg.min(), avg.max()
            y_n = (avg - mn) / (mx - mn + 1e-12)
            ax.plot(t, y_n, color=fc, lw=1.3, alpha=0.85, label=fault)

        ax.set_title(f"Bearing {bid} ({t.max():.1f}hr)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Time [hr]", fontsize=9)
        ax.set_ylabel("NFE (MinMax norm)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_xlim(left=0); ax.set_ylim(-0.05, 1.15)

    plt.tight_layout()
    path = os.path.join(save_dir, "fault_nfe_evolution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[저장] {path}")


def _plot_spectra_comparison(bearing_ids, save_dir):
    """초기 vs 말기 포락선 스펙트럼 비교 (Bearing별, CH3 기준)"""
    CH = 3  # 대표 채널
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle(f"Envelope Spectrum: Early vs Late — CH{CH}\n"
                 "결함 주파수 성분 성장 시각화",
                 fontsize=12, fontweight="bold")

    for i, bid in enumerate(bearing_ids):
        tdms_files = sorted(glob.glob(
            os.path.join(DATA_DIR, f"Train{bid}_Vibration", "*.tdms")))
        n = len(tdms_files)
        op_df   = load_operation(bid)
        rpm_arr = align_rpm(op_df, n)

        ax = axes[i]
        for label, k, color in [
            ("Early (file 5)",   4,   "#4C72B0"),
            ("Late  (file -5)", n-5,  "#C44E52"),
        ]:
            file_idx = k + 1
            rpm      = rpm_arr[k]
            signals  = load_tdms(bid, file_idx)
            if signals is None:
                continue

            x = signals[CH]
            freqs, spec = envelope_spectrum(x, FS)

            # 표시 범위: 0~300 Hz
            mask = freqs <= 300
            ax.plot(freqs[mask], spec[mask], color=color,
                    lw=0.8, alpha=0.85, label=f"{label} (RPM={rpm:.0f})")

        # 결함 주파수 수직선 (평균 RPM 기준)
        mid_rpm = rpm_arr[n//2]
        for fname, f1000 in FAULT_FREQS_1000.items():
            fc = f1000 * mid_rpm / RPM_REF
            ax.axvline(fc, color=FAULT_COLORS[fname], ls="--",
                       lw=0.8, alpha=0.6, label=f"{fname}={fc:.1f}Hz")
            for h in [2, 3]:
                ax.axvline(fc*h, color=FAULT_COLORS[fname], ls=":",
                           lw=0.5, alpha=0.4)

        ax.set_title(f"Bearing {bid}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Frequency [Hz]", fontsize=9)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.legend(fontsize=6, loc="upper right", ncol=2)
        ax.grid(True, ls="--", alpha=0.3)
        ax.set_xlim(0, 300)

    plt.tight_layout()
    path = os.path.join(save_dir, "spectra_early_vs_late.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {path}")


if __name__ == "__main__":
    main()
