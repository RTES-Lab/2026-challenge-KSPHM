"""
PCA 기반 Health Index (HI) 구성 - Baseline
=============================================
대회: 2026 KSPHM-KIMM 기계 데이터 챌린지
작업 범위: SC 폴더 내 전용

설명:
  - 각 Train 베어링의 진동 TDMS 파일로부터 시간/주파수 도메인 특징 추출
  - PCA를 통해 1차원 Health Index 구성 (PC1 기반)
  - HI를 [0, 1]로 정규화 (0: 정상, 1: 고장)
  - 결과 CSV 및 시각화 플롯 저장

채널 정보:
  CH1: Front Vertical Vibration
  CH2: Front Axial Vibration
  CH3: Rear Vertical Vibration
  CH4: Rear Axial Vibration

샘플링 레이트: 25,600 Hz
수집 주기: 10분마다 1분 취득 → 파일 번호 = 수집 인덱스
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless 환경 대응
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import nptdms
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR   = "/data/home/ksphm/2026-challenge-KSPHM"
DATA_DIR   = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "User", "SC", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 25600          # 샘플링 레이트 [Hz]
INTERVAL_SEC = 600  # 10분 주기 [sec]

# 정상 구간: 각 베어링의 초기 N% 구간을 정상으로 가정
NORMAL_RATIO = 0.15  # 앞 15%를 정상 구간으로 사용

# ─────────────────────────────────────────────
# 특징 추출 함수 (시간/주파수 도메인)
# ─────────────────────────────────────────────

def extract_time_features(x: np.ndarray) -> dict:
    """시간 도메인 특징 추출"""
    rms       = np.sqrt(np.mean(x ** 2))
    peak      = np.max(np.abs(x))
    p2p       = np.max(x) - np.min(x)
    mean_abs  = np.mean(np.abs(x))
    std       = np.std(x)
    skewness  = float(stats.skew(x))
    kurtosis  = float(stats.kurtosis(x))  # excess kurtosis
    crest_f   = peak / (rms + 1e-12)
    shape_f   = rms / (mean_abs + 1e-12)
    impulse_f = peak / (mean_abs + 1e-12)
    energy    = np.sum(x ** 2)
    # Kurtosis * RMS^4 → 복합 지표 (KurtRMS)
    kurt_rms  = kurtosis * (rms ** 4)

    return {
        "rms":        rms,
        "peak":       peak,
        "p2p":        p2p,
        "std":        std,
        "skewness":   skewness,
        "kurtosis":   kurtosis,
        "crest_f":    crest_f,
        "shape_f":    shape_f,
        "impulse_f":  impulse_f,
        "energy":     energy,
        "kurt_rms":   kurt_rms,
    }


def extract_freq_features(x: np.ndarray, fs: int = FS) -> dict:
    """주파수 도메인 특징 추출 (Welch PSD 기반)"""
    nperseg = min(4096, len(x) // 8)
    f, psd = welch(x, fs=fs, nperseg=nperseg)

    total_power = np.sum(psd)

    # 주파수 대역별 에너지 비율 (베어링 고장 주파수 대역 중심)
    def band_energy_ratio(f_low, f_high):
        idx = (f >= f_low) & (f < f_high)
        return np.sum(psd[idx]) / (total_power + 1e-12)

    # 대역 정의 (30306 베어링, ~900 RPM 기준 스케일)
    low_band   = band_energy_ratio(0,    500)    # 저주파 성분
    mid_band   = band_energy_ratio(500,  3000)   # 중간 주파수
    high_band  = band_energy_ratio(3000, 8000)   # 고주파 (결함 특성)
    bhigh_band = band_energy_ratio(8000, 12800)  # 초고주파

    # 스펙트럼 통계
    mean_freq = np.sum(f * psd) / (np.sum(psd) + 1e-12)
    freq_std  = np.sqrt(np.sum(((f - mean_freq) ** 2) * psd) / (np.sum(psd) + 1e-12))
    spectral_entropy = -np.sum((psd / (total_power + 1e-12)) *
                                np.log(psd / (total_power + 1e-12) + 1e-12))

    return {
        "total_power":       total_power,
        "low_band":          low_band,
        "mid_band":          mid_band,
        "high_band":         high_band,
        "bhigh_band":        bhigh_band,
        "mean_freq":         mean_freq,
        "freq_std":          freq_std,
        "spectral_entropy":  spectral_entropy,
    }


def extract_all_features(x: np.ndarray, fs: int = FS, ch_name: str = "ch") -> dict:
    """단일 채널에서 모든 특징 추출"""
    tf = extract_time_features(x)
    ff = extract_freq_features(x, fs)
    combined = {}
    for k, v in tf.items():
        combined[f"{ch_name}_{k}"] = v
    for k, v in ff.items():
        combined[f"{ch_name}_{k}"] = v
    return combined


# ─────────────────────────────────────────────
# TDMS 파일 로드 및 특징 추출
# ─────────────────────────────────────────────

def load_tdms_channels(tdms_path: str) -> dict:
    """TDMS 파일에서 4채널 데이터 로드"""
    f = nptdms.TdmsFile(tdms_path)
    group = f["Vibration"]
    channels = {}
    for ch_name in ["CH1", "CH2", "CH3", "CH4"]:
        channels[ch_name] = group[ch_name][:]
    return channels


def extract_features_from_file(tdms_path: str) -> dict:
    """단일 TDMS 파일에서 전체 특징 벡터 추출"""
    channels = load_tdms_channels(tdms_path)
    feat = {}
    for ch_name, data in channels.items():
        ch_feats = extract_all_features(data, fs=FS, ch_name=ch_name.lower())
        feat.update(ch_feats)
    return feat


def process_bearing(bear_id: int, verbose: bool = True) -> pd.DataFrame:
    """단일 베어링의 모든 TDMS 파일 처리 → 특징 DataFrame 반환"""
    vib_dir = os.path.join(DATA_DIR, f"Train{bear_id}_Vibration")
    tdms_files = sorted(glob.glob(os.path.join(vib_dir, "*.tdms")))
    n_files = len(tdms_files)

    if verbose:
        print(f"\n[Bearing {bear_id}] 파일 수: {n_files}개 처리 시작...")

    rows = []
    for i, fpath in enumerate(tdms_files):
        fname = os.path.basename(fpath)
        file_idx = int(os.path.splitext(fname)[0])  # 000001 → 1
        time_sec = (file_idx - 1) * INTERVAL_SEC    # 실제 경과 시간 [sec]

        feat = extract_features_from_file(fpath)
        feat["file_idx"] = file_idx
        feat["time_sec"] = time_sec
        feat["bearing_id"] = bear_id
        rows.append(feat)

        if verbose and (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_files}] 완료")

    df = pd.DataFrame(rows)
    df = df.sort_values("file_idx").reset_index(drop=True)

    if verbose:
        print(f"  완료! 특징 수: {len(df.columns)-3}")
    return df


# ─────────────────────────────────────────────
# PCA 기반 Health Index 구성
# ─────────────────────────────────────────────

FEAT_COLS = None  # 전역 특징 컬럼명 (첫 베어링 처리 후 설정)


# 로그 변환이 필요한 특징 (스케일이 매우 큰 특징들)
LOG_TRANSFORM_KEYWORDS = ["energy", "kurt_rms"]


def log_transform_features(X: np.ndarray, feat_cols: list) -> np.ndarray:
    """energy, kurt_rms 특징에 log1p 변환 적용 (스케일 정규화)"""
    X = X.copy()
    for i, col in enumerate(feat_cols):
        if any(kw in col for kw in LOG_TRANSFORM_KEYWORDS):
            X[:, i] = np.log1p(np.abs(X[:, i]))
    return X


def build_hi_pca(df: pd.DataFrame,
                 normal_ratio: float = NORMAL_RATIO,
                 n_components: int = 3) -> pd.DataFrame:
    """
    PCA 기반 HI 구성
    - energy/kurt_rms 특징에 log1p 변환 적용 (스케일 폭발 방지)
    - 정상 구간(초기 normal_ratio)으로 PCA 학습
    - 전체 데이터에 적용하여 PC1 기반 HI 생성
    - HI를 [0, 1]로 정규화 (min-max, 정상 구간 최솟값=0)
    """
    global FEAT_COLS
    feat_cols = [c for c in df.columns
                 if c not in ["file_idx", "time_sec", "bearing_id"]]
    FEAT_COLS = feat_cols

    X_raw = df[feat_cols].values.astype(np.float32)
    X = log_transform_features(X_raw, feat_cols)  # 스케일 압축

    # 정상 구간 인덱스
    n_normal = max(int(len(df) * normal_ratio), 5)
    X_normal = X[:n_normal]  # log-transformed 정상 구간

    # 스케일링 (정상 구간 기준)
    scaler = StandardScaler()
    scaler.fit(X_normal)
    X_scaled = scaler.transform(X)
    X_normal_scaled = X_scaled[:n_normal]

    # PCA 학습 (정상 구간)
    pca = PCA(n_components=n_components)
    pca.fit(X_normal_scaled)

    # 전체 데이터에 적용
    scores = pca.transform(X_scaled)  # shape: (N, n_components)

    # PC1 점수를 원시 HI로 사용
    # 단, HI는 단조 증가하는 형태여야 하므로 PC1의 방향성 확인
    pc1 = scores[:, 0]
    if pc1[-1] < pc1[0]:  # 말기값이 초기값보다 작으면 부호 반전
        pc1 = -pc1

    # 정규화: 정상 구간 평균을 0으로, 전체 최댓값을 1로
    hi_min = np.mean(pc1[:n_normal])
    hi_max = np.max(pc1)
    if hi_max <= hi_min:
        hi_max = hi_min + 1e-6
    hi = (pc1 - hi_min) / (hi_max - hi_min)
    hi = np.clip(hi, 0, 1)

    # 분산 설명 비율 출력
    explained = pca.explained_variance_ratio_
    print(f"  PCA 분산 설명 비율: PC1={explained[0]:.3f}, "
          f"PC2={explained[1]:.3f}, PC3={explained[2]:.3f} "
          f"(누적: {explained.sum():.3f})")

    df = df.copy()
    df["pc1_raw"] = pc1
    df["HI"] = hi.astype(np.float32)
    return df, pca, scaler


# ─────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def plot_hi_all(hi_dfs: dict, save_path: str):
    """전체 베어링 HI 추이 비교 플롯"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    axes = axes.flatten()

    fig.suptitle("PCA-based Health Index (Baseline)\n"
                 "2026 KSPHM-KIMM Challenge",
                 fontsize=14, fontweight="bold", y=1.01)

    for i, (bear_id, df) in enumerate(hi_dfs.items()):
        ax = axes[i]
        time_hr = df["time_sec"] / 3600  # 시간 단위 [hr]
        hi = df["HI"]

        ax.plot(time_hr, hi, color=COLORS[i], linewidth=1.2, alpha=0.9,
                label=f"Bearing {bear_id}")

        # 정상 구간 표시
        n_normal = max(int(len(df) * NORMAL_RATIO), 5)
        t_normal_end = time_hr.iloc[n_normal - 1]
        ax.axvspan(0, t_normal_end, alpha=0.08, color="green",
                   label="Normal region")
        ax.axvline(t_normal_end, color="green", linestyle="--",
                   linewidth=0.8, alpha=0.7)

        # HI=1 기준선
        ax.axhline(1.0, color="red", linestyle=":", linewidth=0.8, alpha=0.6)

        ax.set_xlim(left=0)
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(f"Bearing {bear_id} "
                     f"(N={len(df)} segments, {time_hr.max():.1f} hr)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Time [hr]", fontsize=9)
        ax.set_ylabel("Health Index (HI)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[저장] HI 플롯 → {save_path}")


def plot_hi_individual(df: pd.DataFrame, bear_id: int, save_dir: str):
    """개별 베어링 상세 플롯 (HI + 특징 상위 4개)"""
    n_normal = max(int(len(df) * NORMAL_RATIO), 5)
    time_hr = df["time_sec"] / 3600

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    color = COLORS[bear_id - 1]

    # 1) HI
    ax = axes[0]
    ax.plot(time_hr, df["HI"], color=color, linewidth=1.4)
    t_normal_end = time_hr.iloc[n_normal - 1]
    ax.axvspan(0, t_normal_end, alpha=0.1, color="green")
    ax.axvline(t_normal_end, color="green", linestyle="--", linewidth=1.0)
    ax.axhline(1.0, color="red", linestyle=":", linewidth=0.8)
    ax.set_ylabel("HI (PCA PC1)", fontsize=10)
    ax.set_title(f"Bearing {bear_id} — PCA Health Index (Baseline)", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, linestyle="--", alpha=0.4)

    # 2) RMS (4채널 평균)
    ax = axes[1]
    for ch, c in zip(["ch1", "ch2", "ch3", "ch4"],
                     ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]):
        col = f"{ch}_rms"
        if col in df.columns:
            ax.plot(time_hr, df[col], linewidth=1.0, alpha=0.8,
                    label=ch.upper(), color=c)
    ax.axvspan(0, t_normal_end, alpha=0.1, color="green")
    ax.axvline(t_normal_end, color="green", linestyle="--", linewidth=1.0)
    ax.set_ylabel("RMS [g]", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)

    # 3) Kurtosis (4채널 평균)
    ax = axes[2]
    for ch, c in zip(["ch1", "ch2", "ch3", "ch4"],
                     ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]):
        col = f"{ch}_kurtosis"
        if col in df.columns:
            ax.plot(time_hr, df[col], linewidth=1.0, alpha=0.8,
                    label=ch.upper(), color=c)
    ax.axvspan(0, t_normal_end, alpha=0.1, color="green")
    ax.axvline(t_normal_end, color="green", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Kurtosis", fontsize=10)
    ax.set_xlabel("Time [hr]", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"Bearing{bear_id}_HI_detail.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [저장] Bearing {bear_id} 상세 플롯 → {save_path}")


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PCA 기반 Health Index 구성 - Baseline")
    print("=" * 60)

    bearing_ids = [1, 2, 3, 4]
    hi_dfs = {}
    all_feat_dfs = []

    # ── Step 1: 특징 추출 ──
    for bear_id in bearing_ids:
        feat_cache = os.path.join(OUTPUT_DIR, f"Bearing{bear_id}_features.csv")

        if os.path.exists(feat_cache):
            print(f"\n[Bearing {bear_id}] 캐시 로드: {feat_cache}")
            df_feat = pd.read_csv(feat_cache)
        else:
            df_feat = process_bearing(bear_id, verbose=True)
            df_feat.to_csv(feat_cache, index=False)
            print(f"  [저장] 특징 캐시 → {feat_cache}")

        all_feat_dfs.append(df_feat)

    # ── Step 2: PCA HI 구성 ──
    print("\n" + "=" * 60)
    print("  PCA Health Index 구성")
    print("=" * 60)

    for bear_id, df_feat in zip(bearing_ids, all_feat_dfs):
        print(f"\n[Bearing {bear_id}]")
        df_hi, pca, scaler = build_hi_pca(df_feat, normal_ratio=NORMAL_RATIO)

        # 저장
        hi_path = os.path.join(OUTPUT_DIR, f"Bearing{bear_id}_HI.csv")
        df_hi[["bearing_id", "file_idx", "time_sec", "HI", "pc1_raw"]].to_csv(
            hi_path, index=False)
        print(f"  [저장] HI CSV → {hi_path}")

        hi_dfs[bear_id] = df_hi

        # 개별 상세 플롯
        plot_hi_individual(df_hi, bear_id, OUTPUT_DIR)

    # ── Step 3: 전체 비교 플롯 ──
    print("\n[전체 비교 플롯 생성 중...]")
    plot_hi_all(hi_dfs, save_path=os.path.join(OUTPUT_DIR, "All_HI_comparison.png"))

    # ── Step 4: 요약 통계 출력 ──
    print("\n" + "=" * 60)
    print("  HI 요약 통계")
    print("=" * 60)
    summary_rows = []
    for bear_id, df in hi_dfs.items():
        n_normal = max(int(len(df) * NORMAL_RATIO), 5)
        row = {
            "Bearing":       bear_id,
            "N_segments":    len(df),
            "Duration_hr":   round(df["time_sec"].max() / 3600, 2),
            "HI_final":      round(df["HI"].iloc[-1], 4),
            "HI_mean_normal": round(df["HI"].iloc[:n_normal].mean(), 4),
            "HI_std_normal": round(df["HI"].iloc[:n_normal].std(), 4),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "HI_summary.csv"), index=False)
    print(f"\n[저장] 요약 통계 → {os.path.join(OUTPUT_DIR, 'HI_summary.csv')}")
    print("\n[완료] 모든 작업이 정상적으로 완료되었습니다.")


if __name__ == "__main__":
    main()
