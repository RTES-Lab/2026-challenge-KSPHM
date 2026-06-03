"""
LL-Cepstrum Band-Energy Health Index (LOO)
==========================================
Reference: SR/0602/Cepstrum_ref.md

핵심 아이디어:
  N_FFT=25600 (1초 윈도우)로 fault frequency의 cepstrum bin 위치를 높여서
  RPM 변동에 따른 bin shift 영향을 줄임.

  Quefrency bin 계산 (N_FFT=FS=25600 → freq_res=1Hz → n ≈ f_Hz):
    BSF  @ 700~950 RPM : 54.6 ~ 74.1 Hz  → n ≈ 55~74
    BPFO @ 700~950 RPM : 65.1 ~ 88.4 Hz  → n ≈ 65~88
    BPFI @ 700~950 RPM : 98.0 ~133.0 Hz  → n ≈ 98~133

  Fault band (n=55~149): 700~950 RPM 전체 RPM 범위에서
  발생 가능한 모든 fault frequency cepstrum 에너지를 합산.
  → 정확한 RPM을 몰라도 해당 구간 에너지가 열화 시 증가.

Feature per file (4채널 평균, 30개 1초 윈도우 평균):
  E_struct : mean(c_tilde[1:55]²)   — 구조적 전달함수 구간 에너지
  E_fault  : mean(c_tilde[55:150]²) — fault frequency 구간 에너지

LOO HI 구성:
  Bearing i의 HI를 구성할 때, Bearing i를 제외한 나머지 베어링의
  초기 15% 정상 데이터로만 레퍼런스 구성 (leakage 없음).
  HI = (E_fault - mu_ref) / (max_E_fault - mu_ref), clipped [0,1]
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nptdms
import warnings

warnings.filterwarnings("ignore")

BASE_DIR   = "/data/home/ksphm/2026-challenge-KSPHM"
DATA_DIR   = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "User", "SR", "0602", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS           = 25600
INTERVAL_SEC = 600
NORMAL_RATIO = 0.15
N_FFT        = 25600   # = FS → freq_res = 1 Hz → quefrency bin n ≈ frequency in Hz
EPS          = 1e-10
N_WINDOWS    = 30      # 1초 윈도우 × 30 = 30초 사용 (파일당 60초 중)

# Quefrency band boundaries (N_FFT=25600, FS=25600 기준)
N_STRUCT_LO  = 1    # structural transfer function 시작
N_STRUCT_HI  = 55   # BSF @ 700 RPM (54.6 Hz) 직전
N_FAULT_LO   = 55   # fault band 시작 (BSF @ 700 RPM)
N_FAULT_HI   = 150  # BPFI @ 950 RPM (133 Hz) + 여유

BEARING_IDS = [1, 2, 3, 4]
COLORS      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ─────────────────────────────────────────────
# LL-Cepstrum 계산 (liftering 없이 전체 반환)
# ─────────────────────────────────────────────

def compute_ll_cepstrum(x: np.ndarray) -> np.ndarray:
    """
    Log-mean-removed cepstrum for one N_FFT-length window.
    liftering 없이 전체 길이 반환 → 이후 band별로 에너지 추출.
    """
    X      = np.fft.rfft(x, n=N_FFT)
    S      = np.log(np.abs(X) + EPS)
    S_mean = S - S.mean()              # mean removal → n=0 cepstrum coeff = 0
    c      = np.fft.irfft(S_mean, n=N_FFT)
    return c                           # shape: (N_FFT,)


def extract_band_energies(signal: np.ndarray) -> tuple:
    """
    1초 윈도우 N_WINDOWS개 평균으로 파일 1개의 band 에너지 추출.

    Returns:
      e_struct : structural band 에너지 (scalar)
      e_fault  : fault band 에너지     (scalar)
    """
    n_avail = len(signal) // N_FFT
    if n_avail == 0:
        x_pad = np.zeros(N_FFT)
        x_pad[:len(signal)] = signal
        c = compute_ll_cepstrum(x_pad)
        return (np.mean(c[N_STRUCT_LO:N_STRUCT_HI] ** 2),
                np.mean(c[N_FAULT_LO:N_FAULT_HI]   ** 2))

    n_use = min(N_WINDOWS, n_avail)
    idxs  = np.linspace(0, n_avail - 1, n_use, dtype=int)

    e_structs, e_faults = [], []
    for idx in idxs:
        seg = signal[idx * N_FFT:(idx + 1) * N_FFT]
        c   = compute_ll_cepstrum(seg)
        e_structs.append(np.mean(c[N_STRUCT_LO:N_STRUCT_HI] ** 2))
        e_faults.append( np.mean(c[N_FAULT_LO:N_FAULT_HI]   ** 2))

    return float(np.mean(e_structs)), float(np.mean(e_faults))


# ─────────────────────────────────────────────
# 데이터 로드 및 파일별 feature 추출
# ─────────────────────────────────────────────

def load_tdms(path: str) -> dict:
    f   = nptdms.TdmsFile(path)
    grp = f["Vibration"]
    return {ch: grp[ch][:] for ch in ["CH1", "CH2", "CH3", "CH4"]}


def extract_file_features(path: str) -> tuple:
    """4채널 평균 band 에너지 반환 → (e_struct, e_fault)"""
    channels   = load_tdms(path)
    e_s_list, e_f_list = [], []
    for sig in channels.values():
        e_s, e_f = extract_band_energies(sig)
        e_s_list.append(e_s)
        e_f_list.append(e_f)
    return float(np.mean(e_s_list)), float(np.mean(e_f_list))


def process_bearing(bid: int) -> pd.DataFrame:
    vib_dir = os.path.join(DATA_DIR, f"Train{bid}_Vibration")
    files   = sorted(glob.glob(os.path.join(vib_dir, "*.tdms")))
    print(f"  Bearing {bid}: {len(files)} files")

    rows = []
    for i, path in enumerate(files):
        file_idx = int(os.path.splitext(os.path.basename(path))[0])
        e_s, e_f = extract_file_features(path)
        rows.append({
            "bearing_id": bid,
            "file_idx":   file_idx,
            "time_sec":   (file_idx - 1) * INTERVAL_SEC,
            "e_struct":   e_s,
            "e_fault":    e_f,
        })
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(files)}]")

    return pd.DataFrame(rows).sort_values("file_idx").reset_index(drop=True)


# ─────────────────────────────────────────────
# LOO HI 구성
# ─────────────────────────────────────────────

def build_hi_loo(all_dfs: list, target_idx: int) -> pd.DataFrame:
    """
    Bearing target_idx의 HI를 LOO 방식으로 구성.
    레퍼런스 = 다른 베어링들의 초기 NORMAL_RATIO 정상 구간 e_fault.

    HI = (e_fault - mu_ref) / (max_e_fault - mu_ref), clipped [0,1]
    """
    # Step 1: 레퍼런스 e_fault 수집 (target 제외)
    ref_e_faults = []
    for i, df in enumerate(all_dfs):
        if i == target_idx:
            continue
        n_norm = max(int(len(df) * NORMAL_RATIO), 5)
        ref_e_faults.extend(df["e_fault"].iloc[:n_norm].tolist())
    ref_e_faults = np.array(ref_e_faults)

    mu_ref = ref_e_faults.mean()
    # 레퍼런스 분산으로 정상 범위 상한 설정 (3-sigma)
    threshold_ref = mu_ref + 3 * ref_e_faults.std()

    # Step 2: target의 e_fault로 HI 계산
    e_fault_tgt = all_dfs[target_idx]["e_fault"].values
    e_fault_max = e_fault_tgt.max()
    denom = max(e_fault_max - mu_ref, 1e-12)

    hi = np.clip((e_fault_tgt - mu_ref) / denom, 0, 1).astype(np.float32)

    print(f"    mu_ref={mu_ref:.6f}  3σ_ref={threshold_ref:.6f}  "
          f"e_fault_max={e_fault_max:.6f}")

    df_out = all_dfs[target_idx].copy()
    df_out["HI"] = hi
    return df_out


# ─────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────

def plot_2x2(hi_dfs: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    fig.suptitle(
        "LL-Cepstrum Fault-Band Energy HI (LOO)\n"
        "2026 KSPHM-KIMM Challenge",
        fontsize=14, fontweight="bold"
    )

    for i, (bid, ax) in enumerate(zip(BEARING_IDS, axes)):
        df       = hi_dfs[bid]
        time_hr  = df["time_sec"] / 3600
        ref_ids  = [b for b in BEARING_IDS if b != bid]
        n_normal = max(int(len(df) * NORMAL_RATIO), 5)

        ax.plot(time_hr, df["HI"], color=COLORS[i],
                linewidth=1.2, alpha=0.9, label="HI (fault-band energy)")

        t_norm_end = time_hr.iloc[n_normal - 1]
        ax.axvspan(0, t_norm_end, alpha=0.08, color="green")
        ax.axvline(t_norm_end, color="green", linestyle="--",
                   linewidth=0.9, alpha=0.7, label="Normal boundary")
        ax.axhline(1.0, color="red", linestyle=":", linewidth=0.8, alpha=0.6)

        ax.set_title(
            f"Bearing {bid}  |  Reference: Bearings {ref_ids}\n"
            f"({len(df)} pts, {time_hr.max():.1f} hr)",
            fontsize=10, fontweight="bold"
        )
        ax.set_xlabel("Time [hr]", fontsize=9)
        ax.set_ylabel("Health Index", fontsize=9)
        ax.set_xlim(left=0)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "HI_cepstrum_faultband_LOO.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  LL-Cepstrum Fault-Band Energy HI  (LOO)")
    print(f"  N_FFT={N_FFT}  fault_band=[{N_FAULT_LO},{N_FAULT_HI})")
    print(f"  normal_ratio={NORMAL_RATIO}")
    print("=" * 60)

    all_dfs = []

    # Step 1: feature 추출 (cache 활용)
    print("\n[1] Extracting band energies...")
    for bid in BEARING_IDS:
        cache = os.path.join(OUTPUT_DIR, f"Bearing{bid}_bandfeats.csv")
        if os.path.exists(cache):
            print(f"  Bearing {bid}: loading cache")
            df = pd.read_csv(cache)
        else:
            df = process_bearing(bid)
            df.to_csv(cache, index=False)
        all_dfs.append(df)

    # Step 2: LOO HI
    print("\n[2] Building LOO HI...")
    hi_dfs = {}
    for i, bid in enumerate(BEARING_IDS):
        ref_ids = [b for b in BEARING_IDS if b != bid]
        print(f"\n  Bearing {bid}  (ref: Bearings {ref_ids})")
        df_hi = build_hi_loo(all_dfs, target_idx=i)
        df_hi.to_csv(
            os.path.join(OUTPUT_DIR, f"Bearing{bid}_HI_cepstrum_faultband.csv"),
            index=False
        )
        hi_dfs[bid] = df_hi

    # Step 3: 2x2 plot
    print("\n[3] Plotting 2x2...")
    plot_2x2(hi_dfs)

    print("\n[Done]")


if __name__ == "__main__":
    main()
