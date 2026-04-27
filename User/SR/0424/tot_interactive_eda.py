# %% [markdown]
# # Tacholess Order Tracking (TOT) & RUL Prediction - Interactive EDA
# 
# 이 스크립트는 백그라운드 스크립트(`tot_rul.py`)에서 수행한 **Tacholess Order Tracking(TOT)** 과정 및 **RUL Prediction** 과정을 단계별로 직접 눈으로 확인하고 시각화할 수 있도록 분해해 놓은 대화형 문서입니다.
# 
# VSCode의 'Run Cell' 혹은 `# %%` 단위 실행 기능을 통해 각 셀을 순서대로 실행하며 중간 산출물을 구경해 보세요!

# %%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, sosfiltfilt, hilbert
from scipy.interpolate import interp1d
from nptdms import TdmsFile
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

# 상수 설정
FS = 25_600
FAULT_HZ_REF = {"BPFI": 140.0, "BPFO": 93.0, "BSF": 78.0, "FTF": 6.7}
SHAFT_HZ_REF = 1000.0 / 60.0
FAULT_ORDERS = {k: v / SHAFT_HZ_REF for k, v in FAULT_HZ_REF.items()}

# 안정적인 런타임을 위해 절대 경로 사용 (FileNotFoundError 방지)
repo_path = Path("/data/home/ksphm/2026-challenge-KSPHM")
sample_file = repo_path / "dataset/Train1_Vibration/000050.tdms"

tdms = TdmsFile.read(sample_file)
df = tdms.as_dataframe()
df.columns = [c.split("/")[-1].strip("'") for c in df.columns]

# CH2(Front Axial) 채널 진동신호
x = df["CH2"].dropna().to_numpy(dtype=np.float32)

t = np.arange(len(x)) / FS
print(f"Loaded {sample_file.name}: {len(x)} samples, approx {t[-1]:.2f} seconds")

# %% [markdown]
# ## Step 1: Harmonic Product Spectrum을 통한 회전주파수 추정

# %%
def estimate_shaft_hz(x, fs, f_range=(10.0, 18.0), n_harm=4):
    f, P = welch(x, fs=fs, nperseg=min(len(x), 1<<16))
    mask = (f >= f_range[0]) & (f <= f_range[1])
    f_cand = f[mask]
    logP = np.log(P + 1e-20)
    scores = np.zeros_like(f_cand)
    for i, f0 in enumerate(f_cand):
        s = 0.0
        for h in range(1, n_harm + 1):
            s += logP[np.argmin(np.abs(f - h * f0))]
        scores[i] = s
    return float(f_cand[np.argmax(scores)]), f, P

f_shaft_approx, f, P = estimate_shaft_hz(x, FS)
print(f"Estimated Shaft Frequency = {f_shaft_approx:.3f} Hz ({f_shaft_approx*60:.1f} RPM)")

plt.figure(figsize=(10,4))
plt.semilogy(f, P)
plt.axvline(f_shaft_approx, color='r', linestyle='--', label=f'1X ({f_shaft_approx:.2f} Hz)')
for h in range(2, 5):
    plt.axvline(f_shaft_approx * h, color='orange', linestyle='--', alpha=0.5)
plt.xlim(0, 100)
plt.title("Power Spectrum & Estimated harmonics")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2: Instantaneous Phase (IP) 추출 
# 1X 대역 주변을 필터링한 후 Hilbert Transform의 각도를 구합니다.

# %%
f_target = f_shaft_approx
bw = 2.0
lo, hi = max(f_target - bw/2, 1.0), min(f_target + bw/2, FS/2 - 1.0)
sos = butter(4, [lo, hi], btype="band", fs=FS, output="sos")
x_bp = sosfiltfilt(sos, x)

x_analytic = hilbert(x_bp)
ip = np.unwrap(np.angle(x_analytic))
ip_monotonic = np.maximum.accumulate(ip)

fig, axes = plt.subplots(3, 1, figsize=(12,8), sharex=True)
axes[0].plot(t[:25600], x[:25600], lw=0.5)
axes[0].set_title("Raw Time Signal (First 1 sec)")
axes[1].plot(t[:25600], x_bp[:25600], color='r', lw=1)
axes[1].set_title(f"Bandpass Filtered around 1X ({lo:.1f} ~ {hi:.1f} Hz)")
axes[2].plot(t[:25600], ip_monotonic[:25600], color='g', lw=2)
axes[2].set_title("Extracted Instantaneous Phase (Radians)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 3: Order Domain 리샘플링

# %%
ip_centered = ip_monotonic - ip_monotonic[0]
total_revs = ip_centered[-1] / (2 * np.pi)
avg_rev_per_sec = total_revs / (t[-1] - t[0])
samples_per_rev = FS / avg_rev_per_sec
n_samples_per_rev = int(np.ceil(samples_per_rev))

target_d_theta = 2 * np.pi / n_samples_per_rev
theta_uniform = np.arange(0, ip_centered[-1], target_d_theta)

inv_interp = interp1d(ip_centered, t, kind='linear', bounds_error=False, fill_value="extrapolate")
t_uniform_angle = inv_interp(theta_uniform)

x_interp = interp1d(t, x, kind='cubic', bounds_error=False, fill_value=0.0)
x_order = x_interp(t_uniform_angle)

print(f"Total Revolutions: {total_revs:.1f}")
print(f"Samples per revolution: {n_samples_per_rev}")
print(f"Resampled length: {len(x_order)} points")

# %% [markdown]
# ## Step 4: Envelope Order Spectrum 계산

# %%
lo_ord, hi_ord = 10, min(100, n_samples_per_rev/2 - 1)
sos_ord = butter(4, [lo_ord, hi_ord], btype="band", fs=n_samples_per_rev, output="sos")
x_filtered = sosfiltfilt(sos_ord, x_order)

env = np.abs(hilbert(x_filtered))
env -= env.mean()

f_ord, P_ord = welch(env, fs=n_samples_per_rev, nperseg=min(len(env), 1<<14))
amp_ord = np.sqrt(P_ord)

plt.figure(figsize=(12,4))
plt.plot(f_ord, amp_ord)
plt.xlim(0, 15)
plt.title("Tacholess Order Tracking Envelope Spectrum")
plt.xlabel("Order (Multiples of Shaft Speed)")
plt.ylabel("Amplitude")

for name, o in FAULT_ORDERS.items():
    plt.axvline(o, color='r', linestyle='--', alpha=0.5)
    plt.text(o, plt.ylim()[1]*0.8, name, fontsize=10, color='red', rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 5: (이미 실행 완료된) RUL 예측 모델 결과 확인

# %%
from IPython.display import Image, display
print("Feature Importances:")
display(Image(filename=str(repo_path / "User/SR/0424/tot_out/feature_importance.png")))
print("\n\nRUL Trajectory vs Real Remaining Time:")
display(Image(filename=str(repo_path / "User/SR/0424/tot_out/rul_trajectory.png")))

# %%
