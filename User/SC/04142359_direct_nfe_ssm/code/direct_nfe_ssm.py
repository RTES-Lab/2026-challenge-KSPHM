"""
Two-Factor SSM with Direct NFE Input
=====================================================
대안 3 구현: 별도의 선형 PCA-SPE 변환 없이
물리적 특징인 'FTF NFE (케이지 결함 포락선 정규화 에너지)'를
직접 Two-Factor SSM의 관측값 y_k로 주입합니다.

NFE는 이미 전체 RMS로 나누어 RPM 정규화 효과가 내재되어 있으나,
Two-Factor SSM의 감마(offset)와 뮤(drift) 파라미터가 
잔여 RPM 효과를 추가로 잡아낼 수 있습니다.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths & Constants
# ─────────────────────────────────────────────
BASE_DIR    = "/data/home/ksphm/2026-challenge-KSPHM"
DATA_DIR    = os.path.join(BASE_DIR, "dataset")
FEAT_DIR    = os.path.join(BASE_DIR, "User", "SC",
                           "04142331_fault_freq_features", "output")
OUTPUT_DIR  = os.path.join(BASE_DIR, "User", "SC",
                           "04142359_direct_nfe_ssm", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEARING_IDS  = [1, 2, 3, 4]
INTERVAL_SEC = 600
N_COND       = 2           # 0=저속, 1=고속
RPM_BOUNDARY = 850         # 실제 패턴 기반 경계
NORMAL_RATIO = 0.15
N_MC         = 2000
DT           = 1.0

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ─────────────────────────────────────────────
# Quality Metrics
# ─────────────────────────────────────────────
def monotonicity(x):
    dx = np.diff(x)
    return abs(np.sum(dx > 0) - np.sum(dx < 0)) / (len(dx) + 1e-12)

def trendability(x):
    rho, _ = spearmanr(x, np.arange(len(x)))
    return abs(rho) if not np.isnan(rho) else 0.0

# ─────────────────────────────────────────────
# Two-Factor SSM (Kalman-based)
# ─────────────────────────────────────────────
class TwoFactorSSM:
    def __init__(self, n_cond=N_COND, dt=DT):
        self.n_cond  = n_cond
        self.dt      = dt
        self.mu      = np.zeros(n_cond)
        self.gamma   = np.zeros(n_cond)
        self.sigma_w = 0.01
        self.sigma_v = 0.05
        self.x0      = 0.0
        self.P0      = 1e-4

    def _to_vec(self):
        return np.concatenate([
            self.mu, self.gamma,
            [np.log(self.sigma_w), np.log(self.sigma_v)]
        ])

    def _from_vec(self, v):
        nc = self.n_cond
        self.mu      = v[:nc]
        self.gamma   = v[nc:2*nc]
        self.sigma_w = max(np.exp(v[2*nc]),   1e-6)
        self.sigma_v = max(np.exp(v[2*nc+1]), 1e-8)

    def kalman_filter(self, y, cond):
        N = len(y)
        x_f = np.zeros(N)
        P_f = np.zeros(N)
        ll  = 0.0
        x_p = self.x0
        P_p = self.P0
        Q   = self.sigma_w ** 2
        R   = self.sigma_v ** 2

        for k in range(N):
            c = cond[k]
            x_pred = x_p + self.mu[c] * self.dt
            P_pred = P_p + Q
            innov  = y[k] - (x_pred + self.gamma[c])
            S      = P_pred + R
            K      = P_pred / S
            x_post = x_pred + K * innov
            P_post = (1 - K) * P_pred
            ll    += -0.5 * (np.log(2*np.pi*S) + innov**2/S)
            x_f[k] = x_post
            P_f[k] = P_post
            x_p, P_p = x_post, P_post

        return x_f, P_f, ll

    def fit(self, y_list, cond_list, verbose=True):
        def neg_ll(v):
            try:
                self._from_vec(v)
                return -sum(self.kalman_filter(y, c)[2]
                            for y, c in zip(y_list, cond_list))
            except Exception:
                return 1e10

        x0 = self._to_vec()
        res = minimize(neg_ll, x0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
        self._from_vec(res.x)
        if verbose:
            print(f"  MLE 수렴: {res.success}  neg_LL={res.fun:.4f}")
            print(f"  mu(drift)    = {np.round(self.mu, 6)}")
            print(f"  gamma(offset)= {np.round(self.gamma, 6)}")
            print(f"  sigma_w={self.sigma_w:.6f}, sigma_v={self.sigma_v:.6f}")

    def predict_rul(self, x_now, P_now, future_cond,
                    threshold=1.0, n_mc=N_MC, max_steps=1000):
        # Prevent premature failure if currently above threshold
        if x_now >= threshold:
            return {"rul_mean_sec": 0.0, "rul_median_sec": 0.0, 
                    "rul_std_sec": 0.0, "rul_5pct_sec": 0.0, 
                    "rul_95pct_sec": 0.0, "rul_samples": np.zeros(n_mc)}

        x_s = np.random.normal(x_now, np.sqrt(P_now), n_mc)
        rul = np.full(n_mc, max_steps, dtype=float)
        alive = np.ones(n_mc, dtype=bool)
        n_f = len(future_cond)
        
        for step in range(max_steps):
            c = future_cond[step % n_f]
            # Add state drift + noise
            x_s[alive] += self.mu[c]*self.dt + np.random.normal(0, self.sigma_w, alive.sum())
            failed = alive & (x_s >= threshold)
            rul[failed] = step + 1
            alive[failed] = False
            if not alive.any():
                break
                
        rul_sec = rul * INTERVAL_SEC
        return {
            "rul_mean_sec":   float(np.mean(rul_sec)),
            "rul_median_sec": float(np.median(rul_sec)),
            "rul_std_sec":    float(np.std(rul_sec)),
            "rul_5pct_sec":   float(np.percentile(rul_sec, 5)),
            "rul_95pct_sec":  float(np.percentile(rul_sec, 95)),
            "rul_samples":    rul_sec,
        }

# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Direct NFE -> Two Factor SSM Pipeline")
    print("  Feature: Channel avg of 'FTF_nfe' (Cage Defect Envelope Energy)")
    print("=" * 70)

    # ── Step 1: 데이터 로드 (NFE)
    bearing_data = {}
    for bid in BEARING_IDS:
        df = pd.read_csv(os.path.join(FEAT_DIR, f"Bearing{bid}_fault_freq_features.csv"))
        rpm = df['rpm'].values.astype(float)
        cond = (rpm >= RPM_BOUNDARY).astype(int)
        
        # FTF nfe 특징 채널 평균 계산
        ftf_cols = [c for c in df.columns if "_FTF_nfe" in c]
        x_raw = df[ftf_cols].mean(axis=1).values.astype(np.float64)
        
        bearing_data[bid] = {"raw_y": x_raw, "cond": cond, "rpm": rpm, "time_sec": df["time_sec"].values}

    # ── Step 2: 간단한 Scaling 
    # SSM 입력의 수치적 안정성을 위해 전체 범위를 0~1 내외로 스케일링합니다
    # 모든 베어링의 데이터를 모아서 Scale 파라미터를 맞춥니다
    all_y = np.concatenate([bd["raw_y"] for bd in bearing_data.values()]).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(all_y)
    
    for bid in BEARING_IDS:
        y_scaled = scaler.transform(bearing_data[bid]["raw_y"].reshape(-1, 1)).flatten()
        bearing_data[bid]["y"] = y_scaled

    print(f"\n[Step 2] Scaling 완료. (min={all_y.min():.5f}, max={all_y.max():.5f} -> MinMaxScaler)")

    # ── Step 3: Two-Factor SSM (Leave-One-Out)
    print(f"\n[Step 3] Two-Factor SSM Run (4-fold LOOCV)")
    print("-" * 70)
    results = []

    for test_bid in BEARING_IDS:
        print(f"\n  [TEST: Bearing {test_bid}]")
        train_bids = [b for b in BEARING_IDS if b != test_bid]
        train_y    = [bearing_data[b]["y"]    for b in train_bids]
        train_cond = [bearing_data[b]["cond"] for b in train_bids]

        model = TwoFactorSSM(n_cond=N_COND, dt=DT)
        model.mu    = np.array([0.001, 0.001])
        model.gamma = np.array([0.0, 0.0])
        model.fit(train_y, train_cond, verbose=True)

        test_y    = bearing_data[test_bid]["y"]
        test_cond = bearing_data[test_bid]["cond"]
        x_f, P_f, _ = model.kalman_filter(test_y, test_cond)

        # FPT Threshold 설정 
        # (NFE가 상승하는 특성을 기반으로, 앞부분 15% 정상치 + 3*std)
        n_normal  = max(int(len(test_y) * NORMAL_RATIO), 5)
        x_n_mean  = x_f[:n_normal].mean()
        x_n_std   = x_f[:n_normal].std()
        
        # 최소 Threshold를 0.3으로 줍니다
        threshold = max(x_n_mean + 4 * x_n_std, 0.3)
        print(f"  Dynamic Threshold: {threshold:.4f}")

        exceed = np.where(x_f >= threshold)[0]
        fpt_idx = int(exceed[0]) if len(exceed) > 0 else len(test_y) - 1
        fpt_sec = fpt_idx * INTERVAL_SEC
        print(f"  FPT 탐지: idx={fpt_idx}/{len(test_y)}, t={fpt_sec/3600:.2f}hr")

        # RUL 모델링
        future_cond = test_cond[max(0, len(test_cond)-6):]
        pred = model.predict_rul(float(x_f[-1]), float(P_f[-1]),
                                 future_cond, threshold=threshold)
        print(f"  RUL 예측: {pred['rul_mean_sec']/3600:.3f}hr (±{pred['rul_std_sec']/3600:.3f}hr)")

        # 시각화 추가
        _plot_ssm(test_bid, test_y, x_f, P_f, threshold,
                  fpt_idx, pred, test_cond,
                  bearing_data[test_bid]["rpm"], OUTPUT_DIR)

        results.append({
            "bearing":     test_bid,
            "fpt_idx":     fpt_idx,
            "fpt_hr":      round(fpt_sec/3600, 3),
            "threshold":   round(threshold, 4),
            "rul_pred_hr": round(pred["rul_mean_sec"]/3600, 3),
            "rul_std_hr":  round(pred["rul_std_sec"]/3600, 3),
            "hi_mon":      round(monotonicity(test_y), 4),
            "hi_tre":      round(trendability(test_y), 4),
        })

    print("\n" + "=" * 70)
    print("  결과 요약")
    print("=" * 70)
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    df_res.to_csv(os.path.join(OUTPUT_DIR, "Direct_NFE_SSM_summary.csv"), index=False)
    
    _plot_all(results, bearing_data, OUTPUT_DIR)

# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def _plot_ssm(bid, hi, x_f, P_f, thr, fpt_idx, pred, cond, rpm, save_dir):
    N = len(hi)
    t = np.arange(N) * INTERVAL_SEC / 3600
    c = COLORS[bid-1]
    std = np.sqrt(P_f)

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=False)
    fig.suptitle(f"Bearing {bid} — Direct FTF NFE -> F2S2 SSM",
                 fontsize=13, fontweight="bold")

    # 1) RPM
    ax = axes[0]
    ax.plot(t, rpm, color="gray", lw=0.8)
    for cv, label, col in [(0, "Low RPM", "#4C72B0"), (1, "High RPM", "#DD8452")]:
        mask = (cond == cv)
        ax.scatter(t[mask], rpm[mask], s=12, color=col, alpha=0.7, label=label)
    ax.axhline(RPM_BOUNDARY, color="red", ls="--", lw=0.8)
    ax.set_ylabel("RPM"); ax.set_title("Operating Condition")
    ax.legend(); ax.grid(True, ls="--", alpha=0.3)

    # 2) HI & State
    ax = axes[1]
    ax.plot(t, hi, color=c, lw=0.8, alpha=0.5, label="NFE obs (y_k)")
    ax.plot(t, x_f, color=c, lw=1.8, label="State (x_k)")
    ax.fill_between(t, x_f-2*std, x_f+2*std, alpha=0.15, color=c)
    ax.axhline(thr, color="red", ls="--", lw=1, label=f"Threshold ({thr:.2f})")
    if fpt_idx < N-1:
        ax.axvline(t[fpt_idx], color="orange", ls=":", lw=1.5, label=f"FPT ({t[fpt_idx]:.1f}hr)")
    ax.set_ylabel("Scaled NFE / State"); ax.set_title("Kalman-Filtered FTF Degradation")
    ax.legend(loc="upper left"); ax.grid(True, ls="--", alpha=0.3)

    # 3) RUL Histogram
    ax = axes[2]
    rul_hr = pred["rul_samples"] / 3600
    ax.hist(rul_hr, bins=60, color=c, alpha=0.7, edgecolor="white")
    ax.axvline(pred["rul_mean_sec"]/3600, color="red", lw=1.5, label=f"Mean: {pred['rul_mean_sec']/3600:.2f}hr")
    ax.set_xlabel("Predicted RUL [hr]"); ax.set_title("RUL Distribution (MC)")
    ax.legend(); ax.grid(True, ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Bearing{bid}_direct_nfe_ssm.png"), dpi=150, bbox_inches="tight")
    plt.close()

def _plot_all(results, bd, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()
    fig.suptitle("Direct NFE SSM Comparison", fontsize=13, fontweight="bold")

    for i, res in enumerate(results):
        bid = res["bearing"]
        ax  = axes[i]
        t   = bd[bid]["time_sec"] / 3600
        c   = COLORS[i]

        y_obs = bd[bid]["y"]
        ax.plot(t, y_obs,  color=c, lw=0.7, alpha=0.4, label="NFE obs")
        ax.axhline(res["threshold"], color="red", ls="--", lw=0.9, label="Threshold")
        if res["fpt_idx"] < len(y_obs)-1:
            ax.axvline(res["fpt_hr"], color="orange", ls=":", lw=1.3, label=f"FPT={res['fpt_hr']:.1f}hr")

        ax.set_xlim(left=0)
        ax.set_title(f"Bearing {bid} (RUL={res['rul_pred_hr']:.1f}hr, Tre={res['hi_tre']:.3f})", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "All_direct_NFE_SSM.png"), dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
