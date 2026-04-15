"""
Signal Transformation v2 + HI 재구성 + Two-Factor SSM
=====================================================
Reference: Li et al. (2019), RESS 186, 88-100.

개선점 (v1 → v2):
  1. RPM 이산화: 3구간 → 2구간 (실제 패턴: 저속~740 ↔ 고속~970)
     - RPM 850 기준으로 이진 분류 (명확한 bimodal 분포)
  2. Signal Transformation 후 상위 특징만 선별
  3. 선별 후 HI 구성 → Two-Factor SSM 적용까지 end-to-end

RPM 패턴 분석 결과:
  Train1~4 모두 2개 RPM 레벨이 1시간 단위로 교번
  저속: 700~780 RPM (주로 720~750)
  고속: 930~990 RPM (주로 960~980)
  경계: ~850 RPM (두 클러스터 사이 gap 존재)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths & Constants
# ─────────────────────────────────────────────
BASE_DIR    = "/data/home/ksphm/2026-challenge-KSPHM"
DATA_DIR    = os.path.join(BASE_DIR, "dataset")
FEAT_DIR    = os.path.join(BASE_DIR, "User", "SC",
                           "04140103_initial_pca_result", "output")
OUTPUT_DIR  = os.path.join(BASE_DIR, "User", "SC",
                           "04142304_signal_transform_v2", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEARING_IDS  = [1, 2, 3, 4]
INTERVAL_SEC = 600
MEAS_WIN_SEC = 60
N_COND       = 2           # ← 2구간 (저속/고속)
RPM_BOUNDARY = 850         # ← 실제 패턴 기반 경계
BASELINE     = 0           # 저속을 baseline (더 많은 시간 존재)
NORMAL_RATIO = 0.15
N_PCA_COMP   = 5
N_MC         = 2000
DT           = 1.0
LOG_KEYWORDS = ["energy", "kurt_rms"]

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_operation(bid):
    df = pd.read_csv(os.path.join(DATA_DIR, f"Train{bid}_Operation.csv"),
                     encoding="cp949")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Time[sec]": "time_sec",
                             "Motor speed[rpm]": "rpm"})
    return df


def align_rpm(op_df, n_files):
    rpm = np.zeros(n_files)
    for k in range(n_files):
        t0, t1 = k * INTERVAL_SEC, k * INTERVAL_SEC + MEAS_WIN_SEC
        mask = (op_df["time_sec"] >= t0) & (op_df["time_sec"] < t1)
        vals = op_df.loc[mask, "rpm"].values
        rpm[k] = vals.mean() if len(vals) > 0 else np.nan
    nans = np.isnan(rpm)
    if nans.any():
        idx = np.arange(n_files)
        rpm[nans] = np.interp(idx[nans], idx[~nans], rpm[~nans])
    return rpm


def discretize_rpm(rpm_arr):
    """850 RPM 기준 이진 분류: 0=저속, 1=고속"""
    return (rpm_arr >= RPM_BOUNDARY).astype(int)


# ─────────────────────────────────────────────
# Signal Transformation (F2S2 논문)
# ─────────────────────────────────────────────

def estimate_transform_params(y, cond, baseline_cond, n_cond):
    N = len(y)
    t_idx = np.arange(N)
    params = {baseline_cond: (1.0, 0.0)}

    bl_mask  = (cond == baseline_cond)
    bl_times = t_idx[bl_mask]
    bl_vals  = y[bl_mask]

    if len(bl_times) < 2:
        for c in range(n_cond):
            params[c] = (1.0, 0.0)
        return params

    for c in range(n_cond):
        if c == baseline_cond:
            continue
        c_mask  = (cond == c)
        c_times = t_idx[c_mask]
        c_vals  = y[c_mask]

        if len(c_times) < 2:
            params[c] = (1.0, 0.0)
            continue

        y_bl_interp = np.interp(c_times, bl_times, bl_vals)
        A = np.column_stack([c_vals, np.ones(len(c_vals))])
        result, _, _, _ = np.linalg.lstsq(A, y_bl_interp, rcond=None)
        params[c] = (float(result[0]), float(result[1]))

    return params


def transform_signal(y, cond, params):
    y_t = np.zeros_like(y)
    for k in range(len(y)):
        c = cond[k]
        a, b = params[c]
        y_t[k] = a * y[k] + b
    return y_t


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
# HI Construction (SPE on transformed features)
# ─────────────────────────────────────────────

def log_transform(X, feat_cols):
    X = X.copy()
    for i, col in enumerate(feat_cols):
        if any(kw in col for kw in LOG_KEYWORDS):
            X[:, i] = np.log1p(np.abs(X[:, i]))
    return X


def build_hi_from_transformed(df_trans, selected_feats,
                               normal_ratio=NORMAL_RATIO,
                               n_comp=N_PCA_COMP):
    avail = [f for f in selected_feats if f in df_trans.columns]
    X_raw = df_trans[avail].values.astype(np.float64)
    X     = log_transform(X_raw, avail)

    n_normal = max(int(len(df_trans) * normal_ratio), 5)
    scaler = StandardScaler()
    scaler.fit(X[:n_normal])
    Xs = scaler.transform(X)

    nc = min(n_comp, n_normal - 1, len(avail))
    pca = PCA(n_components=nc)
    pca.fit(Xs[:n_normal])

    recon = pca.inverse_transform(pca.transform(Xs))
    spe   = np.mean((Xs - recon) ** 2, axis=1)

    s_mean = spe[:n_normal].mean()
    s_max  = spe.max()
    hi = np.clip((spe - s_mean) / (s_max - s_mean + 1e-12), 0, 1)
    return hi.astype(np.float32)


# ─────────────────────────────────────────────
# Two-Factor SSM (simplified Kalman)
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
        self.sigma_v = max(np.exp(v[2*nc+1]), 1e-6)

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
                       options={"maxiter": 5000, "xatol": 1e-6,
                                "fatol": 1e-6})
        self._from_vec(res.x)
        if verbose:
            print(f"  MLE 수렴: {res.success}  neg_LL={res.fun:.4f}")
            print(f"  mu(drift)    = {np.round(self.mu, 6)}")
            print(f"  gamma(offset)= {np.round(self.gamma, 6)}")
            print(f"  sigma_w={self.sigma_w:.6f}, sigma_v={self.sigma_v:.6f}")

    def predict_rul(self, x_now, P_now, future_cond,
                    threshold=0.5, n_mc=N_MC, max_steps=500):
        x_s = np.random.normal(x_now, np.sqrt(P_now), n_mc)
        rul = np.full(n_mc, max_steps, dtype=float)
        alive = np.ones(n_mc, dtype=bool)
        n_f = len(future_cond)
        for step in range(max_steps):
            c = future_cond[step % n_f]
            x_s[alive] += self.mu[c]*self.dt + np.random.normal(
                0, self.sigma_w, alive.sum())
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
    print("  Signal Transformation v2 + HI + Two-Factor SSM")
    print("  RPM 2구간 이산화 (저속<850 / 고속≥850)")
    print("=" * 70)

    # ── Step 0: 데이터 로드 & RPM 정렬
    feat_cols = None
    bearing_data = {}

    for bid in BEARING_IDS:
        df = pd.read_csv(os.path.join(FEAT_DIR,
                                      f"Bearing{bid}_features.csv"))
        if feat_cols is None:
            feat_cols = [c for c in df.columns
                         if c not in ["file_idx","time_sec","bearing_id"]]
        op = load_operation(bid)
        n  = len(df)
        rpm = align_rpm(op, n)
        cond = discretize_rpm(rpm)
        bearing_data[bid] = {"df": df, "rpm": rpm, "cond": cond, "n": n}

    n_low  = sum((bd["cond"]==0).sum() for bd in bearing_data.values())
    n_high = sum((bd["cond"]==1).sum() for bd in bearing_data.values())
    print(f"\n특징 수: {len(feat_cols)}")
    print(f"RPM 구간: 저속(<850)={n_low}회, 고속(≥850)={n_high}회")
    print(f"Baseline: {BASELINE} ({'저속' if BASELINE==0 else '고속'})")

    # ── Step 1: Signal Transformation
    print("\n[Step 1] Signal Transformation (F2S2)")
    transformed_dfs = {}

    for bid in BEARING_IDS:
        df   = bearing_data[bid]["df"]
        cond = bearing_data[bid]["cond"]
        df_t = df[["file_idx", "time_sec", "bearing_id"]].copy()

        for feat in feat_cols:
            y_raw  = df[feat].values.astype(np.float64)
            params = estimate_transform_params(y_raw, cond, BASELINE, N_COND)
            y_t    = transform_signal(y_raw, cond, params)
            df_t[feat] = y_t

        transformed_dfs[bid] = df_t
        # 저장
        df_t.to_csv(os.path.join(OUTPUT_DIR,
                                 f"Bearing{bid}_features_transformed.csv"),
                    index=False)
        print(f"  Bearing {bid}: 변환 완료")

    # ── Step 2: 품질 지표 비교
    print("\n[Step 2] 품질 지표 (Before vs After)")
    before, after = {}, {}
    for f in feat_cols:
        before[f] = {"mon":[], "tre":[]}
        after[f]  = {"mon":[], "tre":[]}

    for bid in BEARING_IDS:
        df_r = bearing_data[bid]["df"]
        df_t = transformed_dfs[bid]
        for f in feat_cols:
            yr = df_r[f].values.astype(np.float64)
            yt = df_t[f].values.astype(np.float64)
            before[f]["mon"].append(monotonicity(yr))
            before[f]["tre"].append(trendability(yr))
            after[f]["mon"].append(monotonicity(yt))
            after[f]["tre"].append(trendability(yt))

    comp = []
    for f in feat_cols:
        mb = np.mean(before[f]["mon"])
        tb = np.mean(before[f]["tre"])
        ma = np.mean(after[f]["mon"])
        ta = np.mean(after[f]["tre"])
        comp.append({
            "feature": f,
            "Mon_bef": round(mb,4), "Tre_bef": round(tb,4),
            "Q_bef":   round((mb+tb)/2, 4),
            "Mon_aft": round(ma,4), "Tre_aft": round(ta,4),
            "Q_aft":   round((ma+ta)/2, 4),
            "dQ":      round((ma+ta)/2 - (mb+tb)/2, 4),
        })

    df_comp = pd.DataFrame(comp).sort_values("Q_aft", ascending=False)
    df_comp = df_comp.reset_index(drop=True)
    df_comp["rank"] = range(1, len(df_comp)+1)
    df_comp.to_csv(os.path.join(OUTPUT_DIR, "quality_comparison_v2.csv"),
                   index=False)

    # 요약 출력
    avg_qb = df_comp["Q_bef"].mean()
    avg_qa = df_comp["Q_aft"].mean()
    n_imp  = (df_comp["dQ"] > 0).sum()
    n_deg  = (df_comp["dQ"] < 0).sum()
    print(f"  평균 Quality: {avg_qb:.4f} → {avg_qa:.4f} (Δ={avg_qa-avg_qb:+.4f})")
    print(f"  개선: {n_imp}/{len(feat_cols)}, 악화: {n_deg}/{len(feat_cols)}")

    # 상위 20개 출력
    print(f"\n{'Rk':>3} {'Feature':<28} {'Q_bef':>7} {'Q_aft':>7} {'ΔQ':>7}")
    print("-" * 60)
    for _, r in df_comp.head(20).iterrows():
        s = "+" if r["dQ"]>0 else ""
        print(f"{int(r['rank']):>3} {r['feature']:<28}"
              f" {r['Q_bef']:>7.4f} {r['Q_aft']:>7.4f} {s}{r['dQ']:>6.4f}")

    # ── Step 3: 상위 특징 선별
    TOP_N = 20
    selected = df_comp.head(TOP_N)["feature"].tolist()
    print(f"\n[Step 3] 상위 {TOP_N}개 특징 선별:")
    for i, f in enumerate(selected):
        print(f"  {i+1:>2}. {f}")

    # ── Step 4: HI 구성 (변환된 특징 → SPE)
    print(f"\n[Step 4] HI 구성 (변환된 상위 {TOP_N}개 특징 → PCA SPE)")
    hi_data = {}
    for bid in BEARING_IDS:
        hi = build_hi_from_transformed(transformed_dfs[bid], selected)
        hi_data[bid] = hi
        print(f"  Bearing {bid}: HI=[{hi.min():.4f}, {hi.max():.4f}]"
              f"  Mon={monotonicity(hi):.4f}, Tre={trendability(hi):.4f}")

    # ── Step 5: Two-Factor SSM (Leave-One-Out)
    print(f"\n[Step 5] Two-Factor SSM (Leave-One-Out 4-fold)")
    print("-" * 70)
    results = []

    for test_bid in BEARING_IDS:
        print(f"\n  [TEST: Bearing {test_bid}]")
        train_bids = [b for b in BEARING_IDS if b != test_bid]
        train_hi   = [hi_data[b]                for b in train_bids]
        train_cond = [bearing_data[b]["cond"]    for b in train_bids]

        model = TwoFactorSSM(n_cond=N_COND, dt=DT)
        model.mu    = np.array([0.003, 0.005])  # 초기값: 고속이 더 빠를 것
        model.gamma = np.array([0.0, 0.0])
        model.fit(train_hi, train_cond, verbose=True)

        test_hi   = hi_data[test_bid]
        test_cond = bearing_data[test_bid]["cond"]
        x_f, P_f, _ = model.kalman_filter(test_hi, test_cond)

        # 임계값
        n_normal   = max(int(len(test_hi) * NORMAL_RATIO), 5)
        x_n_mean   = x_f[:n_normal].mean()
        x_n_std    = x_f[:n_normal].std()
        threshold  = max(x_n_mean + 3 * x_n_std, 0.3)
        print(f"  Threshold: {threshold:.4f}")

        # FPT
        exceed = np.where(x_f >= threshold)[0]
        fpt_idx = int(exceed[0]) if len(exceed) > 0 else len(test_hi) - 1
        fpt_sec = fpt_idx * INTERVAL_SEC
        print(f"  FPT: idx={fpt_idx}, t={fpt_sec/3600:.2f}hr")

        # RUL
        future_cond = test_cond[max(0,len(test_cond)-6):]
        pred = model.predict_rul(float(x_f[-1]), float(P_f[-1]),
                                 future_cond, threshold=threshold)
        print(f"  RUL pred: {pred['rul_mean_sec']/3600:.3f}hr"
              f" (±{pred['rul_std_sec']/3600:.3f}hr)")

        # 시각화
        _plot_ssm(test_bid, test_hi, x_f, P_f, threshold,
                  fpt_idx, pred, bearing_data[test_bid]["cond"],
                  bearing_data[test_bid]["rpm"], OUTPUT_DIR)

        # CSV
        pd.DataFrame({
            "file_idx": np.arange(1, len(test_hi)+1),
            "time_sec": np.arange(len(test_hi)) * INTERVAL_SEC,
            "hi_obs":   test_hi,
            "x_filt":   x_f,
            "P_filt":   P_f,
            "cond":     test_cond,
            "rpm":      bearing_data[test_bid]["rpm"],
        }).to_csv(os.path.join(OUTPUT_DIR,
                               f"Bearing{test_bid}_SSM_result.csv"),
                  index=False)

        results.append({
            "bearing":     test_bid,
            "fpt_idx":     fpt_idx,
            "fpt_hr":      round(fpt_sec/3600, 3),
            "threshold":   round(threshold, 4),
            "rul_pred_hr": round(pred["rul_mean_sec"]/3600, 3),
            "rul_std_hr":  round(pred["rul_std_sec"]/3600, 3),
            "hi_mon":      round(monotonicity(test_hi), 4),
            "hi_tre":      round(trendability(test_hi), 4),
        })

    # ── 요약
    print("\n" + "=" * 70)
    print("  결과 요약")
    print("=" * 70)
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    df_res.to_csv(os.path.join(OUTPUT_DIR, "SSM_summary_v2.csv"), index=False)

    # 전체 비교 플롯
    _plot_all(results, OUTPUT_DIR)

    # Before/After 특징 시계열 비교 플롯
    _plot_feature_comparison(bearing_data, transformed_dfs,
                             selected[:6], OUTPUT_DIR)

    print(f"\n[완료]")


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def _plot_ssm(bid, hi, x_f, P_f, thr, fpt_idx, pred, cond, rpm, save_dir):
    N = len(hi)
    t = np.arange(N) * INTERVAL_SEC / 3600
    c = COLORS[bid-1]
    std = np.sqrt(P_f)

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=False)
    fig.suptitle(f"Bearing {bid} — F2S2 SSM v2 (2-condition)",
                 fontsize=13, fontweight="bold")

    # 1) RPM & Condition
    ax = axes[0]
    ax.plot(t, rpm, color="gray", lw=0.8)
    for cv, label, col in [(0, "Low RPM", "#4C72B0"), (1, "High RPM", "#DD8452")]:
        mask = (cond == cv)
        ax.scatter(t[mask], rpm[mask], s=12, color=col, alpha=0.7, label=label)
    ax.axhline(RPM_BOUNDARY, color="red", ls="--", lw=0.8, label="Boundary (850)")
    ax.set_ylabel("RPM", fontsize=10)
    ax.set_title("Operating Condition (RPM)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)

    # 2) HI & State
    ax = axes[1]
    ax.plot(t, hi, color=c, lw=0.8, alpha=0.5, label="HI obs (y_k)")
    ax.plot(t, x_f, color=c, lw=1.8, label="State (x_k)")
    ax.fill_between(t, x_f-2*std, x_f+2*std, alpha=0.15, color=c)
    ax.axhline(thr, color="red", ls="--", lw=1, label=f"Threshold ({thr:.3f})")
    if fpt_idx < N-1:
        ax.axvline(t[fpt_idx], color="orange", ls=":", lw=1.5,
                   label=f"FPT ({t[fpt_idx]:.1f}hr)")
    ax.set_ylabel("HI / State", fontsize=10)
    ax.set_title("Kalman-Filtered Degradation State", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_ylim(-0.1, 1.2)

    # 3) RUL Distribution
    ax = axes[2]
    rul_hr = pred["rul_samples"] / 3600
    ax.hist(rul_hr, bins=60, color=c, alpha=0.7, edgecolor="white", lw=0.3)
    ax.axvline(pred["rul_mean_sec"]/3600, color="red", lw=1.5,
               label=f"Mean: {pred['rul_mean_sec']/3600:.2f}hr")
    ax.axvline(pred["rul_5pct_sec"]/3600,  color="orange", lw=1, ls="--",
               label=f"5%: {pred['rul_5pct_sec']/3600:.2f}hr")
    ax.axvline(pred["rul_95pct_sec"]/3600, color="green", lw=1, ls="--",
               label=f"95%: {pred['rul_95pct_sec']/3600:.2f}hr")
    ax.set_xlabel("Predicted RUL [hr]", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("RUL Distribution (Monte Carlo)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Bearing{bid}_SSM_v2.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [저장] Bearing{bid}_SSM_v2.png")


def _plot_all(results, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()
    fig.suptitle("F2S2 SSM v2 — All Bearings",
                 fontsize=13, fontweight="bold")

    for i, res in enumerate(results):
        bid = res["bearing"]
        ax  = axes[i]
        csvp = os.path.join(save_dir, f"Bearing{bid}_SSM_result.csv")
        df  = pd.read_csv(csvp)
        t   = df["time_sec"] / 3600
        c   = COLORS[i]

        ax.plot(t, df["hi_obs"],  color=c, lw=0.7, alpha=0.4, label="HI obs")
        ax.plot(t, df["x_filt"], color=c, lw=1.7,             label="State")
        ax.axhline(res["threshold"], color="red", ls="--", lw=0.9,
                   label=f"Thr={res['threshold']:.3f}")
        if res["fpt_idx"] < len(df)-1:
            ax.axvline(res["fpt_hr"], color="orange", ls=":", lw=1.3,
                       label=f"FPT={res['fpt_hr']:.1f}hr")

        ax.set_xlim(left=0); ax.set_ylim(-0.1, 1.2)
        ax.set_title(f"Bearing {bid} (RUL={res['rul_pred_hr']:.1f}hr,"
                     f" Mon={res['hi_mon']:.3f}, Tre={res['hi_tre']:.3f})",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Time [hr]"); ax.set_ylabel("HI / State")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "All_SSM_v2_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] All_SSM_v2_comparison.png")


def _plot_feature_comparison(bd, td, feats, save_dir):
    """상위 특징의 Before/After 시계열 비교 (Bearing 1)"""
    bid = 1
    n  = bd[bid]["n"]
    t  = np.arange(n) * INTERVAL_SEC / 3600
    cond = bd[bid]["cond"]

    fig, axes = plt.subplots(len(feats), 2, figsize=(16, 2.8*len(feats)),
                             sharex=True)
    fig.suptitle(f"Bearing {bid}: Feature Before/After Signal Transform (v2, 2-cond)",
                 fontsize=13, fontweight="bold")

    for i, feat in enumerate(feats):
        yr = bd[bid]["df"][feat].values.astype(np.float64)
        yt = td[bid][feat].values.astype(np.float64)

        yr_n = (yr - yr.min()) / (yr.ptp() + 1e-12)
        yt_n = (yt - yt.min()) / (yt.ptp() + 1e-12)

        # Before
        ax = axes[i, 0]
        for cv, col in [(0,"#4C72B0"),(1,"#DD8452")]:
            m = (cond==cv)
            ax.scatter(t[m], yr_n[m], s=8, color=col, alpha=0.6)
        ax.plot(t, yr_n, color="gray", lw=0.4, alpha=0.4)
        ax.set_ylabel(feat, fontsize=7, rotation=0, ha="right", labelpad=95)
        ax.set_title(f"BEFORE  Mon={monotonicity(yr):.3f} Tre={trendability(yr):.3f}",
                     fontsize=8, loc="left")
        ax.set_ylim(-0.1, 1.2)
        ax.grid(True, ls="--", alpha=0.3)

        # After
        ax = axes[i, 1]
        ax.plot(t, yt_n, color=COLORS[0], lw=1.1, alpha=0.85)
        ax.set_title(f"AFTER   Mon={monotonicity(yt):.3f} Tre={trendability(yt):.3f}",
                     fontsize=8, loc="left")
        ax.set_ylim(-0.1, 1.2)
        ax.grid(True, ls="--", alpha=0.3)

    axes[-1,0].set_xlabel("Time [hr]")
    axes[-1,1].set_xlabel("Time [hr]")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Bearing{bid}_feature_comparison_v2.png"),
                dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[저장] Bearing{bid}_feature_comparison_v2.png")


if __name__ == "__main__":
    main()
