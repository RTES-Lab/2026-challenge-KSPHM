"""
Two-Factor State-Space Model for RUL Prediction
=================================================
Reference: Li et al. (2019), "Remaining useful life prediction of machinery
under time-varying operating conditions based on a two-factor state-space model",
Reliability Engineering and System Safety, 186, 88-100.

Model Structure:
  State Equation (Factor 1 - Degradation Rate):
    x_k = x_{k-1} + mu(c_k) * dt + sigma_w * w_k
    - x_k: latent degradation state at step k
    - mu(c_k): operating-condition-dependent drift (degradation rate)
    - c_k: operating condition at step k (RPM)
    - w_k ~ N(0,1): process noise

  Measurement Equation (Factor 2 - Signal Jump):
    y_k = x_k + gamma(c_k) + sigma_v * v_k
    - y_k: observed HI (from vibration features)
    - gamma(c_k): operating-condition-dependent signal offset (jump correction)
    - v_k ~ N(0,1): measurement noise

  FPT: First time x_k >= threshold D_f
  RUL: E[T_failure | x_k, history] via Monte Carlo simulation

Operating Condition Mapping:
  - RPM from Operation CSV (10s sampling)
  - Each vibration file (10-min interval) → mean RPM during [k*600, k*600+60) s
  - RPM discretized into n_cond conditions for tractable parameter estimation

Data Special Notes:
  - Train1~4: each bearing has different type of pre-induced micro-fault
  - Variable speed: 700~950 RPM (changes every ~1 hour)
  - Sampling: 25,600 Hz vibration, 10s for operation data

Dataset Split Strategy (only Train data available):
  - Leave-One-Out: train on 3 bearings, validate on 1
  - Rotation: 4 splits → average performance
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR    = "/data/home/ksphm/2026-challenge-KSPHM"
DATA_DIR    = os.path.join(BASE_DIR, "dataset")
FEAT_DIR    = os.path.join(BASE_DIR, "User", "SC",
                           "04140103_initial_pca_result", "output")
SEL_FEAT    = os.path.join(BASE_DIR, "User", "SC",
                           "04140144_feature_quality_analysis",
                           "output", "selected_features.txt")
OUTPUT_DIR  = os.path.join(BASE_DIR, "User", "SC",
                           "04142243_two_factor_ssm", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
BEARING_IDS  = [1, 2, 3, 4]
INTERVAL_SEC = 600          # 10분 주기 [s]
MEAS_WIN_SEC = 60           # 1분 측정 창 [s]
NORMAL_RATIO = 0.15         # 정상 구간 비율
N_COND       = 3            # RPM 이산 조건 수 (저속/중속/고속)
N_PCA_COMP   = 5            # HI 구성용 PCA 성분 수
N_MC         = 2000         # Monte Carlo 샘플 수
DT           = 1.0          # 시간 스텝 단위 (파일 인덱스 단위)
LOG_KEYWORDS = ["energy", "kurt_rms"]

# ─────────────────────────────────────────────
# Step 0: Load & Align Operation Data
# ─────────────────────────────────────────────

def load_operation(bear_id: int) -> pd.DataFrame:
    """
    Operation CSV 로드 (10s 샘플링)
    컬럼: Time[sec], Torque[Nm], Motor speed[rpm], TC SP Front, TC SP Rear
    """
    path = os.path.join(DATA_DIR, f"Train{bear_id}_Operation.csv")
    df = pd.read_csv(path, encoding="cp949")
    df.columns = [c.strip() for c in df.columns]
    # 컬럼명 정규화
    df = df.rename(columns={
        "Time[sec]":        "time_sec",
        "Torque[Nm]":       "torque",
        "Motor speed[rpm]": "rpm",
        "TC SP Front[℃]":   "temp_front",
        "TC SP Rear[℃]":    "temp_rear",
    })
    # 불필요 컬럼 제거
    df = df[[c for c in ["time_sec","torque","rpm","temp_front","temp_rear"]
             if c in df.columns]]
    return df


def align_rpm_to_vibration(op_df: pd.DataFrame, n_files: int) -> np.ndarray:
    """
    진동 파일 인덱스(0-based)별 대표 RPM 계산.
    파일 k (1-based index) → 시간 구간 [(k-1)*600, (k-1)*600+60) s
    해당 구간의 op_df['rpm'] 평균을 대표 RPM으로 사용.

    Returns: shape (n_files,) - 각 파일의 평균 RPM
    """
    rpm_per_file = np.zeros(n_files)
    for k in range(n_files):
        t_start = k * INTERVAL_SEC
        t_end   = t_start + MEAS_WIN_SEC
        mask = (op_df["time_sec"] >= t_start) & (op_df["time_sec"] < t_end)
        vals = op_df.loc[mask, "rpm"].values
        rpm_per_file[k] = vals.mean() if len(vals) > 0 else np.nan

    # NaN 보간 (선형)
    nans = np.isnan(rpm_per_file)
    if nans.any():
        idx = np.arange(n_files)
        rpm_per_file[nans] = np.interp(
            idx[nans], idx[~nans], rpm_per_file[~nans])

    return rpm_per_file


def discretize_rpm(rpm_array: np.ndarray, n_cond: int = N_COND) -> np.ndarray:
    """
    RPM을 n_cond개의 이산 조건으로 분류.
    분위수 기반 균등 분할.
    Returns: int array, shape (n_files,), values in {0, ..., n_cond-1}
    """
    quantiles = np.linspace(0, 100, n_cond + 1)
    thresholds = np.percentile(rpm_array, quantiles)
    cond = np.zeros(len(rpm_array), dtype=int)
    for j in range(1, n_cond):
        cond[rpm_array >= thresholds[j]] = j
    return cond


# ─────────────────────────────────────────────
# Step 1: Health Indicator (HI) Construction
# ─────────────────────────────────────────────

def log_transform(X: np.ndarray, feat_cols: list) -> np.ndarray:
    X = X.copy()
    for i, col in enumerate(feat_cols):
        if any(kw in col for kw in LOG_KEYWORDS):
            X[:, i] = np.log1p(np.abs(X[:, i]))
    return X


def build_hi(bear_id: int,
             selected_feats: list,
             normal_ratio: float = NORMAL_RATIO,
             n_components: int = N_PCA_COMP) -> np.ndarray:
    """
    선별 특징 기반 PCA SPE(재구성 오차) HI 구성.
    정상 구간으로 PCA 학습 → 전체에 SPE 계산 → [0,1] 정규화.
    """
    feat_path = os.path.join(FEAT_DIR, f"Bearing{bear_id}_features.csv")
    df = pd.read_csv(feat_path)

    avail = [f for f in selected_feats if f in df.columns]
    X_raw = df[avail].values.astype(np.float64)
    X     = log_transform(X_raw, avail)

    n_normal = max(int(len(df) * normal_ratio), 5)
    scaler = StandardScaler()
    scaler.fit(X[:n_normal])
    Xs = scaler.transform(X)

    n_comp = min(n_components, n_normal - 1, len(avail))
    pca = PCA(n_components=n_comp)
    pca.fit(Xs[:n_normal])

    # SPE
    recon  = pca.inverse_transform(pca.transform(Xs))
    spe    = np.mean((Xs - recon) ** 2, axis=1)

    # [0,1] 정규화
    s_mean = spe[:n_normal].mean()
    s_max  = spe.max()
    hi = np.clip((spe - s_mean) / (s_max - s_mean + 1e-12), 0, 1)

    return hi.astype(np.float32)


# ─────────────────────────────────────────────
# Step 2: Two-Factor State-Space Model
# ─────────────────────────────────────────────

class TwoFactorSSM:
    """
    Two-Factor State-Space Model (Li et al., 2019)

    Parameters (per operating condition c):
      mu[c]    : drift rate  (degradation rate under condition c)
      gamma[c] : signal jump offset (HI offset under condition c)
      sigma_w  : process noise std (shared across conditions)
      sigma_v  : measurement noise std (shared)

    State equation:   x_k = x_{k-1} + mu[c_k] * dt + sigma_w * w_k
    Measurement eq.:  y_k = x_k + gamma[c_k] + sigma_v * v_k
    """

    def __init__(self, n_cond: int = N_COND, dt: float = DT):
        self.n_cond  = n_cond
        self.dt      = dt
        # Parameters (to be estimated)
        self.mu      = np.zeros(n_cond)     # drift per condition
        self.gamma   = np.zeros(n_cond)     # offset per condition
        self.sigma_w = 0.01                  # process noise
        self.sigma_v = 0.05                  # measurement noise
        self.x0      = 0.0                   # initial state
        self.P0      = 1e-4                  # initial variance

    def _params_to_vec(self):
        return np.concatenate([
            self.mu, self.gamma,
            [np.log(self.sigma_w), np.log(self.sigma_v)]
        ])

    def _vec_to_params(self, vec):
        nc = self.n_cond
        self.mu      = vec[:nc]
        self.gamma   = vec[nc:2*nc]
        self.sigma_w = np.exp(vec[2*nc])
        self.sigma_v = np.exp(vec[2*nc + 1])

    # ── Kalman Filter (linear Gaussian → exact solution)
    def kalman_filter(self, y: np.ndarray, cond: np.ndarray):
        """
        Kalman filter for state estimation.

        State transition:  x_k = x_{k-1} + mu[c_k]*dt  (F=1, B=mu[c_k]*dt)
        Measurement:       y_k = x_k + gamma[c_k]       (H=1, d=gamma[c_k])

        Returns:
          x_filt  : filtered state means   (N,)
          P_filt  : filtered state variances (N,)
          log_lik : total log-likelihood (scalar)
        """
        N       = len(y)
        x_f     = np.zeros(N)
        P_f     = np.zeros(N)
        log_lik = 0.0

        x_prior = self.x0
        P_prior = self.P0

        Q = self.sigma_w ** 2   # process noise variance
        R = self.sigma_v ** 2   # measurement noise variance

        for k in range(N):
            c = cond[k]
            # Predict
            x_pred = x_prior + self.mu[c] * self.dt
            P_pred = P_prior + Q

            # Update
            innov   = y[k] - (x_pred + self.gamma[c])
            S       = P_pred + R               # innovation variance
            K       = P_pred / S              # Kalman gain
            x_post  = x_pred + K * innov
            P_post  = (1 - K) * P_pred

            # Log-likelihood contribution
            log_lik += -0.5 * (np.log(2 * np.pi * S) + innov**2 / S)

            x_f[k]  = x_post
            P_f[k]  = P_post
            x_prior = x_post
            P_prior = P_post

        return x_f, P_f, log_lik

    # ── Parameter Estimation via MLE
    def fit(self, y_list: list, cond_list: list, verbose: bool = True):
        """
        MLE 파라미터 추정 (전체 베어링 데이터의 로그가능도 합산 최적화).

        y_list    : list of HI arrays per bearing
        cond_list : list of condition arrays per bearing
        """
        def neg_log_lik(vec):
            try:
                self._vec_to_params(vec)
                # Clipping to avoid explosion
                self.sigma_w = max(self.sigma_w, 1e-6)
                self.sigma_v = max(self.sigma_v, 1e-6)
                total_ll = 0.0
                for y, cond in zip(y_list, cond_list):
                    _, _, ll = self.kalman_filter(y, cond)
                    total_ll += ll
                return -total_ll
            except Exception:
                return 1e10

        x0 = self._params_to_vec()
        result = minimize(
            neg_log_lik, x0,
            method="Nelder-Mead",
            options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-5,
                     "disp": False}
        )
        self._vec_to_params(result.x)
        self.sigma_w = max(self.sigma_w, 1e-6)
        self.sigma_v = max(self.sigma_v, 1e-6)

        if verbose:
            print(f"  MLE 수렴: {result.success}  |  neg_LL={result.fun:.4f}")
            print(f"  mu(drift)    = {np.round(self.mu, 5)}")
            print(f"  gamma(offset)= {np.round(self.gamma, 5)}")
            print(f"  sigma_w={self.sigma_w:.6f}, sigma_v={self.sigma_v:.6f}")

        return result

    # ── RUL Prediction via Monte Carlo
    def predict_rul(self,
                    x_now: float,
                    P_now: float,
                    future_cond: np.ndarray,
                    threshold: float = 0.5,
                    n_mc: int = N_MC,
                    max_steps: int = 500) -> dict:
        """
        현재 상태 x_now ~ N(x_now, P_now)에서 출발하여
        Monte Carlo로 미래 궤적을 시뮬레이션하고 RUL 분포 추정.

        future_cond: 미래 운전 조건 시퀀스 (반복 패턴)
        threshold  : 고장 임계값 (HI 척도)
        """
        # 초기 상태 샘플링
        x_samples = np.random.normal(x_now, np.sqrt(P_now), n_mc)
        rul_samples = np.full(n_mc, max_steps, dtype=float)
        alive = np.ones(n_mc, dtype=bool)

        n_future = len(future_cond)
        Q = self.sigma_w ** 2

        for step in range(max_steps):
            c = future_cond[step % n_future]
            # State transition
            x_samples[alive] = (
                x_samples[alive]
                + self.mu[c] * self.dt
                + np.random.normal(0, self.sigma_w, alive.sum())
            )
            # Check failure
            failed = alive & (x_samples >= threshold)
            rul_samples[failed] = step + 1
            alive[failed] = False
            if not alive.any():
                break

        rul_sec = rul_samples * INTERVAL_SEC   # 파일 단위 → 초 단위

        return {
            "rul_mean_sec":   float(np.mean(rul_sec)),
            "rul_median_sec": float(np.median(rul_sec)),
            "rul_std_sec":    float(np.std(rul_sec)),
            "rul_5pct_sec":   float(np.percentile(rul_sec, 5)),
            "rul_95pct_sec":  float(np.percentile(rul_sec, 95)),
            "rul_samples":    rul_sec,
        }


# ─────────────────────────────────────────────
# Step 3: FPT Detection
# ─────────────────────────────────────────────

def detect_fpt(hi: np.ndarray, x_filt: np.ndarray,
               threshold: float) -> int:
    """
    필터링된 상태 x_filt가 threshold를 처음 초과하는 인덱스.
    초과하지 않으면 마지막 인덱스 반환.
    """
    exceed = np.where(x_filt >= threshold)[0]
    return int(exceed[0]) if len(exceed) > 0 else len(hi) - 1


# ─────────────────────────────────────────────
# Step 4: Evaluation (RMSE, MAE, Score)
# ─────────────────────────────────────────────

def eval_score(true_rul_sec: float, pred_rul_sec: float) -> float:
    """
    대회 평가 지표:
      Er = 100 * (true - pred) / true
      if Er <= 0 (과대 예측): A = exp(-ln(0.5) * Er / 20)
      if Er > 0  (과소 예측): A = exp(+ln(0.5) * Er / 50)
    """
    Er = 100.0 * (true_rul_sec - pred_rul_sec) / (true_rul_sec + 1e-10)
    if Er <= 0:
        return float(np.exp(-np.log(0.5) * Er / 20))
    else:
        return float(np.exp(np.log(0.5) * Er / 50))


# ─────────────────────────────────────────────
# Step 5: Main Pipeline (Leave-One-Out)
# ─────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Two-Factor State-Space Model for RUL Prediction")
    print("  Li et al. (2019) - RESS 186, 88-100")
    print("=" * 70)

    # ── 선별 특징 로드
    with open(SEL_FEAT, "r") as f:
        selected_feats = [line.strip() for line in f if line.strip()]
    print(f"\n선별 특징 수: {len(selected_feats)}")

    # ── 각 베어링 데이터 준비
    print("\n[Step 0-1] 데이터 준비 (HI 구성 + 운전조건 정렬)")
    bearing_data = {}
    rpm_global = []

    for bid in BEARING_IDS:
        op_df  = load_operation(bid)
        hi     = build_hi(bid, selected_feats)
        n      = len(hi)
        rpm_arr = align_rpm_to_vibration(op_df, n)
        rpm_global.append(rpm_arr)

        bearing_data[bid] = {
            "hi":    hi,
            "rpm":   rpm_arr,
            "n":     n,
            "op_df": op_df,
        }
        print(f"  Bearing {bid}: n={n}, HI=[{hi.min():.4f},{hi.max():.4f}],"
              f" RPM=[{rpm_arr.min():.0f},{rpm_arr.max():.0f}]")

    # ── 전체 RPM 분위수로 공통 이산화 경계 결정
    all_rpm = np.concatenate(rpm_global)
    quantiles = np.linspace(0, 100, N_COND + 1)
    rpm_thresholds = np.percentile(all_rpm, quantiles)
    print(f"\n RPM 구간 경계: {np.round(rpm_thresholds, 0)}")

    def discretize_common(rpm_arr):
        cond = np.zeros(len(rpm_arr), dtype=int)
        for j in range(1, N_COND):
            cond[rpm_arr >= rpm_thresholds[j]] = j
        return cond

    for bid in BEARING_IDS:
        bearing_data[bid]["cond"] = discretize_common(bearing_data[bid]["rpm"])

    # ── Leave-One-Out 교차 검증
    print("\n[Step 2-5] Leave-One-Out 교차 검증")
    print("-" * 70)

    results = []
    for test_bid in BEARING_IDS:
        print(f"\n  [TEST: Bearing {test_bid}]")
        train_bids = [b for b in BEARING_IDS if b != test_bid]

        # 학습 셋 준비 (정상 구간 HI만 사용하거나 전체 사용)
        # 논문: 완전 수명 데이터로 파라미터 학습
        train_hi   = [bearing_data[b]["hi"]   for b in train_bids]
        train_cond = [bearing_data[b]["cond"] for b in train_bids]

        # ── MLE 파라미터 추정
        model = TwoFactorSSM(n_cond=N_COND, dt=DT)
        # 초기값: 각 운전조건별로 약간 다른 drift 부여
        model.mu    = np.linspace(0.001, 0.005, N_COND)
        model.gamma = np.zeros(N_COND)
        model.fit(train_hi, train_cond, verbose=True)

        # ── 테스트 베어링 상태 추정
        test_hi   = bearing_data[test_bid]["hi"]
        test_cond = bearing_data[test_bid]["cond"]
        x_filt, P_filt, _ = model.kalman_filter(test_hi, test_cond)

        # ── 고장 임계값: 정상 구간 상태 + 마진
        n_normal     = max(int(len(test_hi) * NORMAL_RATIO), 5)
        normal_x     = x_filt[:n_normal]
        threshold    = normal_x.mean() + 3 * normal_x.std()
        if threshold < 0.3:
            threshold = 0.3  # 최소 임계값
        print(f"  고장 임계값: {threshold:.4f}")

        # ── FPT 탐지
        fpt_idx = detect_fpt(test_hi, x_filt, threshold)
        fpt_sec = fpt_idx * INTERVAL_SEC
        true_total_sec = (len(test_hi) - 1) * INTERVAL_SEC

        # 실제 RUL (FPT 기준)
        true_rul_from_fpt = (len(test_hi) - 1 - fpt_idx) * INTERVAL_SEC
        print(f"  FPT: idx={fpt_idx}, t={fpt_sec/3600:.2f}hr  |"
              f" True RUL(FPT기준)={true_rul_from_fpt/3600:.2f}hr")

        # ── RUL 예측 (마지막 관측 시점 기준)
        # 미래 운전 조건: 마지막 1시간의 조건 패턴 반복
        last_cond = test_cond[max(0, len(test_cond)-6):]
        pred = model.predict_rul(
            x_now       = float(x_filt[-1]),
            P_now       = float(P_filt[-1]),
            future_cond = last_cond,
            threshold   = threshold,
            n_mc        = N_MC,
        )
        pred_rul_sec = pred["rul_mean_sec"]

        # 실제 RUL (마지막 관측 기준)
        # 대회 정의: 마지막 데이터 기준 RUL = 남은 실제 수명
        # Train에서는 마지막 파일 = 고장 직전으로 가정
        # RUL_true ≈ (고장 시점) - (마지막 관측)
        # 보수적 가정: 마지막 파일 이후 최대 1개 파일 시간(600s) 내 고장
        true_rul_last = 300  # 마지막 관측 후 평균 ~300초 (불측정 구간 가정)

        # 대회 점수 계산
        score = eval_score(true_rul_last, pred_rul_sec)

        print(f"  RUL 예측: {pred_rul_sec/3600:.3f}hr "
              f"(±{pred['rul_std_sec']/3600:.3f}hr)  "
              f"[5%: {pred['rul_5pct_sec']/3600:.3f}hr, "
              f"95%: {pred['rul_95pct_sec']/3600:.3f}hr]")
        print(f"  대회 Score(참고용): {score:.4f}")

        # ── 시각화
        _plot_bearing_result(
            test_bid, test_hi, x_filt, P_filt,
            threshold, fpt_idx, pred, OUTPUT_DIR
        )

        # CSV 저장
        df_out = pd.DataFrame({
            "file_idx":  np.arange(1, len(test_hi)+1),
            "time_sec":  np.arange(len(test_hi)) * INTERVAL_SEC,
            "hi_obs":    test_hi,
            "x_filt":    x_filt,
            "P_filt":    P_filt,
            "cond":      test_cond,
            "rpm":       bearing_data[test_bid]["rpm"],
        })
        csv_path = os.path.join(OUTPUT_DIR, f"Bearing{test_bid}_SSM_result.csv")
        df_out.to_csv(csv_path, index=False)

        results.append({
            "bearing":        test_bid,
            "fpt_idx":        fpt_idx,
            "fpt_hr":         round(fpt_sec/3600, 3),
            "threshold":      round(threshold, 4),
            "rul_pred_sec":   round(pred_rul_sec, 1),
            "rul_pred_hr":    round(pred_rul_sec/3600, 3),
            "rul_std_hr":     round(pred["rul_std_sec"]/3600, 3),
            "score_ref":      round(score, 4),
            "mu":             np.round(model.mu, 6).tolist(),
            "gamma":          np.round(model.gamma, 6).tolist(),
            "sigma_w":        round(model.sigma_w, 6),
            "sigma_v":        round(model.sigma_v, 6),
        })

    # ── 요약 출력 및 저장
    print("\n" + "=" * 70)
    print("  결과 요약")
    print("=" * 70)
    df_res = pd.DataFrame(results)
    print(df_res[["bearing","fpt_hr","threshold","rul_pred_hr",
                  "rul_std_hr","score_ref"]].to_string(index=False))
    sum_path = os.path.join(OUTPUT_DIR, "SSM_summary.csv")
    df_res.to_csv(sum_path, index=False)
    print(f"\n[저장] 요약 → {sum_path}")

    # ── 전체 비교 플롯
    _plot_all_results(bearing_data, results, OUTPUT_DIR)

    print("\n[완료]")


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def _plot_bearing_result(bear_id, hi, x_filt, P_filt,
                         threshold, fpt_idx, pred_dict, save_dir):
    """개별 베어링 SSM 결과 플롯"""
    N   = len(hi)
    t   = np.arange(N) * INTERVAL_SEC / 3600  # hr
    c   = COLORS[bear_id - 1]
    std = np.sqrt(P_filt)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(f"Bearing {bear_id} — Two-Factor SSM (Li et al., 2019)",
                 fontsize=13, fontweight="bold")

    # ── subplot 1: 관측 HI + 필터링된 상태
    ax = axes[0]
    ax.plot(t, hi, color=c, lw=0.8, alpha=0.5, label="Observed HI (y_k)")
    ax.plot(t, x_filt, color=c, lw=1.8, label="Filtered state (x_k)")
    ax.fill_between(t, x_filt - 2*std, x_filt + 2*std,
                    alpha=0.15, color=c, label="95% CI")
    ax.axhline(threshold, color="red", ls="--", lw=1.0,
               label=f"Failure threshold ({threshold:.3f})")
    ax.axvline(t[fpt_idx], color="orange", ls=":", lw=1.5,
               label=f"FPT ({t[fpt_idx]:.1f}hr)")
    ax.set_ylabel("Health Index / State", fontsize=10)
    ax.set_title("Kalman-Filtered Degradation State  (State Equation)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_ylim(-0.05, 1.15)

    # ── subplot 2: RUL 예측 분포
    ax = axes[1]
    rul_samples_hr = pred_dict["rul_samples"] / 3600
    ax.hist(rul_samples_hr, bins=60, color=c, alpha=0.7,
            edgecolor="white", linewidth=0.3)
    ax.axvline(pred_dict["rul_mean_sec"]/3600, color="red", lw=1.5,
               label=f"Mean RUL: {pred_dict['rul_mean_sec']/3600:.2f}hr")
    ax.axvline(pred_dict["rul_5pct_sec"]/3600,  color="orange", lw=1.0,
               ls="--", label=f"5th pct: {pred_dict['rul_5pct_sec']/3600:.2f}hr")
    ax.axvline(pred_dict["rul_95pct_sec"]/3600, color="green", lw=1.0,
               ls="--", label=f"95th pct: {pred_dict['rul_95pct_sec']/3600:.2f}hr")
    ax.set_xlabel("Predicted RUL [hr]", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("RUL Distribution (Monte Carlo Simulation)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(save_dir, f"Bearing{bear_id}_SSM.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [저장] {path}")


def _plot_all_results(bearing_data, results, save_dir):
    """4개 베어링 필터링 상태 한 번에 비교"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()
    fig.suptitle("Two-Factor SSM — All Bearings: Kalman Filtered States",
                 fontsize=13, fontweight="bold")

    for i, res in enumerate(results):
        bid  = res["bearing"]
        ax   = axes[i]
        c    = COLORS[i]
        csvp = os.path.join(save_dir, f"Bearing{bid}_SSM_result.csv")
        df   = pd.read_csv(csvp)
        t    = df["time_sec"] / 3600

        ax.plot(t, df["hi_obs"],  color=c, lw=0.7, alpha=0.45, label="HI obs")
        ax.plot(t, df["x_filt"], color=c, lw=1.7,              label="State x_k")
        ax.axhline(res["threshold"], color="red", ls="--", lw=0.9,
                   label=f"Thr={res['threshold']:.3f}")
        ax.axvline(res["fpt_hr"], color="orange", ls=":", lw=1.3,
                   label=f"FPT={res['fpt_hr']:.1f}hr")

        ax.set_xlim(left=0); ax.set_ylim(-0.05, 1.15)
        ax.set_title(f"Bearing {bid} (RUL pred: {res['rul_pred_hr']:.2f}hr)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Time [hr]", fontsize=9)
        ax.set_ylabel("HI / State", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(save_dir, "All_SSM_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] 전체 비교 → {path}")


if __name__ == "__main__":
    main()
