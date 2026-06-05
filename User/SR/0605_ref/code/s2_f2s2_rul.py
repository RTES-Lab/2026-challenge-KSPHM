"""
Stage 2 + 3: F2S2 State Estimation and RUL Prediction
=======================================================
Paper: Li et al. (20xx) "Remaining Useful Life Prediction … Two-Factor State-Space Model"

State-space model (Wiener process with time-varying degradation rate):
  State transition:  x_k = x_{k-1} + η · r(rpm_k) · Δt + σ_B · √(r(rpm_k)·Δt) · w_k
  Measurement:       y_k = a · x_k^c + b + ε_k,   ε_k ~ N(0, σ²_m)

  where:
    x_k   ∈ [0, 1]  – hidden degradation state (0=new, 1=failure)
    y_k           – CNN-estimated DEI (observation)
    r(rpm) = rpm / rpm_baseline  – continuous condition coefficient
    Δt    = 1 (in cycle units; convert to seconds by × 600)

Parameter estimation from training bearings:
  η, σ_B²  via MLE on inverse-Gaussian failure time distribution
  a, b, c, σ²_m via least-squares fit of measurement function

Particle Filter (SIS + resampling):
  N_particles = 2000
  Effective sample size (ESS) threshold = N/2 → systematic resampling

RUL prediction:
  At last observation k, estimated state x̄_k:
    μ_rul   = (1 - x̄_k) / (η · r_future)
    λ_rul   = (1 - x̄_k)² / (σ_B² · r_future)   [shape parameter]
    T ~ InvGaussian(mean=μ_rul, shape=λ_rul)
    RUL_point = ppf(PERCENTILE, T) × INTERVAL_SEC seconds

  Asymmetric scoring penalty (Er≤0 penalized at /20, Er>0 at /50):
    → use PERCENTILE = 0.40 to slightly under-predict and avoid over-prediction.

Usage:
  python s2_f2s2_rul.py
  python s2_f2s2_rul.py --percentile 0.42 --n_particles 3000
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.stats import invgauss
from scipy.signal import butter, sosfiltfilt

warnings.filterwarnings("ignore")

# ── add s1 to path so we can import helpers ───────────────────────────────────
CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))
from s1_dei_cnn import BearingCNN, predict_dei_sequence, WIN_SIZE, DEVICE

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path("/data/home/ksphm/2026-challenge-KSPHM")
DATA_DIR  = BASE_DIR / "dataset"
OUT_DEI   = BASE_DIR / "User/SR/0605_ref/output/dei"
OUT_MDL   = BASE_DIR / "User/SR/0605_ref/output/models"
OUT_PRED  = BASE_DIR / "User/SR/0605_ref/output/predictions"
OUT_PRED.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
INTERVAL_SEC  = 600          # seconds per measurement cycle
RPM_BASELINE  = 825.0        # (700+950)/2
TRAIN_EOL     = {1: 126, 2: 114, 3: 89, 4: 137}   # failure in cycles
N_PARTICLES   = 2000
PERCENTILE    = 0.40         # RUL point estimate percentile (under-predict bias)
EPS_STATE     = 1e-6


# ── load DEI sequences ────────────────────────────────────────────────────────

def load_dei_sequence(bearing_type: str, b_id: int) -> np.ndarray:
    """Load previously saved CNN-estimated DEI sequence."""
    prefix = "Train" if bearing_type == "train" else "Test"
    csv    = OUT_DEI / f"{prefix}{b_id}_DEI_CNN.csv"
    return pd.read_csv(csv)["dei_cnn"].values.astype(np.float64)


def load_or_compute_dei(model: BearingCNN) -> dict:
    """
    Load DEI CSVs if they exist, otherwise recompute from CNN.
    Returns {'train': {1:arr,...,4:arr}, 'test': {1:arr,...,6:arr}}
    """
    dei = {"train": {}, "test": {}}
    for b in range(1, 5):
        csv = OUT_DEI / f"Train{b}_DEI_CNN.csv"
        if csv.exists():
            dei["train"][b] = pd.read_csv(csv)["dei_cnn"].values.astype(np.float64)
        else:
            print(f"  Computing DEI for Train{b} …")
            dei["train"][b] = predict_dei_sequence(
                model, DATA_DIR / f"Train{b}_Vibration"
            ).astype(np.float64)

    for t in range(1, 7):
        csv = OUT_DEI / f"Test{t}_DEI_CNN.csv"
        if csv.exists():
            dei["test"][t] = pd.read_csv(csv)["dei_cnn"].values.astype(np.float64)
        else:
            print(f"  Computing DEI for Test{t} …")
            dei["test"][t] = predict_dei_sequence(
                model, DATA_DIR / "Test" / f"Test{t}"
            ).astype(np.float64)

    return dei


# ── RPM condition coefficient ─────────────────────────────────────────────────

def load_rpm_sequence(b_id: int) -> np.ndarray:
    """Load actual RPM for training bearing b_id (one value per TDMS file)."""
    op_csv = DATA_DIR / f"Train{b_id}_Operation.csv"
    df     = pd.read_csv(op_csv, encoding="cp949")
    df.columns = df.columns.str.strip()
    n_files = TRAIN_EOL[b_id]
    rpms    = []
    for k in range(1, n_files + 1):
        t_center = (k - 1) * INTERVAL_SEC + 30
        idx      = (df["Time[sec]"] - t_center).abs().idxmin()
        rpms.append(float(df.loc[idx, "Motor speed[rpm]"]))
    return np.array(rpms, dtype=np.float64)


def r_coeff(rpm: float) -> float:
    """Continuous condition coefficient: ratio to baseline RPM."""
    return rpm / RPM_BASELINE


def tau(rpm_seq: np.ndarray) -> float:
    """Integrated condition coefficient (time-scale transformation)."""
    return float(np.sum(r_coeff(rpm_seq)))   # Δt=1 cycle each


# ── F2S2 parameter estimation ─────────────────────────────────────────────────

def _ig_logpdf(t: float, eta: float, sigma_b2: float) -> float:
    """Log-PDF of InvGaussian failure time for unit reaching D=1 after τ cycles."""
    # T ~ IG(mean = D/η = 1/η, lambda = D²/σ_B² = 1/σ_B²) in transformed scale
    # Actual failure time (transformed) τ = T (1 cycle each, r applied)
    mu_ig  = 1.0 / eta
    lam_ig = 1.0 / sigma_b2
    if t <= 0 or eta <= 0 or sigma_b2 <= 0:
        return -1e9
    # log f(t) = 0.5*(log λ - log(2π) - 3*log t) - λ(t-μ)²/(2μ²t)
    return (0.5 * (np.log(lam_ig) - np.log(2 * np.pi) - 3 * np.log(t))
            - lam_ig * (t - mu_ig) ** 2 / (2 * mu_ig ** 2 * t))


def estimate_state_transition_params(dei_dict: dict) -> tuple:
    """
    MLE for η (degradation rate) and σ_B² (diffusion) from training failure times.
    Failure time in transformed time-scale: τ_n = Σ r(rpm_k) * Δt
    """
    tau_list = []
    for b in range(1, 5):
        rpm_seq   = load_rpm_sequence(b)
        tau_n     = tau(rpm_seq)      # time-scale transformed failure time (cycles)
        tau_list.append(tau_n)
    tau_arr = np.array(tau_list)

    def neg_loglik(params):
        eta, log_sb2 = params
        sb2 = np.exp(log_sb2)
        if eta <= 0 or sb2 <= 0:
            return 1e9
        ll = sum(_ig_logpdf(t, eta, sb2) for t in tau_arr)
        return -ll

    # initial guess: η ~ 1/mean_tau, σ_B² ~ 1/mean_tau³
    mean_tau = float(tau_arr.mean())
    x0 = [1.0 / mean_tau, np.log(1.0 / mean_tau)]
    res = minimize(neg_loglik, x0, method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    eta_hat  = float(res.x[0])
    sb2_hat  = float(np.exp(res.x[1]))
    print(f"  MLE:  η={eta_hat:.6f} cycle⁻¹,  σ_B²={sb2_hat:.6f}")
    return eta_hat, sb2_hat


def estimate_measurement_params(dei_dict: dict) -> tuple:
    """
    Fit measurement function: y_k = a * x_k^c + b
    x_k ∈ [0,1] is approximated as normalized time (k / K_total).
    Parameters a, b, c, σ²_m estimated jointly via nonlinear LS.
    """
    xs, ys = [], []
    for b in range(1, 5):
        y_seq = dei_dict["train"][b]
        K     = len(y_seq)
        x_seq = np.linspace(0, 1, K, endpoint=True)  # approximate state as linear time
        xs.append(x_seq)
        ys.append(y_seq)
    X_all = np.concatenate(xs)
    Y_all = np.concatenate(ys)

    # clip x to avoid 0^c issues
    X_all = np.clip(X_all, EPS_STATE, 1.0 - EPS_STATE)

    def residuals(params):
        a, b, log_c = params
        c = np.exp(log_c)
        y_hat = a * X_all ** c + b
        return float(np.mean((Y_all - y_hat) ** 2))

    # initial: a=1, b=0, c=1 → y≈x (linear)
    res = minimize(residuals, [1.0, 0.0, 0.0], method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 10000})
    a_hat, b_hat = float(res.x[0]), float(res.x[1])
    c_hat        = float(np.exp(res.x[2]))

    # measurement noise variance
    y_hat  = a_hat * X_all ** c_hat + b_hat
    sm2    = float(np.mean((Y_all - y_hat) ** 2))
    sm2    = max(sm2, 1e-6)   # floor

    print(f"  Measurement: a={a_hat:.4f}, b={b_hat:.4f}, c={c_hat:.4f}, σ²_m={sm2:.6f}")
    return a_hat, b_hat, c_hat, sm2


# ── Particle Filter ───────────────────────────────────────────────────────────

def particle_filter(y_obs: np.ndarray,
                    rpm_seq: np.ndarray,
                    eta: float, sigma_b2: float,
                    a: float, b: float, c: float, sigma_m2: float,
                    n_particles: int = N_PARTICLES,
                    resample_thresh_frac: float = 0.5) -> np.ndarray:
    """
    Run SIS particle filter with systematic resampling.

    Parameters
    ----------
    y_obs     : observed DEI sequence (length K)
    rpm_seq   : RPM at each observation step (length K); use RPM_BASELINE if unknown
    eta, sigma_b2 : state transition parameters
    a, b, c, sigma_m2 : measurement function parameters
    n_particles : number of particles

    Returns
    -------
    x_est : ndarray (K,)  – estimated state at each step (weighted mean)
    """
    K      = len(y_obs)
    # Initialize all particles at 0 (new bearing, "as-good-as-new")
    particles = np.zeros(n_particles, dtype=np.float64)
    weights   = np.ones(n_particles) / n_particles
    x_est     = np.zeros(K, dtype=np.float64)

    for k in range(K):
        r_k = r_coeff(rpm_seq[k])
        dt  = 1.0                             # 1 cycle

        # ── Prediction ────────────────────────────────────────────────────────
        noise      = np.random.randn(n_particles)
        particles  = particles + eta * r_k * dt + np.sqrt(sigma_b2 * r_k * dt) * noise
        particles  = np.clip(particles, EPS_STATE, 1.0 - EPS_STATE)

        # ── Update ────────────────────────────────────────────────────────────
        y_pred     = a * particles ** c + b
        # Gaussian likelihood
        log_w      = -0.5 * ((y_obs[k] - y_pred) ** 2) / sigma_m2
        log_w     -= log_w.max()              # numerical stability
        w_new      = np.exp(log_w)
        w_new     /= w_new.sum() + 1e-300

        weights    = w_new

        # ── State estimate ────────────────────────────────────────────────────
        x_est[k]   = float(np.dot(weights, particles))

        # ── Resampling (systematic) ───────────────────────────────────────────
        ess = 1.0 / (np.sum(weights ** 2) + 1e-300)
        if ess < n_particles * resample_thresh_frac:
            # systematic resampling
            cumsum = np.cumsum(weights)
            u      = (np.arange(n_particles) + np.random.rand()) / n_particles
            idx    = np.searchsorted(cumsum, u)
            idx    = np.clip(idx, 0, n_particles - 1)
            particles = particles[idx]
            weights   = np.ones(n_particles) / n_particles

    return x_est


# ── RUL prediction from Inverse Gaussian ─────────────────────────────────────

def predict_rul_cycles(x_bar: float, eta: float, sigma_b2: float,
                       r_future: float = 1.0,
                       percentile: float = PERCENTILE) -> float:
    """
    Compute RUL in cycles as the `percentile` quantile of the IG distribution.

    T ~ IG(mean = (1-x̄)/(η·r_future),  shape = (1-x̄)²/(σ_B²·r_future))
    scipy: invgauss(mu = mean/shape, scale = shape)
    """
    rem    = max(1.0 - x_bar, EPS_STATE)
    eta_e  = eta * r_future
    sb2_e  = sigma_b2 * r_future

    mu_ig  = rem / eta_e         # mean RUL in cycles
    lam_ig = rem ** 2 / sb2_e   # shape parameter

    if mu_ig <= 0 or lam_ig <= 0:
        return float("nan")

    # scipy invgauss(mu_sc, scale) with mean = mu_sc * scale
    # → mu_sc = mu_ig / lam_ig,  scale = lam_ig
    try:
        rul = float(invgauss.ppf(percentile, mu=mu_ig / lam_ig, scale=lam_ig))
    except Exception:
        rul = float("nan")
    return rul


# ── LOOCV evaluation on training bearings ────────────────────────────────────

def loocv_evaluation(dei_dict: dict,
                     eta: float, sigma_b2: float,
                     a: float, b: float, c: float, sigma_m2: float,
                     percentile: float) -> pd.DataFrame:
    """
    Leave-one-bearing-out cross-validation on training bearings.
    For each test bearing, uses state at last available observation
    to predict RUL, then compares to actual RUL (which is 0 for train
    bearings at EOL — so actual RUL = 0 cycles, prediction should be small).
    """
    # Since training bearings run to failure, the 'actual' remaining time at
    # the LAST observation is 0 (or 1 cycle, the last measurement before EOL).
    # We evaluate at 80% lifetime and compare estimated vs remaining cycles.
    records = []
    for b in range(1, 5):
        rpm_seq = load_rpm_sequence(b)
        y_obs   = dei_dict["train"][b]
        K       = len(y_obs)
        eval_at = int(K * 0.80)   # predict at 80% lifetime

        x_est = particle_filter(y_obs[:eval_at], rpm_seq[:eval_at],
                                 eta, sigma_b2, a, b, c, sigma_m2)
        x_bar     = x_est[-1]
        r_future  = float(np.mean(r_coeff(rpm_seq[eval_at:]))) if eval_at < K else 1.0

        rul_pred  = predict_rul_cycles(x_bar, eta, sigma_b2, r_future, percentile)
        rul_true  = float(K - eval_at)
        er        = 100.0 * (rul_true - rul_pred) / (rul_true + 1e-9)

        records.append({"bearing": b, "eval_cycle": eval_at, "K": K,
                        "x_est": round(x_bar, 4),
                        "rul_pred_cyc": round(rul_pred, 2),
                        "rul_true_cyc": round(rul_true, 2),
                        "Er_pct": round(er, 2)})
        print(f"  B{b} @{eval_at}/{K}: x̄={x_bar:.3f}  "
              f"pred={rul_pred:.1f} cyc  true={rul_true:.1f} cyc  Er={er:.1f}%")

    return pd.DataFrame(records)


# ── scoring helper ────────────────────────────────────────────────────────────

def comp_score(rul_true_s: float, rul_pred_s: float) -> float:
    """Competition asymmetric score (Er≤0 → /20, Er>0 → /50)."""
    if rul_true_s <= 0:
        return float("nan")
    Er = 100.0 * (rul_true_s - rul_pred_s) / rul_true_s
    if Er <= 0:
        return float(np.exp(-np.log(0.5) * Er / 20.0))
    return float(np.exp(np.log(0.5) * Er / 50.0))


# ── main prediction ───────────────────────────────────────────────────────────

def predict_test(model: BearingCNN,
                 dei_dict: dict,
                 eta: float, sigma_b2: float,
                 a: float, b: float, c: float, sigma_m2: float,
                 percentile: float) -> pd.DataFrame:
    """
    Run particle filter on each test bearing, predict RUL [seconds].
    """
    # Try to load tacholess RPM if available
    rpm_npy = BASE_DIR / "User/SR/0605_ref/output/rpm/test_rpm_estimates.npy"
    if rpm_npy.exists():
        try:
            test_rpm_dict = np.load(rpm_npy, allow_pickle=True).item()
            print("  Loaded tacholess RPM estimates for test bearings.")
        except Exception:
            test_rpm_dict = {}
    else:
        test_rpm_dict = {}

    records   = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for ax, t_id in zip(axes.flat, range(1, 7)):
        y_obs = dei_dict["test"][t_id]
        K     = len(y_obs)

        # RPM for each observation
        if t_id in test_rpm_dict:
            rpm_seq = test_rpm_dict[t_id].astype(np.float64)
        else:
            rpm_seq = np.full(K, RPM_BASELINE, dtype=np.float64)

        # Run particle filter over full available sequence
        x_est = particle_filter(y_obs, rpm_seq, eta, sigma_b2, a, b, c, sigma_m2)
        x_bar = x_est[-1]

        # Future condition: use same average as observed
        r_future  = float(np.mean(r_coeff(rpm_seq)))
        rul_cyc   = predict_rul_cycles(x_bar, eta, sigma_b2, r_future, percentile)
        rul_sec   = rul_cyc * INTERVAL_SEC

        records.append({"test_id": t_id, "n_obs": K,
                        "x_est_last": round(x_bar, 4),
                        "r_future": round(r_future, 4),
                        "rul_cycles": round(rul_cyc, 2),
                        "rul_seconds": round(rul_sec, 1)})

        # Plot
        ax.plot(np.arange(1, K + 1), y_obs,  "b-",  lw=1.2, label="DEI (CNN)")
        ax.plot(np.arange(1, K + 1), x_est,  "r--", lw=1.5, label="State x̂")
        ax.axhline(1.0, color="k", ls=":", lw=0.8, label="Failure threshold")
        ax.set_xlabel("Measurement cycle"); ax.set_ylabel("DEI / State")
        ax.set_title(f"Test {t_id}  |  RUL={rul_sec/3600:.2f} h  ({rul_sec:.0f} s)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        print(f"  Test{t_id}: x̄={x_bar:.3f}  RUL={rul_sec:.0f} s  ({rul_sec/3600:.2f} h)")

    plt.suptitle(f"F2S2 Particle Filter — Test RUL (percentile={percentile:.2f})",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_PRED / "test_rul_prediction.png", dpi=120)
    plt.close()
    print(f"  Plot saved → {OUT_PRED / 'test_rul_prediction.png'}")

    return pd.DataFrame(records)


# ── main ──────────────────────────────────────────────────────────────────────

def main(percentile: float = PERCENTILE, n_particles: int = N_PARTICLES):
    print("=" * 60)
    print("Stage 2+3 – F2S2 State Estimation + RUL Prediction")
    print("=" * 60)

    # ── Load CNN model ────────────────────────────────────────────────────────
    print("\n[1/5] Loading trained CNN …")
    model = BearingCNN(WIN_SIZE).to(DEVICE)
    state_dict = torch.load(OUT_MDL / "bearing_cnn.pt", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("      CNN loaded.")

    # ── Load DEI sequences ────────────────────────────────────────────────────
    print("\n[2/5] Loading DEI sequences …")
    dei_dict = load_or_compute_dei(model)
    for k, v in dei_dict["train"].items():
        print(f"  Train{k}: {len(v)} steps, range [{v.min():.3f}, {v.max():.3f}]")
    for k, v in dei_dict["test"].items():
        print(f"  Test{k}:  {len(v)} steps, range [{v.min():.3f}, {v.max():.3f}]")

    # ── Estimate F2S2 parameters ──────────────────────────────────────────────
    print("\n[3/5] Estimating F2S2 parameters …")
    print("  State transition (MLE on failure times):")
    eta, sigma_b2 = estimate_state_transition_params(dei_dict)
    print("  Measurement function (NLS):")
    a, b, c, sigma_m2 = estimate_measurement_params(dei_dict)

    params = {"eta": eta, "sigma_b2": sigma_b2,
              "a": a, "b": b, "c": c, "sigma_m2": sigma_m2}
    pd.DataFrame([params]).to_csv(OUT_PRED / "f2s2_params.csv", index=False)

    # ── LOOCV on training bearings ────────────────────────────────────────────
    print("\n[4/5] LOOCV evaluation (80% lifetime cut) …")
    loocv_df = loocv_evaluation(dei_dict, eta, sigma_b2, a, b, c, sigma_m2, percentile)
    loocv_df.to_csv(OUT_PRED / "loocv_summary.csv", index=False)
    print(loocv_df.to_string(index=False))

    # ── State trajectories for training bearings (diagnostic plot) ────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, b in zip(axes.flat, range(1, 5)):
        rpm_seq = load_rpm_sequence(b)
        y_obs   = dei_dict["train"][b]
        K       = len(y_obs)
        x_est   = particle_filter(y_obs, rpm_seq, eta, sigma_b2, a, b, c, sigma_m2,
                                   n_particles)
        ax.plot(np.arange(1, K + 1), y_obs,  "b-",  lw=1.2, label="DEI")
        ax.plot(np.arange(1, K + 1), x_est,  "r--", lw=1.5, label="State x̂")
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("Cycle"); ax.set_ylabel("DEI / State")
        ax.set_title(f"Training Bearing {b} (EOL={TRAIN_EOL[b]} cyc)"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.suptitle("F2S2: State Estimation on Training Bearings", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_PRED / "train_state_estimation.png", dpi=120)
    plt.close()

    # ── Predict test RUL ──────────────────────────────────────────────────────
    print(f"\n[5/5] Predicting test RUL (percentile={percentile}) …")
    results_df = predict_test(model, dei_dict, eta, sigma_b2, a, b, c, sigma_m2,
                               percentile)

    results_df.to_csv(OUT_PRED / "test_rul_predictions.csv", index=False)
    print("\n── Test RUL Predictions ──────────────────────────────")
    print(results_df[["test_id", "n_obs", "x_est_last",
                       "rul_cycles", "rul_seconds"]].to_string(index=False))
    print(f"\nResults saved → {OUT_PRED / 'test_rul_predictions.csv'}")

    # ── RUL PDF visualization (test bearings) ─────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, row in zip(axes.flat, results_df.itertuples()):
        x_bar    = row.x_est_last
        r_future = row.r_future
        rem      = max(1.0 - x_bar, EPS_STATE)
        eta_e    = eta * r_future
        sb2_e    = sigma_b2 * r_future
        mu_ig    = rem / eta_e
        lam_ig   = rem ** 2 / sb2_e

        t_max = float(invgauss.ppf(0.995, mu=mu_ig / lam_ig, scale=lam_ig))
        t_arr = np.linspace(EPS_STATE, t_max, 500)
        pdf   = invgauss.pdf(t_arr, mu=mu_ig / lam_ig, scale=lam_ig)

        rul_p40 = row.rul_cycles
        rul_p50 = float(invgauss.ppf(0.50, mu=mu_ig / lam_ig, scale=lam_ig))

        ax.plot(t_arr * INTERVAL_SEC / 3600, pdf / INTERVAL_SEC * 3600,
                "b-", lw=1.5)
        ax.axvline(rul_p40 * INTERVAL_SEC / 3600, color="r", ls="--",
                   label=f"p{int(percentile*100)}={rul_p40*INTERVAL_SEC/3600:.2f}h")
        ax.axvline(rul_p50 * INTERVAL_SEC / 3600, color="g", ls=":",
                   label=f"p50={rul_p50*INTERVAL_SEC/3600:.2f}h")
        ax.set_xlabel("RUL [hours]"); ax.set_ylabel("PDF")
        ax.set_title(f"Test {row.test_id}"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle("RUL Probability Density (Inverse Gaussian)", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_PRED / "rul_pdf.png", dpi=120)
    plt.close()
    print(f"PDF plot saved → {OUT_PRED / 'rul_pdf.png'}")

    print("\nStage 2+3 complete.")
    return results_df


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--percentile",  type=float, default=PERCENTILE)
    parser.add_argument("--n_particles", type=int,   default=N_PARTICLES)
    args = parser.parse_args()
    main(percentile=args.percentile, n_particles=args.n_particles)
