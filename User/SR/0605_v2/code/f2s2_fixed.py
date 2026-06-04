"""
F2S2 Core (bug-fixed) — Two-Factor State-Space Model for RUL
=============================================================
Paper: Li et al., "Remaining useful life prediction of machinery under
time-varying operating conditions based on a two-factor state-space model",
RESS 186 (2019).

Fixes vs original implementation (User/SC/HI/06031410_f2s2_rul_raw):
  1. tau_path   — correct index: tau[k]=0 at k=0, tau[k]=Σ_{j<k}R[p_j] for k>0
  2. b_B        — mean of per-unit b_{B,n}=sm[0]/a_{B,n}, not mean(sm[0])/mean(a_{B,n})
  3. future_regime — uses actual condition profile with last-regime padding,
                     not a hard-coded artificial schedule
  4. sigma2     — stored as actual noise variance σ² (not σ²/a_B²);
                  particle-filter weight updated accordingly (same result, cleaner)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

ROOT     = Path("/data/home/ksphm/2026-challenge-KSPHM")
DATASET  = ROOT / "dataset"
BEARINGS = [1, 2, 3, 4]
TEST_IDS = [1, 2, 3, 4, 5, 6]
EOL      = {1: 126, 2: 114, 3: 89, 4: 137}
INTERVAL_SEC = 600
FS           = 25600
RPM_SPLIT    = 850.0
D_FAIL       = 1.0
EPS          = 1e-9


# ── condition profiles ────────────────────────────────────────────────────────

def regime_profile_train(bearing: int, n_cycle: int) -> np.ndarray:
    """Per-cycle regime from Train Operation.csv  (0=low, 1=high)."""
    op  = pd.read_csv(DATASET / f"Train{bearing}_Operation.csv", encoding="latin-1")
    t   = op.iloc[:, 0].values.astype(float)
    rpm = op.iloc[:, 2].values.astype(float)
    lab = np.zeros(n_cycle, dtype=int)
    for k in range(n_cycle):
        m = (t >= k * INTERVAL_SEC) & (t < (k + 1) * INTERVAL_SEC)
        if m.sum() > 0:
            lab[k] = int(rpm[m].mean() > RPM_SPLIT)
        elif k > 0:
            lab[k] = lab[k - 1]
    return lab


def regime_profile_test(test_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract ch3_rms and regime from test TDMS files.

    Regime detection: CH2 FFT dominant peak in 8–20 Hz → shaft RPM.
    Uses first 131072 (2^17) samples for speed (freq-resolution ≈ 0.2 Hz).

    Returns (ch3_rms, regime)  both length = number of .tdms files.
    """
    import nptdms
    tdms_dir = DATASET / "Test" / f"Test{test_id}"
    files    = sorted(tdms_dir.glob("*.tdms"))
    N_FFT    = 131072                      # 2^17 ≈ 5.1 s at 25600 Hz
    freqs    = np.fft.rfftfreq(N_FFT, 1.0 / FS)
    mask_rpm = (freqs >= 8.0) & (freqs <= 20.0)

    ch3_rms_arr, regime_arr = [], []
    for fp in files:
        tf   = nptdms.TdmsFile(fp)
        vib  = tf["Vibration"]
        ch2  = vib["CH2"][:N_FFT]
        ch3  = vib["CH3"][:]
        # ch3 RMS
        ch3_rms_arr.append(float(np.sqrt(np.mean(ch3 ** 2))))
        # shaft-frequency peak → RPM → regime
        amp       = np.abs(np.fft.rfft(ch2, n=N_FFT))
        peak_freq = freqs[mask_rpm][np.argmax(amp[mask_rpm])]
        regime_arr.append(int(peak_freq * 60.0 > RPM_SPLIT))

    return np.array(ch3_rms_arr, dtype=float), np.array(regime_arr, dtype=int)


def future_regime(full_profile: np.ndarray, k_start: int, n_future: int) -> np.ndarray:
    """FIX: use actual profile from k_start; cyclically tile historical profile beyond known data.

    Cyclic tiling preserves the empirical low/high ratio and avoids the bias of
    last-regime padding (if the last cycle is HIGH with r_high<1, padding with HIGH
    makes all future tau increments tiny → systematic over-prediction of RUL).

    For train LOO: actual known profile fills most of the window.
    For test:      known 50 cycles + tiled historical pattern for extrapolation.
    """
    avail = full_profile[k_start:k_start + n_future]
    if len(avail) >= n_future:
        return avail[:n_future]
    n_pad = n_future - len(avail)
    L     = max(len(full_profile), 1)
    tile  = np.tile(full_profile, (n_pad // L) + 1)[:n_pad]
    return np.concatenate([avail, tile])


# ── state-transition MLE  (eq 22–27) ─────────────────────────────────────────

def _phi_vectors(profiles: list[np.ndarray]) -> np.ndarray:
    """phi[n, c] = total cycles spent in condition c for unit n."""
    return np.array([[np.sum(p == c) for c in (0, 1)] for p in profiles], dtype=float)


def fit_state_transition(profiles: list[np.ndarray]) -> dict:
    """Return {r, eta, sigma2_B}.  r_low ≡ 1 fixed; r_high optimised via 1-D MLE."""
    phi = _phi_vectors(profiles)
    N   = len(phi)

    def negloglik(r_high: float) -> float:
        R   = np.array([1.0, r_high])
        tau = phi @ R                       # τ_n  (N,)
        if np.any(tau <= 0):
            return 1e18
        eta = N / np.sum(tau)               # eq 27
        s2  = max(np.mean((tau * eta - D_FAIL) ** 2 / tau), 1e-12)
        last_r = np.array([R[p[-1]] for p in profiles])
        ll = (-N / 2 * np.log(2 * np.pi * s2)
              + np.sum(np.log(last_r))
              - np.sum(1.5 * np.log(tau))
              - np.sum((tau * eta - D_FAIL) ** 2 / (2 * s2 * tau)))
        return -ll

    res    = minimize_scalar(negloglik, bounds=(0.2, 8.0), method="bounded")
    r_high = float(res.x)
    R      = np.array([1.0, r_high])
    tau    = phi @ R
    eta    = float(N / np.sum(tau))
    s2     = float(max(np.mean((tau * eta - D_FAIL) ** 2 / tau), 1e-12))
    return {"r": R, "eta": eta, "sigma2_B": s2}


def tau_path(profile: np.ndarray, R: np.ndarray) -> np.ndarray:
    """FIX: τ[k] = Σ_{j=0}^{k-1} R[p_j] · Δt,  τ[0] = 0.

    Old code: inc[j] = R[profile[max(j-1,0)]] for j in range(K)
    → duplicates profile[0] at j=0 and j=1, giving wrong values for alternating profiles.

    Correct: τ is the CUMULATIVE transformed time BEFORE cycle k.
    """
    inc = np.array([R[p] for p in profile])            # Δτ at each cycle
    return np.concatenate([[0.0], np.cumsum(inc)[:-1]]) # τ[0]=0, τ[k]=Σ_{j<k}


# ── signal transformation  (eq 31–33) ────────────────────────────────────────

def fit_signal_transform(series_list: list[np.ndarray],
                         profiles: list[np.ndarray]) -> dict:
    """Eq 32–33: fit ȳ_baseline ≈ α·y_high + β  pooled over all units.
    Returns {0:(1,0), 1:(α,β)}.
    """
    Y_high, Ybar = [], []
    for y, p in zip(series_list, profiles):
        idx  = np.arange(len(y))
        low  = idx[p == 0]
        high = idx[p == 1]
        if len(low) < 2 or len(high) == 0:
            continue
        ybar_high = np.interp(high, low, y[low])
        Y_high.append(y[high])
        Ybar.append(ybar_high)
    if not Y_high:
        return {0: (1.0, 0.0), 1: (1.0, 0.0)}
    yh = np.concatenate(Y_high)
    yb = np.concatenate(Ybar)

    def J(a: float) -> float:
        b = np.mean(yb - a * yh)           # eq 33
        return float(np.sum((yb - a * yh - b) ** 2))

    res = minimize_scalar(J, bounds=(1e-3, 1e3), method="bounded")
    a   = float(res.x)
    b   = float(np.mean(yb - a * yh))
    return {0: (1.0, 0.0), 1: (a, b)}


def transform_to_baseline(y: np.ndarray, profile: np.ndarray, trans: dict) -> np.ndarray:
    yb = y.astype(float).copy()
    for c, (a, b) in trans.items():
        m       = profile == c
        yb[m]   = a * y[m] + b
    return yb


# ── measurement-function parameters  (eq 34–37) ──────────────────────────────

def loess_smooth(y: np.ndarray, frac: float = 0.3) -> np.ndarray:
    """Locally-weighted linear LOESS smoother."""
    n = len(y)
    x = np.arange(n, dtype=float)
    r = max(int(np.ceil(frac * n)), 3)
    out = np.empty(n)
    for i in range(n):
        d = np.abs(x - x[i])
        h = max(np.sort(d)[min(r, n - 1)], EPS)
        w = np.clip(1 - (d / h) ** 3, 0, 1) ** 3
        W = np.diag(w)
        X = np.vstack([np.ones(n), x]).T
        try:
            beta   = np.linalg.solve(X.T @ W @ X + 1e-9 * np.eye(2), X.T @ W @ y)
            out[i] = beta[0] + beta[1] * x[i]
        except np.linalg.LinAlgError:
            out[i] = y[i]
    return out


def fit_measurement(yb_list: list[np.ndarray], eta: float,
                    R: np.ndarray, profiles: list[np.ndarray],
                    loess_frac: float = 0.3) -> dict:
    """Eq 34–37: estimate a_B, b_B (FIX: per-unit average), sigma2 (actual), c."""
    sm_list = [loess_smooth(yb, loess_frac) for yb in yb_list]

    # FIX: a_Bn and b_Bn per unit, then average  (eq 35)
    a_Bns = [float(sm[-1] - sm[0]) for sm in sm_list]
    a_B   = float(np.mean(a_Bns))
    if abs(a_B) < EPS:
        a_B = 1.0

    b_Bns = []
    for sm, a_Bn in zip(sm_list, a_Bns):
        b_Bns.append(float(sm[0] / a_Bn) if abs(a_Bn) > EPS else float(sm[0]))
    b_B = float(np.mean(b_Bns))

    # FIX: sigma2 = actual noise variance (not divided by a_B^2)
    sigma2 = float(max(np.mean([np.mean((yb - sm) ** 2)
                                for yb, sm in zip(yb_list, sm_list)]), 1e-9))

    # c estimation  (eq 36–37); FIX: uses corrected tau_path
    taus = [tau_path(p, R) for p in profiles]

    def err_c(c: float) -> float:
        tot = 0.0
        for sm, tau in zip(sm_list, taus):
            denom = sm[-1] - sm[0]
            denom = denom if abs(denom) > EPS else EPS
            ratio = np.clip((sm - sm[0]) / denom, 1e-6, 1.0)
            xhat  = ratio ** (1.0 / c)
            tot  += float(np.sum((xhat - eta * tau) ** 2))
        return tot

    res = minimize_scalar(err_c, bounds=(0.5, 6.0), method="bounded")
    c   = float(res.x)
    return {"a_B": a_B, "b_B": b_B, "sigma2": sigma2, "c": c, "smoothed": sm_list}


# ── particle filter  (eq 38–40) ──────────────────────────────────────────────

def particle_filter(y: np.ndarray, profile: np.ndarray,
                    params: dict, n_particles: int = 800,
                    seed: int = 0) -> np.ndarray:
    """Estimate system state x̄_k (median of particles) per cycle."""
    R, eta, s2B = params["r"], params["eta"], params["sigma2_B"]
    a_B, b_B, c = params["a_B"], params["b_B"], params["c"]
    s2   = params["sigma2"]     # FIX: actual σ² (not σ²/a_B²)
    trans = params["trans"]
    rng  = np.random.default_rng(seed)
    K    = len(y)
    x    = np.zeros(n_particles)
    w    = np.full(n_particles, 1.0 / n_particles)
    x_bar = np.zeros(K)

    for k in range(K):
        # ── prediction (eq 38) ──
        if k > 0:
            r_prev = float(R[profile[k - 1]])
            x = (x + r_prev * eta
                 + rng.normal(0, np.sqrt(max(r_prev * s2B, 1e-12)), n_particles))
            x = np.clip(x, 0.0, None)

        # ── update (eq 39–40) ──
        a_t, b_t = trans[int(profile[k])]
        yB        = a_t * y[k] + b_t                           # baseline-transformed obs
        mean_meas = a_B * (b_B + np.clip(x, 0.0, None) ** c)  # eq 15 (baseline)
        # FIX: weight uses actual σ² directly
        logw  = -(yB - mean_meas) ** 2 / (2.0 * s2 + EPS)
        logw -= logw.max()
        w     = w * np.exp(logw)
        sw    = w.sum()
        if sw <= 0 or not np.isfinite(sw):
            w = np.full(n_particles, 1.0 / n_particles)
        else:
            w /= sw

        # ── resample (systematic) ──
        idx   = _systematic_resample(w, rng)
        x     = x[idx]
        w     = np.full(n_particles, 1.0 / n_particles)
        x_bar[k] = float(np.median(x))

    return x_bar


def _systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n    = len(w)
    pos  = (rng.random() + np.arange(n)) / n
    csum = np.cumsum(w); csum[-1] = 1.0
    return np.searchsorted(csum, pos)


# ── RUL prediction  (eq 42) ──────────────────────────────────────────────────

def predict_rul(x_bar_k: float, full_profile: np.ndarray,
                k: int, params: dict, max_future: int = 400) -> float:
    """FIX: uses actual condition profile (with last-regime padding beyond data)."""
    R, eta, s2B = params["r"], params["eta"], params["sigma2_B"]
    rem = D_FAIL - x_bar_k
    if rem <= 1e-4:
        return 1.0

    fut  = future_regime(full_profile, k + 1, max_future)
    dtau = np.array([R[c] for c in fut])    # r_{p(l+t_k)} = Jacobian term
    tau_cum = np.cumsum(dtau)               # Δτ cumulated = τ_f in eq 42
    l    = np.arange(1, len(tau_cum) + 1)

    pdf = (dtau * rem
           / np.sqrt(2 * np.pi * s2B * tau_cum ** 3 + EPS)
           * np.exp(-((tau_cum * eta - rem) ** 2) / (2 * s2B * tau_cum + EPS)))
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)

    if pdf.sum() <= 0:                      # fallback: deterministic mean
        target = rem / (eta + EPS)
        return float(np.clip(np.searchsorted(tau_cum, target) + 1, 1.0, max_future))

    return float(np.clip(np.sum(l * pdf) / np.sum(pdf), 1.0, max_future))


# ── metrics ───────────────────────────────────────────────────────────────────

def competition_score(rul_true: float, rul_pred: float) -> float:
    if rul_true <= 0:
        return float("nan")
    er = 100.0 * (rul_true - rul_pred) / rul_true
    return float(np.exp(-np.log(0.5) * er / 20.0) if er <= 0
                 else np.exp(np.log(0.5) * er / 50.0))


def monotonicity(s: np.ndarray) -> float:
    s = np.asarray(s, float)
    if len(s) <= 1: return 0.0
    d = np.diff(s)
    return float(abs(np.sum(d > 0) - np.sum(d < 0)) / len(d))


def trendability(s: np.ndarray) -> float:
    s = np.asarray(s, float)
    if len(s) <= 1: return 0.0
    rho, _ = spearmanr(np.arange(len(s)), s)
    return float(abs(rho) if not np.isnan(rho) else 0.0)
