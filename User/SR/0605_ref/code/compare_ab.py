"""
compare_ab.py  —  Compare Approach A (Improved CNN) vs Approach B (Direct HI)
==============================================================================
Shared F2S2 pipeline (particle filter + IG RUL) applied to both HI sources.

Produces:
  1. Training HI diagnostic: FDR label vs CNN(A) vs Direct(B) side-by-side
  2. Test HI comparison: A vs B for all 6 test bearings
  3. LOOCV on training bearings: RUL error at 80% lifetime (same F2S2 params)
  4. Test RUL predictions: both approaches, table + plot
  5. output/predictions/comparison_summary.csv

Usage:
  python compare_ab.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import invgauss

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from c2_hi_pipeline import load_operation, align_rpm, discretize_rpm, BEARING_IDS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path("/data/home/ksphm/2026-challenge-KSPHM")
DATA_DIR  = BASE_DIR / "dataset"
HI_DIR    = BASE_DIR / "User/SR/0605_ref/output/hi"
PRED_DIR  = BASE_DIR / "User/SR/0605_ref/output/predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL_SEC  = 600
RPM_BASELINE  = 825.0
TRAIN_EOL     = {1: 126, 2: 114, 3: 89, 4: 137}
N_PARTICLES   = 2000
PERCENTILE    = 0.40
EPS           = 1e-6


# ── Load HI sequences ─────────────────────────────────────────────────────────

def load_train_hi(b: int) -> np.ndarray:
    return pd.read_csv(HI_DIR / f"Bearing{b}_HI.csv")["HI"].values.astype(np.float64)


def load_test_hi(t: int, approach: str) -> np.ndarray:
    col = "HI_CNN" if approach == "A" else "HI_Direct"
    csv = HI_DIR / f"Test{t}_HI_{'CNN' if approach == 'A' else 'Direct'}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} — run s1a or c2_test_hi first")
    return pd.read_csv(csv)[col].values.astype(np.float64)


# ── RPM condition coefficient ─────────────────────────────────────────────────

def build_rpm_seqs():
    """Return {b: r_array} where r_array[k] = rpm_k / RPM_BASELINE."""
    result = {}
    for b in BEARING_IDS:
        op  = load_operation(b)
        rpm = align_rpm(op, TRAIN_EOL[b])
        result[b] = rpm / RPM_BASELINE
    return result


# ── F2S2 parameter estimation ─────────────────────────────────────────────────

def ig_logpdf(t, eta, sb2):
    if t <= 0 or eta <= 0 or sb2 <= 0:
        return -1e9
    mu_ig  = 1.0 / eta
    lam_ig = 1.0 / sb2
    return (0.5 * (np.log(lam_ig) - np.log(2 * np.pi) - 3 * np.log(t))
            - lam_ig * (t - mu_ig)**2 / (2 * mu_ig**2 * t))


def estimate_state_params(r_seqs: dict):
    tau_list = [float(r.sum()) for r in r_seqs.values()]
    def neg_ll(params):
        eta, log_sb2 = params
        sb2 = np.exp(log_sb2)
        return -sum(ig_logpdf(t, eta, sb2) for t in tau_list)
    mean_tau = np.mean(tau_list)
    res = minimize(neg_ll, [1.0 / mean_tau, np.log(1.0 / mean_tau)],
                   method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    return float(res.x[0]), float(np.exp(res.x[1]))


def _loess_smooth(y: np.ndarray, window: int = 13) -> np.ndarray:
    """Savitzky-Golay smoothing as a practical LOESS substitute."""
    from scipy.signal import savgol_filter
    n   = len(y)
    win = min(window, n - (0 if n % 2 == 1 else 1))
    win = win if win % 2 == 1 else win - 1
    win = max(win, 3)
    pol = min(2, win - 1)
    return np.clip(savgol_filter(y, window_length=win, polyorder=pol), 0.0, 1.0)


def estimate_meas_params(hi_seqs: dict):
    """
    F2S2 Section 3.2.3: proper measurement-function parameter estimation.

    Steps
    -----
    1. LOESS-smooth each training HI sequence → ỹ_k
    2. Boundary conditions  (x_1=0, x_K=1) →
           b_B = mean(ỹ_{k=1})   (initial HI when bearing is new)
           a_B = mean(ỹ_{k=K} - ỹ_{k=1})  (total HI range at failure)
    3. Infer state from observation: x̂_k = ((ỹ_k - b_B)/a_B)^(1/c)
       Find c  that minimises  Σ_n Σ_k  (x̂_k - k/K)²
       → makes the inferred state as linear as possible, matching
         the Wiener-process (linear-drift) assumption.
    4. σ²_m = variance of raw HI around the model y = a*(k/K)^c + b
       (noise of the raw signal around the fitted trajectory).
    """
    from scipy.optimize import minimize_scalar

    smoothed = {b: _loess_smooth(hi) for b, hi in hi_seqs.items()}

    # ── Step 2: boundary-condition estimates ─────────────────────────────────
    a_vals, b_vals = [], []
    for hi_s in smoothed.values():
        b_vals.append(float(hi_s[0]))
        a_vals.append(float(hi_s[-1] - hi_s[0]))
    a_B = float(np.mean(a_vals))
    b_B = float(np.mean(b_vals))
    # Safety floor: a_B should be positive and not too small
    a_B = max(a_B, 0.10)

    # ── Step 3: estimate c ────────────────────────────────────────────────────
    def obj(log_c):
        c = np.exp(log_c)
        err = 0.0
        for hi_s in smoothed.values():
            K     = len(hi_s)
            x_ref = np.linspace(0.0, 1.0, K)
            y_norm = np.clip((hi_s - b_B) / (a_B + EPS), EPS, 1.0)
            x_inf  = y_norm ** (1.0 / c)
            err   += float(np.mean((x_inf - x_ref) ** 2))
        return err

    res = minimize_scalar(obj, bounds=(-3.0, 3.0), method="bounded")
    c   = float(np.exp(res.x))

    # ── Step 4: σ²_m = var(smoothed_HI − model) ─────────────────────────────
    # σ²_m represents the uncertainty in the measurement equation y = a*x^c + b.
    # Three components exist:
    #   (a) local noise: var(raw − smoothed) → captures HI sensor noise
    #   (b) model bias: var(smoothed − model) → captures degradation curve mismatch
    #
    # The F2S2 paper uses LOESS-smoothed signals and then estimates σ² from the
    # variance of smoothed residuals around the model.  We use the same:
    #   σ²_m = mean_n[ var_k( ỹ_k − (a_B * (k/K)^c + b_B) ) ]
    # This is larger than local noise (captures real HI-to-model mismatch)
    # but smaller than raw residuals (LOESS removes signal-level noise).
    smoothed_resid = []
    for b, hi in hi_seqs.items():
        hi_s  = smoothed[b]
        K     = len(hi_s)
        x_ref = np.linspace(0.0, 1.0, K)
        y_mod = a_B * (x_ref ** c) + b_B
        smoothed_resid.extend((hi_s - y_mod).tolist())
    sm2 = max(float(np.var(smoothed_resid)), 1e-6)

    print(f"  Meas params (F2S2 §3.2.3): "
          f"a_B={a_B:.4f}, b_B={b_B:.4f}, c={c:.4f}, σ²_m={sm2:.6f}"
          f"  (smoothed-HI vs model)")
    return a_B, b_B, c, sm2


# ── Particle filter ───────────────────────────────────────────────────────────

def _pf_step(particles, weights, y_k, r_k, eta, sb2, a, b, c, sm2, n):
    """Single particle-filter step: predict → update → resample."""
    # Predict
    particles = (particles + eta * r_k
                 + np.sqrt(sb2 * r_k) * np.random.randn(n))
    particles = np.clip(particles, EPS, 1.0 - EPS)

    # Update
    y_pred = a * particles**c + b
    log_w  = -0.5 * (y_k - y_pred)**2 / sm2
    log_w -= log_w.max()
    w      = np.exp(log_w)
    weights = w / (w.sum() + 1e-300)

    x_est = float(np.dot(weights, particles))

    # Resample
    ess = 1.0 / (np.sum(weights**2) + 1e-300)
    if ess < n * 0.5:
        idx       = np.searchsorted(np.cumsum(weights),
                                     (np.arange(n) + np.random.rand()) / n)
        particles = particles[np.clip(idx, 0, n - 1)]
        weights   = np.ones(n) / n

    return particles, weights, x_est


def particle_filter(y_obs, r_seq, eta, sb2, a, b, c, sm2,
                    n=N_PARTICLES):
    """Standard PF starting from x_0 = 0."""
    K         = len(y_obs)
    particles = np.zeros(n)
    weights   = np.ones(n) / n
    x_est     = np.zeros(K)

    for k in range(K):
        particles, weights, x_est[k] = _pf_step(
            particles, weights, y_obs[k], float(r_seq[k]),
            eta, sb2, a, b, c, sm2, n)

    return x_est


def detect_fpt(hi: np.ndarray, threshold: float = 0.25,
               min_consecutive: int = 3) -> int:
    """
    First Prediction Time: first index where HI exceeds `threshold`
    for at least `min_consecutive` consecutive cycles.
    Returns 0 if bearing starts above threshold or never reaches it.

    The plateau phase (HI ≈ constant below threshold) tends to mislead
    the particle filter because the flat signal implies a low hidden state
    while the prior keeps drifting up.  Skipping the plateau and
    initialising from the estimated state at FPT avoids this conflict.
    """
    for i in range(len(hi) - min_consecutive + 1):
        if np.all(hi[i: i + min_consecutive] >= threshold):
            return i
    return 0   # fall back: use full sequence


def particle_filter_fpt(y_obs: np.ndarray, r_seq: np.ndarray,
                        eta: float, sb2: float,
                        a: float, b: float, c: float, sm2: float,
                        fpt_threshold: float = 0.25,
                        n: int = N_PARTICLES) -> np.ndarray:
    """
    FPT-aware particle filter.

    1.  Detect FPT = first cycle where HI exceeds fpt_threshold
        (3 consecutive cycles to avoid false triggers).

    2.  Before FPT: x̂_k = k * η * r̄   (pure prior, no measurement update)
        — the flat plateau phase carries no state information.

    3.  At FPT: initialise particles from the prior distribution
            x_FPT  ~ N( η * τ_FPT,  σ²_B * τ_FPT )
        where τ_FPT = Σ_{k=0}^{FPT-1} r_k  (time-scale transformed time).

    4.  From FPT onwards: standard particle filter with measurement update.
    """
    K   = len(y_obs)
    fpt = detect_fpt(y_obs, fpt_threshold)

    x_est = np.zeros(K)

    # ── Phase 1: before FPT (pure drift estimate) ─────────────────────────
    tau = 0.0
    for k in range(fpt):
        tau     += float(r_seq[k])
        x_est[k] = eta * tau   # deterministic prior mean

    if fpt == K:                          # FPT never reached → all prior
        return x_est

    # ── Phase 2: initialise at FPT ────────────────────────────────────────
    tau_fpt   = sum(float(r_seq[k]) for k in range(fpt))
    mu_fpt    = eta * tau_fpt
    sigma_fpt = np.sqrt(sb2 * tau_fpt) if tau_fpt > 0 else 1e-4

    particles = np.random.normal(mu_fpt, sigma_fpt, n)
    particles = np.clip(particles, EPS, 1.0 - EPS)
    weights   = np.ones(n) / n

    # ── Phase 3: PF from FPT to end ───────────────────────────────────────
    for k in range(fpt, K):
        particles, weights, x_est[k] = _pf_step(
            particles, weights, y_obs[k], float(r_seq[k]),
            eta, sb2, a, b, c, sm2, n)

    return x_est


def predict_rul(x_bar, eta, sb2, r_future=1.0, pct=PERCENTILE):
    rem   = max(1.0 - x_bar, EPS)
    mu_ig = rem / (eta * r_future)
    lam   = rem**2 / (sb2 * r_future)
    if mu_ig <= 0 or lam <= 0:
        return float("nan")
    try:
        return float(invgauss.ppf(pct, mu=mu_ig / lam, scale=lam))
    except Exception:
        return float("nan")


# ── LOOCV on training ─────────────────────────────────────────────────────────

def loocv_training(hi_seqs, r_seqs, eta, sb2, a, b, c, sm2,
                   use_fpt: bool = False, fpt_threshold: float = 0.25):
    """
    LOOCV at 80% lifetime — with or without FPT.
    When use_fpt=True the particle filter skips the plateau phase.
    """
    records = []
    pf_fn   = particle_filter_fpt if use_fpt else particle_filter

    for bid in BEARING_IDS:
        hi   = hi_seqs[bid]
        r    = r_seqs[bid]
        K    = len(hi)
        cut  = int(K * 0.80)

        if use_fpt:
            x_est = pf_fn(hi[:cut], r[:cut], eta, sb2, a, b, c, sm2,
                          fpt_threshold=fpt_threshold)
        else:
            x_est = pf_fn(hi[:cut], r[:cut], eta, sb2, a, b, c, sm2)

        x_bar = x_est[-1]
        r_fut = float(np.mean(r[cut:])) if cut < K else 1.0
        rul_p = predict_rul(x_bar, eta, sb2, r_fut) * INTERVAL_SEC
        rul_t = (K - cut) * INTERVAL_SEC
        Er    = 100.0 * (rul_t - rul_p) / (rul_t + 1e-9)

        fpt_k = detect_fpt(hi[:cut], fpt_threshold) if use_fpt else None
        records.append({"bearing": bid, "cut_cyc": cut, "K": K,
                        "fpt": fpt_k,
                        "x_bar": round(x_bar, 4),
                        "rul_pred_s": round(rul_p, 0),
                        "rul_true_s": round(rul_t, 0),
                        "Er_pct": round(Er, 2)})
    return pd.DataFrame(records)


# ── Scoring ───────────────────────────────────────────────────────────────────

def comp_score(true_s, pred_s):
    if true_s <= 0:
        return float("nan")
    Er = 100.0 * (true_s - pred_s) / true_s
    if Er <= 0:
        return float(np.exp(-np.log(0.5) * Er / 20.0))
    return float(np.exp(np.log(0.5) * Er / 50.0))


def find_optimal_percentile(hi_seqs: dict, r_seqs: dict,
                             eta, sb2, a, b, c, sm2,
                             use_fpt: bool = True,
                             fpt_threshold: float = 0.25) -> tuple:
    """
    Grid-search over percentile ∈ [0.01, 0.44] to maximise the LOOCV
    average competition score on the 4 training bearings (80% cut).

    Returns (best_pct, best_score, score_table_df).
    """
    pcts  = np.round(np.arange(0.01, 0.45, 0.01), 3)
    pf_fn = particle_filter_fpt if use_fpt else particle_filter

    # Precompute final state estimates (expensive, done once per bearing)
    x_finals, r_futs, rul_trues = {}, {}, {}
    for bid in BEARING_IDS:
        hi  = hi_seqs[bid]
        r   = r_seqs[bid]
        K   = len(hi)
        cut = int(K * 0.80)
        if use_fpt:
            xe = pf_fn(hi[:cut], r[:cut], eta, sb2, a, b, c, sm2,
                       fpt_threshold=fpt_threshold)
        else:
            xe = pf_fn(hi[:cut], r[:cut], eta, sb2, a, b, c, sm2)
        x_finals[bid]  = xe[-1]
        r_futs[bid]    = float(np.mean(r[cut:])) if cut < K else 1.0
        rul_trues[bid] = (K - cut) * INTERVAL_SEC

    best_score, best_pct = 0.0, 0.40
    rows = []
    for pct in pcts:
        scores = []
        for bid in BEARING_IDS:
            rul_p = predict_rul(x_finals[bid], eta, sb2, r_futs[bid], pct) * INTERVAL_SEC
            sc    = comp_score(rul_trues[bid], rul_p)
            if not np.isnan(sc):
                scores.append(sc)
        mean_sc = float(np.mean(scores))
        rows.append({"pct": pct, "score": round(mean_sc, 5)})
        if mean_sc > best_score:
            best_score, best_pct = mean_sc, pct

    return best_pct, best_score, pd.DataFrame(rows)


# ── Test prediction ───────────────────────────────────────────────────────────

def predict_test(approach, hi_seqs_test, r_test_seqs,
                 eta, sb2, a, b, c, sm2):
    records = []
    for t in range(1, 7):
        hi     = hi_seqs_test[t]
        r      = r_test_seqs[t]
        x_est  = particle_filter(hi, r, eta, sb2, a, b, c, sm2)
        x_bar  = x_est[-1]
        r_fut  = float(np.mean(r))
        rul_c  = predict_rul(x_bar, eta, sb2, r_fut)
        rul_s  = rul_c * INTERVAL_SEC
        records.append({"approach": approach, "test_id": t,
                        "x_est": round(x_bar, 4),
                        "rul_cyc": round(rul_c, 2),
                        "rul_s": round(rul_s, 0),
                        "rul_h": round(rul_s / 3600, 2)})
    return records


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_train_comparison(hi_seqs, r_seqs, eta, sb2, a, b, c, sm2):
    """Show FDR label + state estimate for all training bearings."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, bid in zip(axes.flat, BEARING_IDS):
        hi    = hi_seqs[bid]
        r     = r_seqs[bid]
        x_est = particle_filter(hi, r, eta, sb2, a, b, c, sm2, n=1000)
        K     = len(hi)
        ax.plot(np.arange(K), hi,    "b-",  lw=1.2, label="HI (FDR)")
        ax.plot(np.arange(K), x_est, "r--", lw=1.5, label="State x̂")
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_title(f"Bearing {bid}"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle("Training: F2S2 state estimation on FDR HI", fontsize=11)
    plt.tight_layout()
    plt.savefig(PRED_DIR / "train_state_loocv.png", dpi=120)
    plt.close()


def plot_test_comparison(hi_A, hi_B, x_est_A, x_est_B, rul_A, rul_B):
    fig, axes = plt.subplots(6, 2, figsize=(14, 20))
    for row, t in enumerate(range(1, 7)):
        for col, (hi, x_est, rul, label, color) in enumerate([
            (hi_A[t], x_est_A[t], rul_A[t], "A: CNN",    "tab:blue"),
            (hi_B[t], x_est_B[t], rul_B[t], "B: Direct", "tab:orange"),
        ]):
            ax = axes[row, col]
            K  = len(hi)
            ax.plot(np.arange(K), hi,    "-",  color=color, lw=1.2, alpha=0.7, label="HI")
            ax.plot(np.arange(K), x_est, "--", color="red",  lw=1.5,           label="State x̂")
            ax.axhline(1.0, color="k", ls=":", lw=0.8)
            ax.set_title(f"Test {t}  [{label}]  RUL={rul:.0f}s ({rul/3600:.2f}h)")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.suptitle("Test: Approach A (CNN) vs B (Direct HI) — F2S2 state + RUL",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(PRED_DIR / "test_comparison_AB.png", dpi=120)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Approach A vs B — F2S2 Comparison")
    print("=" * 60)

    # ── Load training HI ──────────────────────────────────────────────────────
    print("\n[1] Loading training HI …")
    hi_train  = {b: load_train_hi(b) for b in BEARING_IDS}
    r_train   = build_rpm_seqs()
    for b in BEARING_IDS:
        n = min(len(hi_train[b]), len(r_train[b]))
        hi_train[b] = hi_train[b][:n]
        r_train[b]  = r_train[b][:n]
        print(f"  Bearing {b}: {n} cycles")

    # ── Estimate F2S2 parameters ──────────────────────────────────────────────
    print("\n[2] Estimating F2S2 parameters from training bearings …")
    eta, sb2 = estimate_state_params(r_train)
    print(f"  η={eta:.6f}  σ_B²={sb2:.6f}")
    a, b, c, sm2 = estimate_meas_params(hi_train)
    print(f"  a={a:.4f}, b={b:.4f}, c={c:.4f}, σ²_m={sm2:.6f}")

    pd.DataFrame([{"eta": eta, "sb2": sb2, "a": a, "b": b, "c": c, "sm2": sm2}]
                 ).to_csv(PRED_DIR / "f2s2_params.csv", index=False)

    # ── LOOCV diagnostic on training — standard + FPT ─────────────────────────
    print("\n[3] LOOCV on training (80% lifetime cutoff) …")
    loocv_std = loocv_training(hi_train, r_train, eta, sb2, a, b, c, sm2,
                                use_fpt=False)
    loocv_fpt = loocv_training(hi_train, r_train, eta, sb2, a, b, c, sm2,
                                use_fpt=True, fpt_threshold=0.25)

    print("\n  Standard PF:")
    print(loocv_std[["bearing","cut_cyc","K","x_bar",
                      "rul_pred_s","rul_true_s","Er_pct"]].to_string(index=False))
    print("\n  FPT PF (threshold=0.25):")
    print(loocv_fpt[["bearing","cut_cyc","K","fpt","x_bar",
                      "rul_pred_s","rul_true_s","Er_pct"]].to_string(index=False))

    # Competition score comparison
    def avg_comp_score(df):
        scores = [comp_score(r["rul_true_s"], r["rul_pred_s"])
                  for _, r in df.iterrows()]
        return float(np.nanmean(scores))

    sc_std = avg_comp_score(loocv_std)
    sc_fpt = avg_comp_score(loocv_fpt)
    print(f"\n  LOOCV avg competition score — Std: {sc_std:.4f}  |  FPT: {sc_fpt:.4f}")

    loocv_std.to_csv(PRED_DIR / "loocv_training.csv", index=False)
    loocv_fpt.to_csv(PRED_DIR / "loocv_training_fpt.csv", index=False)
    plot_train_comparison(hi_train, r_train, eta, sb2, a, b, c, sm2)

    # ── Grid-search optimal percentile ────────────────────────────────────────
    print("\n[3b] Grid-searching optimal percentile (FPT mode) …")
    best_pct, best_loocv_score, pct_table = find_optimal_percentile(
        hi_train, r_train, eta, sb2, a, b, c, sm2,
        use_fpt=True, fpt_threshold=0.25)
    pct_table.to_csv(PRED_DIR / "percentile_grid.csv", index=False)
    print(f"  Optimal percentile: {best_pct:.2f}  "
          f"→ LOOCV score: {best_loocv_score:.4f}")

    # show top-10 percentile-score table
    print(pct_table.sort_values("score", ascending=False).head(10).to_string(index=False))

    # Plot percentile vs score curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pct_table["pct"], pct_table["score"], "b-o", ms=3)
    ax.axvline(best_pct, color="r", ls="--",
               label=f"best pct={best_pct:.2f}, score={best_loocv_score:.4f}")
    ax.axvline(PERCENTILE, color="gray", ls=":", label=f"default pct=0.40")
    ax.set_xlabel("Percentile"); ax.set_ylabel("Avg LOOCV Competition Score")
    ax.set_title("Percentile Grid Search (FPT PF)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PRED_DIR / "percentile_grid.png", dpi=120)
    plt.close()
    print(f"  Grid plot → {PRED_DIR / 'percentile_grid.png'}")

    # ── Load test HI (A & B) ──────────────────────────────────────────────────
    print("\n[4] Loading test HI …")
    hi_test_A, hi_test_B = {}, {}
    ok_A = ok_B = True
    for t in range(1, 7):
        try:
            hi_test_A[t] = load_test_hi(t, "A")
        except FileNotFoundError as e:
            print(f"  Approach A missing: {e}")
            ok_A = False
        try:
            hi_test_B[t] = load_test_hi(t, "B")
        except FileNotFoundError as e:
            print(f"  Approach B missing: {e}")
            ok_B = False

    if not ok_A:
        print("  → Run s1a_cnn_improved.py first for Approach A")
    if not ok_B:
        print("  → Run c2_test_hi.py first for Approach B")
    if not (ok_A or ok_B):
        return

    # test RPM: use average condition (no operation CSV for test)
    r_test = {t: np.ones(len(hi_test_A.get(t, hi_test_B.get(t))),
                          dtype=np.float64)
              for t in range(1, 7)}

    # ── Run F2S2 + RUL for both approaches (standard + FPT) ──────────────────
    print(f"\n[5] Predicting test RUL (using optimal pct={best_pct:.2f}) …")
    all_records = []
    x_est_A_dict, x_est_B_dict = {}, {}
    rul_A_dict,   rul_B_dict   = {}, {}

    # Approach B uses FPT (better HI → FPT more meaningful)
    # Approach A uses standard PF (HI is flat, FPT triggers immediately)
    for t in range(1, 7):
        r = r_test[t]
        if ok_A:
            hi  = hi_test_A[t]
            xe  = particle_filter(hi, r[:len(hi)], eta, sb2, a, b, c, sm2)
            rul = predict_rul(xe[-1], eta, sb2, pct=best_pct) * INTERVAL_SEC
            x_est_A_dict[t] = xe
            rul_A_dict[t]   = rul
            all_records.append({"approach": "A (CNN)",  "test_id": t,
                                 "x_est": round(xe[-1], 4),
                                 "fpt": None,
                                 "rul_s": round(rul, 0),
                                 "rul_h": round(rul / 3600, 2)})
        if ok_B:
            hi   = hi_test_B[t]
            fpt  = detect_fpt(hi, threshold=0.25)
            xe   = particle_filter_fpt(hi, r[:len(hi)], eta, sb2, a, b, c, sm2,
                                        fpt_threshold=0.25)
            rul  = predict_rul(xe[-1], eta, sb2, pct=best_pct) * INTERVAL_SEC
            x_est_B_dict[t] = xe
            rul_B_dict[t]   = rul
            all_records.append({"approach": "B (Direct+FPT)", "test_id": t,
                                 "x_est": round(xe[-1], 4),
                                 "fpt": int(fpt),
                                 "rul_s": round(rul, 0),
                                 "rul_h": round(rul / 3600, 2)})

    result_df = pd.DataFrame(all_records)
    result_df.to_csv(PRED_DIR / "comparison_summary.csv", index=False)

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n── Test RUL Predictions ──────────────────────────────")
    pivot = result_df.pivot(index="test_id", columns="approach",
                            values=["x_est", "rul_h"])
    print(pivot.to_string())

    # ── Comparison plots ──────────────────────────────────────────────────────
    if ok_A and ok_B:
        plot_test_comparison(hi_test_A, hi_test_B,
                             x_est_A_dict, x_est_B_dict,
                             rul_A_dict, rul_B_dict)

    # ── Single-approach plots if one is missing ───────────────────────────────
    for approach, ok, hi_dict, x_est_d, rul_d in [
        ("A (CNN)",    ok_A, hi_test_A, x_est_A_dict, rul_A_dict),
        ("B (Direct)", ok_B, hi_test_B, x_est_B_dict, rul_B_dict),
    ]:
        if not ok:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, t in zip(axes.flat, range(1, 7)):
            hi = hi_dict[t]; xe = x_est_d[t]; rul = rul_d[t]
            ax.plot(hi, "b-", lw=1.2, label="HI")
            ax.plot(xe, "r--", lw=1.5, label="State x̂")
            ax.axhline(1.0, color="k", ls=":", lw=0.8)
            ax.set_title(f"Test {t}  RUL={rul:.0f}s ({rul/3600:.2f}h)")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        plt.suptitle(f"Approach {approach} — Test Predictions", fontsize=11)
        plt.tight_layout()
        tag = "A" if "A" in approach else "B"
        plt.savefig(PRED_DIR / f"test_rul_{tag}.png", dpi=120)
        plt.close()

    print(f"\nAll results → {PRED_DIR}")
    print("\nApproach A vs B comparison complete.")


if __name__ == "__main__":
    main()
