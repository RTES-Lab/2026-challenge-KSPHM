"""
GLCT-GWO: Generalized Linear Chirplet Transform with Grey Wolf Optimizer.

Reference: R. Duan, Y. Liao, L. Yang, "Adaptive tacholess order tracking method
based on generalized linear chirplet transform...", ISA Trans. 127 (2022) 324-341.

Pipeline:
  1) Down-sample raw signal to ~3*fr_max
  2) Bandpass around expected shaft frequency
  3) GWO searches optimal window length L and demodulator count N by maximizing
     Gini index of the TF plane (concentrated ridge = high GI)
  4) GLCT yields TF plane; ridge extraction => instantaneous rotating frequency (IRF)

Used by preprocess.py to estimate shaft IRF from vibration alone (no tacho).
"""
from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt, get_window, decimate


# --------------------------------------------------------------------------
# Down-sampling and band-pass helpers
# --------------------------------------------------------------------------
def downsample(x: np.ndarray, fs: float, fs_new: float) -> tuple[np.ndarray, float]:
    """Polyphase decimation toward fs_new. Returns (y, fs_actual)."""
    q = max(int(round(fs / fs_new)), 1)
    if q == 1:
        return x.astype(np.float32), float(fs)
    y = x
    while q > 13:
        y = decimate(y, 13, ftype="iir", zero_phase=True)
        q = max(int(round(q / 13)), 1)
    if q > 1:
        y = decimate(y, q, ftype="iir", zero_phase=True)
    fs_actual = fs / max(int(round(fs / fs_new)), 1)
    return y.astype(np.float32), float(fs_actual)


def bandpass(x: np.ndarray, fs: float, fc: float, bw: float, order: int = 4) -> np.ndarray:
    low = max(fc - bw / 2, 0.5)
    high = min(fc + bw / 2, fs / 2 - 0.5)
    if high <= low:
        return x.astype(np.float32)
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


# --------------------------------------------------------------------------
# Gini index (Eq. 19, Duan et al. 2022)
# --------------------------------------------------------------------------
def gini_index(x: np.ndarray) -> float:
    """
    Gini index of a 1-D array, elements sorted small-to-large.
    GI → 1 for concentrated (sparse) signals; GI → 0 for uniform.
    Higher GI = sharper TF ridge = better GLCT parameters.
    """
    x = np.abs(x).ravel().astype(np.float64)
    x_sorted = np.sort(x)
    n = len(x_sorted)
    l1 = x_sorted.sum()
    if l1 == 0.0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    coeff = (n - idx + 0.5) / n
    gi = 1.0 - 2.0 * float(np.dot(x_sorted / l1, coeff))
    return float(np.clip(gi, 0.0, 1.0))


# --------------------------------------------------------------------------
# GLCT core (vectorized — no Python loop over time frames)
# --------------------------------------------------------------------------
def glct(
    x: np.ndarray,
    fs: float,
    win_len: int,
    N: int,
    window: str = "gaussian",
    hop: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generalized Linear Chirplet Transform.

    Parameters
    ----------
    x       : 1-D preprocessed signal (down-sampled & band-passed)
    fs      : sampling frequency of x
    win_len : analysis window length L
    N       : number of demodulators (1..10)
    hop     : time-axis decimation for output TF

    Returns
    -------
    tf : (n_freq, n_frames) magnitude TF plane
    t  : time vector (s)
    f  : frequency vector (Hz)
    """
    n = len(x)
    win_len = max(32, min(win_len, n))
    if win_len % 2 == 1:
        win_len += 1
    N = max(1, min(N, 10))
    half = win_len // 2

    # Gaussian window (best noise immunity per paper Section 3.1)
    if window == "gaussian":
        kk = np.arange(win_len, dtype=np.float32) - half
        sigma = win_len / 6.0
        w = np.exp(-(kk ** 2) / (2.0 * sigma ** 2))
    else:
        w = get_window(window, win_len).astype(np.float32)

    # Demodulator angles α (Eq. 6): N values excluding ±π/2 endpoints
    if N == 1:
        alphas = np.array([0.0])
    else:
        alphas = -np.pi / 2.0 + np.arange(1, N + 1) * np.pi / (N + 1)

    # Chirp demodulation yields complex-valued segments → must use fft, not rfft
    n_freq = win_len // 2 + 1
    f_all = np.fft.fftfreq(win_len, d=1.0 / fs)
    f = f_all[:n_freq].astype(np.float32)
    f[-1] = abs(f[-1])  # Nyquist bin is negative in fftfreq; make positive

    # All frames at once via stride tricks (zero-copy view)
    # sliding_window_view(x, win_len)[::hop] → (n_frames, win_len)
    # frame j is centered at half + j*hop
    all_windows = sliding_window_view(x.astype(np.float32), win_len)[::hop]
    n_frames = all_windows.shape[0]
    t = (half + hop * np.arange(n_frames, dtype=np.float32)) / fs

    kk_s = (np.arange(win_len, dtype=np.float32) - half) / fs  # (kk / fs)
    quad = 0.5 * kk_s ** 2  # chirp demod exponent factor

    tf = np.zeros((n_freq, n_frames), dtype=np.float32)

    for a in alphas:
        c = float(np.tan(a)) * fs  # chirp rate (Eq. 4)
        demod = np.exp(-1j * c * quad).astype(np.complex64)
        we = (w * demod).astype(np.complex64)  # (win_len,)

        # Complex input → full fft, take first n_freq bins (DC to Nyquist)
        seg = all_windows.astype(np.complex64) * we[np.newaxis, :]  # (n_frames, win_len)
        S = np.abs(np.fft.fft(seg, axis=1))[:, :n_freq].T.astype(np.float32)  # (n_freq, n_frames)
        np.maximum(tf, S, out=tf)

    return tf, t, f


# --------------------------------------------------------------------------
# Grey Wolf Optimizer (Mirjalili et al., 2014)
# --------------------------------------------------------------------------
def grey_wolf_optimizer(
    objective,
    lb: np.ndarray,
    ub: np.ndarray,
    n_agents: int = 20,
    max_iter: int = 60,
    seed: int | None = 42,
) -> tuple[np.ndarray, float]:
    """
    Minimizes objective(x) over [lb, ub] using GWO.

    Returns
    -------
    best_pos   : optimal parameter vector
    best_score : objective value at best_pos
    """
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    dim = len(lb)

    pos = rng.uniform(lb, ub, size=(n_agents, dim))

    # Evaluate initial population
    scores = np.array([objective(pos[i]) for i in range(n_agents)])
    order = np.argsort(scores)
    alpha_pos, alpha_score = pos[order[0]].copy(), scores[order[0]]
    beta_pos,  beta_score  = pos[order[1]].copy(), scores[order[1]]
    delta_pos, delta_score = pos[order[2]].copy(), scores[order[2]]

    for iteration in range(max_iter):
        a = 2.0 * (1.0 - iteration / max_iter)  # linearly 2 → 0

        for i in range(n_agents):
            new_pos = np.empty(dim)
            for leader, leader_pos in zip(
                (alpha_score, beta_score, delta_score),
                (alpha_pos, beta_pos, delta_pos),
            ):
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                D = np.abs(C * leader_pos - pos[i])
                new_pos += leader_pos - A * D
            pos[i] = np.clip(new_pos / 3.0, lb, ub)

        for i in range(n_agents):
            score = objective(pos[i])
            if score < alpha_score:
                delta_pos, delta_score = beta_pos.copy(), beta_score
                beta_pos,  beta_score  = alpha_pos.copy(), alpha_score
                alpha_pos, alpha_score = pos[i].copy(), score
            elif score < beta_score:
                delta_pos, delta_score = beta_pos.copy(), beta_score
                beta_pos,  beta_score  = pos[i].copy(), score
            elif score < delta_score:
                delta_pos, delta_score = pos[i].copy(), score

    return alpha_pos, alpha_score


# --------------------------------------------------------------------------
# Ridge (IRF) extraction
# --------------------------------------------------------------------------
def extract_ridge(
    tf: np.ndarray, f: np.ndarray, f_lo: float, f_hi: float, smooth: int = 5
) -> np.ndarray:
    """Per-time argmax in [f_lo, f_hi]. Returns IRF in Hz."""
    mask = (f >= f_lo) & (f <= f_hi)
    if not mask.any():
        raise ValueError(f"No frequency bins in [{f_lo}, {f_hi}] Hz")
    irf = f[mask][np.argmax(tf[mask, :], axis=0)].astype(np.float32)
    if smooth > 1:
        from scipy.signal import medfilt
        k = smooth if smooth % 2 == 1 else smooth + 1
        irf = medfilt(irf, kernel_size=k).astype(np.float32)
    return irf


# --------------------------------------------------------------------------
# High-level: estimate IRF via GLCT-GWO
# --------------------------------------------------------------------------
def estimate_irf(
    x: np.ndarray,
    fs: float,
    fr_min: float,
    fr_max: float,
    fs_target: float | None = None,
    bp_bw: float | None = None,
    win_len_frac: float = 1 / 3,
    hop_s: float = 0.05,
    n_agents: int = 20,
    max_iter: int = 60,
) -> dict:
    """
    GLCT-GWO: GWO searches optimal (L, N) by maximizing Gini index of TF plane.

    During optimization, a 5× coarser hop is used for speed; the final TF is
    computed at full resolution with the optimal parameters.

    Parameters
    ----------
    fr_min, fr_max : expected shaft-frequency range (Hz)
    fs_target      : down-sample target. Default: max(50, 3*fr_max)
    bp_bw          : bandpass bandwidth. Default: 1.5*(fr_max-fr_min) + 4 Hz
    hop_s          : output time resolution (s)
    n_agents       : GWO search agents (paper: 20)
    max_iter       : GWO iterations   (paper: 60)

    Returns dict keys:
      irf    : IRF array (Hz)
      t      : time vector (s)
      tf     : TF plane at optimal (L, N)
      f      : frequency vector (Hz)
      fs_d   : actual down-sampled fs
      N      : optimal N
      L      : optimal window length
      gi     : achieved Gini index
    """
    fs_target = fs_target or max(50.0, 3.0 * fr_max)
    bp_bw = bp_bw or (1.5 * (fr_max - fr_min) + 4.0)
    bp_fc = 0.5 * (fr_min + fr_max)

    y, fs_d = downsample(x, fs, fs_target)
    y = bandpass(y, fs_d, bp_fc, bp_bw)

    default_L = max(64, int(len(y) * win_len_frac))
    hop = max(1, int(round(hop_s * fs_d)))
    hop_opt = max(hop, hop * 5)  # coarser hop during GWO for speed

    # GWO search bounds: L ∈ [default/2, 3*default/2], N ∈ [1, 10]
    L_min = max(32, default_L // 2)
    L_max = min(len(y) // 2, int(default_L * 1.5))
    lb = np.array([float(L_min), 1.0])
    ub = np.array([float(L_max), 10.0])

    def objective(params: np.ndarray) -> float:
        L_int = max(32, int(round(params[0])))
        N_int = max(1, min(10, int(round(params[1]))))
        tf_, _, _ = glct(y, fs_d, win_len=L_int, N=N_int, hop=hop_opt)
        # Maximize GI (= minimize 1 - GI): concentrated TF ridge is optimal
        return 1.0 - gini_index(tf_)

    best_pos, best_score = grey_wolf_optimizer(
        objective, lb, ub, n_agents=n_agents, max_iter=max_iter
    )

    opt_L = max(32, int(round(best_pos[0])))
    opt_N = max(1, min(10, int(round(best_pos[1]))))

    # Final TF at full time resolution with optimal parameters
    tf, t, f = glct(y, fs_d, win_len=opt_L, N=opt_N, hop=hop)
    irf = extract_ridge(tf, f, fr_min - 1.0, fr_max + 1.0, smooth=5)

    return {
        "irf": irf,
        "t": t,
        "tf": tf,
        "f": f,
        "fs_d": fs_d,
        "N": opt_N,
        "L": opt_L,
        "gi": 1.0 - best_score,
    }
