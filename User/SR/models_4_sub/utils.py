"""Shared utilities: data loading, scoring, feature extraction."""

import numpy as np
import pandas as pd
from pathlib import Path

INTERVAL_SEC = 600   # seconds per measurement cycle
FEAT_NAMES   = ["hi_mean", "hi_slope", "hi_end", "hi_std",
                "hi_range", "hi_max", "hi_recent", "regime_frac"]

# Default competition bearing configuration
COMP_EOL = {1: 126, 2: 114, 3: 89, 4: 137}    # total lifespan (cycles)
COMP_NU  = {1: 89,  2: 92,  3: 62, 4: 78}      # end of normal phase (cycles)


def load_hi_csv(csv_path):
    """
    Load HI data from a single CSV file.

    Required columns : id, HI
    Optional columns : regime

    Each (id, row-order) pair is one time step for that bearing.
    Rows for each id must be in chronological order.

    Returns: {id: {"hi": ndarray, "regime": ndarray|None, "n": int}}

    Example CSV (train)          Example CSV (test)
    ─────────────────────        ──────────────────────
    id,HI,regime                 id,HI
    1,0.001,0                    1,0.012
    1,0.014,0                    1,0.025
    ...                          ...
    2,0.003,1                    2,0.008
    ...                          ...
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    missing = [c for c in ("id", "HI") if c not in df.columns]
    if missing:
        raise KeyError(
            f"{csv_path.name}: missing required columns {missing}. "
            f"Found: {list(df.columns)}. "
            f"Required format: id, HI  (optional: regime)"
        )

    data = {}
    for bid, group in df.groupby("id", sort=True):
        hi     = group["HI"].values.astype(float)
        regime = (group["regime"].values.astype(int)
                  if "regime" in group.columns else None)
        data[int(bid)] = {"hi": hi, "regime": regime, "n": len(hi)}
    return data


def rul_labels(n_total, eol, normal_until=None):
    """
    RUL label vector.
    - If normal_until given: flat at (eol - normal_until) up to normal_until,
      then linearly decreasing to 0.
    - Otherwise: max(eol - t, 0) throughout.
    """
    idx = np.arange(n_total, dtype=float)
    if normal_until is not None:
        flat = float(eol - normal_until)
        return np.where(idx <= normal_until, flat,
                        np.maximum(eol - idx, 0.0))
    return np.maximum(eol - idx, 0.0)


def comp_score(rul_true, rul_pred):
    """
    Competition asymmetric scoring.
    Er = 100*(true-pred)/true
      Er < 0 (over-predict): penalized with /20 → steeper
      Er > 0 (under-predict): penalized with /50 → gentler
    Returns NaN if rul_true <= 0.
    """
    if rul_true <= 0:
        return np.nan
    Er = 100.0 * (rul_true - rul_pred) / rul_true
    if Er <= 0:
        return float(np.exp(-np.log(0.5) * Er / 20.0))
    return float(np.exp(np.log(0.5) * Er / 50.0))


def avg_score(trues, preds):
    """Mean competition score, ignoring NaN entries."""
    scores = [comp_score(t, p) for t, p in zip(trues, preds)]
    valid  = [s for s in scores if not np.isnan(s)]
    return float(np.mean(valid)) if valid else np.nan


def extract_window_features(hi_window, regime_window=None):
    """
    Extract an 8-dim feature vector from a fixed-size HI window.
    Order matches FEAT_NAMES.
    """
    W = len(hi_window)
    t = np.arange(W, dtype=float)
    slope = float(np.polyfit(t, hi_window, 1)[0]) if W > 1 else 0.0
    seg   = max(1, W // 5)
    early = float(np.mean(hi_window[:seg]))
    late  = float(np.mean(hi_window[-seg:]))
    rfrac = float(np.mean(regime_window)) if regime_window is not None else 0.5
    return np.array([
        float(np.mean(hi_window)),  # hi_mean
        slope,                       # hi_slope (per cycle)
        float(hi_window[-1]),       # hi_end
        float(np.std(hi_window)),   # hi_std
        late - early,               # hi_range (trend proxy)
        float(np.max(hi_window)),   # hi_max
        late,                        # hi_recent (last 20%)
        rfrac,                      # regime_frac
    ], dtype=float)


def build_window_dataset(data, train_bids, eol_dict, nu_dict, win_size):
    """
    Build (X, y) matrix from sliding windows over all training bearings.
    No leakage: only uses train_bids data.
    """
    X_list, y_list = [], []
    for b in train_bids:
        hi     = data[b]["hi"]
        regime = data[b]["regime"]
        rul    = rul_labels(len(hi), eol_dict[b], nu_dict[b])
        N = len(hi)
        for t_s in range(0, N - win_size + 1):
            t_e     = t_s + win_size
            hi_win  = hi[t_s:t_e]
            reg_win = regime[t_s:t_e] if regime is not None else None
            X_list.append(extract_window_features(hi_win, reg_win))
            y_list.append(float(rul[t_e - 1]))
    return np.array(X_list, dtype=float), np.array(y_list, dtype=float)
