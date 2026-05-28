import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

# ── PATHS & CONSTANTS ────────────────────────────────────────────────────────
BASE_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM")
TRAIN_FEAT_DIR = BASE_DIR / "User/TH/common_source"
TEST_FEAT_DIR = BASE_DIR / "User/TH/FI/06_v6/output/validation_features"
OUT_DIR = BASE_DIR / "User/SR/0526/rul/output/pure_svdd_exp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS = [1, 2, 3, 4]
TEST_IDS = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH = 10
INTERVAL_SEC = 600

MAIN_ALL_FEATS = ["ch3_total_power", "ch3_energy", "ch3_rms", "ch3_std", "ch3_p2p"]
NORMAL_UNTIL = {1: 89, 2: 92, 3: 62, 4: 78}
EOL = {1: 126, 2: 114, 3: 89, 4: 137}
MEAN_TRAIN_LIFE = 116.5

# ── HELPER FUNCTIONS ────────────────────────────────────────────────────────
def ema_smooth(x, alpha=0.2):
    s = np.zeros_like(x)
    s[0] = x[0]
    for i in range(1, len(x)):
        s[i] = alpha * x[i] + (1 - alpha) * s[i-1]
    return s

def moving_average(x, w=7):
    pad = w // 2
    x_pad = np.pad(x, pad, mode='edge')
    return np.convolve(x_pad, np.ones(w)/w, mode='valid')[:len(x)]

def competition_score(rul_true, rul_pred):
    if rul_true <= 0:
        return np.nan
    Er = 100.0 * (rul_true - rul_pred) / rul_true
    return (np.exp(-np.log(0.5) * Er / 20.0) if Er <= 0
            else np.exp(np.log(0.5) * Er / 50.0))

# ── DEGRADATION PROGNOSIS MODELS ─────────────────────────────────────────────
def _exp_fn(x, a, b, c):
    return a * np.exp(b * x) + c

def predict_wiener_degraded(hi_prefix, hi_threshold=1.0):
    n = len(hi_prefix)
    t = np.arange(n, dtype=float)
    if n < 2 or hi_prefix.std() < 1e-6:
        return float(50.0)
    r, b = np.polyfit(t, hi_prefix, 1)
    if r <= 1e-6:
        return float(50.0)
    t_fail = (hi_threshold - b) / r
    t_fail = min(t_fail, 180.0)
    return float(max(0.0, t_fail - (n - 1)))

def predict_exp_fit_degraded(hi_prefix, hi_threshold=1.0):
    n = len(hi_prefix)
    if n < 5:
        return predict_wiener_degraded(hi_prefix, hi_threshold)
    t = np.arange(n, dtype=float)
    try:
        popt, _ = curve_fit(_exp_fn, t, hi_prefix,
                            p0=[0.01, 0.05, hi_prefix[0]],
                            maxfev=2000)
        a, b, c = popt
        if a <= 1e-8 or b <= 1e-8 or (hi_threshold - c) / a <= 0:
            return predict_wiener_degraded(hi_prefix, hi_threshold)
        t_fail = np.log((hi_threshold - c) / a) / b
        t_fail = min(t_fail, 180.0)
        return float(max(0.0, t_fail - (n - 1)))
    except Exception:
        return predict_wiener_degraded(hi_prefix, hi_threshold)

def predict_rul_physical(hi_full, obs, t_d, mean_life=MEAN_TRAIN_LIFE):
    if obs < t_d:
        return float(max(0.0, mean_life - obs))
    else:
        hi_degraded = hi_full[t_d:obs]
        return predict_exp_fit_degraded(hi_degraded, hi_threshold=1.0)

# ── SVDD FUNCTIONS ──────────────────────────────────────────────────────────
def train_global_svdd(train_dfs, train_bids=BEARINGS):
    healthy_features = []
    for b in train_bids:
        df = train_dfs[b]
        healthy_features.append(df[MAIN_ALL_FEATS].values[:50])
    X_healthy = np.vstack(healthy_features)
    
    scaler = StandardScaler()
    X_healthy_scaled = scaler.fit_transform(X_healthy)
    
    ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
    ocsvm.fit(X_healthy_scaled)
    return ocsvm, scaler

def compute_smoothed_svdd_scores(df, ocsvm, scaler):
    X = df[MAIN_ALL_FEATS].values
    X_scaled = scaler.transform(X)
    raw = -ocsvm.decision_function(X_scaled)
    smooth = ema_smooth(raw, alpha=0.2)
    smooth = moving_average(smooth, w=7)
    return raw, smooth

# ── MAIN EVALUATION & INFERENCE ──────────────────────────────────────────────
def run_all(train_dfs, test_dfs):
    print("Evaluating LOOCV with Physical Two-Phase SVDD Extrapolation Model...")
    
    predictions = {}
    
    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        
        # 1. LOO-SVDD
        ocsvm_loo, scaler_loo = train_global_svdd(train_dfs, train_bids)
        
        # 2. Recompute smoothed scores
        raw_scores_loo = {}
        smoothed_scores_loo = {}
        for b in BEARINGS:
            raw, smooth = compute_smoothed_svdd_scores(train_dfs[b], ocsvm_loo, scaler_loo)
            raw_scores_loo[b] = raw
            smoothed_scores_loo[b] = smooth
            
        # 3. Global LOO normalize bounds
        all_train_loo_smoothed = np.concatenate([smoothed_scores_loo[b] for b in train_bids])
        global_min_loo = all_train_loo_smoothed.min()
        global_max_loo = all_train_loo_smoothed.max()
        
        hi_train_loo = {}
        for b in BEARINGS:
            hi = (smoothed_scores_loo[b] - global_min_loo) / (global_max_loo - global_min_loo + 1e-8)
            hi_train_loo[b] = np.clip(hi, 0.0, 1.0)
            
        # 4. Onset detection (t_d) using raw LOO scores
        raw_val = raw_scores_loo[test_bid]
        t_d_loo = None
        for i_c in range(len(raw_val) - 2):
            if raw_val[i_c] > 0.2 and raw_val[i_c+1] > 0.2 and raw_val[i_c+2] > 0.2:
                t_d_loo = i_c
                break
        if t_d_loo is None:
            t_d_loo = len(raw_val)
            
        hi_val = hi_train_loo[test_bid]
        N_val = len(hi_val)
        obs_pts = np.arange(SEQ_LENGTH, N_val)
        
        preds = []
        for obs in obs_pts:
            p = predict_rul_physical(hi_val, obs, t_d_loo, mean_life=MEAN_TRAIN_LIFE)
            preds.append(p)
            
        predictions[test_bid] = {
            "N": N_val,
            "obs_pts": obs_pts,
            "preds": np.array(preds)
        }
        
    # Search for the best calibration factor cf
    best_cf, best_score = 1.0, -np.inf
    for cf in np.arange(0.20, 2.51, 0.02):
        scores = []
        for b in BEARINGS:
            data = predictions[b]
            N = data["N"]
            obs_pts = data["obs_pts"]
            sc = np.nanmean([competition_score(N - obs, p * cf) for obs, p in zip(obs_pts, data["preds"])])
            scores.append(sc)
        mean_sc = np.mean(scores)
        if mean_sc > best_score:
            best_score = mean_sc
            best_cf = cf
            
    print(f"\n[LOOCV Results (best_cf = {best_cf:.2f})]")
    print(f"  LOOCV Mean Score: {best_score:.4f}")
    for b in BEARINGS:
        data = predictions[b]
        sc = np.nanmean([competition_score(data["N"] - obs, p * best_cf) for obs, p in zip(data["obs_pts"], data["preds"])])
        print(f"    Bearing {b}: score = {sc:.4f}")

    print("\n" + "="*50)
    print("Running Test Inference...")
    # Train global SVDD on all 4 training bearings
    ocsvm_global, scaler_global = train_global_svdd(train_dfs, BEARINGS)
    
    # Smooth and globally normalize all training scores for final global bounds
    smoothed_train = {}
    for b in BEARINGS:
        _, smooth = compute_smoothed_svdd_scores(train_dfs[b], ocsvm_global, scaler_global)
        smoothed_train[b] = smooth
    all_train_smoothed = np.concatenate([smoothed_train[b] for b in BEARINGS])
    global_min = all_train_smoothed.min()
    global_max = all_train_smoothed.max()
    
    # Process each test bearing
    for tid in TEST_IDS:
        df_test = test_dfs[tid]
        raw_test, smooth_test = compute_smoothed_svdd_scores(df_test, ocsvm_global, scaler_global)
        hi_test = (smooth_test - global_min) / (global_max - global_min + 1e-8)
        hi_test = np.clip(hi_test, 0.0, 1.0)
        
        # Detect onset t_d using raw scores
        t_d = None
        for i_c in range(len(raw_test) - 2):
            if raw_test[i_c] > 0.2 and raw_test[i_c+1] > 0.2 and raw_test[i_c+2] > 0.2:
                t_d = i_c
                break
        
        obs = len(hi_test)
        if t_d is None or t_d >= obs:
            # Healthy countdown
            p_cyc = float(max(0.0, MEAN_TRAIN_LIFE - obs))
            status = "HEALTHY"
        else:
            # Degraded exponential fitting
            p_cyc = predict_rul_physical(hi_test, obs, t_d, mean_life=MEAN_TRAIN_LIFE)
            status = "DEGRADED"
            
        p_calibrated = p_cyc * best_cf
        p_hours = p_calibrated * INTERVAL_SEC / 3600.0
        
        print(f"  Test {tid} ({status}): t_d={t_d if t_d is not None else 'None'}, predicted RUL = {p_hours:.2f} hr (cycles={p_calibrated:.1f})")

if __name__ == "__main__":
    print("Loading training data...")
    train_dfs = {b: pd.read_csv(TRAIN_FEAT_DIR / f"Bearing{b}_features.csv") for b in BEARINGS}
    print("Loading test data...")
    test_dfs = {t: pd.read_csv(TEST_FEAT_DIR / f"Test{t}_features.csv") for t in TEST_IDS}
    run_all(train_dfs, test_dfs)
