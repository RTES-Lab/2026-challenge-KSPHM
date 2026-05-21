"""
SR/0520 — Exp E: A-full v7_advanced RUL Pipeline
=====================================================================
목적:
- A-full dynamic baseline recalculation 아키텍처 연동
- LSTM-B 다차원 피처 ([window_norm, raw_hi, delta_from_start, obs_frac]) 데이터 로더
- Segment-Start Matching (구간 다차원 매칭) 기반 start_obs 추정
- Dynamic Ensemble (예측 편차 기반 실시간 가중치 시프트)
- Irreversible Corrections (Rolling median, Upward jump 제약, HI-end 감쇠 보정)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE      = Path("/data/home/ksphm/2026-challenge-KSPHM")
TH_HI_DIR = BASE / "User/TH/FI/07_v7/output/v7_4_2_conditional_aux_boost"
V721_DIR  = BASE / "User/TH/FI/07_v7/output/v7_2_1_rpm_baseline_relative"
RPM_DIR   = BASE / "User/TH/FI/07_v7/output/v7_1_1_rpm_estimator_validation"
TRAIN_FEAT_DIR = BASE / "User/TH/common_source"
TEST_FEAT_DIR  = BASE / "User/TH/FI/06_v6/output/validation_features"

# SR/0520 출력 디렉토리
OUT_DIR   = BASE / "User/SR/0520/rul/output/th742_v7_advanced"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH      = 10
MEAN_TRAIN_LIFE = 116.5
INTERVAL_SEC    = 600
SEEDS           = [42, 7, 123, 0, 99]

NORMAL_UNTIL = {1: 89, 2: 92, 3: 62, 4: 78}
EOL          = {1: 126, 2: 114, 3: 89, 4: 137}

# ── HI computation constants ───────────────────────────────────────────────
EPS               = 1e-8
NORMAL_RATIO      = 0.15
EMA_ALPHA         = 0.20
SMOOTH_WINDOW     = 7
Z_CAP             = 10.0
USE_LOG_COMPRESS  = True

MAIN_FEATURE_GROUPS = {
    "energy":    ["ch3_total_power", "ch3_energy", "ch3_rms"],
    "variation": ["ch3_std", "ch3_p2p"],
}
MAIN_ALL_FEATS = [f for fs in MAIN_FEATURE_GROUPS.values() for f in fs]

AUX_FEATURE_CANDIDATES = ["ch3_high_band", "ch4_high_band", "ch3_mean_freq"]
AUX_DIRECTION_MAP = {"ch3_high_band": -1.0, "ch4_high_band": -1.0, "ch3_mean_freq": -1.0}

# v7_4_2 gate params
AUX_BETA        = 1.00
GATE_MAX        = 0.90
FAIL_RECENT_TH  = 0.18
FAIL_MAX_TH     = 0.22
AUX_RECENT_TH   = 0.10
AUX_MAX_TH      = 0.20
AUX_BOOST_MAX   = 1.50
AUX_GATE_POWER  = 0.50

# ── Advanced Config constants ──────────────────────────────────────────────
START_MATCH_LEN       = 18       # test segment length used for matching
START_SEARCH_MIN      = 0        # train search start index
START_SEARCH_MARGIN   = 5        # allow windows until len(train)-segment_len-margin

GAIN_MIN_START_TH     = 0.25
GAIN_MIN_START_SPAN   = 0.35
GAIN_MIN_START_MAX    = 25

ROLL_MED_WIN          = 3
MAX_UP_STEP           = 1.0      # allowed upward jump in cycles per step
MIN_RUL_CYCLES        = 1.0

DIFF_REF              = 35.0     # cycles; larger difference -> trust LSTM more
MAX_LSTM_SHIFT        = 0.25     # maximum extra weight moved to LSTM

HI_END_CORR_TH        = 0.45
HI_END_CORR_STRENGTH  = 0.18

# ── HI utilities ──────────────────────────────────────────────────────────
def ema_smooth(x, alpha=EMA_ALPHA):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if len(x) == 0:
        return y
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def moving_average(x, window=SMOOTH_WINDOW):
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    return pd.Series(x).rolling(window=window, center=True, min_periods=1).mean().values

def smooth_score(x):
    x = ema_smooth(x)
    x = moving_average(x)
    return np.clip(x, 0.0, None)

def robust_transform_z(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -Z_CAP, Z_CAP)
    if USE_LOG_COMPRESS:
        z = np.sign(z) * np.log1p(np.abs(z))
    return np.maximum(z, 0.0)

def exp_calibrate(raw, offset, tau):
    x  = np.maximum(np.asarray(raw, dtype=float) - offset, 0.0)
    hi = 1.0 - np.exp(-x / max(tau, EPS))
    hi = ema_smooth(hi)
    hi = moving_average(hi)
    return np.clip(hi, 0.0, 1.0)

def minmax_norm(x):
    x = np.asarray(x, dtype=float)
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

def slope_of(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(x)), x, 1)[0])

def rolling_median_np(x, win=3):
    x = np.asarray(x, dtype=float)
    if win <= 1 or len(x) < win:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(win, center=True, min_periods=1).median().values

def clip01(x):
    return float(np.clip(x, 0.0, 1.0))

# ── Load saved TH parameters ───────────────────────────────────────────────
def load_th_params():
    params   = pd.read_csv(TH_HI_DIR / "v7_4_2_params.csv").iloc[0].to_dict()
    sigma_df = pd.read_csv(V721_DIR / "v7_2_1_train_regime_sigma.csv")
    feat_q   = pd.read_csv(V721_DIR / "v7_2_1_feature_q_weights.csv")
    grp_q    = pd.read_csv(V721_DIR / "v7_2_1_group_q_weights.csv")
    aux_w_df = pd.read_csv(TH_HI_DIR / "v7_4_2_aux_feature_weights.csv")
    aux_s_df = pd.read_csv(TH_HI_DIR / "v7_4_2_aux_regime_sigma.csv")

    sigma    = {(int(r["regime"]), r["feature"]): float(r["sigma"])
                for _, r in sigma_df.iterrows()}
    feat_w   = dict(zip(feat_q["feature"], feat_q["weight"]))
    grp_w    = dict(zip(grp_q["group"],    grp_q["group_weight"]))
    aux_w    = dict(zip(aux_w_df["feature"], aux_w_df["weight"]))
    aux_s    = {(int(r["regime"]), r["feature"]): float(r["sigma"])
                for _, r in aux_s_df.iterrows()}

    aux_features = [f for f in AUX_FEATURE_CANDIDATES if f in aux_w]
    return params, sigma, feat_w, grp_w, aux_w, aux_s, aux_features

def load_train_features():
    return {b: pd.read_csv(TRAIN_FEAT_DIR / f"Bearing{b}_features.csv") for b in BEARINGS}

def load_test_features():
    return {t: pd.read_csv(TEST_FEAT_DIR / f"Test{t}_features.csv") for t in TEST_IDS}

def load_train_regimes():
    rpm_df = pd.read_csv(RPM_DIR / "v7_1_1_train_rpm_eval_results.csv")
    result = {}
    for b in BEARINGS:
        sub = rpm_df[rpm_df["bearing"] == b].sort_values("file_idx")
        result[b] = sub["true_regime"].values.astype(int)
    return result

def load_test_regimes():
    rpm_df = pd.read_csv(RPM_DIR / "v7_1_1_test_estimated_rpm.csv")
    result = {}
    for t in TEST_IDS:
        sub = rpm_df[rpm_df["test_id"] == t].sort_values("file_idx")
        col = "est_regime_smooth" if "est_regime_smooth" in sub.columns else "est_regime"
        result[t] = sub[col].values.astype(int)
    return result

# ── Train-baseline computation (A-full) ──────────────────────────────
def compute_train_baseline(train_dfs, train_regimes, features):
    baseline = {}
    for feat in features:
        all_vals = []
        for b, df in train_dfs.items():
            n0 = max(3, int(len(df) * NORMAL_RATIO))
            if feat in df.columns:
                all_vals.extend(df.iloc[:n0][feat].values.astype(float).tolist())
        global_mean = float(np.nanmean(all_vals)) if all_vals else 0.0

        for regime in [0, 1]:
            vals = []
            for b, df in train_dfs.items():
                n0 = max(3, int(len(df) * NORMAL_RATIO))
                r  = train_regimes[b]
                mask = (np.arange(min(len(df), len(r))) < n0) & (r[:min(len(df), len(r))] == regime)
                idx  = np.where(mask)[0]
                if len(idx) >= 1 and feat in df.columns:
                    vals.extend(df.iloc[idx][feat].values.astype(float).tolist())
            baseline[(regime, feat)] = float(np.nanmean(vals)) if len(vals) >= 2 else global_mean
    return baseline

# ── HI computation ──────────────────────────────────────────────────────────
def make_main_raw(feat_df, regime_arr, sigma, feat_w, grp_w, baseline):
    n   = len(feat_df)
    raw = np.zeros(n, dtype=float)
    for group, feats in MAIN_FEATURE_GROUPS.items():
        group_score = np.zeros(n, dtype=float)
        grp_feat_total = max(sum(feat_w.get(f, 0.0) for f in feats), EPS)
        for feat in feats:
            if feat not in feat_df.columns:
                continue
            x  = feat_df[feat].values.astype(float)
            w  = feat_w.get(feat, 1.0 / len(feats)) / grp_feat_total
            z  = np.zeros(n, dtype=float)
            for regime in [0, 1]:
                idx = np.where(regime_arr == regime)[0]
                if len(idx) == 0:
                    continue
                base = baseline.get((regime, feat), baseline.get((0, feat), 0.0))
                sig  = sigma.get((regime, feat), EPS)
                z[idx] = (x[idx] - base) / max(sig, EPS)
            z = robust_transform_z(z)
            group_score += w * z
        group_score = smooth_score(group_score)
        raw += grp_w.get(group, 0.5) * group_score
    return raw

def make_aux_raw(feat_df, regime_arr, aux_features, aux_w, aux_s, baseline):
    n   = len(feat_df)
    raw = np.zeros(n, dtype=float)
    aux_total_w = max(sum(aux_w.get(f, 0.0) for f in aux_features), EPS)
    for feat in aux_features:
        if feat not in feat_df.columns:
            continue
        x         = feat_df[feat].values.astype(float)
        direction = AUX_DIRECTION_MAP.get(feat, 1.0)
        w         = aux_w.get(feat, 1.0 / len(aux_features)) / aux_total_w
        z         = np.zeros(n, dtype=float)
        for regime in [0, 1]:
            idx  = np.where(regime_arr == regime)[0]
            if len(idx) == 0:
                continue
            base = baseline.get((regime, feat), baseline.get((0, feat), 0.0))
            sig  = aux_s.get((regime, feat), EPS)
            z[idx] = direction * (x[idx] - base) / max(sig, EPS)
        z = robust_transform_z(z)
        raw += w * z
    return smooth_score(raw)

def compute_gate(hi_main, hi_aux):
    hi_main, hi_aux = np.asarray(hi_main, float), np.asarray(hi_aux, float)
    main_recent = float(np.mean(hi_main[-5:]))
    main_max    = float(np.max(hi_main))
    aux_recent  = float(np.mean(hi_aux[-5:]))
    aux_max     = float(np.max(hi_aux))

    fail_recent   = clip01((FAIL_RECENT_TH - main_recent) / max(FAIL_RECENT_TH, EPS))
    fail_max      = clip01((FAIL_MAX_TH    - main_max)    / max(FAIL_MAX_TH,    EPS))
    main_failure  = 0.65 * fail_recent + 0.35 * fail_max

    aux_rec_sc    = clip01((aux_recent - AUX_RECENT_TH) / max(0.45 - AUX_RECENT_TH, EPS))
    aux_max_sc    = clip01((aux_max    - AUX_MAX_TH)    / max(0.55 - AUX_MAX_TH,    EPS))
    aux_reliable  = 0.85 * aux_rec_sc + 0.15 * aux_max_sc

    aux_boost = 1.0 + AUX_BOOST_MAX * main_failure
    gate      = float(np.clip(GATE_MAX * main_failure * (aux_reliable ** AUX_GATE_POWER),
                               0.0, GATE_MAX))
    return gate, aux_boost

def fuse_hi(hi_main, hi_aux, gate, aux_boost=1.0):
    hi_main, hi_aux = np.asarray(hi_main, float), np.asarray(hi_aux, float)
    aux_reflected = np.clip(AUX_BETA * aux_boost * hi_aux, 0.0, 1.0)
    candidate     = np.maximum(hi_main, aux_reflected)
    hi_final      = (1.0 - gate) * hi_main + gate * candidate
    hi_final      = ema_smooth(hi_final)
    hi_final      = moving_average(hi_final)
    return np.clip(hi_final, 0.0, 1.0)

def compute_bearing_hi(feat_df, regime_arr, params, sigma, feat_w, grp_w,
                       aux_features, aux_w, aux_s,
                       main_baseline, aux_baseline):
    n = min(len(feat_df), len(regime_arr))
    feat_df   = feat_df.iloc[:n].copy()
    regime_arr = regime_arr[:n]

    # Main branch
    main_raw = make_main_raw(feat_df, regime_arr, sigma, feat_w, grp_w, main_baseline)
    hi_main  = exp_calibrate(main_raw, params["main_offset"], params["main_tau"])

    # Aux branch
    aux_raw  = make_aux_raw(feat_df, regime_arr, aux_features, aux_w, aux_s, aux_baseline)
    hi_aux   = exp_calibrate(aux_raw, params["aux_offset"], params["aux_tau"])

    # Gate + fusion
    gate, aux_boost = compute_gate(hi_main, hi_aux)
    hi_final = fuse_hi(hi_main, hi_aux, gate, aux_boost)
    return hi_final, hi_main, hi_aux, gate

# ── Competition scoring ────────────────────────────────────────────────────
def competition_score(rul_true, rul_pred):
    if rul_true <= 0:
        return np.nan
    Er = 100.0 * (rul_true - rul_pred) / rul_true
    return (np.exp(-np.log(0.5) * Er / 20.0) if Er <= 0
            else np.exp(np.log(0.5) * Er / 50.0))

def rul_labels(n_total, bid):
    nu, eol = NORMAL_UNTIL[bid], EOL[bid]
    idx = np.arange(n_total)
    return np.where(idx <= nu, eol - nu, np.maximum(eol - idx, 0)).astype(float)

# ── Advanced LightGBM features ─────────────────────────────────────────────
def make_lgbm_features(hi_array, start_obs=0):
    features, targets = [], []
    hi_array = np.asarray(hi_array, dtype=float)
    N = len(hi_array)
    hi0 = float(hi_array[0])

    for i in range(SEQ_LENGTH, N):
        window = hi_array[i - SEQ_LENGTH: i]
        window_norm = minmax_norm(window)
        slope_norm  = slope_of(window_norm)
        slope_raw   = slope_of(window)
        obs_frac    = float(np.clip((start_obs + i) / MEAN_TRAIN_LIFE, 0.0, 2.0))
        gain_from_start = float(window[-1] - hi0)

        feats = (
            list(window_norm)
            + [
                slope_norm,
                float(window[-1]),
                float(window.mean()),
                float(window.max()),
                float(window.min()),
                float(window.std()),
                slope_raw,
                float(window[-1] - window[0]),
                gain_from_start,
                obs_frac,
                float(window[-1] * obs_frac),
            ]
        )
        features.append(feats)
        targets.append(float(N - i)) # Trained on Linear RUL target! (Proven peak LOOCV)
    return np.array(features), np.array(targets)

def lgbm_asymmetric_obj(y_pred, dataset):
    y_true = dataset.get_label()
    diff   = y_true - y_pred
    weight = np.where(diff < 0, 2.8, 1.0)
    return -diff * weight, np.ones_like(diff) * weight

def train_lgbm(hi_dict, train_bids):
    X_all, y_all = [], []
    for b in train_bids:
        x, y = make_lgbm_features(hi_dict[b], start_obs=0)
        X_all.append(x); y_all.append(y)
    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    dtrain = lgb.Dataset(X, label=y)
    params = {
        "num_leaves": 15,
        "learning_rate": 0.04,
        "min_child_samples": 5,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "verbose": -1,
        "objective": lgbm_asymmetric_obj,
    }
    return lgb.train(params, dtrain, num_boost_round=260)

def predict_lgbm(model, hi_array, start_obs=0):
    X, _ = make_lgbm_features(hi_array, start_obs=start_obs)
    return np.maximum(model.predict(X), 0.0)

# ── LSTM-B ──────────────────────────────────────────────────────────────────
N_FEAT = 4

def make_seqs(hi_arr, rul_arr, start_obs=0):
    X, y = [], []
    hi_arr = np.asarray(hi_arr, dtype=float)
    n = len(hi_arr)
    hi0 = float(hi_arr[0])

    for i in range(n - SEQ_LENGTH):
        window = hi_arr[i: i + SEQ_LENGTH].copy()
        window_norm = minmax_norm(window)
        raw_hi = window.copy()
        delta_from_start = window - hi0
        obs_frac = np.clip(
            (start_obs + i + np.arange(SEQ_LENGTH)) / MEAN_TRAIN_LIFE,
            0.0,
            2.0,
        )
        X.append(np.stack([window_norm, raw_hi, delta_from_start, obs_frac], axis=1))
        y.append(float(rul_arr[i + SEQ_LENGTH]))
    return np.array(X), np.array(y)

class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=N_FEAT,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)

def train_lstm(X_train, y_train, rul_scale, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_norm = y_train / rul_scale
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_norm, dtype=torch.float32)

    n_val = max(1, int(len(Xt) * 0.1))
    tr_dl = DataLoader(TensorDataset(Xt[:-n_val], yt[:-n_val]), batch_size=64, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xt[-n_val:], yt[-n_val:]), batch_size=64)

    model = LSTMRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.MSELoss()

    best_val = np.inf
    patience = 0
    best_state = None

    for _ in range(220):
        model.train()
        for xb, yb in tr_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vals = []
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                vals.append(crit(model(xb), yb).item())
            vl = float(np.mean(vals))

        if vl < best_val:
            best_val = vl
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 25:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def train_lstm_ensemble(hi_dict, train_bids, device):
    X_all, y_all = [], []
    for b in train_bids:
        X, y = make_seqs(hi_dict[b], rul_labels(len(hi_dict[b]), b), start_obs=0)
        X_all.append(X); y_all.append(y)
    X_train = np.concatenate(X_all)
    y_train = np.concatenate(y_all)
    rul_scale = float(y_train.max())
    models = [train_lstm(X_train, y_train, rul_scale, s, device) for s in SEEDS]
    return models, rul_scale

def predict_lstm(models, rul_scale, hi_arr, start_obs, device):
    X, _ = make_seqs(hi_arr, np.zeros(len(hi_arr)), start_obs=start_obs)
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    all_preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            all_preds.append(np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0))
    return np.median(all_preds, axis=0)

# ── Dynamic Ensemble ──────────────────────────────────────────────────────────
def dynamic_ensemble(pred_lgbm, pred_lstm, base_w_lgbm, base_w_lstm):
    pred_lgbm = np.asarray(pred_lgbm, dtype=float)
    pred_lstm = np.asarray(pred_lstm, dtype=float)
    diff = np.abs(pred_lgbm - pred_lstm)
    shift = MAX_LSTM_SHIFT * np.clip(diff / DIFF_REF, 0.0, 1.0)

    w_lstm = np.clip(base_w_lstm + shift, 0.25, 0.90)
    w_lgbm = 1.0 - w_lstm
    pred = w_lgbm * pred_lgbm + w_lstm * pred_lstm
    return pred, w_lgbm, w_lstm

# ── Irreversible Corrections ───────────────────────────────────────────────
def limited_upward_rul_correction(preds, max_up_step=1.0, roll_win=3):
    y = np.asarray(preds, dtype=float).copy()
    y = rolling_median_np(y, win=roll_win)
    y = np.maximum(y, MIN_RUL_CYCLES)

    for i in range(1, len(y)):
        if y[i] > y[i - 1] + max_up_step:
            y[i] = y[i - 1] + max_up_step
    return y

def hi_end_mild_correction(preds, hi_end):
    y = np.asarray(preds, dtype=float).copy()
    excess = max(float(hi_end) - HI_END_CORR_TH, 0.0)
    factor = 1.0 - HI_END_CORR_STRENGTH * clip01(excess / 0.35)
    return np.maximum(y * factor, MIN_RUL_CYCLES), factor

def apply_prediction_corrections(preds, hi_end):
    y = limited_upward_rul_correction(preds, MAX_UP_STEP, ROLL_MED_WIN)
    y, hi_factor = hi_end_mild_correction(y, hi_end)
    return y, hi_factor

# ── Segment descriptor & matching ───────────────────────────────────────────
def segment_descriptor(seg):
    seg = np.asarray(seg, dtype=float)
    return {
        "start": float(seg[0]),
        "end": float(seg[-1]),
        "mean": float(seg.mean()),
        "max": float(seg.max()),
        "slope": slope_of(seg),
        "gain": float(seg[-1] - seg[0]),
        "shape": minmax_norm(seg),
    }

def segment_distance(test_seg, train_seg):
    td = segment_descriptor(test_seg)
    rd = segment_descriptor(train_seg)

    d_start = abs(rd["start"] - td["start"]) / 0.25
    d_end   = abs(rd["end"] - td["end"]) / 0.25
    d_mean  = abs(rd["mean"] - td["mean"]) / 0.25
    d_gain  = abs(rd["gain"] - td["gain"]) / 0.25
    d_slope = abs(rd["slope"] - td["slope"]) / 0.03
    d_shape = float(np.mean(np.abs(rd["shape"] - td["shape"])))

    dist = (
        0.08 * d_start
        + 0.30 * d_end
        + 0.22 * d_mean
        + 0.25 * d_gain
        + 0.10 * d_slope
        + 0.05 * d_shape
    )
    return float(dist)

def estimate_start_obs_segment(hi_train_all, hi_target, seg_len=START_MATCH_LEN):
    hi_target = np.asarray(hi_target, dtype=float)
    L = int(min(seg_len, len(hi_target)))
    test_seg = hi_target[:L]

    candidates = []
    per_bearing_best = []

    for b in BEARINGS:
        hi = np.asarray(hi_train_all[b], dtype=float)
        max_start = max(START_SEARCH_MIN + 1, len(hi) - L - START_SEARCH_MARGIN)
        best = None
        for s in range(START_SEARCH_MIN, max_start):
            train_seg = hi[s: s + L]
            if len(train_seg) != L:
                continue
            dist = segment_distance(test_seg, train_seg)
            item = (dist, s, b)
            candidates.append(item)
            if best is None or dist < best[0]:
                best = item
        if best is not None:
            per_bearing_best.append(best)

    if not candidates:
        return 0

    candidates = sorted(candidates, key=lambda x: x[0])
    top_global = candidates[: min(8, len(candidates))]
    top_bearing = sorted(per_bearing_best, key=lambda x: x[0])

    selected = top_global + top_bearing
    weights, positions = [], []
    for dist, s, b in selected:
        w = 1.0 / (dist + 1e-6)
        weights.append(w)
        positions.append(s)

    est = int(round(np.average(positions, weights=weights)))

    hi_gain = float(hi_target[-1] - hi_target[0])
    gain_ratio = np.clip((hi_gain - GAIN_MIN_START_TH) / GAIN_MIN_START_SPAN, 0.0, 1.0)
    gain_based_min_start = int(round(gain_ratio * GAIN_MIN_START_MAX))
    est = max(est, gain_based_min_start)

    est = int(np.clip(est, 0, int(MEAN_TRAIN_LIFE)))
    return est

# ── Calibration search ────────────────────────────────────────────────────
def search_calibration(preds_dict, results, label=""):
    best_cf, best_score = 1.0, -np.inf
    for cf in np.arange(0.60, 1.21, 0.02):
        scores = []
        for bid in BEARINGS:
            obs_pts = results[bid]["obs_pts"]
            N = results[bid]["N"]
            preds = [p * cf for p in preds_dict[bid]]
            sc = np.nanmean([competition_score(N - obs, p) for obs, p in zip(obs_pts, preds)])
            scores.append(sc)
        mean_sc = float(np.mean(scores))
        if mean_sc > best_score:
            best_score, best_cf = mean_sc, float(cf)
    print(f"  [{label}] best cf={best_cf:.2f} → {best_score:.4f}")
    return best_cf, best_score

# ── LOOCV ──────────────────────────────────────────────────────────────────
def run_loocv(train_dfs, train_regs, params, sigma, feat_w, grp_w,
              aux_features, aux_w, aux_s):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results = {}
    sc_lgbm_list, sc_lstm_list, sc_ens_list, sc_corr_list = [], [], [], []

    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        print(f"{'='*60}")
        print(f"[LOOCV] Bearing {test_bid} held out")

        # 1. LOO-baseline 계산 (test_bid 제외)
        train_dfs_loo = {b: train_dfs[b] for b in train_bids}
        train_regs_loo = {b: train_regs[b] for b in train_bids}
        train_baseline_loo = compute_train_baseline(train_dfs_loo, train_regs_loo, MAIN_ALL_FEATS)
        train_aux_baseline_loo = compute_train_baseline(train_dfs_loo, train_regs_loo, aux_features)

        # 2. LOO-baseline을 통해 4개 베어링 HI 동적 재계산
        hi_train_loo = {}
        for b in BEARINGS:
            hi_corr, _, _, _ = compute_bearing_hi(
                train_dfs[b], train_regs[b], params, sigma, feat_w, grp_w,
                aux_features, aux_w, aux_s, train_baseline_loo, train_aux_baseline_loo)
            hi_train_loo[b] = hi_corr

        N_test = len(hi_train_loo[test_bid])
        obs_pts = np.arange(SEQ_LENGTH, N_test)

        # 3. LGBM & LSTM-B 학습 (train_bids 기준)
        lgbm_model = train_lgbm(hi_train_loo, train_bids)
        preds_lgbm = predict_lgbm(lgbm_model, hi_train_loo[test_bid], start_obs=0)

        print(f"  LSTM-B training ({len(SEEDS)} seeds)...")
        lstm_models, rul_scale = train_lstm_ensemble(hi_train_loo, train_bids, device)
        preds_lstm = predict_lstm(lstm_models, rul_scale, hi_train_loo[test_bid], start_obs=0, device=device)

        def avg_score(preds):
            return float(np.nanmean([competition_score(N_test - obs, p)
                                     for obs, p in zip(obs_pts, preds)]))
        sc_l = avg_score(preds_lgbm)
        sc_a = avg_score(preds_lstm)

        w_l = float(np.clip(sc_l / (sc_l + sc_a + 1e-12), 0.1, 0.7))
        w_a = 1.0 - w_l
        preds_ens, dyn_w_lgbm, dyn_w_lstm = dynamic_ensemble(preds_lgbm, preds_lstm, w_l, w_a)
        preds_corr, hi_factor = apply_prediction_corrections(preds_ens, hi_train_loo[test_bid][-1])

        sc_e = avg_score(preds_ens)
        sc_c = avg_score(preds_corr)
        print(f"  → LGBM={sc_l:.4f}  LSTM-B={sc_a:.4f}  Ens={sc_e:.4f}  Corr={sc_c:.4f}")

        sc_lgbm_list.append(sc_l)
        sc_lstm_list.append(sc_a)
        sc_ens_list.append(sc_e)
        sc_corr_list.append(sc_c)

        results[test_bid] = {
            "lgbm": list(preds_lgbm), "lstm_b": list(preds_lstm), "ens": list(preds_ens),
            "ens_corr": list(preds_corr), "obs_pts": list(obs_pts), "N": N_test,
            "sc_lgbm": sc_l, "sc_lstm": sc_a, "sc_ens": sc_e, "sc_corr": sc_c,
            "w_lgbm_base": w_l, "w_lstm_base": w_a,
            "w_lgbm_dyn_mean": float(np.mean(dyn_w_lgbm)),
            "w_lstm_dyn_mean": float(np.mean(dyn_w_lstm)),
            "hi_factor": hi_factor,
        }

    print(f"\n{'='*60}")
    print(f"  {'Bearing':>8} | {'LGBM':>8} | {'LSTM-B':>8} | {'Ens':>8} | {'Corr':>8}")
    print(f"  {'-'*58}")
    for b in BEARINGS:
        r = results[b]
        print(f"  {b:>8} | {r['sc_lgbm']:>8.4f} | {r['sc_lstm']:>8.4f} | "
              f"{r['sc_ens']:>8.4f} | {r['sc_corr']:>8.4f}")
    print(f"  {'Mean':>8} | {np.mean(sc_lgbm_list):>8.4f} | {np.mean(sc_lstm_list):>8.4f} | "
          f"{np.mean(sc_ens_list):>8.4f} | {np.mean(sc_corr_list):>8.4f}")

    print("\n  [Calibration search]")
    cf_lstm, sc_lstm_cf = search_calibration(
        {b: results[b]["lstm_b"] for b in BEARINGS}, results, "LSTM-B")
    cf_ens, sc_ens_cf = search_calibration(
        {b: results[b]["ens"] for b in BEARINGS}, results, "Dynamic Ensemble")
    cf_corr, sc_corr_cf = search_calibration(
        {b: results[b]["ens_corr"] for b in BEARINGS}, results, "Corrected Ensemble")

    # Pick best config by LOOCV score after calibration
    candidates = {
        "lstm_b": (cf_lstm, sc_lstm_cf),
        "ens": (cf_ens, sc_ens_cf),
        "ens_corr": (cf_corr, sc_corr_cf),
    }
    best_config = max(candidates, key=lambda k: candidates[k][1])
    best_cf = candidates[best_config][0]
    best_score = candidates[best_config][1]
    print(f"\n  Best config: {best_config} (cf={best_cf:.2f}) -> score={best_score:.4f}")

    # 결과 로그 파일 작성
    with open(OUT_DIR / "loocv_log.txt", "w", encoding="utf-8") as f:
        f.write("SR/0520 — Exp E: A-full v7_advanced RUL Pipeline\n")
        f.write(f"LGBM avg:      {np.mean(sc_lgbm_list):.4f}\n")
        f.write(f"LSTM-B avg:    {np.mean(sc_lstm_list):.4f}\n")
        f.write(f"Ens avg:       {np.mean(sc_ens_list):.4f}\n")
        f.write(f"Corr Ens avg:  {np.mean(sc_corr_list):.4f}\n\n")
        f.write(f"After cf: LSTM-B cf={cf_lstm:.2f}->{sc_lstm_cf:.4f}, "
                f"Ens cf={cf_ens:.2f}->{sc_ens_cf:.4f}, "
                f"Corr cf={cf_corr:.2f}->{sc_corr_cf:.4f}\n")
        f.write(f"Best config: {best_config}, cf={best_cf:.2f}, score={best_score:.4f}\n\n")
        f.write("Fold detail:\n")
        for b in BEARINGS:
            r = results[b]
            f.write(f"  B{b}: LGBM={r['sc_lgbm']:.4f}  LSTM-B={r['sc_lstm']:.4f}  "
                    f"Ens={r['sc_ens']:.4f}  Corr={r['sc_corr']:.4f}\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SR/0520 A-full v7_advanced LOOCV", fontsize=12)
    for i, test_bid in enumerate(BEARINGS):
        ax  = axes.flatten()[i]
        r   = results[test_bid]
        obs = r["obs_pts"]
        ax.plot(obs, [r["N"] - o for o in obs], "k-",  lw=2, label="True RUL")
        ax.plot(obs, r["lgbm"],  "r--", lw=1, alpha=0.7, label="LGBM")
        ax.plot(obs, r["lstm_b"],"g--", lw=1, alpha=0.7, label="LSTM-B")
        ax.plot(obs, r["ens"],   "m-",  lw=1.5, alpha=0.7, label="Dyn Ens")
        ax.plot(obs, r["ens_corr"], "b-",  lw=2, label="Corrected")
        ax.set_title(f"B{test_bid} LGBM={r['sc_lgbm']:.3f} LSTM-B={r['sc_lstm']:.3f} Ens={r['sc_ens']:.3f} Corr={r['sc_corr']:.3f}")
        ax.set_xlabel("Observation cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loocv_predictions.png", dpi=150)
    plt.close()

    return results, best_config, best_cf, best_score

# ── Test inference with Train-baseline corrected HI & start_obs Estimation ──
def run_test_inference(hi_train_all, results, best_config, best_cf,
                       test_dfs, test_regs, params, sigma, feat_w, grp_w,
                       aux_features, aux_w, aux_s,
                       train_baseline, train_aux_baseline):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Test inference — config={best_config}, cf={best_cf:.2f}")
    print(f"  HI mode: Train-baseline + Align start_obs\n")

    # 4개 베어링 전체 HI로 최종 모델 재학습
    lgbm_model = train_lgbm(hi_train_all, BEARINGS)
    print("  LSTM-B training on all 4 bearings...")
    lstm_models, rul_scale = train_lstm_ensemble(hi_train_all, BEARINGS, device)

    w_lgbm = float(np.mean([results[b]["w_lgbm_base"] for b in BEARINGS]))
    w_lstm = float(np.mean([results[b]["w_lstm_base"] for b in BEARINGS]))
    total  = w_lgbm + w_lstm
    w_lgbm, w_lstm = w_lgbm / total, w_lstm / total
    print(f"  Weights: LGBM={w_lgbm:.3f}, LSTM-B={w_lstm:.3f}")

    REF = {
        "0514": {1: 5.05, 2: 5.08, 3: 4.69, 4: 3.43, 5: 7.46, 6: 5.40},
        "v4":   {1: 7.40, 2: 8.58, 3: 5.21, 4: 9.78, 5: 8.31, 6: 6.79},
    }

    summary_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"SR/0520 Test RUL — Exp E A-full v7_advanced  config={best_config}, cf={best_cf:.2f}", fontsize=12)

    for i, tid in enumerate(TEST_IDS):
        feat_df    = test_dfs[tid]
        regime_arr = test_regs[tid]

        # 1. Train-baseline 기준 HI 계산
        hi_corr, hi_main, hi_aux, gate = compute_bearing_hi(
            feat_df, regime_arr, params, sigma, feat_w, grp_w,
            aux_features, aux_w, aux_s, train_baseline, train_aux_baseline)

        # 2. start_obs 추정 (Segment matching)
        start_obs = estimate_start_obs_segment(hi_train_all, hi_corr)
        obs_pts = np.arange(SEQ_LENGTH, len(hi_corr))
        print(f"  [Test{tid}] HI: [{hi_corr[0]:.4f} → {hi_corr[-1]:.4f}]  "
              f"Segment start_obs: {start_obs} cycles (obs_frac starts at {start_obs/MEAN_TRAIN_LIFE:.3f})")

        # 3. RUL 예측
        preds_lstm  = predict_lstm(lstm_models, rul_scale, hi_corr, start_obs=start_obs, device=device)
        preds_lgbm  = predict_lgbm(lgbm_model, hi_corr, start_obs=start_obs)
        preds_ens, dyn_w_lgbm, dyn_w_lstm = dynamic_ensemble(preds_lgbm, preds_lstm, w_lgbm, w_lstm)
        preds_corr, hi_factor = apply_prediction_corrections(preds_ens, hi_corr[-1])

        if best_config == "lstm_b":
            preds_selected = preds_lstm
        elif best_config == "ens":
            preds_selected = preds_ens
        else:
            preds_selected = preds_corr

        preds_final = np.maximum(preds_selected * best_cf, MIN_RUL_CYCLES)
        final_cyc = float(preds_final[-1])
        final_hr = final_cyc * INTERVAL_SEC / 3600

        # Save result CSV
        pd.DataFrame({
            "obs_cycle": obs_pts,
            "rul_lgbm": preds_lgbm,
            "rul_lstm_b": preds_lstm,
            "rul_dyn_ens": preds_ens,
            "rul_corr": preds_corr,
            "rul_final": preds_final,
            "rul_hours": preds_final * INTERVAL_SEC / 3600,
            "dyn_w_lgbm": dyn_w_lgbm,
            "dyn_w_lstm": dyn_w_lstm,
        }).to_csv(OUT_DIR / f"Test{tid}_RUL.csv", index=False)

        summary_rows.append({
            "test_id": tid,
            "base0514": REF["0514"][tid],
            "v4_hours": REF["v4"][tid],
            "rul_hours": final_hr,
            "estimated_start_obs": start_obs,
            "hi_corr_start": hi_corr[0],
            "hi_corr_end": hi_corr[-1]
        })

        # Plot prediction
        ax  = axes.flatten()[i]
        ax2 = ax.twinx()
        ax.plot(obs_pts, preds_lgbm,  "r--", lw=1, alpha=0.6, label="LGBM")
        ax.plot(obs_pts, preds_lstm,  "g--", lw=1, alpha=0.6, label="LSTM-B")
        ax.plot(obs_pts, preds_final, "m-",  lw=2,             label="Final")
        ax2.plot(obs_pts, hi_corr[SEQ_LENGTH:], "b-",  lw=1.5,           label="HI corr")
        ax2.set_ylabel("HI", color="b"); ax2.tick_params(axis="y", labelcolor="b")
        ax2.set_ylim(-0.05, 1.05)
        ax.set_title(
            f"Test{tid} (start_obs={start_obs}) "
            f"RUL={final_hr:.1f}hr  (v4={REF['v4'][tid]:.1f})")
        ax.set_xlabel("Observation cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_predictions.png", dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary.csv", index=False)

    print(f"\n  {'Test':>5} | {'0514':>6} | {'v4':>6} | {'v7_adv':>8} | {'start_obs':>9} | {'hi_start':>10} | {'hi_end':>8}")
    print(f"  {'-'*80}")
    for row in summary_rows:
        print(f"  {row['test_id']:>5} | {row['base0514']:>6.2f} | "
              f"{row['v4_hours']:>6.2f} | {row['rul_hours']:>8.2f} | "
              f"{row['estimated_start_obs']:>9} | {row['hi_corr_start']:>10.4f} | "
              f"{row['hi_corr_end']:.4f}")

    with open(OUT_DIR / "loocv_log.txt", "a", encoding="utf-8") as f:
        f.write("\nTest inference (Train-baseline HI & Align start_obs):\n")
        f.write(df_sum.to_string(index=False) + "\n")

    print(f"\n[완료] 결과 저장 위치: {OUT_DIR}")

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("SR/0520 — Exp E: A-full v7_advanced RUL Pipeline")
    print("=" * 60)

    # 1. 파라미터 로드 (Saved TH CSVs)
    print("\nLoading TH saved parameters...")
    params, sigma, feat_w, grp_w, aux_w, aux_s, aux_features = load_th_params()
    print(f"  main_offset={params['main_offset']:.5f}, main_tau={params['main_tau']:.4f}")
    print(f"  aux_features: {aux_features}")

    # 2. Features 및 Regime 로드
    print("\nLoading features and regime labels...")
    train_dfs   = load_train_features()
    test_dfs    = load_test_features()
    train_regs  = load_train_regimes()
    test_regs   = load_test_regimes()

    for b in BEARINGS:
        print(f"  Bearing{b}: feat={len(train_dfs[b])}, regime={len(train_regs[b])}")
    for t in TEST_IDS:
        print(f"  Test{t}:    feat={len(test_dfs[t])}, regime={len(test_regs[t])}")

    # 3. Train 전체 기준의 baseline 및 HI 궤적 사전 계산
    print("\nComputing global Train-baseline (4 bearings)...")
    train_baseline_all     = compute_train_baseline(train_dfs, train_regs, MAIN_ALL_FEATS)
    train_aux_baseline_all = compute_train_baseline(train_dfs, train_regs, aux_features)

    print("\nRecomputing 4 Train bearings' HI using global Train-baseline...")
    hi_train_all = {}
    for b in BEARINGS:
        hi_corr, _, _, _ = compute_bearing_hi(
            train_dfs[b], train_regs[b], params, sigma, feat_w, grp_w,
            aux_features, aux_w, aux_s, train_baseline_all, train_aux_baseline_all)
        hi_train_all[b] = hi_corr
        print(f"  Bearing{b}: n={len(hi_corr)}, range=[{hi_corr.min():.3f}, {hi_corr.max():.3f}]")

    # 4. LOOCV 실행
    print("\nStarting LOOCV...")
    results, best_config, best_cf, loocv_score = run_loocv(
        train_dfs, train_regs, params, sigma, feat_w, grp_w,
        aux_features, aux_w, aux_s)

    # 5. Test Inference 실행
    print("\nStarting Test Inference...")
    run_test_inference(
        hi_train_all, results, best_config, best_cf,
        test_dfs, test_regs, params, sigma, feat_w, grp_w,
        aux_features, aux_w, aux_s,
        train_baseline_all, train_aux_baseline_all)
