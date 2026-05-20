"""
SR/0518 v6 — TH v7_4_2 HI + Solution A: Train-baseline HI for test bearings
=============================================================================
v4/v5 대비 변경:
  Test inference 시 HI를 Train-baseline으로 재계산.

  핵심 문제 (v4까지):
    TH v7_4_2의 own-baseline 방식:
      z = (x - own_baseline) / sigma
      own_baseline = test 베어링 자신의 초반 15% 평균
    → HI는 항상 ~0에서 시작: 관측 시작 전 쌓인 열화량 소실

  Solution A:
    Test 베어링의 raw_score를 Train-baseline으로 재계산:
      z = (x - train_normal_baseline) / sigma
      train_normal_baseline = Train 4개 베어링 초반 15% 평균
    → 관측 시작 시점부터 Train 정상 상태 대비 열화 수준 반영

  LOOCV: TH 사전계산 HI(own-baseline) 그대로 사용 → 점수 v4와 동일
  Test inference: Train-baseline으로 재계산한 HI 사용

v4 결과 (baseline):
  LOOCV: LGBM=0.4102  LSTM-A=0.4261  Ensemble=0.5004
  Test v4:  T1=7.40 T2=8.58 T3=5.21 T4=9.78 T5=8.31 T6=6.79 hr

코드 출처:
  HI 계산 유틸리티 함수: TH/FI/07_v7/code/hi_v7_2_1_rpm_baseline_relative.py
                         TH/FI/07_v7/code/hi_v7_4_2_conditional_aux_boost.py
  를 User/SR/0518에 복사 후 Train-baseline 방식으로 수정.
"""

import numpy as np
import pandas as pd
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
OUT_DIR   = BASE / "User/SR/0518/rul/output/th742_v6"
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

# ── HI computation constants (copied from TH v7_2_1 / v7_4_2) ─────────────
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

# ── HI utilities (copied from TH v7_4_2) ─────────────────────────────────
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
    """v7_4_2 version — includes EMA+MA smoothing on HI."""
    x  = np.maximum(np.asarray(raw, dtype=float) - offset, 0.0)
    hi = 1.0 - np.exp(-x / max(tau, EPS))
    hi = ema_smooth(hi)
    hi = moving_average(hi)
    return np.clip(hi, 0.0, 1.0)

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

    # which aux features are available
    aux_features = [f for f in AUX_FEATURE_CANDIDATES if f in aux_w]

    return params, sigma, feat_w, grp_w, aux_w, aux_s, aux_features

# ── Load features and regime labels ───────────────────────────────────────
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

# ── Train-baseline computation (Solution A) ───────────────────────────────
def compute_train_baseline(train_dfs, train_regimes, features):
    """
    Train 4개 베어링의 초반 NORMAL_RATIO 구간에서 regime별 feature 평균 계산.
    이 값을 Test 베어링 z-score의 baseline으로 사용.
    """
    baseline = {}
    for feat in features:
        # Global fallback (no regime split)
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
                # first n0 obs that belong to this regime
                mask = (np.arange(min(len(df), len(r))) < n0) & (r[:min(len(df), len(r))] == regime)
                idx  = np.where(mask)[0]
                if len(idx) >= 1 and feat in df.columns:
                    vals.extend(df.iloc[idx][feat].values.astype(float).tolist())
            baseline[(regime, feat)] = float(np.nanmean(vals)) if len(vals) >= 2 else global_mean

    return baseline

# ── Main branch raw_score with Train-baseline ──────────────────────────────
def make_main_raw_train_baseline(feat_df, regime_arr, sigma, feat_w, grp_w, train_baseline):
    """
    v7_2_1의 make_raw_score와 동일하되, own-baseline 대신 train_baseline 사용.
    (copied & modified from TH/FI/07_v7/code/hi_v7_2_1_rpm_baseline_relative.py)
    """
    n   = len(feat_df)
    raw = np.zeros(n, dtype=float)

    for group, feats in MAIN_FEATURE_GROUPS.items():
        group_score = np.zeros(n, dtype=float)

        # Feature weights within this group (normalized)
        grp_feat_total = sum(feat_w.get(f, 0.0) for f in feats)
        grp_feat_total = max(grp_feat_total, EPS)

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
                base = train_baseline.get((regime, feat),
                        train_baseline.get((0, feat), 0.0))
                sig  = sigma.get((regime, feat), EPS)
                z[idx] = (x[idx] - base) / max(sig, EPS)

            z = robust_transform_z(z)
            group_score += w * z

        group_score = smooth_score(group_score)
        raw += grp_w.get(group, 0.5) * group_score

    return raw

# ── Aux branch raw_score with Train-baseline ───────────────────────────────
def make_aux_raw_train_baseline(feat_df, regime_arr, aux_features, aux_w, aux_s,
                                 train_aux_baseline):
    """
    v7_4_2의 make_aux_raw와 동일하되, own-baseline 대신 train_aux_baseline 사용.
    (copied & modified from TH/FI/07_v7/code/hi_v7_4_2_conditional_aux_boost.py)
    """
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
            base = train_aux_baseline.get((regime, feat),
                    train_aux_baseline.get((0, feat), 0.0))
            sig  = aux_s.get((regime, feat), EPS)
            z[idx] = direction * (x[idx] - base) / max(sig, EPS)

        z = robust_transform_z(z)
        raw += w * z

    return smooth_score(raw)

# ── Gate + fusion (copied from TH v7_4_2, unchanged) ─────────────────────
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

# ── Full corrected HI for one test bearing ────────────────────────────────
def compute_test_hi_corrected(feat_df, regime_arr, params, sigma, feat_w, grp_w,
                               aux_features, aux_w, aux_s,
                               train_baseline, train_aux_baseline):
    """
    Test 베어링에 대해 Train-baseline으로 HI_v7_4_2를 재계산.
    """
    n = min(len(feat_df), len(regime_arr))
    feat_df   = feat_df.iloc[:n].copy()
    regime_arr = regime_arr[:n]

    # Main branch
    main_raw = make_main_raw_train_baseline(
        feat_df, regime_arr, sigma, feat_w, grp_w, train_baseline)
    hi_main  = exp_calibrate(main_raw, params["main_offset"], params["main_tau"])

    # Aux branch
    aux_raw  = make_aux_raw_train_baseline(
        feat_df, regime_arr, aux_features, aux_w, aux_s, train_aux_baseline)
    hi_aux   = exp_calibrate(aux_raw, params["aux_offset"], params["aux_tau"])

    # Gate + fusion
    gate, aux_boost = compute_gate(hi_main, hi_aux)
    hi_final = fuse_hi(hi_main, hi_aux, gate, aux_boost)

    return hi_final, hi_main, hi_aux, gate

# ── HI Loading ────────────────────────────────────────────────────────────
def load_train_hi():
    return {b: pd.read_csv(TH_HI_DIR / f"v7_4_2_Bearing{b}_HI.csv")["HI_v7_4_2"].values.astype(float)
            for b in BEARINGS}

def load_test_hi():
    return {t: pd.read_csv(TH_HI_DIR / f"v7_4_2_Test{t}_HI.csv")["HI_v7_4_2"].values.astype(float)
            for t in TEST_IDS}

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

# ── LightGBM features ─────────────────────────────────────────────────────
def make_lgbm_features(hi_array, start_obs=0):
    features, targets = [], []
    N = len(hi_array)
    for i in range(SEQ_LENGTH, N):
        window = hi_array[i - SEQ_LENGTH: i]
        w_min, w_max = window.min(), window.max()
        window_norm  = (window - w_min) / (w_max - w_min + 1e-8)
        slope_norm   = float(np.polyfit(np.arange(SEQ_LENGTH), window_norm, 1)[0])
        slope_raw    = float(np.polyfit(np.arange(SEQ_LENGTH), window, 1)[0])
        obs_frac     = float(np.clip((start_obs + i) / MEAN_TRAIN_LIFE, 0.0, 2.0))
        feats = (list(window_norm) +
                 [slope_norm, float(window[-1]), float(window.mean()),
                  float(window.max()), slope_raw, float(window[-1] - window[0]), obs_frac])
        features.append(feats)
        targets.append(float(N - i))
    return np.array(features), np.array(targets)

def lgbm_asymmetric_obj(y_pred, dataset):
    y_true = dataset.get_label()
    diff   = y_true - y_pred
    weight = np.where(diff < 0, 2.5, 1.0)
    return -diff * weight, np.ones_like(diff) * weight

def train_lgbm(hi_dict, train_bids):
    X_all, y_all = [], []
    for b in train_bids:
        x, y = make_lgbm_features(hi_dict[b], start_obs=0)
        X_all.append(x); y_all.append(y)
    dtrain = lgb.Dataset(np.concatenate(X_all), label=np.concatenate(y_all))
    return lgb.train({"num_leaves": 15, "learning_rate": 0.05,
                      "min_child_samples": 5, "verbose": -1,
                      "objective": lgbm_asymmetric_obj},
                     dtrain, num_boost_round=200)

def predict_lgbm(model, hi_array, start_obs=0):
    X, _ = make_lgbm_features(hi_array, start_obs=start_obs)
    return np.maximum(model.predict(X), 0.0)

# ── LSTM-A ────────────────────────────────────────────────────────────────
N_FEAT = 2

def make_seqs(hi_arr, rul_arr, start_obs=0):
    X, y = [], []
    n = len(hi_arr)
    for i in range(n - SEQ_LENGTH):
        window = hi_arr[i:i + SEQ_LENGTH].copy()
        w_min, w_max = window.min(), window.max()
        window_norm = (window - w_min) / (w_max - w_min + 1e-8)
        obs_frac = np.clip(
            (start_obs + i + np.arange(SEQ_LENGTH)) / MEAN_TRAIN_LIFE, 0.0, 2.0)
        X.append(np.stack([window_norm, obs_frac], axis=1))
        y.append(float(rul_arr[i + SEQ_LENGTH]))
    return np.array(X), np.array(y)

class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_FEAT, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)

def train_lstm(X_train, y_train, rul_scale, seed, device):
    torch.manual_seed(seed)
    y_norm = y_train / rul_scale
    Xt  = torch.tensor(X_train, dtype=torch.float32)
    yt  = torch.tensor(y_norm,  dtype=torch.float32)
    n_val  = max(1, int(len(Xt) * 0.1))
    tr_dl  = DataLoader(TensorDataset(Xt[:-n_val], yt[:-n_val]), batch_size=64, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xt[-n_val:], yt[-n_val:]), batch_size=64)
    model  = LSTMRegressor().to(device)
    opt, crit = torch.optim.Adam(model.parameters(), lr=1e-3), nn.MSELoss()
    best_val, patience, best_state = np.inf, 0, None
    for _ in range(200):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad(); crit(model(xb.to(device)), yb.to(device)).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([crit(model(xb.to(device)), yb.to(device)).item()
                          for xb, yb in val_dl])
        if vl < best_val:
            best_val, patience = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20: break
    model.load_state_dict(best_state)
    return model

def train_lstm_ensemble(hi_dict, train_bids, device):
    X_all, y_all = [], []
    for b in train_bids:
        X, y = make_seqs(hi_dict[b], rul_labels(len(hi_dict[b]), b))
        X_all.append(X); y_all.append(y)
    X_train   = np.concatenate(X_all)
    y_train   = np.concatenate(y_all)
    rul_scale = float(y_train.max())
    models    = [train_lstm(X_train, y_train, rul_scale, s, device) for s in SEEDS]
    return models, rul_scale

def predict_lstm(models, rul_scale, hi_arr, start_obs, device):
    X, _ = make_seqs(hi_arr, np.zeros(len(hi_arr)), start_obs)
    Xt   = torch.tensor(X, dtype=torch.float32).to(device)
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0))
    return np.median(preds, axis=0)

# ── Calibration search ────────────────────────────────────────────────────
def search_calibration(preds_dict, results, label=""):
    best_cf, best_score = 1.0, -np.inf
    for cf in np.arange(0.60, 1.21, 0.02):
        scores = []
        for bid in BEARINGS:
            obs_pts = results[bid]["obs_pts"]
            N       = results[bid]["N"]
            preds   = [p * cf for p in preds_dict[bid]]
            sc = np.nanmean([competition_score(N - obs, p)
                             for obs, p in zip(obs_pts, preds)])
            scores.append(sc)
        mean_sc = float(np.mean(scores))
        if mean_sc > best_score:
            best_score, best_cf = mean_sc, float(cf)
    print(f"  [{label}] best cf={best_cf:.2f} → {best_score:.4f}")
    return best_cf, best_score

# ── LOOCV (unchanged from v4 — own-baseline TH HI) ───────────────────────
def run_loocv(hi_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results = {}
    sc_lgbm_list, sc_lstm_list = [], []

    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        print(f"{'='*55}")
        print(f"[LOOCV] Bearing {test_bid} held out")

        N_test     = len(hi_train[test_bid])
        lgbm_model = train_lgbm(hi_train, train_bids)
        preds_lgbm = predict_lgbm(lgbm_model, hi_train[test_bid], start_obs=0)

        print(f"  LSTM training ({len(SEEDS)} seeds)...")
        lstm_models, rul_scale = train_lstm_ensemble(hi_train, train_bids, device)
        preds_lstm = predict_lstm(lstm_models, rul_scale,
                                  hi_train[test_bid], start_obs=0, device=device)

        obs_pts = np.arange(SEQ_LENGTH, N_test)
        def avg_score(preds):
            return float(np.nanmean([competition_score(N_test - obs, p)
                                     for obs, p in zip(obs_pts, preds)]))
        sc_l = avg_score(preds_lgbm)
        sc_a = avg_score(preds_lstm)
        print(f"  → LGBM: {sc_l:.4f}  LSTM-A: {sc_a:.4f}")
        sc_lgbm_list.append(sc_l)
        sc_lstm_list.append(sc_a)

        w_l = float(np.clip(sc_l / (sc_l + sc_a + 1e-12), 0.1, 0.7))
        w_a = 1.0 - w_l
        preds_ens = [w_l * l + w_a * a for l, a in zip(preds_lgbm, preds_lstm)]

        results[test_bid] = {
            "lgbm": list(preds_lgbm), "lstm_a": list(preds_lstm), "ens": preds_ens,
            "obs_pts": list(obs_pts), "N": N_test,
            "sc_lgbm": sc_l, "sc_lstm": sc_a, "w_lgbm": w_l, "w_lstm": w_a,
        }

    ens_scores = []
    print(f"\n{'='*55}")
    print(f"  {'Bearing':>8} | {'LGBM':>8} | {'LSTM-A':>8} | {'Ensemble':>8}")
    print(f"  {'-'*50}")
    for test_bid in BEARINGS:
        r  = results[test_bid]
        sc = float(np.nanmean([competition_score(r["N"] - obs, p)
                               for obs, p in zip(r["obs_pts"], r["ens"])]))
        ens_scores.append(sc)
        results[test_bid]["sc_ens"] = sc
        print(f"  {test_bid:>8} | {r['sc_lgbm']:>8.4f} | {r['sc_lstm']:>8.4f} | {sc:>8.4f}")
    print(f"  {'Mean':>8} | {np.mean(sc_lgbm_list):>8.4f} | "
          f"{np.mean(sc_lstm_list):>8.4f} | {np.mean(ens_scores):>8.4f}")

    print("\n  [Calibration search]")
    cf_lstm, sc_lstm_cf = search_calibration(
        {b: results[b]["lstm_a"] for b in BEARINGS}, results, "LSTM-A")
    cf_ens, sc_ens_cf = search_calibration(
        {b: results[b]["ens"] for b in BEARINGS}, results, "Ensemble")

    V4 = {"lgbm": 0.4102, "lstm": 0.4261, "ens": 0.5004}
    print(f"\n  v4 baseline: LGBM={V4['lgbm']:.4f}  LSTM-A={V4['lstm']:.4f}  Ens={V4['ens']:.4f}")
    print(f"  v6 LOOCV:    LGBM={np.mean(sc_lgbm_list):.4f}  "
          f"LSTM-A={np.mean(sc_lstm_list):.4f}  Ens={np.mean(ens_scores):.4f}")

    with open(OUT_DIR / "loocv_log.txt", "w", encoding="utf-8") as f:
        f.write("SR/0518 v6 — TH v7_4_2 HI + Solution A (Train-baseline for test)\n")
        f.write(f"LOOCV uses own-baseline HI from TH (same as v4)\n\n")
        f.write(f"LGBM avg:   {np.mean(sc_lgbm_list):.4f}\n")
        f.write(f"LSTM-A avg: {np.mean(sc_lstm_list):.4f}\n")
        f.write(f"Ensemble:   {np.mean(ens_scores):.4f}\n")
        f.write(f"cf: LSTM-A cf={cf_lstm:.2f}→{sc_lstm_cf:.4f}  "
                f"Ens cf={cf_ens:.2f}→{sc_ens_cf:.4f}\n\n")
        for b in BEARINGS:
            r = results[b]
            f.write(f"  B{b}: LGBM={r['sc_lgbm']:.4f}  LSTM-A={r['sc_lstm']:.4f}  "
                    f"Ens={r['sc_ens']:.4f}\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SR/0518 v6 LOOCV — TH v7_4_2 HI (own-baseline, unchanged)", fontsize=11)
    for i, test_bid in enumerate(BEARINGS):
        ax  = axes.flatten()[i]
        r   = results[test_bid]
        obs = r["obs_pts"]
        ax.plot(obs, [r["N"] - o for o in obs], "k-",  lw=2, label="True RUL")
        ax.plot(obs, r["lgbm"],  "r--", lw=1, alpha=0.7, label="LGBM")
        ax.plot(obs, r["lstm_a"],"g--", lw=1, alpha=0.7, label="LSTM-A")
        ax.plot(obs, r["ens"],   "m-",  lw=2,             label="Ensemble")
        ax.set_title(f"B{test_bid}  LGBM={r['sc_lgbm']:.3f}  "
                     f"LSTM-A={r['sc_lstm']:.3f}  Ens={r['sc_ens']:.3f}")
        ax.set_xlabel("Observation cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loocv_predictions.png", dpi=150)
    plt.close()

    best_config = "lstm_a" if sc_lstm_cf >= sc_ens_cf else "ens"
    best_cf     = cf_lstm  if best_config == "lstm_a" else cf_ens
    print(f"\n  Best config: {'LSTM-A' if best_config == 'lstm_a' else 'LGBM+LSTM-A'} "
          f"(cf={best_cf:.2f})")
    return results, best_config, best_cf, np.mean(ens_scores)

# ── Test inference with Train-baseline corrected HI ───────────────────────
def run_test_inference(hi_train, hi_test_own, results, best_config, best_cf,
                       test_feat_dfs, test_regimes, params, sigma, feat_w, grp_w,
                       aux_features, aux_w, aux_s,
                       train_baseline, train_aux_baseline):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Test inference — "
          f"{'LSTM-A' if best_config == 'lstm_a' else 'LGBM+LSTM-A'}, cf={best_cf:.2f}")
    print(f"  HI mode: Train-baseline (Solution A)\n")

    lgbm_model = train_lgbm(hi_train, BEARINGS)
    print("  LSTM training on all 4 bearings...")
    lstm_models, rul_scale = train_lstm_ensemble(hi_train, BEARINGS, device)

    if best_config == "ens":
        w_lgbm = float(np.mean([results[b]["w_lgbm"] for b in BEARINGS]))
        w_lstm = float(np.mean([results[b]["w_lstm"]  for b in BEARINGS]))
        total  = w_lgbm + w_lstm
        w_lgbm, w_lstm = w_lgbm / total, w_lstm / total
        print(f"  Weights: LGBM={w_lgbm:.3f}, LSTM-A={w_lstm:.3f}")

    REF = {
        "0514": {1: 5.05, 2: 5.08, 3: 4.69, 4: 3.43, 5: 7.46, 6: 5.40},
        "v4":   {1: 7.40, 2: 8.58, 3: 5.21, 4: 9.78, 5: 8.31, 6: 6.79},
    }

    summary_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"SR/0518 v6 Test RUL — Train-baseline HI (Solution A)  cf={best_cf:.2f}",
        fontsize=11)

    for i, tid in enumerate(TEST_IDS):
        feat_df    = test_feat_dfs[tid]
        regime_arr = test_regimes[tid]
        hi_own     = hi_test_own[tid]

        # ── Compute corrected HI (Train-baseline) ──────────────────────
        hi_corr, hi_main, hi_aux, gate = compute_test_hi_corrected(
            feat_df, regime_arr, params, sigma, feat_w, grp_w,
            aux_features, aux_w, aux_s, train_baseline, train_aux_baseline)

        obs_pts = np.arange(SEQ_LENGTH, len(hi_corr))
        print(f"  [Test{tid}]  own HI:  [{hi_own[0]:.4f} → {hi_own[-1]:.4f}]")
        print(f"             corr HI: [{hi_corr[0]:.4f} → {hi_corr[-1]:.4f}]  gate={gate:.3f}")

        # obs_fraction: start_obs=0 since corr HI already reflects absolute level
        preds_lstm  = predict_lstm(lstm_models, rul_scale, hi_corr,
                                   start_obs=0, device=device)
        preds_lgbm  = predict_lgbm(lgbm_model, hi_corr, start_obs=0)
        preds_raw   = (w_lgbm * preds_lgbm + w_lstm * preds_lstm
                       if best_config == "ens" else preds_lstm)
        preds_final = preds_raw * best_cf

        final_cyc = float(preds_final[-1])
        final_hr  = final_cyc * INTERVAL_SEC / 3600
        print(f"  → {final_hr:.2f}hr  (v4={REF['v4'][tid]:.2f}, 0514={REF['0514'][tid]:.2f})\n")

        pd.DataFrame({
            "obs_cycle":    obs_pts,
            "hi_own":       hi_own[SEQ_LENGTH:],
            "hi_corrected": hi_corr[SEQ_LENGTH:],
            "hi_main":      hi_main[SEQ_LENGTH:],
            "hi_aux":       hi_aux[SEQ_LENGTH:],
            "rul_lgbm":     preds_lgbm,
            "rul_lstm":     preds_lstm,
            "rul_final":    preds_final,
            "rul_hours":    preds_final * INTERVAL_SEC / 3600,
        }).to_csv(OUT_DIR / f"Test{tid}_RUL.csv", index=False)

        summary_rows.append({
            "test_id":       tid,
            "hi_own_start":  round(float(hi_own[0]),  4),
            "hi_own_end":    round(float(hi_own[-1]), 4),
            "hi_corr_start": round(float(hi_corr[0]),  4),
            "hi_corr_end":   round(float(hi_corr[-1]), 4),
            "gate":          round(gate, 3),
            "rul_hours":     round(final_hr, 2),
            "v4_hours":      REF["v4"][tid],
            "base0514":      REF["0514"][tid],
        })

        ax  = axes.flatten()[i]
        ax2 = ax.twinx()
        ax.plot(obs_pts, preds_lgbm,  "r--", lw=1, alpha=0.6, label="LGBM")
        ax.plot(obs_pts, preds_lstm,  "g--", lw=1, alpha=0.6, label="LSTM-A")
        ax.plot(obs_pts, preds_final, "m-",  lw=2,             label="Final")
        ax2.plot(obs_pts, hi_own[SEQ_LENGTH:],  "b:",  lw=1, alpha=0.5, label="HI own")
        ax2.plot(obs_pts, hi_corr[SEQ_LENGTH:], "b-",  lw=1.5,           label="HI corr")
        ax2.set_ylabel("HI", color="b"); ax2.tick_params(axis="y", labelcolor="b")
        ax.set_title(
            f"Test{tid}  [{hi_own[0]:.3f}→{hi_own[-1]:.3f}] "
            f"corr=[{hi_corr[0]:.3f}→{hi_corr[-1]:.3f}]  "
            f"RUL={final_hr:.1f}hr  (v4={REF['v4'][tid]:.1f})")
        ax.set_xlabel("Observation cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.4)
        ax2.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_predictions.png", dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary.csv", index=False)

    print(f"\n  {'Test':>5} | {'0514':>6} | {'v4':>6} | {'v6':>6} | own_start | corr_start | corr_end")
    print(f"  {'-'*72}")
    for row in summary_rows:
        print(f"  {row['test_id']:>5} | {row['base0514']:>6.2f} | "
              f"{row['v4_hours']:>6.2f} | {row['rul_hours']:>6.2f} | "
              f"{row['hi_own_start']:>9.4f} | {row['hi_corr_start']:>10.4f} | "
              f"{row['hi_corr_end']:.4f}")

    with open(OUT_DIR / "loocv_log.txt", "a", encoding="utf-8") as f:
        f.write("\nTest inference (Train-baseline HI):\n")
        f.write(df_sum.to_string(index=False) + "\n")

    print(f"\n[Done] {OUT_DIR}")

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("SR/0518 v6 — Solution A: Train-baseline HI for test bearings")
    print("=" * 60)

    # Load TH params (saved CSVs — no code from TH modified, data only)
    print("\nLoading TH saved parameters...")
    params, sigma, feat_w, grp_w, aux_w, aux_s, aux_features = load_th_params()
    print(f"  main_offset={params['main_offset']:.5f}, main_tau={params['main_tau']:.4f}")
    print(f"  aux_offset={params['aux_offset']:.5f},  aux_tau={params['aux_tau']:.4f}")
    print(f"  aux_features: {aux_features}")

    # Load features and regime labels
    print("\nLoading features and regime labels...")
    train_dfs   = load_train_features()
    test_dfs    = load_test_features()
    train_regs  = load_train_regimes()
    test_regs   = load_test_regimes()

    for b in BEARINGS:
        print(f"  Bearing{b}: feat={len(train_dfs[b])}, regime={len(train_regs[b])}")
    for t in TEST_IDS:
        print(f"  Test{t}:    feat={len(test_dfs[t])}, regime={len(test_regs[t])}")

    # Compute Train-baseline for main and aux features
    print("\nComputing Train-baseline (main + aux)...")
    train_baseline     = compute_train_baseline(train_dfs, train_regs, MAIN_ALL_FEATS)
    train_aux_baseline = compute_train_baseline(train_dfs, train_regs, aux_features)

    print("  Main baseline (regime=0):  " +
          "  ".join(f"{f}={train_baseline[(0,f)]:.5f}" for f in MAIN_ALL_FEATS))
    print("  Aux  baseline (regime=0):  " +
          "  ".join(f"{f}={train_aux_baseline.get((0,f), 0):.5f}" for f in aux_features))

    # Preview corrected HI for test bearings (before LOOCV)
    print("\nTest HI preview (Train-baseline vs own-baseline):")
    for t in TEST_IDS:
        n = min(len(test_dfs[t]), len(test_regs[t]))
        hi_corr, hi_main, hi_aux, gate = compute_test_hi_corrected(
            test_dfs[t].iloc[:n], test_regs[t][:n],
            params, sigma, feat_w, grp_w,
            aux_features, aux_w, aux_s,
            train_baseline, train_aux_baseline)
        hi_own = pd.read_csv(TH_HI_DIR / f"v7_4_2_Test{t}_HI.csv")["HI_v7_4_2"].values
        print(f"  Test{t}: own=[{hi_own[0]:.4f}→{hi_own[-1]:.4f}]  "
              f"corr=[{hi_corr[0]:.4f}→{hi_corr[-1]:.4f}]  "
              f"gate={gate:.3f}")

    # Load TH own-baseline HI for LOOCV (unchanged)
    print("\nLoading TH v7_4_2 HI (own-baseline) for LOOCV...")
    hi_train = load_train_hi()
    hi_test_own = load_test_hi()
    for b in BEARINGS:
        hi = hi_train[b]
        print(f"  Bearing{b}: n={len(hi)}, range=[{hi.min():.3f}, {hi.max():.3f}]")

    print()
    results, best_config, best_cf, loocv_ens = run_loocv(hi_train)

    run_test_inference(
        hi_train, hi_test_own, results, best_config, best_cf,
        test_dfs, test_regs, params, sigma, feat_w, grp_w,
        aux_features, aux_w, aux_s,
        train_baseline, train_aux_baseline)
