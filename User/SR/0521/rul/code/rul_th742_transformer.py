"""
User/SR/0521 — A-full Baseline & Align start_obs Bearing RUL Pipeline (Transformer version)
========================================================================================
작업자: SR/0521
핵심 접근법:
  1. SC 팀원의 Transformer 아키텍처를 도입하여 시계열 특징량의 잠재적 표현력(B1=0.61, B2=0.63 등)을 획득.
  2. LOOCV 시 dynamic LOO-baseline recalculation 구조를 적용하여 스케일 차이에 따른 외삽 붕괴를 방지.
  3. Test Inference 시 Train-baseline을 기준으로 HI 및 start_obs를 절대적 물리 궤적에 정렬.
  4. LightGBM과 Transformer의 앙상블을 수행하여 최적의 RUL 예측 신뢰성 확보.
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

# SR/0521 출력 디렉토리
OUT_DIR   = BASE / "User/SR/0521/rul/output/th742_transformer"
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

# ── HI utilities ─────────────────────────────────────────────────────────
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

# ── Main branch raw_score with baseline ──────────────────────────────
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

# ── Aux branch raw_score with baseline ───────────────────────────────
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

# ── Gate + fusion ─────────────────────
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

# ── Dynamic HI Calculator using external baseline ────────────────────────────────
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

# ── Transformer-A (Adapted for 2D inputs and LOOCV Recalculation) ──────────
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

class TransformerRegressor(nn.Module):
    def __init__(self, n_feat=N_FEAT, seq_len=SEQ_LENGTH, hidden=64, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.pos = nn.Parameter(torch.randn(seq_len, hidden) * 0.02)
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden*2,
                                         dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.fc   = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(32, 1))
    def forward(self, x):
        # x: (B, seq_len, n_feat)
        z = self.proj(x) + self.pos
        z = self.enc(z)
        # Global mean pooling over sequence length
        return self.fc(z.mean(dim=1)).squeeze(-1)

def train_transformer(X_train, y_train, rul_scale, seed, device):
    torch.manual_seed(seed)
    y_norm = y_train / rul_scale
    Xt  = torch.tensor(X_train, dtype=torch.float32)
    yt  = torch.tensor(y_norm,  dtype=torch.float32)
    n_val  = max(1, int(len(Xt) * 0.1))
    tr_dl  = DataLoader(TensorDataset(Xt[:-n_val], yt[:-n_val]), batch_size=64, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xt[-n_val:], yt[-n_val:]), batch_size=64)
    model  = TransformerRegressor().to(device)
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

def train_transformer_ensemble(hi_dict, train_bids, device):
    X_all, y_all = [], []
    for b in train_bids:
        X, y = make_seqs(hi_dict[b], rul_labels(len(hi_dict[b]), b))
        X_all.append(X); y_all.append(y)
    X_train   = np.concatenate(X_all)
    y_train   = np.concatenate(y_all)
    rul_scale = float(y_train.max())
    models    = [train_transformer(X_train, y_train, rul_scale, s, device) for s in SEEDS]
    return models, rul_scale

def predict_transformer(models, rul_scale, hi_arr, start_obs, device):
    X, _ = make_seqs(hi_arr, np.zeros(len(hi_arr)), start_obs)
    Xt   = torch.tensor(X, dtype=torch.float32).to(device)
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0))
    return np.median(preds, axis=0)

# ── start_obs estimation via Train HI alignment ───────────────────────────
def estimate_start_obs(hi_test_start, hi_train_dict):
    """
    테스트 베어링의 초기 HI 값을 Train HI 궤적과 매치시켜
     가장 유사한 시점의 사이클 인덱스(start_obs)를 추정.
    """
    candidate_cycles = []
    for b, hi_tr in hi_train_dict.items():
        idx = np.where(hi_tr >= hi_test_start)[0]
        if len(idx) > 0:
            candidate_cycles.append(idx[0])
        else:
            candidate_cycles.append(len(hi_tr) - 1)
    start_obs = int(np.mean(candidate_cycles))
    return np.clip(start_obs, 0, 100)

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

# ── LOOCV (Dynamic A-full Baseline version) ──────────────────────────────
def run_loocv(train_dfs, train_regs, params, sigma, feat_w, grp_w, aux_features, aux_w, aux_s):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results = {}
    sc_lgbm_list, sc_tf_list = [], []

    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        print(f"{'='*60}")
        print(f"[LOOCV] Bearing {test_bid} held out (LOO-baseline HI recalculation)")

        # 1. LOO-baseline 계산
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

        # 3. LGBM & Transformer 학습 (train_bids 기준)
        lgbm_model = train_lgbm(hi_train_loo, train_bids)
        preds_lgbm = predict_lgbm(lgbm_model, hi_train_loo[test_bid], start_obs=0)

        tf_models, rul_scale = train_transformer_ensemble(hi_train_loo, train_bids, device)
        preds_tf = predict_transformer(tf_models, rul_scale, hi_train_loo[test_bid], start_obs=0, device=device)

        def avg_score(preds):
            return float(np.nanmean([competition_score(N_test - obs, p)
                                     for obs, p in zip(obs_pts, preds)]))
        sc_l = avg_score(preds_lgbm)
        sc_t = avg_score(preds_tf)
        print(f"  → LGBM: {sc_l:.4f}  Transformer-A: {sc_t:.4f}")
        sc_lgbm_list.append(sc_l)
        sc_tf_list.append(sc_t)

        w_l = float(np.clip(sc_l / (sc_l + sc_t + 1e-12), 0.1, 0.7))
        w_t = 1.0 - w_l
        preds_ens = [w_l * l + w_t * t for l, t in zip(preds_lgbm, preds_tf)]

        results[test_bid] = {
            "lgbm": list(preds_lgbm), "transformer": list(preds_tf), "ens": preds_ens,
            "obs_pts": list(obs_pts), "N": N_test,
            "sc_lgbm": sc_l, "sc_tf": sc_t, "w_lgbm": w_l, "w_tf": w_t,
        }

    ens_scores = []
    print(f"\n{'='*60}")
    print(f"  {'Bearing':>8} | {'LGBM':>8} | {'Transformer-A':>8} | {'Ensemble':>8}")
    print(f"  {'-'*55}")
    for test_bid in BEARINGS:
        r  = results[test_bid]
        sc = float(np.nanmean([competition_score(r["N"] - obs, p)
                               for obs, p in zip(r["obs_pts"], r["ens"])]))
        ens_scores.append(sc)
        results[test_bid]["sc_ens"] = sc
        print(f"  {test_bid:>8} | {r['sc_lgbm']:>8.4f} | {r['sc_tf']:>8.4f} | {sc:>8.4f}")
    print(f"  {'Mean':>8} | {np.mean(sc_lgbm_list):>8.4f} | "
          f"{np.mean(sc_tf_list):>8.4f} | {np.mean(ens_scores):>8.4f}")

    print("\n  [Calibration search]")
    cf_tf, sc_tf_cf = search_calibration(
        {b: results[b]["transformer"] for b in BEARINGS}, results, "Transformer-A")
    cf_ens, sc_ens_cf = search_calibration(
        {b: results[b]["ens"] for b in BEARINGS}, results, "Ensemble")

    best_config = "transformer" if sc_tf_cf >= sc_ens_cf else "ens"
    best_cf     = cf_tf  if best_config == "transformer" else cf_ens
    print(f"\n  Best config: {'Transformer-A' if best_config == 'transformer' else 'LGBM+Transformer-A'} (cf={best_cf:.2f})")

    # 결과 로그 파일 작성
    with open(OUT_DIR / "loocv_log.txt", "w", encoding="utf-8") as f:
        f.write("SR/0521 — Transformer-A-full Bearing RUL Pipeline\n")
        f.write(f"LOOCV uses DYNAMIC LOO-baseline HI recalculation\n\n")
        f.write(f"LGBM avg:        {np.mean(sc_lgbm_list):.4f}\n")
        f.write(f"Transformer avg: {np.mean(sc_tf_list):.4f}\n")
        f.write(f"Ensemble:        {np.mean(ens_scores):.4f}\n")
        f.write(f"cf: Transformer cf={cf_tf:.2f}→{sc_tf_cf:.4f}  Ens cf={cf_ens:.2f}→{sc_ens_cf:.4f}\n\n")
        for b in BEARINGS:
            r = results[b]
            f.write(f"  B{b}: LGBM={r['sc_lgbm']:.4f}  Transformer-A={r['sc_tf']:.4f}  Ens={r['sc_ens']:.4f}\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SR/0521 Transformer-A-full LOOCV — Recalculated LOO-baseline HI", fontsize=12)
    for i, test_bid in enumerate(BEARINGS):
        ax  = axes.flatten()[i]
        r   = results[test_bid]
        obs = r["obs_pts"]
        ax.plot(obs, [r["N"] - o for o in obs], "k-",  lw=2, label="True RUL")
        ax.plot(obs, r["lgbm"],  "r--", lw=1, alpha=0.7, label="LGBM")
        ax.plot(obs, r["transformer"],"g--", lw=1, alpha=0.7, label="Transformer-A")
        ax.plot(obs, r["ens"],   "m-",  lw=2,             label="Ensemble")
        ax.set_title(f"B{test_bid}  LGBM={r['sc_lgbm']:.3f}  "
                     f"Transformer={r['sc_tf']:.3f}  Ens={r['sc_ens']:.3f}")
        ax.set_xlabel("Observation cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loocv_predictions.png", dpi=150)
    plt.close()

    return results, best_config, best_cf, np.mean(ens_scores)

# ── Test inference with Train-baseline corrected HI & start_obs Estimation ──
def run_test_inference(hi_train_all, results, best_config, best_cf,
                       test_dfs, test_regs, params, sigma, feat_w, grp_w,
                       aux_features, aux_w, aux_s,
                       train_baseline, train_aux_baseline):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Test inference — "
          f"{'Transformer-A' if best_config == 'transformer' else 'LGBM+Transformer-A'}, cf={best_cf:.2f}")
    print(f"  HI mode: Train-baseline + Align start_obs\n")

    lgbm_model = train_lgbm(hi_train_all, BEARINGS)
    print("  Transformer training on all 4 bearings...")
    tf_models, rul_scale = train_transformer_ensemble(hi_train_all, BEARINGS, device)

    if best_config == "ens":
        w_lgbm = float(np.mean([results[b]["w_lgbm"] for b in BEARINGS]))
        w_tf   = float(np.mean([results[b]["w_tf"]  for b in BEARINGS]))
        total  = w_lgbm + w_tf
        w_lgbm, w_tf = w_lgbm / total, w_tf / total
        print(f"  Weights: LGBM={w_lgbm:.3f}, Transformer-A={w_tf:.3f}")

    REF = {
        "0514": {1: 5.05, 2: 5.08, 3: 4.69, 4: 3.43, 5: 7.46, 6: 5.40},
        "v4":   {1: 7.40, 2: 8.58, 3: 5.21, 4: 9.78, 5: 8.31, 6: 6.79},
        "afull": {1: 7.08, 2: 0.44, 3: 2.38, 4: 5.31, 5: 1.90, 6: 0.56},
    }

    summary_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"SR/0521 Test RUL — Train-baseline HI & Align start_obs  cf={best_cf:.2f}", fontsize=12)

    for i, tid in enumerate(TEST_IDS):
        feat_df    = test_dfs[tid]
        regime_arr = test_regs[tid]

        # 1. Train-baseline 기준 HI 계산
        hi_corr, hi_main, hi_aux, gate = compute_bearing_hi(
            feat_df, regime_arr, params, sigma, feat_w, grp_w,
            aux_features, aux_w, aux_s, train_baseline, train_aux_baseline)

        # 2. start_obs 추정
        start_obs = estimate_start_obs(hi_corr[0], hi_train_all)
        obs_pts = np.arange(SEQ_LENGTH, len(hi_corr))
        print(f"  [Test{tid}] HI: [{hi_corr[0]:.4f} → {hi_corr[-1]:.4f}]  "
              f"Estimated start_obs: {start_obs} cycles (obs_frac starts at {start_obs/MEAN_TRAIN_LIFE:.3f})")

        # 3. 정렬된 start_obs 기반 RUL 예측
        preds_tf    = predict_transformer(tf_models, rul_scale, hi_corr, start_obs=start_obs, device=device)
        preds_lgbm  = predict_lgbm(lgbm_model, hi_corr, start_obs=start_obs)
        preds_raw   = (w_lgbm * preds_lgbm + w_tf * preds_tf if best_config == "ens" else preds_tf)
        preds_final = preds_raw * best_cf

        final_cyc = float(preds_final[-1])
        final_hr  = final_cyc * INTERVAL_SEC / 3600
        print(f"  → 최종 RUL = {final_hr:.2f}hr  (참고용: afull={REF['afull'][tid]:.2f}, v4={REF['v4'][tid]:.2f})\n")

        # CSV 출력 저장
        pd.DataFrame({
            "obs_cycle":    obs_pts,
            "hi_corrected": hi_corr[SEQ_LENGTH:],
            "hi_main":      hi_main[SEQ_LENGTH:],
            "hi_aux":       hi_aux[SEQ_LENGTH:],
            "rul_lgbm":     preds_lgbm,
            "rul_tf":       preds_tf,
            "rul_final":    preds_final,
            "rul_hours":    preds_final * INTERVAL_SEC / 3600,
        }).to_csv(OUT_DIR / f"Test{tid}_RUL.csv", index=False)

        summary_rows.append({
            "test_id":       tid,
            "hi_corr_start": round(float(hi_corr[0]),  4),
            "hi_corr_end":   round(float(hi_corr[-1]), 4),
            "estimated_start_obs": start_obs,
            "rul_hours":     round(final_hr, 2),
            "afull_hours":   REF["afull"][tid],
            "base0514":      REF["0514"][tid],
        })

        ax  = axes.flatten()[i]
        ax2 = ax.twinx()
        ax.plot(obs_pts, preds_lgbm,  "r--", lw=1, alpha=0.6, label="LGBM")
        ax.plot(obs_pts, preds_tf,    "g--", lw=1, alpha=0.6, label="Transformer-A")
        ax.plot(obs_pts, preds_final, "m-",  lw=2,             label="Final")
        ax2.plot(obs_pts, hi_corr[SEQ_LENGTH:], "b-",  lw=1.5,           label="HI corr")
        ax2.set_ylabel("HI", color="b"); ax2.tick_params(axis="y", labelcolor="b")
        ax2.set_ylim(-0.05, 1.05)
        ax.set_title(
            f"Test{tid} (start_obs={start_obs}) "
            f"RUL={final_hr:.1f}hr  (afull={REF['afull'][tid]:.1f})")
        ax.set_xlabel("Observation cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.4)
        ax2.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_predictions.png", dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary.csv", index=False)

    print(f"\n  {'Test':>5} | {'0514':>6} | {'afull':>6} | {'v6_tf':>8} | start_obs | corr_start | corr_end")
    print(f"  {'-'*80}")
    for row in summary_rows:
        print(f"  {row['test_id']:>5} | {row['base0514']:>6.2f} | "
              f"{row['afull_hours']:>6.2f} | {row['rul_hours']:>8.2f} | "
              f"{row['estimated_start_obs']:>9} | {row['hi_corr_start']:>10.4f} | "
              f"{row['hi_corr_end']:.4f}")

    with open(OUT_DIR / "loocv_log.txt", "a", encoding="utf-8") as f:
        f.write("\nTest inference (Train-baseline HI & Align start_obs):\n")
        f.write(df_sum.to_string(index=False) + "\n")

    print(f"\n[완료] 결과 저장 위치: {OUT_DIR}")

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("SR/0521 — Transformer-A-full RUL Pipeline")
    print("=" * 60)

    # 1. 파라미터 로드
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
    results, best_config, best_cf, loocv_ens = run_loocv(
        train_dfs, train_regs, params, sigma, feat_w, grp_w,
        aux_features, aux_w, aux_s)

    # 5. Test Inference 실행
    print("\nStarting Test Inference...")
    run_test_inference(
        hi_train_all, results, best_config, best_cf,
        test_dfs, test_regs, params, sigma, feat_w, grp_w,
        aux_features, aux_w, aux_s,
        train_baseline_all, train_aux_baseline_all)
