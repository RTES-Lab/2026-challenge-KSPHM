"""
앙상블 RUL 예측 v2 — LGBM + LSTM-A (window-minmax HI) + LSTM-C
================================================================
LSTM-A 추가 (Approach A):
  - v4 FDR HI를 window 내부 minmax로 정규화 → shape만 학습
  - Bearing3 max(0.152) < Bearing4 min(0.244) HI 범위 역전 문제 우회
  - obs_fraction(obs_idx / MEAN_TRAIN_LIFE) 추가로 temporal context 제공
  - Piecewise Linear RUL 라벨 (LSTM-C와 동일)

모델 구성:
  - LGBM   : HI-A flat 피처 (직전 10개 HI + 기울기/mean/std/max) + 비대칭 custom obj
  - LSTM-A : [window_minmax_HI, obs_fraction] 시퀀스, piecewise RUL, 5 seeds median
  - LSTM-C : raw 8 signal features + piecewise RUL, 5 seeds median
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ── 경로 ──────────────────────────────────────────────────────────────────
BASE           = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0514")
TRAIN_FEAT_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SC/HI/04142304_signal_transform_v2/output")
TEST_FEAT_DIR  = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SC/HI/05072245_signal_transform_v5_test/output")
HI_A_TEST      = BASE / "hi/output/test_v4"
OUT_DIR        = BASE / "rul/output/ensemble_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH      = 10
MEAN_TRAIN_LIFE = 116.5   # (126+114+89+137)/4
INTERVAL_SEC    = 600
SEEDS           = [42, 7, 123, 0, 99]

BR_A, ALPHA_A = 0.25, 0.1

# ── HI-A 피처 ────────────────────────────────────────────────────────────
FEATURE_Q = {
    "ch3_high_band":   0.4315732105779938,
    "ch4_high_band":   0.41934581236265145,
    "ch3_std":         0.4143663846438889,
    "ch3_total_power": 0.41207524516168137,
    "ch3_energy":      0.41108403369865365,
    "ch3_rms":         0.41108403369865365,
    "ch3_p2p":         0.3665441916808586,
}
FEATURE_GROUPS = {
    "highfreq":  ["ch3_high_band", "ch4_high_band"],
    "energy":    ["ch3_total_power", "ch3_energy", "ch3_rms"],
    "variation": ["ch3_std", "ch3_p2p"],
}
ALL_FEATS = list(FEATURE_Q.keys())


# =========================================================
# V4 FDR HI 인라인 계산 헬퍼
# =========================================================
def _moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, np.ones(window) / window, mode="valid")[:len(x)]


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def _train_anchored_scale(x: np.ndarray, p5: float, p95: float) -> np.ndarray:
    denom = p95 - p5
    if abs(denom) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return np.clip((x - p5) / denom, 0.0, 1.0)


def _fdr_ratios(feat_matrix: np.ndarray, feature_names: list,
                cond: np.ndarray, baseline: dict, eps: float = 1e-8) -> np.ndarray:
    ratios = np.zeros_like(feat_matrix, dtype=float)
    for regime in [0, 1]:
        idx = np.where(cond == regime)[0]
        if len(idx) == 0:
            continue
        bvec = np.array([baseline[(regime, f)] for f in feature_names])
        bvec = np.where(np.abs(bvec) < eps, eps, bvec)
        ratios[idx] = (feat_matrix[idx] - bvec) / (np.abs(bvec) + eps)
    return ratios


def _fdr_baseline(dfs: dict, br: float, exclude_bid: int, feat_list: list) -> dict:
    feat_vals = {(r, f): [] for r in [0, 1] for f in feat_list}
    for bid, df in dfs.items():
        if bid == exclude_bid:
            continue
        cond = df["cond"].values
        n_base = max(3, int(len(df) * br))
        for regime in [0, 1]:
            idx_regime = np.where(cond == regime)[0]
            base_idx = idx_regime[:n_base]
            if len(base_idx) == 0:
                continue
            for f in feat_list:
                if f in df.columns:
                    feat_vals[(regime, f)].extend(df[f].values[base_idx].tolist())
    baseline = {}
    for regime in [0, 1]:
        for f in feat_list:
            vals = feat_vals[(regime, f)]
            baseline[(regime, f)] = float(np.mean(vals)) if vals else 1.0
    return baseline


def _group_stats(dfs: dict, br: float, exclude_bid: int,
                 feat_groups: dict, feat_q: dict, feat_list: list,
                 baseline: dict) -> dict:
    all_scores = {g: [] for g in feat_groups}
    dir_votes  = {g: [] for g in feat_groups}
    for bid, df in dfs.items():
        if bid == exclude_bid:
            continue
        cond = df["cond"].values
        for gname, feats in feat_groups.items():
            available = [f for f in feats if f in df.columns]
            if not available:
                continue
            mat     = df[available].values
            ratios  = _fdr_ratios(mat, available, cond, baseline)
            weights = np.array([feat_q[f] for f in available], dtype=float)
            weights /= weights.sum() + 1e-12
            score   = (ratios * weights.reshape(1, -1)).sum(axis=1)
            rho, _  = spearmanr(np.arange(len(score)), score)
            dir_votes[gname].append(+1 if (not np.isnan(rho) and rho >= 0) else -1)
            all_scores[gname].extend(score.tolist())
    stats = {}
    for gname in feat_groups:
        direction = +1 if sum(dir_votes[gname]) >= 0 else -1
        arr = np.array(all_scores[gname]) * direction
        stats[gname] = {
            "direction": direction,
            "p5":  float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }
    return stats


def _v4fdr_hi(df: pd.DataFrame, baseline: dict, group_stats: dict,
              feat_groups: dict, feat_q: dict, ema_alpha: float) -> np.ndarray:
    cond = df["cond"].values
    sub_his, group_weights = {}, {}
    for gname, feats in feat_groups.items():
        available = [f for f in feats if f in df.columns]
        if not available:
            continue
        mat     = df[available].values
        ratios  = _fdr_ratios(mat, available, cond, baseline)
        weights = np.array([feat_q[f] for f in available], dtype=float)
        weights /= weights.sum() + 1e-12
        score   = (ratios * weights.reshape(1, -1)).sum(axis=1)
        score   = score * group_stats[gname]["direction"]
        score   = _train_anchored_scale(score, group_stats[gname]["p5"], group_stats[gname]["p95"])
        score   = _ema(score, alpha=ema_alpha)
        sub_his[gname]       = np.clip(score, 0.0, 1.0)
        group_weights[gname] = np.mean([feat_q[f] for f in available])
    if not sub_his:
        return np.zeros(len(df))
    sub_mat = np.column_stack([sub_his[g] for g in sub_his])
    w = np.array([group_weights[g] for g in sub_his], dtype=float)
    w /= w.sum() + 1e-12
    final_hi = (sub_mat * w.reshape(1, -1)).sum(axis=1)
    return np.clip(_moving_avg(final_hi, 7), 0.0, 1.0)


def load_train_features() -> dict:
    dfs = {}
    for bid in BEARINGS:
        df = pd.read_csv(TRAIN_FEAT_DIR / f"Bearing{bid}_features_transformed.csv")
        cond = pd.read_csv(TRAIN_FEAT_DIR / f"Bearing{bid}_SSM_result.csv")["cond"].values
        df["cond"] = cond
        dfs[bid] = df
    return dfs


def compute_fold_hi_a(dfs: dict, exclude_bid: int) -> dict:
    baseline_a    = _fdr_baseline(dfs, BR_A, exclude_bid, ALL_FEATS)
    group_stats_a = _group_stats(dfs, BR_A, exclude_bid,
                                  FEATURE_GROUPS, FEATURE_Q, ALL_FEATS, baseline_a)
    return {b: _v4fdr_hi(dfs[b], baseline_a, group_stats_a,
                          FEATURE_GROUPS, FEATURE_Q, ALPHA_A) for b in BEARINGS}


# ── 대회 채점 함수 ─────────────────────────────────────────────────────────
def competition_score(rul_true: float, rul_pred: float) -> float:
    if rul_true <= 0:
        return np.nan
    Er = 100.0 * (rul_true - rul_pred) / rul_true
    if Er <= 0:
        return np.exp(-np.log(0.5) * Er / 20.0)
    else:
        return np.exp(np.log(0.5) * Er / 50.0)


# ── LightGBM ──────────────────────────────────────────────────────────────
def make_lgbm_features(hi_array, seq_length=SEQ_LENGTH):
    features, targets = [], []
    N = len(hi_array)
    for i in range(seq_length, N):
        window = hi_array[i - seq_length: i]
        slope  = float(np.polyfit(np.arange(seq_length), window, 1)[0])
        feats  = list(window) + [
            slope,
            float(window.mean()),
            float(window.std()),
            float(window.max()),
            float(window[-1]),
            float(window[-1] - window[0]),
        ]
        features.append(feats)
        targets.append(float(N - i))
    return np.array(features), np.array(targets)


def lgbm_asymmetric_obj(y_pred, dataset):
    y_true = dataset.get_label()
    diff = y_true - y_pred
    weight = np.where(diff < 0, 2.5, 1.0)
    grad = -diff * weight
    hess = np.ones_like(grad) * weight
    return grad, hess


def train_lgbm(hi_dict, train_bids):
    X_list, Y_list = [], []
    for b in train_bids:
        x, y = make_lgbm_features(hi_dict[b])
        X_list.append(x); Y_list.append(y)
    X_train = np.concatenate(X_list)
    Y_train = np.concatenate(Y_list)
    dtrain = lgb.Dataset(X_train, label=Y_train)
    params = {
        "num_leaves": 15,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": 5,
        "verbose": -1,
        "objective": lgbm_asymmetric_obj,
    }
    return lgb.train(params, dtrain, num_boost_round=200)


def predict_lgbm(model, hi_array):
    X, _ = make_lgbm_features(hi_array)
    return np.maximum(model.predict(X), 0.0)


# =========================================================
# LSTM-A: window-minmax HI + obs_fraction (Approach A)
# =========================================================
N_FEAT_A = 2  # [window_minmax_hi, obs_fraction]

# Piecewise RUL 라벨 파라미터 (v3 HI > 0.3 최초 돌파 기준)
NORMAL_UNTIL_A = {1: 89, 2: 92, 3: 62, 4: 78}
EOL_A          = {1: 126, 2: 114, 3: 89, 4: 137}


def rul_labels_a(n_total: int, bid: int) -> np.ndarray:
    nu  = NORMAL_UNTIL_A[bid]
    eol = EOL_A[bid]
    idx = np.arange(n_total)
    return np.where(idx <= nu, eol - nu, np.maximum(eol - idx, 0)).astype(float)


def make_seqs_a(hi_arr: np.ndarray, rul_arr: np.ndarray,
                seq_len: int, start_obs: int = 0):
    """window-minmax HI + obs_fraction 시퀀스 생성."""
    n = len(hi_arr)
    X, y = [], []
    for i in range(n - seq_len):
        window = hi_arr[i:i + seq_len].copy()
        w_min, w_max = window.min(), window.max()
        window_norm = (window - w_min) / (w_max - w_min + 1e-8)
        obs_idx  = start_obs + i + np.arange(seq_len)
        obs_frac = np.clip(obs_idx / MEAN_TRAIN_LIFE, 0.0, 2.0)
        seq = np.stack([window_norm, obs_frac], axis=1)  # (seq_len, 2)
        X.append(seq)
        y.append(float(rul_arr[i + seq_len]))
    return np.array(X), np.array(y)


class LSTMRegressorA(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_FEAT_A, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)


def train_model_a(X_train: np.ndarray, y_train: np.ndarray,
                  rul_scale: float, seed: int, device) -> LSTMRegressorA:
    torch.manual_seed(seed)
    y_norm = y_train / rul_scale
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_norm,  dtype=torch.float32)
    n_val = max(1, int(len(Xt) * 0.1))
    tr_dl = DataLoader(TensorDataset(Xt[:-n_val], yt[:-n_val]),
                       batch_size=64, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xt[-n_val:], yt[-n_val:]), batch_size=64)

    model = LSTMRegressorA().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.MSELoss()

    best_val, patience, best_state = np.inf, 0, None
    for _ in range(200):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([crit(model(xb.to(device)), yb.to(device)).item()
                          for xb, yb in val_dl])
        if vl < best_val:
            best_val, patience = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    model.load_state_dict(best_state)
    return model


def predict_lstm_a_ensemble(hi_dict: dict, train_bids: list, test_bid: int,
                             device) -> np.ndarray:
    """LSTM-A 5-seed median 앙상블 (LOOCV 용)."""
    X_list, y_list = [], []
    for b in train_bids:
        X, y = make_seqs_a(hi_dict[b], rul_labels_a(len(hi_dict[b]), b), SEQ_LENGTH)
        X_list.append(X); y_list.append(y)
    X_train = np.concatenate(X_list)
    y_train = np.concatenate(y_list)
    rul_scale = float(y_train.max())

    X_test, _ = make_seqs_a(hi_dict[test_bid],
                             rul_labels_a(len(hi_dict[test_bid]), test_bid),
                             SEQ_LENGTH)
    Xt = torch.tensor(X_test, dtype=torch.float32)

    all_preds = []
    for s in SEEDS:
        m = train_model_a(X_train, y_train, rul_scale, s, device)
        m.eval()
        with torch.no_grad():
            p = m(Xt.to(device)).cpu().numpy()
        all_preds.append(np.maximum(p * rul_scale, 0.0))
    return np.median(all_preds, axis=0)


def build_lstm_a_for_test(hi_dict: dict, device):
    """전체 Train 4개 베어링으로 LSTM-A 학습."""
    X_list, y_list = [], []
    for b in BEARINGS:
        X, y = make_seqs_a(hi_dict[b], rul_labels_a(len(hi_dict[b]), b), SEQ_LENGTH)
        X_list.append(X); y_list.append(y)
    X_train = np.concatenate(X_list)
    y_train = np.concatenate(y_list)
    rul_scale = float(y_train.max())

    models = []
    for s in SEEDS:
        m = train_model_a(X_train, y_train, rul_scale, s, device)
        models.append(m)
        print(f"    LSTM-A seed={s} 완료")
    return models, rul_scale


def predict_lstm_a_from_models(models: list, rul_scale: float,
                                hi_arr: np.ndarray, start_obs: int,
                                device) -> np.ndarray:
    """사전 학습된 LSTM-A 모델로 test HI 예측."""
    n = len(hi_arr)
    X = []
    for i in range(n - SEQ_LENGTH):
        window = hi_arr[i:i + SEQ_LENGTH].copy()
        w_min, w_max = window.min(), window.max()
        window_norm = (window - w_min) / (w_max - w_min + 1e-8)
        obs_idx  = start_obs + i + np.arange(SEQ_LENGTH)
        obs_frac = np.clip(obs_idx / MEAN_TRAIN_LIFE, 0.0, 2.0)
        seq = np.stack([window_norm, obs_frac], axis=1)
        X.append(seq)
    Xt = torch.tensor(np.array(X), dtype=torch.float32).to(device)

    all_preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            p = m(Xt).cpu().numpy()
        all_preds.append(np.maximum(p * rul_scale, 0.0))
    return np.median(all_preds, axis=0)


# =========================================================
# LSTM-C: raw signal features (기존 유지)
# =========================================================
INPUT_COLS_C = [
    "ch3_high_band", "ch4_high_band",
    "ch3_std", "ch3_total_power", "ch3_energy",
    "ch3_rms", "ch3_p2p",
    "ch4_rms",
]
N_FEAT_C = len(INPUT_COLS_C)

NORMAL_UNTIL_C = {1: 89, 2: 92, 3: 62, 4: 78}
EOL_C          = {1: 126, 2: 114, 3: 89, 4: 137}


def rul_labels_c(n_total: int, bid: int) -> np.ndarray:
    nu  = NORMAL_UNTIL_C[bid]
    eol = EOL_C[bid]
    idx = np.arange(n_total)
    return np.where(idx <= nu, eol - nu, np.maximum(eol - idx, 0)).astype(float)


def make_seqs_c(feat_arr: np.ndarray, rul_arr: np.ndarray, seq_len: int):
    n = len(feat_arr)
    X = np.array([feat_arr[i:i + seq_len] for i in range(n - seq_len)])
    y = rul_arr[seq_len:]
    return X, y


class LSTMRegressorC(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_FEAT_C, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)


def train_model_c(X_train: np.ndarray, y_train: np.ndarray,
                  rul_scale: float, seed: int, device) -> LSTMRegressorC:
    torch.manual_seed(seed)
    y_norm = y_train / rul_scale
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_norm,  dtype=torch.float32)
    n_val = max(1, int(len(Xt) * 0.1))
    tr_dl = DataLoader(TensorDataset(Xt[:-n_val], yt[:-n_val]),
                       batch_size=64, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xt[-n_val:], yt[-n_val:]), batch_size=64)

    model = LSTMRegressorC().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.MSELoss()

    best_val, patience, best_state = np.inf, 0, None
    for _ in range(200):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([crit(model(xb.to(device)), yb.to(device)).item()
                          for xb, yb in val_dl])
        if vl < best_val:
            best_val, patience = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    model.load_state_dict(best_state)
    return model


def predict_lstm_c_ensemble(dfs: dict, train_bids: list, test_bid: int,
                             device) -> np.ndarray:
    scaler = StandardScaler()
    scaler.fit(np.concatenate([dfs[b][INPUT_COLS_C].values for b in train_bids]))

    X_list, y_list = [], []
    for b in train_bids:
        feat_s = scaler.transform(dfs[b][INPUT_COLS_C].values)
        X, y   = make_seqs_c(feat_s, rul_labels_c(len(dfs[b]), b), SEQ_LENGTH)
        X_list.append(X); y_list.append(y)
    X_train = np.concatenate(X_list)
    y_train = np.concatenate(y_list)
    rul_scale = float(y_train.max())

    feat_test = scaler.transform(dfs[test_bid][INPUT_COLS_C].values)
    X_test, _ = make_seqs_c(feat_test, rul_labels_c(len(dfs[test_bid]), test_bid), SEQ_LENGTH)

    all_preds = []
    for s in SEEDS:
        m = train_model_c(X_train, y_train, rul_scale, s, device)
        m.eval()
        with torch.no_grad():
            p = m(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        all_preds.append(np.maximum(p * rul_scale, 0.0))
    return np.median(all_preds, axis=0)


def build_lstm_c_for_test(dfs: dict, device):
    scaler = StandardScaler()
    scaler.fit(np.concatenate([dfs[b][INPUT_COLS_C].values for b in BEARINGS]))

    X_list, y_list = [], []
    for b in BEARINGS:
        feat_s = scaler.transform(dfs[b][INPUT_COLS_C].values)
        X, y   = make_seqs_c(feat_s, rul_labels_c(len(dfs[b]), b), SEQ_LENGTH)
        X_list.append(X); y_list.append(y)
    X_train = np.concatenate(X_list)
    y_train = np.concatenate(y_list)
    rul_scale = float(y_train.max())

    models = []
    for s in SEEDS:
        m = train_model_c(X_train, y_train, rul_scale, s, device)
        models.append(m)
        print(f"    LSTM-C seed={s} 완료")
    return models, scaler, rul_scale


def predict_lstm_c_from_models(models: list, scaler, rul_scale: float,
                                test_feat: np.ndarray, device) -> np.ndarray:
    feat_s = scaler.transform(test_feat)
    n = len(feat_s)
    X_test = np.array([feat_s[i:i + SEQ_LENGTH] for i in range(n - SEQ_LENGTH)])
    Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
    all_preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            p = m(Xt).cpu().numpy()
        all_preds.append(np.maximum(p * rul_scale, 0.0))
    return np.median(all_preds, axis=0)


# ── 가중치 결정 ────────────────────────────────────────────────────────────
def clamp_weights(scores: dict, lo=0.1, hi_w=0.6):
    keys = list(scores.keys())
    vals = np.array([scores[k] for k in keys], dtype=float)
    vals = np.clip(vals, 1e-6, None)
    w = vals / vals.sum()
    w = np.clip(w, lo, hi_w)
    w = w / w.sum()
    return {k: float(w[i]) for i, k in enumerate(keys)}


# ── LOOCV ─────────────────────────────────────────────────────────────────
def run_loocv():
    print("  Train feature 로드 중...")
    dfs = load_train_features()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    results = {b: {} for b in BEARINGS}
    model_fold_scores = {"lgbm": [], "lstm_a": [], "lstm_c": []}
    fold_weights = {}

    for test_bid in BEARINGS:
        print(f"\n{'='*60}")
        print(f"[LOOCV] Test Bearing {test_bid}")
        train_bids = [b for b in BEARINGS if b != test_bid]

        print(f"  HI-A v4 계산 (Bearing{test_bid} 제외)...")
        hi_a = compute_fold_hi_a(dfs, exclude_bid=test_bid)
        N_test = len(hi_a[test_bid])

        # 1. LGBM
        print("  학습: LGBM (HI-A flat)...")
        lgbm_model = train_lgbm(hi_a, train_bids)
        preds_lgbm = predict_lgbm(lgbm_model, hi_a[test_bid])

        # 2. LSTM-A (window-minmax HI + obs_fraction)
        print("  학습: LSTM-A (window-minmax HI + obs_frac, 5 seeds)...")
        preds_a = predict_lstm_a_ensemble(hi_a, train_bids, test_bid, device)

        # 3. LSTM-C (raw 8 signal features)
        print("  학습: LSTM-C (raw 8feat, piecewise RUL, 5 seeds)...")
        preds_c = predict_lstm_c_ensemble(dfs, train_bids, test_bid, device)

        obs_pts = np.arange(SEQ_LENGTH, N_test)

        def avg_score(preds):
            scores = [competition_score(N_test - obs, p)
                      for obs, p in zip(obs_pts, preds)]
            return float(np.nanmean(scores))

        sc_lgbm = avg_score(preds_lgbm)
        sc_a    = avg_score(preds_a)
        sc_c    = avg_score(preds_c)

        # LSTM-A 수렴 여부 진단 (전 구간 0 예측 체크)
        a_collapse = float(np.mean(preds_a)) < 1.0
        print(f"  → LGBM: {sc_lgbm:.4f}, LSTM-A: {sc_a:.4f} {'[COLLAPSE?]' if a_collapse else ''}, "
              f"LSTM-C: {sc_c:.4f}")

        model_fold_scores["lgbm"].append(sc_lgbm)
        model_fold_scores["lstm_a"].append(sc_a)
        model_fold_scores["lstm_c"].append(sc_c)

        w = clamp_weights({"lgbm": sc_lgbm, "lstm_a": sc_a, "lstm_c": sc_c})
        fold_weights[test_bid] = w
        print(f"  → 가중치: LGBM={w['lgbm']:.3f}, LSTM-A={w['lstm_a']:.3f}, LSTM-C={w['lstm_c']:.3f}")

        preds_ens = w["lgbm"] * preds_lgbm + w["lstm_a"] * preds_a + w["lstm_c"] * preds_c
        sc_ens = avg_score(preds_ens)
        print(f"  → 앙상블: {sc_ens:.4f}")

        results[test_bid] = {
            "lgbm":     list(preds_lgbm),
            "lstm_a":   list(preds_a),
            "lstm_c":   list(preds_c),
            "ensemble": list(preds_ens),
            "obs_pts":  list(obs_pts),
            "N":        N_test,
        }

    # LOOCV 요약
    print(f"\n{'='*60}")
    print("  LOOCV 요약")
    print(f"{'='*60}")
    print(f"  {'Bearing':>8} | {'LGBM':>8} | {'LSTM-A':>8} | {'LSTM-C':>8} | {'Ensemble':>9}")
    print(f"  {'-'*55}")
    ens_scores = []
    for test_bid in BEARINGS:
        obs_pts = results[test_bid]["obs_pts"]
        N_test  = results[test_bid]["N"]
        s_ens = [competition_score(N_test - obs, p)
                 for obs, p in zip(obs_pts, results[test_bid]["ensemble"])]
        ens_scores.append(float(np.nanmean(s_ens)))
        i = test_bid - 1
        print(f"  {test_bid:>8} | {model_fold_scores['lgbm'][i]:>8.4f} | "
              f"{model_fold_scores['lstm_a'][i]:>8.4f} | "
              f"{model_fold_scores['lstm_c'][i]:>8.4f} | "
              f"{ens_scores[-1]:>9.4f}")

    overall = {k: float(np.mean(v)) for k, v in model_fold_scores.items()}
    print(f"  {'평균':>8} | {overall['lgbm']:>8.4f} | {overall['lstm_a']:>8.4f} | "
          f"{overall['lstm_c']:>8.4f} | {np.mean(ens_scores):>9.4f}")

    # Calibration factor 그리드 서치
    print("\n  [Calibration] factor 그리드 서치 (0.7~1.0)...")
    best_cf, best_cf_score = 1.0, -np.inf
    for cf in np.arange(0.70, 1.01, 0.02):
        cf_scores = []
        for test_bid in BEARINGS:
            obs_pts = results[test_bid]["obs_pts"]
            N_test  = results[test_bid]["N"]
            preds   = [p * cf for p in results[test_bid]["ensemble"]]
            s = [competition_score(N_test - obs, p) for obs, p in zip(obs_pts, preds)]
            cf_scores.append(float(np.nanmean(s)))
        mean_cf = float(np.mean(cf_scores))
        print(f"    cf={cf:.2f} → {mean_cf:.4f}")
        if mean_cf > best_cf_score:
            best_cf_score, best_cf = mean_cf, float(cf)

    print(f"\n  → Best calibration factor: {best_cf:.2f} (score={best_cf_score:.4f})")

    # 로그 저장
    with open(OUT_DIR / "loocv_log.txt", "w", encoding="utf-8") as f:
        f.write("LOOCV 결과 (v4 FDR 인라인 HI-A + LSTM-A window-minmax + LSTM-C raw)\n")
        f.write(f"LGBM   (HI-A flat):              {overall['lgbm']:.4f}\n")
        f.write(f"LSTM-A (window-minmax HI):        {overall['lstm_a']:.4f}\n")
        f.write(f"LSTM-C (raw 8feat piecewise RUL): {overall['lstm_c']:.4f}\n")
        f.write(f"Ensemble (3 models):              {np.mean(ens_scores):.4f}\n")
        f.write(f"Best calibration factor:          {best_cf:.2f}\n")
        f.write(f"\nFold 상세:\n")
        for test_bid in BEARINGS:
            i = test_bid - 1
            f.write(f"  B{test_bid}: LGBM={model_fold_scores['lgbm'][i]:.4f}  "
                    f"LSTM-A={model_fold_scores['lstm_a'][i]:.4f}  "
                    f"LSTM-C={model_fold_scores['lstm_c'][i]:.4f}  "
                    f"Ensemble={ens_scores[i]:.4f}\n")

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Ensemble v2 LOOCV — LGBM + LSTM-A (window-minmax) + LSTM-C", fontsize=12)
    axes = axes.flatten()
    for i, test_bid in enumerate(BEARINGS):
        ax = axes[i]
        obs_pts = results[test_bid]["obs_pts"]
        N_test  = results[test_bid]["N"]
        true_rul = [N_test - obs for obs in obs_pts]
        ax.plot(obs_pts, true_rul,                     "k-",  lw=1.5, label="True RUL")
        ax.plot(obs_pts, results[test_bid]["lgbm"],    "r--", lw=1,   alpha=0.7, label="LGBM")
        ax.plot(obs_pts, results[test_bid]["lstm_a"],  "g--", lw=1,   alpha=0.7, label="LSTM-A")
        ax.plot(obs_pts, results[test_bid]["lstm_c"],  "c--", lw=1,   alpha=0.7, label="LSTM-C")
        ax.plot(obs_pts, results[test_bid]["ensemble"],"m-",  lw=2,   label="Ensemble")
        ax.set_title(f"Bearing {test_bid}  (LSTM-A: {model_fold_scores['lstm_a'][i]:.3f})")
        ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ensemble_v2_loocv_predictions.png", dpi=150)
    plt.close()

    return overall, best_cf, fold_weights


# ── Test 추론 ──────────────────────────────────────────────────────────────
def run_test_inference(best_cf: float, fold_weights: dict):
    print(f"\n{'='*60}")
    print("  Test 추론 (전체 Train 4개 베어링으로 재학습)")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not all((HI_A_TEST / f"Test{t}_best.csv").exists() for t in TEST_IDS):
        print("[경고] HI-A Test 파일 없음. hi_test_v4.py를 먼저 실행하세요.")
        return

    print("  Train HI-A v4 계산 (전체 4개 베어링 기준)...")
    dfs = load_train_features()
    hi_a = compute_fold_hi_a(dfs, exclude_bid=0)

    print("  LGBM 학습...")
    lgbm_model = train_lgbm(hi_a, BEARINGS)

    print("  LSTM-A 학습 (window-minmax HI, 5 seeds)...")
    lstm_a_models, lstm_a_scale = build_lstm_a_for_test(hi_a, device)

    print("  LSTM-C 학습 (raw features, 5 seeds)...")
    lstm_c_models, lstm_c_scaler, lstm_c_scale = build_lstm_c_for_test(dfs, device)

    # LOOCV fold 평균 가중치
    w_lgbm = float(np.mean([fold_weights[b]["lgbm"]   for b in BEARINGS]))
    w_a    = float(np.mean([fold_weights[b]["lstm_a"]  for b in BEARINGS]))
    w_c    = float(np.mean([fold_weights[b]["lstm_c"]  for b in BEARINGS]))
    total  = w_lgbm + w_a + w_c
    w_lgbm, w_a, w_c = w_lgbm / total, w_a / total, w_c / total
    print(f"\n  Test 가중치: LGBM={w_lgbm:.3f}, LSTM-A={w_a:.3f}, LSTM-C={w_c:.3f}")
    print(f"  Calibration factor: {best_cf:.2f}")

    summary_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Ensemble v2 RUL — Test (LGBM + LSTM-A + LSTM-C)", fontsize=12)
    axes = axes.flatten()

    for i, tid in enumerate(TEST_IDS):
        hi_a_t = pd.read_csv(HI_A_TEST / f"Test{tid}_best.csv")["HI"].values
        N = len(hi_a_t)
        obs_pts = np.arange(SEQ_LENGTH, N)

        preds_lgbm = predict_lgbm(lgbm_model, hi_a_t)
        preds_a    = predict_lstm_a_from_models(lstm_a_models, lstm_a_scale,
                                                hi_a_t, start_obs=0, device=device)
        test_raw_feat = pd.read_csv(TEST_FEAT_DIR / f"Test{tid}_features.csv")[INPUT_COLS_C].values
        preds_c    = predict_lstm_c_from_models(lstm_c_models, lstm_c_scaler,
                                                lstm_c_scale, test_raw_feat, device)

        preds_ens = (w_lgbm * preds_lgbm + w_a * preds_a + w_c * preds_c) * best_cf

        final_rul_cyc = float(preds_ens[-1])
        final_rul_hr  = final_rul_cyc * INTERVAL_SEC / 3600
        print(f"  [Test{tid}] {N}슬롯({N*INTERVAL_SEC/3600:.1f}hr) | "
              f"최종 RUL={final_rul_hr:.2f}hr ({final_rul_cyc:.1f}cycles)")

        df_out = pd.DataFrame({
            "obs_cycle":         obs_pts,
            "rul_pred_lgbm":     preds_lgbm,
            "rul_pred_lstm_a":   preds_a,
            "rul_pred_lstm_c":   preds_c,
            "rul_pred_ensemble": preds_ens,
            "rul_pred_hours":    preds_ens * INTERVAL_SEC / 3600,
        })
        df_out.to_csv(OUT_DIR / f"Test{tid}_ensemble_v2_RUL.csv", index=False)

        summary_rows.append({
            "test_id":          tid,
            "observed_slots":   N,
            "observed_hours":   round(N * INTERVAL_SEC / 3600, 2),
            "final_rul_cycles": round(final_rul_cyc, 2),
            "final_rul_hours":  round(final_rul_hr,  2),
            "w_lgbm":           round(w_lgbm, 3),
            "w_lstm_a":         round(w_a, 3),
            "w_lstm_c":         round(w_c, 3),
            "calib_factor":     round(best_cf, 2),
        })

        ax = axes[i]
        ax.plot(obs_pts, preds_lgbm, "r--", lw=1,  alpha=0.6, label="LGBM")
        ax.plot(obs_pts, preds_a,    "g--", lw=1,  alpha=0.6, label="LSTM-A")
        ax.plot(obs_pts, preds_c,    "c--", lw=1,  alpha=0.6, label="LSTM-C")
        ax.plot(obs_pts, preds_ens,  "m-",  lw=2,  label="Ensemble")
        ax.axvline(N, color="gray", ls="--", lw=1)
        ax.set_title(f"Test{tid}  Final={final_rul_hr:.1f}hr")
        ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "ensemble_v2_test_predictions.png", dpi=150)
    plt.close()

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / "Test_ensemble_v2_summary.csv", index=False)
    print(f"\n  최종 요약:")
    print(df_summary.to_string(index=False))
    print(f"\n[완료] {OUT_DIR}")


# ── 메인 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    overall, best_cf, fold_weights = run_loocv()
    run_test_inference(best_cf, fold_weights)
