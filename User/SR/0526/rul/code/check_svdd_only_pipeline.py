import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Paths & Constants
BASE = Path("/data/home/ksphm/2026-challenge-KSPHM")
TRAIN_FEAT_DIR = BASE / "User/TH/common_source"
BEARINGS = [1, 2, 3, 4]
SEQ_LENGTH = 10
MEAN_TRAIN_LIFE = 116.5
INTERVAL_SEC = 600
SEEDS = [42, 7, 123, 0, 99]

EOL = {1: 126, 2: 114, 3: 89, 4: 137}
MAIN_ALL_FEATS = ["ch3_total_power", "ch3_energy", "ch3_rms", "ch3_std", "ch3_p2p"]

# Helper functions
def ema_smooth(x, alpha=0.2):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if len(x) == 0: return y
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def moving_average(x, window=7):
    x = np.asarray(x, dtype=float)
    return pd.Series(x).rolling(window=window, center=True, min_periods=1).mean().values

def competition_score(y_true, y_pred):
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-np.log(0.5) * (d / 20.0)), np.exp(np.log(0.5) * (d / 20.0)))
    return score

# Load Train Features
train_dfs = {}
for b in BEARINGS:
    df = pd.read_csv(TRAIN_FEAT_DIR / f"Bearing{b}_features.csv")
    train_dfs[b] = df

# 1. Train Global SVDD on healthy data (first 50 cycles of each training bearing)
healthy_features = []
for b in BEARINGS:
    healthy_features.append(train_dfs[b][MAIN_ALL_FEATS].values[:50])
X_healthy = np.vstack(healthy_features)

scaler = StandardScaler()
X_healthy_scaled = scaler.fit_transform(X_healthy)

ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
ocsvm.fit(X_healthy_scaled)

# Calculate smoothed SVDD scores for all training bearings
raw_svdd_scores = {}
smoothed_svdd_scores = {}
for b in BEARINGS:
    X = train_dfs[b][MAIN_ALL_FEATS].values
    X_scaled = scaler.transform(X)
    scores = -ocsvm.decision_function(X_scaled)
    raw_svdd_scores[b] = scores
    
    # Smooth
    s_scores = ema_smooth(scores)
    s_scores = moving_average(s_scores)
    smoothed_svdd_scores[b] = s_scores

# Determine global normalization bounds using all training bearings
all_smoothed_scores = np.concatenate(list(smoothed_svdd_scores.values()))
global_min = all_smoothed_scores.min()
global_max = all_smoothed_scores.max()

# SVDD HI: normalized to [0, 1] individually for training consistency
svdd_hi_train = {}
for b in BEARINGS:
    hi = (smoothed_svdd_scores[b] - smoothed_svdd_scores[b].min()) / (smoothed_svdd_scores[b].max() - smoothed_svdd_scores[b].min() + 1e-8)
    svdd_hi_train[b] = np.clip(hi, 0.0, 1.0)
    print(f"Bearing {b} SVDD HI: start={svdd_hi_train[b][0]:.4f}, end={svdd_hi_train[b][-1]:.4f}")

# ── RUL Dataset Utilities ────────────────────────────────────────────────
def rul_labels(length, bid):
    N = EOL[bid]
    return np.array([N - i for i in range(length)])

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

# LSTM & Transformer definitions
class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)

class TransformerRegressor(nn.Module):
    def __init__(self, n_feat=2, seq_len=10, hidden=64, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.pos = nn.Parameter(torch.randn(seq_len, hidden) * 0.02)
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden*2, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.fc   = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))
    def forward(self, x):
        z = self.proj(x) + self.pos
        z = self.enc(z)
        return self.fc(z.mean(dim=1)).squeeze(-1)

def lgbm_asymmetric_obj(y_pred, dataset):
    y_true = dataset.get_label()
    diff   = y_true - y_pred
    weight = np.where(diff < 0, 2.5, 1.0)
    return -diff * weight, np.ones_like(diff) * weight

# ── Train functions ──────────────────────────────────────────────────────
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
            vl = np.mean([crit(model(xb.to(device)), yb.to(device)).item() for xb, yb in val_dl])
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
            vl = np.mean([crit(model(xb.to(device)), yb.to(device)).item() for xb, yb in val_dl])
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

# ── LOOCV ────────────────────────────────────────────────────────────────
print("\n--- Running Pure SVDD-HI LOOCV ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = {}
for test_bid in BEARINGS:
    train_bids = [b for b in BEARINGS if b != test_bid]
    N_test = EOL[test_bid]
    obs_pts = np.arange(SEQ_LENGTH, N_test)
    
    # Train Models
    lgbm_model = train_lgbm(svdd_hi_train, train_bids)
    lstm_models, rul_scale_lstm = train_lstm_ensemble(svdd_hi_train, train_bids, device)
    tf_models, rul_scale_tf = train_transformer_ensemble(svdd_hi_train, train_bids, device)
    
    # Predict
    X_lgb, _ = make_lgbm_features(svdd_hi_train[test_bid], start_obs=0)
    preds_lgb = np.maximum(lgbm_model.predict(X_lgb), 0.0)
    
    X_seq, _ = make_seqs(svdd_hi_train[test_bid], np.zeros(N_test), start_obs=0)
    Xt = torch.tensor(X_seq, dtype=torch.float32).to(device)
    
    # LSTM
    preds_lstm = []
    for m in lstm_models:
        m.eval()
        with torch.no_grad():
            preds_lstm.append(m(Xt).cpu().numpy() * rul_scale_lstm)
    preds_lstm = np.maximum(np.median(preds_lstm, axis=0), 0.0)
    
    # Transformer
    preds_tf = []
    for m in tf_models:
        m.eval()
        with torch.no_grad():
            preds_tf.append(m(Xt).cpu().numpy() * rul_scale_tf)
    preds_tf = np.maximum(np.median(preds_tf, axis=0), 0.0)
    
    # Ensemble
    sc_l = float(np.nanmean([competition_score(N_test - obs, p) for obs, p in zip(obs_pts, preds_lgb)]))
    sc_lm = float(np.nanmean([competition_score(N_test - obs, p) for obs, p in zip(obs_pts, preds_lstm)]))
    sc_t = float(np.nanmean([competition_score(N_test - obs, p) for obs, p in zip(obs_pts, preds_tf)]))
    
    w_l, w_lm, w_tf = max(sc_l, 1e-4), max(sc_lm, 1e-4), max(sc_t, 1e-4)
    total_w = w_l + w_lm + w_tf
    w_l, w_lm, w_tf = w_l / total_w, w_lm / total_w, w_tf / total_w
    
    preds_ens = w_l * preds_lgb + w_lm * preds_lstm + w_tf * preds_tf
    sc_ens = float(np.nanmean([competition_score(N_test - obs, p) for obs, p in zip(obs_pts, preds_ens)]))
    
    print(f"Bearing {test_bid}: LGBM={sc_l:.4f}, LSTM={sc_lm:.4f}, TF={sc_t:.4f}, Ens={sc_ens:.4f}")
