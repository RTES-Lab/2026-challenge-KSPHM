"""New model definitions for Exp-K."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

EPS = 1e-8
MIN_RUL = 1.0

def minmax_norm(x):
    x = np.asarray(x, float)
    return (x - x.min()) / (x.max() - x.min() + EPS)

def slope_of(x):
    x = np.asarray(x, float)
    return float(np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x) >= 2 else 0.0

# ── Ridge ─────────────────────────────────────────────────────
def train_ridge(X, Y, alpha=10.0):
    return Ridge(alpha=alpha).fit(X, Y)

def predict_ridge(model, X):
    return np.maximum(model.predict(X), MIN_RUL)

# ── TCN-Res ───────────────────────────────────────────────────
class TCNRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(4, 32, 3, padding=1)
        self.c2 = nn.Conv1d(32, 32, 3, dilation=2, padding=2)
        self.c3 = nn.Conv1d(32, 32, 3, dilation=4, padding=4)
        self.c4 = nn.Conv1d(32, 32, 3, dilation=8, padding=8)
        self.skip = nn.Conv1d(4, 32, 1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.10)
        self.fc = nn.Sequential(nn.Linear(32, 24), nn.ReLU(), nn.Linear(24, 1))
    def forward(self, x):
        z = x.transpose(1, 2)
        s = self.skip(z)
        z = self.drop(self.act(self.c1(z)))
        z = self.drop(self.act(self.c2(z))) + s
        z = self.drop(self.act(self.c3(z)))
        z = self.act(self.c4(z)) + z
        return self.fc(z[:, :, -1]).squeeze(-1)

# ── MiniTransformer ───────────────────────────────────────────
class MiniTransformer(nn.Module):
    def __init__(self, d_in=4, d_model=32, nhead=2, nlayers=2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=0.2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.fc = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        z = self.proj(x)
        z = self.enc(z)
        return self.fc(z[:, -1, :]).squeeze(-1)

# ── BiLSTM ────────────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(4, 48, num_layers=2, batch_first=True,
                          dropout=0.15, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        y, _ = self.rnn(x)
        return self.fc(y[:, -1, :]).squeeze(-1)

# ── LGBM feature variants ────────────────────────────────────
def make_tabular_wide(hi_arr, seq=20, hi_mean_eol=0.75):
    """Wide window (SEQ=20) tabular features."""
    hi = np.asarray(hi_arr, float); hi0 = float(hi[0])
    x, obs = [], []
    for i in range(seq, len(hi)):
        w = hi[i - seq:i]; wn = minmax_norm(w)
        hf = float(np.clip(w[-1] / hi_mean_eol, 0.0, 2.0))
        x.append(list(wn) + [slope_of(wn), float(w[-1]), float(w.mean()),
                  float(w.max()), float(w.min()), float(w.std()),
                  slope_of(w), float(w[-1]-w[0]), float(w[-1]-hi0), hf, float(w[-1]*hf)])
        obs.append(i)
    return np.asarray(x), np.asarray(obs)

def make_tabular_hifeat(hi_arr, seq=10, hi_mean_eol=0.75):
    """Standard window + extra HI features (curvature, CV, accel)."""
    hi = np.asarray(hi_arr, float); hi0 = float(hi[0])
    x, obs = [], []
    for i in range(seq, len(hi)):
        w = hi[i - seq:i]; wn = minmax_norm(w)
        hf = float(np.clip(w[-1] / hi_mean_eol, 0.0, 2.0))
        grad2 = float(np.gradient(np.gradient(w))[-1])
        cv = float(w.std() / (w.mean() + EPS))
        half = seq // 2
        accel = slope_of(w[half:]) - slope_of(w[:half])
        x.append(list(wn) + [slope_of(wn), float(w[-1]), float(w.mean()),
                  float(w.max()), float(w.min()), float(w.std()),
                  slope_of(w), float(w[-1]-w[0]), float(w[-1]-hi0), hf, float(w[-1]*hf),
                  grad2, cv, accel])
        obs.append(i)
    return np.asarray(x), np.asarray(obs)

# ── DTW variants ──────────────────────────────────────────────
def seg_dist_exp(a, b, sigma=1.0):
    """Exponential distance weighting for DTW."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (0.25*abs(a[-1]-b[-1])/0.25 + 0.20*abs(a.mean()-b.mean())/0.25
         + 0.20*abs((a[-1]-a[0])-(b[-1]-b[0]))/0.25
         + 0.15*abs(slope_of(a)-slope_of(b))/0.03
         + 0.20*float(np.mean(np.abs(minmax_norm(a)-minmax_norm(b)))))
    return d

def predict_dtw_exp(hi_train, train_bids, hi_target, match_len=18, k=6, sigma=1.5):
    """DTW with exponential distance weighting."""
    hi_target = np.asarray(hi_target, float)
    obs_pts = np.arange(10, len(hi_target))
    preds = []
    for obs in obs_pts:
        l = min(match_len, obs)
        seg = hi_target[obs-l:obs]
        cands = []
        for b in train_bids:
            hi = np.asarray(hi_train[b], float)
            for end in range(l, len(hi)):
                d = seg_dist_exp(seg, hi[end-l:end])
                cands.append((d, max(len(hi)-end, MIN_RUL)))
        top = sorted(cands, key=lambda c: c[0])[:k]
        wt = np.asarray([np.exp(-d/sigma) for d, p in top])
        pv = np.asarray([p for d, p in top])
        preds.append(float(np.average(pv, weights=wt+EPS)))
    return obs_pts, np.asarray(preds)

def predict_dtw_adaptive_k(hi_train, train_bids, hi_target, match_len=18, k_max=10, dist_thresh=3.0):
    """DTW with adaptive k: only use neighbors within distance threshold."""
    hi_target = np.asarray(hi_target, float)
    obs_pts = np.arange(10, len(hi_target))
    preds = []
    for obs in obs_pts:
        l = min(match_len, obs)
        seg = hi_target[obs-l:obs]
        cands = []
        for b in train_bids:
            hi = np.asarray(hi_train[b], float)
            for end in range(l, len(hi)):
                d = seg_dist_exp(seg, hi[end-l:end])
                cands.append((d, max(len(hi)-end, MIN_RUL)))
        top = sorted(cands, key=lambda c: c[0])[:k_max]
        # Filter by threshold
        filtered = [(d, p) for d, p in top if d < dist_thresh]
        if len(filtered) < 3:
            filtered = top[:3]
        wt = np.asarray([1.0/(d+EPS) for d, p in filtered])
        pv = np.asarray([p for d, p in filtered])
        preds.append(float(np.average(pv, weights=wt)))
    return obs_pts, np.asarray(preds)
