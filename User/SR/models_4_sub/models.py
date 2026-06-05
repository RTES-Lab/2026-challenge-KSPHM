"""
RUL model implementations — all share a common interface:

  model.fit(data, train_bids, eol_dict, nu_dict, win_size)
  model.predict(hi_window, regime_window=None)  →  float  (RUL in cycles)

LOOCV leakage contract
-----------------------
- fit() may only touch data[b] for b in train_bids
- All normalisation statistics computed inside fit() from train_bids only
- predict() uses only the supplied window; no external state from test bearing
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from utils import rul_labels, extract_window_features, build_window_dataset

try:
    from sklearn.neighbors    import KNeighborsRegressor
    from sklearn.ensemble     import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.svm          import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════════════════
class BaseRULModel:
    name = "Base"

    def fit(self, data, train_bids, eol_dict, nu_dict, win_size):
        raise NotImplementedError

    def predict(self, hi_window, regime_window=None):
        raise NotImplementedError

    def __repr__(self):
        return self.name


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Slope Extrapolate
# ═══════════════════════════════════════════════════════════════════════════════
class SlopeExtrapolate(BaseRULModel):
    """
    Fit a line to the HI window; extrapolate forward to failure_hi.
    failure_hi and flat_rul are learned from training bearings → no leakage.
    Falls back to flat_rul when slope ≤ 0 (normal phase).
    """
    name = "SlopeExtrapolate"

    def fit(self, data, train_bids, eol_dict, nu_dict, win_size):
        self.failure_hi = float(np.mean([data[b]["hi"][-1] for b in train_bids]))
        self.flat_rul   = float(np.mean([eol_dict[b] - nu_dict[b]
                                         for b in train_bids]))
        # Learn typical degradation slope from the degradation phase of each
        # training bearing so we can gate out near-zero slopes (flat phase).
        deg_slopes = []
        for b in train_bids:
            hi  = data[b]["hi"]
            nu  = nu_dict[b]
            eol = eol_dict[b]
            if eol > nu + 5:
                t = np.arange(eol - nu, dtype=float)
                s = float(np.polyfit(t, hi[nu:eol], 1)[0])
                if s > 0:
                    deg_slopes.append(s)
        # Threshold = 5% of typical degradation slope to filter normal-phase noise
        self.slope_threshold = (float(np.mean(deg_slopes)) * 0.05
                                 if deg_slopes else 1e-4)
        # Maximum sensible prediction: cap at 3 × flat_rul to prevent wild outliers
        self.max_rul = self.flat_rul * 3.0

    def predict(self, hi_window, regime_window=None):
        W = len(hi_window)
        if W < 2:
            return self.flat_rul
        t      = np.arange(W, dtype=float)
        slope  = float(np.polyfit(t, hi_window, 1)[0])
        hi_now = float(hi_window[-1])
        if slope < self.slope_threshold:
            return self.flat_rul
        remaining = (self.failure_hi - hi_now) / slope
        return float(np.clip(remaining, 0.0, self.max_rul))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Level Lookup  (histogram-based)
# ═══════════════════════════════════════════════════════════════════════════════
class LevelLookup(BaseRULModel):
    """
    Bin HI end-value into N_BINS; return mean RUL of training windows with
    the same HI level.  No ML training required.
    """
    name = "LevelLookup"
    N_BINS = 25

    def fit(self, data, train_bids, eol_dict, nu_dict, win_size):
        bin_ruls = [[] for _ in range(self.N_BINS)]
        for b in train_bids:
            hi  = data[b]["hi"]
            rul = rul_labels(len(hi), eol_dict[b], nu_dict[b])
            for i in range(win_size - 1, len(hi)):
                idx = min(int(hi[i] * self.N_BINS), self.N_BINS - 1)
                bin_ruls[max(idx, 0)].append(float(rul[i]))
        all_ruls = [r for g in bin_ruls for r in g]
        self.bin_means = np.array([
            np.mean(g) if g else (np.mean(all_ruls) if all_ruls else 0.0)
            for g in bin_ruls
        ])

    def predict(self, hi_window, regime_window=None):
        hi_end = float(hi_window[-1])
        idx = min(int(hi_end * self.N_BINS), self.N_BINS - 1)
        return float(self.bin_means[max(idx, 0)])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Linear (Ridge)
# ═══════════════════════════════════════════════════════════════════════════════
class LinearModel(BaseRULModel):
    """Ridge regression on 8-dim window feature vector."""
    name = "Linear"

    def fit(self, data, train_bids, eol_dict, nu_dict, win_size):
        assert HAS_SKLEARN
        X, y = build_window_dataset(data, train_bids, eol_dict, nu_dict, win_size)
        self.scaler = StandardScaler().fit(X)
        self.model  = Ridge(alpha=1.0).fit(self.scaler.transform(X), y)

    def predict(self, hi_window, regime_window=None):
        f = extract_window_features(hi_window, regime_window)
        return max(float(self.model.predict(
            self.scaler.transform(f.reshape(1, -1)))[0]), 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. k-NN
# ═══════════════════════════════════════════════════════════════════════════════
class KNNModel(BaseRULModel):
    """k-Nearest Neighbours (inverse-distance weighted) on window features."""
    name = "KNN"

    def __init__(self, k=10):
        self.k = k

    def fit(self, data, train_bids, eol_dict, nu_dict, win_size):
        assert HAS_SKLEARN
        X, y = build_window_dataset(data, train_bids, eol_dict, nu_dict, win_size)
        self.scaler = StandardScaler().fit(X)
        Xn = self.scaler.transform(X)
        self.knn = KNeighborsRegressor(
            n_neighbors=min(self.k, len(y)),
            weights="distance", metric="euclidean",
        ).fit(Xn, y)

    def predict(self, hi_window, regime_window=None):
        f = extract_window_features(hi_window, regime_window)
        fn = self.scaler.transform(f.reshape(1, -1))
        return max(float(self.knn.predict(fn)[0]), 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Random Forest
# ═══════════════════════════════════════════════════════════════════════════════
class RFModel(BaseRULModel):
    """Random Forest regression on window features."""
    name = "RF"

    def __init__(self, n_estimators=300, random_state=42):
        self.n_estimators  = n_estimators
        self.random_state  = random_state

    def fit(self, data, train_bids, eol_dict, nu_dict, win_size):
        assert HAS_SKLEARN
        X, y = build_window_dataset(data, train_bids, eol_dict, nu_dict, win_size)
        self.scaler = StandardScaler().fit(X)
        self.model  = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(self.scaler.transform(X), y)

    def predict(self, hi_window, regime_window=None):
        f = extract_window_features(hi_window, regime_window)
        return max(float(self.model.predict(
            self.scaler.transform(f.reshape(1, -1)))[0]), 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LightGBM
# ═══════════════════════════════════════════════════════════════════════════════
class LGBMModel(BaseRULModel):
    """LightGBM gradient-boosting regression on window features."""
    name = "LGBM"

    def __init__(self, n_estimators=400, learning_rate=0.05, random_state=42):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.random_state  = random_state

    def fit(self, data, train_bids, eol_dict, nu_dict, win_size):
        assert HAS_LGBM, "lightgbm not installed"
        X, y = build_window_dataset(data, train_bids, eol_dict, nu_dict, win_size)
        self.scaler = StandardScaler().fit(X)
        self.model  = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=-1,
        ).fit(self.scaler.transform(X), y)

    def predict(self, hi_window, regime_window=None):
        f = extract_window_features(hi_window, regime_window)
        return max(float(self.model.predict(
            self.scaler.transform(f.reshape(1, -1)))[0]), 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SVR
# ═══════════════════════════════════════════════════════════════════════════════
class SVRModel(BaseRULModel):
    """Support Vector Regression (RBF kernel) on window features."""
    name = "SVR"

    def fit(self, data, train_bids, eol_dict, nu_dict, win_size):
        assert HAS_SKLEARN
        X, y = build_window_dataset(data, train_bids, eol_dict, nu_dict, win_size)
        self.scaler = StandardScaler().fit(X)
        Xn = self.scaler.transform(X)
        self.model = SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)
        self.model.fit(Xn, y)

    def predict(self, hi_window, regime_window=None):
        f = extract_window_features(hi_window, regime_window)
        return max(float(self.model.predict(
            self.scaler.transform(f.reshape(1, -1)))[0]), 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Gaussian Process
# ═══════════════════════════════════════════════════════════════════════════════
class GPRModel(BaseRULModel):
    """
    Gaussian Process Regression (RBF + White noise kernel).
    Capped at 2000 training samples for speed.
    """
    name = "GPR"
    MAX_TRAIN = 2000

    def fit(self, data, train_bids, eol_dict, nu_dict, win_size):
        assert HAS_SKLEARN
        X, y = build_window_dataset(data, train_bids, eol_dict, nu_dict, win_size)
        self.scaler = StandardScaler().fit(X)
        Xn = self.scaler.transform(X)
        if len(Xn) > self.MAX_TRAIN:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(Xn), self.MAX_TRAIN, replace=False)
            Xn, y = Xn[idx], y[idx]
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        self.model = GaussianProcessRegressor(
            kernel=kernel, alpha=0.0, n_restarts_optimizer=2, normalize_y=True
        ).fit(Xn, y)

    def predict(self, hi_window, regime_window=None):
        f = extract_window_features(hi_window, regime_window)
        mu = float(self.model.predict(
            self.scaler.transform(f.reshape(1, -1)))[0])
        return max(mu, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. LSTM  (optional — requires PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════
if HAS_TORCH:
    class _LSTMNet(nn.Module):
        def __init__(self, n_in, hidden, n_layers):
            super().__init__()
            drop = 0.1 if n_layers > 1 else 0.0
            self.lstm = nn.LSTM(n_in, hidden, num_layers=n_layers,
                                batch_first=True, dropout=drop)
            self.fc   = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1)
            )

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h[-1]).squeeze(-1)

    class LSTMModel(BaseRULModel):
        """
        LSTM on sliding sequences of length seq_len.
        Input per time step: [hi_normalised, regime].
        Ensemble of n_seeds models for stability.
        No obs_frac → fully leakage-free.
        """
        name = "LSTM"

        def __init__(self, seq_len=10, hidden=64, n_layers=2,
                     epochs=150, n_seeds=5):
            self.seq_len  = seq_len
            self.hidden   = hidden
            self.n_layers = n_layers
            self.epochs   = epochs
            self.n_seeds  = n_seeds

        def _build_sequences(self, data, train_bids, eol_dict, nu_dict):
            X_list, y_list = [], []
            for b in train_bids:
                hi     = (data[b]["hi"] - self.hi_mean) / self.hi_std
                regime = data[b]["regime"]
                rul    = rul_labels(len(data[b]["hi"]),
                                    eol_dict[b], nu_dict[b])
                N = len(hi)
                for t_e in range(self.seq_len, N + 1):
                    t_s    = t_e - self.seq_len
                    hi_seq = hi[t_s:t_e]
                    rg_seq = (regime[t_s:t_e].astype(float)
                              if regime is not None
                              else np.full(self.seq_len, 0.5))
                    X_list.append(np.stack([hi_seq, rg_seq], axis=1))
                    y_list.append(float(rul[t_e - 1]) / self.rul_scale)
            return (torch.tensor(np.array(X_list), dtype=torch.float32),
                    torch.tensor(np.array(y_list),  dtype=torch.float32))

        def fit(self, data, train_bids, eol_dict, nu_dict, win_size):
            assert HAS_TORCH
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            all_hi = np.concatenate([data[b]["hi"] for b in train_bids])
            self.hi_mean   = float(all_hi.mean())
            self.hi_std    = float(all_hi.std() + 1e-8)
            self.rul_scale = float(max(eol_dict[b] for b in train_bids))

            X, y = self._build_sequences(data, train_bids, eol_dict, nu_dict)
            n_val  = max(1, int(len(X) * 0.1))
            tr_dl  = DataLoader(TensorDataset(X[:-n_val], y[:-n_val]),
                                batch_size=64, shuffle=True)
            val_dl = DataLoader(TensorDataset(X[-n_val:], y[-n_val:]),
                                batch_size=64)

            self.models = []
            crit = nn.MSELoss()
            for seed in range(self.n_seeds):
                torch.manual_seed(seed)
                m = _LSTMNet(2, self.hidden, self.n_layers).to(self.device)
                opt = torch.optim.Adam(m.parameters(), lr=1e-3)
                best_val, patience, best_state = np.inf, 0, None
                for _ in range(self.epochs):
                    m.train()
                    for xb, yb in tr_dl:
                        opt.zero_grad()
                        crit(m(xb.to(self.device)), yb.to(self.device)).backward()
                        opt.step()
                    m.eval()
                    with torch.no_grad():
                        vl = float(np.mean([
                            crit(m(xb.to(self.device)), yb.to(self.device)).item()
                            for xb, yb in val_dl
                        ]))
                    if vl < best_val:
                        best_val, patience = vl, 0
                        best_state = {k: v.cpu().clone()
                                      for k, v in m.state_dict().items()}
                    else:
                        patience += 1
                        if patience >= 20:
                            break
                if best_state:
                    m.load_state_dict(best_state)
                self.models.append(m)

        def predict(self, hi_window, regime_window=None):
            tail = (hi_window[-self.seq_len:]
                    if len(hi_window) >= self.seq_len
                    else np.pad(hi_window,
                                (self.seq_len - len(hi_window), 0), "edge"))
            hi_norm = (tail - self.hi_mean) / self.hi_std
            rg = (regime_window[-self.seq_len:].astype(float)
                  if regime_window is not None
                  else np.full(self.seq_len, 0.5))
            if len(rg) < self.seq_len:
                rg = np.pad(rg, (self.seq_len - len(rg), 0), "edge")
            seq = np.stack([hi_norm, rg], axis=1)
            x   = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(self.device)
            preds = []
            for m in self.models:
                m.eval()
                with torch.no_grad():
                    preds.append(float(m(x).item()) * self.rul_scale)
            return max(float(np.median(preds)), 0.0)

else:
    class LSTMModel(BaseRULModel):
        name = "LSTM"
        def fit(self, *a, **kw):
            raise ImportError("PyTorch not installed")
        def predict(self, *a, **kw):
            raise ImportError("PyTorch not installed")


# ═══════════════════════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════════════════════
BASE_MODELS = {
    "SlopeExtrapolate": SlopeExtrapolate,
    "LevelLookup":      LevelLookup,
    "Linear":           LinearModel,
    "KNN":              KNNModel,
    "RF":               RFModel,
    "LGBM":             LGBMModel if HAS_LGBM else None,
    "SVR":              SVRModel,
    "GPR":              GPRModel,
    "LSTM":             LSTMModel,
}

# Remove None entries (packages not available)
BASE_MODELS = {k: v for k, v in BASE_MODELS.items() if v is not None}
