"""
RUL Regime v6: 3-way ensemble (LGBM + Ridge + LSTM) with LOO-based weights
================================================================
HI source: SR/0603_v3/output/train/BearingX_HI.csv
           SR/0603_v3/output/test/TestX_HI.csv

v4 대비 변경:
  [추가] Ridge Regression (LGBM과 동일 피처, StandardScaler 전처리)
  [앙상블] LGBM + Ridge + LSTM 3-way
           가중치: LOO score 기반 자동 (타 3개 bearing 평균 score 비례)
           w_lgbm  = sc_lgbm  / (sc_lgbm + sc_ridge + sc_lstm)
           w_ridge = sc_ridge / (sc_lgbm + sc_ridge + sc_lstm)
           w_lstm  = sc_lstm  / (sc_lgbm + sc_ridge + sc_lstm)
  [유지]   conservative bias: T2/T5/T6 ×0.875
  [유지]   slope5/10/delta 피처, LSTM global norm

판단 기준:
  Ens_raw > 0.466 → v6 채택
  Ens_raw ≤ 0.466 → v4 유지
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
SR_BASE  = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0603_v3")
HI_TRAIN = SR_BASE / "output/train"
HI_TEST  = SR_BASE / "output/test"
OUT_DIR  = SR_BASE / "output/rul"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH      = 10
MEAN_TRAIN_LIFE = 116.5
INTERVAL_SEC    = 600
SEEDS           = [42, 7, 123, 0, 99]

NORMAL_UNTIL = {1: 89, 2: 92, 3: 62, 4: 78}
EOL          = {1: 126, 2: 114, 3: 89, 4: 137}

N_FEAT = 3  # [hi_norm, obs_frac, regime]


# ══════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════
def load_train_hi():
    data = {}
    for bid in BEARINGS:
        df = pd.read_csv(HI_TRAIN / f"Bearing{bid}_HI.csv")
        data[bid] = {
            "hi":     df["HI"].values.astype(float),
            "regime": df["regime"].values.astype(int),
        }
    return data


def load_test_hi():
    data = {}
    for tid in TEST_IDS:
        df = pd.read_csv(HI_TEST / f"Test{tid}_HI.csv")
        data[tid] = {
            "hi":     df["HI"].values.astype(float),
            "regime": df["regime"].values.astype(int),
        }
    return data


def rul_labels(n_total: int, bid: int) -> np.ndarray:
    nu, eol = NORMAL_UNTIL[bid], EOL[bid]
    idx = np.arange(n_total)
    return np.where(idx <= nu, eol - nu, np.maximum(eol - idx, 0)).astype(float)


# ══════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════
def comp_score(rul_true: float, rul_pred: float) -> float:
    if rul_true <= 0:
        return np.nan
    Er = 100.0 * (rul_true - rul_pred) / rul_true
    return (np.exp(-np.log(0.5) * Er / 20.0) if Er <= 0
            else np.exp(np.log(0.5) * Er / 50.0))


def avg_score(N: int, obs_pts, preds) -> float:
    return float(np.nanmean([comp_score(N - obs, p)
                              for obs, p in zip(obs_pts, preds)]))


# ══════════════════════════════════════════════════════════════════
# 시작 위치 추정
# ══════════════════════════════════════════════════════════════════
def estimate_start_frac(hi_start: float, train_data: dict,
                         ref_bids: list) -> float:
    fracs = []
    for bid in ref_bids:
        hi = train_data[bid]["hi"]
        n  = len(hi)
        if hi[0] >= hi_start:
            continue
        exceed = np.where(hi >= hi_start)[0]
        if len(exceed) == 0:
            fracs.append(float(n) / MEAN_TRAIN_LIFE)
        else:
            fracs.append(float(exceed[0]) / MEAN_TRAIN_LIFE)
    return float(np.mean(fracs)) if fracs else 0.0


# ══════════════════════════════════════════════════════════════════
# HI global statistics (LOO or full)
# ══════════════════════════════════════════════════════════════════
def compute_hi_stats(train_data: dict, bids: list) -> tuple:
    all_hi = np.concatenate([train_data[b]["hi"] for b in bids])
    return float(all_hi.mean()), float(all_hi.std() + 1e-8)


# ══════════════════════════════════════════════════════════════════
# Shared features (LGBM & Ridge use the same)
# ══════════════════════════════════════════════════════════════════
def make_lgbm_features(hi_arr: np.ndarray, regime_arr: np.ndarray,
                        seq_len: int = SEQ_LENGTH,
                        start_frac: float = 0.0):
    feats, targets = [], []
    N = len(hi_arr)
    t = np.arange(seq_len, dtype=float)
    hi_obs_start = float(hi_arr[0])
    for i in range(seq_len, N):
        win   = hi_arr[i - seq_len: i]
        reg_w = regime_arr[i - seq_len: i]
        slope        = float(np.polyfit(t, win, 1)[0])
        elapsed_frac = np.clip(start_frac + float(i) / MEAN_TRAIN_LIFE, 0.0, 3.0)
        hi_slope_5   = float((win[-1] - win[-6]) / 5)
        hi_slope_10  = float((win[-1] - win[0]) / (seq_len - 1))
        hi_delta     = float(win[-1] - hi_obs_start)
        feats.append([
            *win,
            slope,
            float(win.mean()),
            float(win.std()),
            float(win.max()),
            float(win[-1]),
            float(win[-1] - win[0]),
            float(regime_arr[i]),
            float(reg_w.mean()),
            elapsed_frac,
            hi_slope_5,
            hi_slope_10,
            hi_delta,
        ])
        targets.append(float(N - i))
    return np.array(feats), np.array(targets)


# ══════════════════════════════════════════════════════════════════
# LGBM
# ══════════════════════════════════════════════════════════════════
def lgbm_asym_obj(y_pred, dataset):
    y_true = dataset.get_label()
    diff   = y_true - y_pred
    w      = np.where(diff < 0, 2.5, 1.0)
    return -diff * w, np.ones_like(diff) * w


def train_lgbm(train_data: dict, train_bids: list) -> lgb.Booster:
    X_list, y_list = [], []
    for b in train_bids:
        x, y = make_lgbm_features(train_data[b]["hi"], train_data[b]["regime"],
                                    start_frac=0.0)
        X_list.append(x); y_list.append(y)
    dtrain = lgb.Dataset(np.concatenate(X_list), label=np.concatenate(y_list))
    return lgb.train(
        {"num_leaves": 15, "learning_rate": 0.05,
         "min_child_samples": 5, "verbose": -1,
         "objective": lgbm_asym_obj},
        dtrain, num_boost_round=200,
    )


def predict_lgbm(model: lgb.Booster, hi_arr: np.ndarray,
                  regime_arr: np.ndarray,
                  start_frac: float = 0.0) -> np.ndarray:
    X, _ = make_lgbm_features(hi_arr, regime_arr, start_frac=start_frac)
    return np.maximum(model.predict(X), 0.0)


# ══════════════════════════════════════════════════════════════════
# Ridge
# ══════════════════════════════════════════════════════════════════
def train_ridge(train_data: dict, train_bids: list) -> Pipeline:
    X_list, y_list = [], []
    for b in train_bids:
        x, y = make_lgbm_features(train_data[b]["hi"], train_data[b]["regime"],
                                    start_frac=0.0)
        X_list.append(x); y_list.append(y)
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    model.fit(X, y)
    return model


def predict_ridge(model: Pipeline, hi_arr: np.ndarray,
                   regime_arr: np.ndarray,
                   start_frac: float = 0.0) -> np.ndarray:
    X, _ = make_lgbm_features(hi_arr, regime_arr, start_frac=start_frac)
    return np.maximum(model.predict(X), 0.0)


# ══════════════════════════════════════════════════════════════════
# LSTM
# ══════════════════════════════════════════════════════════════════
def make_seqs(hi_arr: np.ndarray, regime_arr: np.ndarray,
              rul_arr: np.ndarray, seq_len: int,
              hi_mean: float, hi_std: float,
              start_obs: int = 0):
    n = len(hi_arr)
    X, y = [], []
    for i in range(n - seq_len):
        win      = hi_arr[i: i + seq_len].copy()
        win_norm = (win - hi_mean) / hi_std
        obs_frac = np.clip(
            (start_obs + i + np.arange(seq_len)) / MEAN_TRAIN_LIFE, 0.0, 2.0)
        reg_seq  = regime_arr[i: i + seq_len].astype(float)
        X.append(np.stack([win_norm, obs_frac, reg_seq], axis=1))
        y.append(float(rul_arr[i + seq_len]))
    return np.array(X), np.array(y)


class LSTMRegressor(nn.Module):
    def __init__(self, n_feat: int = N_FEAT, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=n_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               rul_scale: float, seed: int, device) -> LSTMRegressor:
    torch.manual_seed(seed)
    y_norm = y_train / rul_scale
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_norm,  dtype=torch.float32)
    n_val  = max(1, int(len(Xt) * 0.1))
    tr_dl  = DataLoader(TensorDataset(Xt[:-n_val], yt[:-n_val]),
                        batch_size=64, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xt[-n_val:], yt[-n_val:]),
                        batch_size=64)
    model = LSTMRegressor().to(device)
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
            vl = float(np.mean([
                crit(model(xb.to(device)), yb.to(device)).item()
                for xb, yb in val_dl
            ]))
        if vl < best_val:
            best_val, patience = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    model.load_state_dict(best_state)
    return model


def build_lstm_preds(train_data: dict, train_bids: list,
                      test_bid: int, device,
                      hi_mean: float, hi_std: float) -> tuple:
    X_list, y_list = [], []
    for b in train_bids:
        hi_b  = train_data[b]["hi"]
        reg_b = train_data[b]["regime"]
        rul_b = rul_labels(len(hi_b), b)
        X, y  = make_seqs(hi_b, reg_b, rul_b, SEQ_LENGTH,
                           hi_mean, hi_std, start_obs=0)
        X_list.append(X); y_list.append(y)
    X_train   = np.concatenate(X_list)
    y_train   = np.concatenate(y_list)
    rul_scale = float(max(EOL[b] for b in train_bids))

    hi_t  = train_data[test_bid]["hi"]
    reg_t = train_data[test_bid]["regime"]
    rul_t = rul_labels(len(hi_t), test_bid)
    X_test, _ = make_seqs(hi_t, reg_t, rul_t, SEQ_LENGTH,
                           hi_mean, hi_std, start_obs=0)
    Xt = torch.tensor(X_test, dtype=torch.float32).to(device)

    all_preds = []
    for s in SEEDS:
        m = train_lstm(X_train, y_train, rul_scale, s, device)
        m.eval()
        with torch.no_grad():
            all_preds.append(
                np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0))
    return np.median(all_preds, axis=0), rul_scale


# ══════════════════════════════════════════════════════════════════
# CF search (leakage-free)
# ══════════════════════════════════════════════════════════════════
def find_fold_cf(results: dict, test_bid: int,
                 pred_key: str = "ens_raw") -> tuple:
    other_bids = [b for b in BEARINGS if b != test_bid]
    best_cf, best_sc = 1.0, -np.inf
    for cf in np.arange(0.60, 1.41, 0.01):
        sc_list = []
        for bid in other_bids:
            N   = results[bid]["N"]
            obs = results[bid]["obs_pts"]
            preds = [p * cf for p in results[bid][pred_key]]
            sc_list.append(avg_score(N, obs, preds))
        mean_sc = float(np.mean(sc_list))
        if mean_sc > best_sc:
            best_sc, best_cf = mean_sc, float(cf)
    return best_cf, best_sc


def fold_ensemble_weights(results: dict, test_bid: int) -> tuple:
    other_bids = [b for b in BEARINGS if b != test_bid]
    sc_lgbm  = float(np.mean([results[b]["sc_lgbm"]  for b in other_bids]))
    sc_ridge = float(np.mean([results[b]["sc_ridge"]  for b in other_bids]))
    sc_lstm  = float(np.mean([results[b]["sc_lstm"]   for b in other_bids]))
    total    = sc_lgbm + sc_ridge + sc_lstm + 1e-12
    w_lgbm   = float(sc_lgbm  / total)
    w_ridge  = float(sc_ridge / total)
    w_lstm   = float(sc_lstm  / total)
    return w_lgbm, w_ridge, w_lstm


# ══════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════
def _draw_rul_ax(ax, obs, true_rul, r, title):
    ax.plot(obs, true_rul,         "k-",  lw=1.5, label="True RUL")
    ax.plot(obs, r["preds_lgbm"],  "r--", lw=1,   alpha=0.6,
            label=f"LGBM {r['sc_lgbm']:.3f}")
    ax.plot(obs, r["preds_ridge"], "g--", lw=1,   alpha=0.6,
            label=f"Ridge {r['sc_ridge']:.3f}")
    ax.plot(obs, r["preds_lstm"],  "b--", lw=1,   alpha=0.6,
            label=f"LSTM {r['sc_lstm']:.3f}")
    ax.plot(obs, r["preds_cal"],   "m-",  lw=2,
            label=f"Ens+CF {r['sc_cal']:.3f}")
    ax.set_title(
        f"{title}  [w=({r['w_lgbm']:.2f},{r['w_ridge']:.2f},{r['w_lstm']:.2f})  cf={r['fold_cf']:.2f}]",
        fontsize=8)
    ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.4)


def _draw_test_ax(ax, obs, preds_lgbm, preds_ridge, preds_lstm, preds_final,
                   cf, start_frac, title):
    ax.plot(obs, preds_lgbm,  "r--", lw=1, alpha=0.6, label="LGBM")
    ax.plot(obs, preds_ridge, "g--", lw=1, alpha=0.6, label="Ridge")
    ax.plot(obs, preds_lstm,  "b--", lw=1, alpha=0.6, label="LSTM")
    ax.plot(obs, preds_final, "m-",  lw=2,
            label=f"Final (cf={cf:.2f})")
    final_hr = float(preds_final[-1]) * INTERVAL_SEC / 3600
    sf_pct   = int(start_frac * 100)
    ax.set_title(f"{title}  RUL={final_hr:.1f}hr  (start≈{sf_pct}%)", fontsize=9)
    ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.4)


# ══════════════════════════════════════════════════════════════════
# LOOCV
# ══════════════════════════════════════════════════════════════════
def run_loocv():
    print("=" * 70)
    print("  RUL Regime v6 — 3-way ensemble: LGBM + Ridge + LSTM (LOO weights)")
    print("=" * 70)

    train_data = load_train_hi()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    results = {}

    # Step 1: raw predictions per fold
    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        hi_t  = train_data[test_bid]["hi"]
        reg_t = train_data[test_bid]["regime"]
        N     = len(hi_t)
        obs_pts = np.arange(SEQ_LENGTH, N)

        hi_mean, hi_std = compute_hi_stats(train_data, train_bids)

        print(f"[Fold B{test_bid}]  train: {train_bids}"
              f"  hi_mean={hi_mean:.4f}  hi_std={hi_std:.4f}")

        # LGBM
        lgbm_model = train_lgbm(train_data, train_bids)
        preds_lgbm = predict_lgbm(lgbm_model, hi_t, reg_t, start_frac=0.0)
        sc_lgbm    = avg_score(N, obs_pts, preds_lgbm)

        # Ridge
        ridge_model = train_ridge(train_data, train_bids)
        preds_ridge = predict_ridge(ridge_model, hi_t, reg_t, start_frac=0.0)
        sc_ridge    = avg_score(N, obs_pts, preds_ridge)

        # LSTM
        preds_lstm, _ = build_lstm_preds(
            train_data, train_bids, test_bid, device, hi_mean, hi_std)
        sc_lstm = avg_score(N, obs_pts, preds_lstm)

        print(f"  LGBM={sc_lgbm:.4f}  Ridge={sc_ridge:.4f}  LSTM={sc_lstm:.4f}")
        results[test_bid] = {
            "N": N, "obs_pts": list(obs_pts),
            "preds_lgbm":  list(preds_lgbm),
            "preds_ridge": list(preds_ridge),
            "preds_lstm":  list(preds_lstm),
            "sc_lgbm": sc_lgbm, "sc_ridge": sc_ridge, "sc_lstm": sc_lstm,
        }

    # Step 2: ensemble weights (LOO score-based, 3-way)
    for test_bid in BEARINGS:
        w_lgbm, w_ridge, w_lstm = fold_ensemble_weights(results, test_bid)
        preds_ens = [
            w_lgbm * l + w_ridge * rg + w_lstm * a
            for l, rg, a in zip(results[test_bid]["preds_lgbm"],
                                 results[test_bid]["preds_ridge"],
                                 results[test_bid]["preds_lstm"])
        ]
        sc_ens = avg_score(results[test_bid]["N"],
                           results[test_bid]["obs_pts"], preds_ens)
        results[test_bid].update({
            "ens_raw":    preds_ens,
            "w_lgbm":     w_lgbm,
            "w_ridge":    w_ridge,
            "w_lstm":     w_lstm,
            "sc_ens_raw": sc_ens,
        })

    # Step 3: CF search
    print("\n[Fold-wise CF search (other 3 bearings only, range 0.60~1.40)]")
    for test_bid in BEARINGS:
        cf, cf_ref_sc = find_fold_cf(results, test_bid, pred_key="ens_raw")
        preds_cal     = [p * cf for p in results[test_bid]["ens_raw"]]
        sc_cal        = avg_score(
            results[test_bid]["N"], results[test_bid]["obs_pts"], preds_cal)
        results[test_bid].update({
            "fold_cf": cf, "preds_cal": preds_cal,
            "sc_cal": sc_cal, "cf_ref_score": cf_ref_sc,
        })
        r = results[test_bid]
        print(f"  B{test_bid}: w=({r['w_lgbm']:.2f},{r['w_ridge']:.2f},{r['w_lstm']:.2f})"
              f"  cf={cf:.2f} → sc_ens={r['sc_ens_raw']:.4f}  sc_cal={sc_cal:.4f}")

    # Step 4: summary
    print(f"\n{'='*70}")
    print("  LOOCV Summary (v6 — 3-way)")
    print(f"{'='*70}")
    print(f"  {'Bearing':>8} | {'LGBM':>8} | {'Ridge':>8} | {'LSTM':>8} | {'Ens(raw)':>10} | {'CF':>5} | {'Ens+CF':>8}")
    print(f"  {'-'*72}")
    sc_lists = {k: [] for k in ["lgbm", "ridge", "lstm", "ens", "cal"]}
    for test_bid in BEARINGS:
        r = results[test_bid]
        print(f"  {test_bid:>8} | {r['sc_lgbm']:>8.4f} | {r['sc_ridge']:>8.4f} | "
              f"{r['sc_lstm']:>8.4f} | {r['sc_ens_raw']:>10.4f} | "
              f"{r['fold_cf']:>5.2f} | {r['sc_cal']:>8.4f}")
        for k in ["lgbm", "ridge", "lstm", "ens", "cal"]:
            sc_lists[k].append(r[f"sc_{k}" if k != "ens" else "sc_ens_raw"])

    mean_ens = np.mean(sc_lists["ens"])
    mean_cal = np.mean(sc_lists["cal"])
    print(f"  {'avg':>8} | {np.mean(sc_lists['lgbm']):>8.4f} | "
          f"{np.mean(sc_lists['ridge']):>8.4f} | "
          f"{np.mean(sc_lists['lstm']):>8.4f} | "
          f"{mean_ens:>10.4f} | {'—':>5} | {mean_cal:>8.4f}")
    v4_baseline = 0.466
    mark = "★ BETTER than v4" if mean_ens > v4_baseline else "✗ NOT better than v4"
    print(f"\n  Ens_raw={mean_ens:.4f}  (v4 baseline={v4_baseline})  {mark}")
    print(f"  Ens+CF ={mean_cal:.4f}")

    # Log
    with open(OUT_DIR / "loocv_log_v6.txt", "w") as f:
        f.write("RUL Regime v6 — 3-way ensemble: LGBM + Ridge + LSTM\n")
        f.write(f"LGBM  mean: {np.mean(sc_lists['lgbm']):.4f}\n")
        f.write(f"Ridge mean: {np.mean(sc_lists['ridge']):.4f}\n")
        f.write(f"LSTM  mean: {np.mean(sc_lists['lstm']):.4f}\n")
        f.write(f"Ens (raw):  {mean_ens:.4f}\n")
        f.write(f"Ens + CF :  {mean_cal:.4f}\n\n")
        for test_bid in BEARINGS:
            r = results[test_bid]
            f.write(f"  B{test_bid}: LGBM={r['sc_lgbm']:.4f}  Ridge={r['sc_ridge']:.4f}  "
                    f"LSTM={r['sc_lstm']:.4f}  Ens={r['sc_ens_raw']:.4f}  "
                    f"CF={r['fold_cf']:.2f}  "
                    f"w=({r['w_lgbm']:.3f},{r['w_ridge']:.3f},{r['w_lstm']:.3f})  "
                    f"final={r['sc_cal']:.4f}\n")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("RUL Regime v6 — 3-way Ensemble LOOCV", fontsize=12)
    axes = axes.flatten()
    for i, test_bid in enumerate(BEARINGS):
        r   = results[test_bid]
        obs = r["obs_pts"]
        N   = r["N"]
        true_rul = [N - o for o in obs]
        _draw_rul_ax(axes[i], obs, true_rul, r, f"Bearing{test_bid}")

        fig_i, ax_i = plt.subplots(figsize=(9, 5))
        _draw_rul_ax(ax_i, obs, true_rul, r, f"Bearing{test_bid}")
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Bearing{test_bid}_RUL_v6.png", dpi=150)
        plt.close(fig_i)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "loocv_predictions_v6.png", dpi=150)
    plt.close()

    return results, mean_ens


# ══════════════════════════════════════════════════════════════════
# Test Inference
# ══════════════════════════════════════════════════════════════════
def run_test_inference(loocv_results: dict):
    print(f"\n{'='*70}")
    print("  Test inference — 전체 Train 4개 + 시작 위치 추정")
    print(f"{'='*70}")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = load_train_hi()
    test_data  = load_test_hi()

    hi_mean, hi_std = compute_hi_stats(train_data, BEARINGS)
    print(f"  Full train HI: mean={hi_mean:.4f}  std={hi_std:.4f}")

    # Test weights: average of LOO weights across all 4 folds
    w_lgbm  = float(np.mean([loocv_results[b]["w_lgbm"]  for b in BEARINGS]))
    w_ridge = float(np.mean([loocv_results[b]["w_ridge"]  for b in BEARINGS]))
    w_lstm  = float(np.mean([loocv_results[b]["w_lstm"]   for b in BEARINGS]))
    # re-normalize in case of floating point drift
    total   = w_lgbm + w_ridge + w_lstm
    w_lgbm /= total; w_ridge /= total; w_lstm /= total
    cf_test = float(np.mean([loocv_results[b]["fold_cf"]  for b in BEARINGS]))
    print(f"  Test weights: LGBM={w_lgbm:.3f}, Ridge={w_ridge:.3f}, LSTM={w_lstm:.3f}")
    print(f"  Test CF:      {cf_test:.2f}")

    # Train models on full dataset
    print("  LGBM training (full train)...")
    lgbm_model  = train_lgbm(train_data, BEARINGS)

    print("  Ridge training (full train)...")
    ridge_model = train_ridge(train_data, BEARINGS)

    print("  LSTM training (full train, 5 seeds)...")
    X_all, y_all = [], []
    for b in BEARINGS:
        hi_b  = train_data[b]["hi"]
        reg_b = train_data[b]["regime"]
        rul_b = rul_labels(len(hi_b), b)
        X, y  = make_seqs(hi_b, reg_b, rul_b, SEQ_LENGTH,
                           hi_mean, hi_std, start_obs=0)
        X_all.append(X); y_all.append(y)
    X_train   = np.concatenate(X_all)
    y_train   = np.concatenate(y_all)
    rul_scale = float(max(EOL.values()))

    lstm_models = []
    for s in SEEDS:
        m = train_lstm(X_train, y_train, rul_scale, s, device)
        lstm_models.append(m)
        print(f"    seed={s} done")

    summary_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"RUL Regime v6 — Test  w=({w_lgbm:.2f},{w_ridge:.2f},{w_lstm:.2f})  "
        f"cf={cf_test:.2f}  bias×0.875 for T2/5/6", fontsize=11)
    axes = axes.flatten()

    for i, tid in enumerate(TEST_IDS):
        hi_t    = test_data[tid]["hi"]
        reg_t   = test_data[tid]["regime"]
        N       = len(hi_t)
        obs_pts = np.arange(SEQ_LENGTH, N)

        hi_start   = float(hi_t[0])
        start_frac = estimate_start_frac(hi_start, train_data, BEARINGS)
        start_obs  = int(start_frac * MEAN_TRAIN_LIFE)
        print(f"\n  [Test{tid}] hi_start={hi_start:.3f}  "
              f"→ start_frac={start_frac:.3f}  start_obs≈{start_obs}cyc")

        # LGBM
        preds_lgbm = predict_lgbm(lgbm_model, hi_t, reg_t, start_frac=start_frac)

        # Ridge
        preds_ridge = predict_ridge(ridge_model, hi_t, reg_t, start_frac=start_frac)

        # LSTM
        X_test = []
        for j in range(N - SEQ_LENGTH):
            win      = hi_t[j: j + SEQ_LENGTH].copy()
            win_norm = (win - hi_mean) / hi_std
            obs_frac = np.clip(
                (start_obs + j + np.arange(SEQ_LENGTH)) / MEAN_TRAIN_LIFE,
                0.0, 2.0)
            rg = reg_t[j: j + SEQ_LENGTH].astype(float)
            X_test.append(np.stack([win_norm, obs_frac, rg], axis=1))
        Xt = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)

        all_lstm = []
        for m in lstm_models:
            m.eval()
            with torch.no_grad():
                all_lstm.append(
                    np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0))
        preds_lstm = np.median(all_lstm, axis=0)

        # 3-way ensemble + CF
        preds_ens   = w_lgbm * preds_lgbm + w_ridge * preds_ridge + w_lstm * preds_lstm
        preds_final = preds_ens * cf_test

        # Conservative bias: T2/T5/T6
        FLAT_IDS            = {2, 5, 6}
        CONSERVATIVE_FACTOR = 0.875
        if tid in FLAT_IDS:
            preds_final = preds_final * CONSERVATIVE_FACTOR
            bias_note   = f"  conservative×{CONSERVATIVE_FACTOR}"
        else:
            bias_note   = ""

        final_cyc = float(preds_final[-1])
        final_hr  = final_cyc * INTERVAL_SEC / 3600
        print(f"    RUL={final_hr:.2f}hr ({final_cyc:.1f}cyc)  "
              f"LGBM={preds_lgbm[-1]:.1f}  Ridge={preds_ridge[-1]:.1f}  "
              f"LSTM={preds_lstm[-1]:.1f}{bias_note}")

        pd.DataFrame({
            "obs_cycle":    obs_pts,
            "preds_lgbm":   preds_lgbm,
            "preds_ridge":  preds_ridge,
            "preds_lstm":   preds_lstm,
            "preds_final":  preds_final,
            "rul_hours":    preds_final * INTERVAL_SEC / 3600,
        }).to_csv(OUT_DIR / f"Test{tid}_RUL_v6.csv", index=False)

        summary_rows.append({
            "test_id":          tid,
            "hi_start":         round(hi_start, 4),
            "start_frac":       round(start_frac, 3),
            "start_obs_est":    start_obs,
            "final_rul_cycles": round(final_cyc, 2),
            "final_rul_hours":  round(final_hr, 2),
            "cf":               round(cf_test, 2),
            "w_lgbm":           round(w_lgbm, 3),
            "w_ridge":          round(w_ridge, 3),
            "w_lstm":           round(w_lstm, 3),
        })

        _draw_test_ax(axes[i], obs_pts,
                      preds_lgbm, preds_ridge, preds_lstm,
                      preds_final, cf_test, start_frac, f"Test{tid}")

        fig_i, ax_i = plt.subplots(figsize=(9, 5))
        _draw_test_ax(ax_i, obs_pts,
                      preds_lgbm, preds_ridge, preds_lstm,
                      preds_final, cf_test, start_frac, f"Test{tid}")
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Test{tid}_RUL_v6.png", dpi=150)
        plt.close(fig_i)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_predictions_v6.png", dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary_v6.csv", index=False)
    print(f"\n  Test summary:")
    print(df_sum.to_string(index=False))
    print(f"\n[Done] {OUT_DIR}")


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    loocv_results, loocv_ens = run_loocv()
    if loocv_ens > 0.466:
        print(f"\n  → v6 ADOPTED (Ens_raw={loocv_ens:.4f} > 0.466)")
        run_test_inference(loocv_results)
    else:
        print(f"\n  → v6 REJECTED (Ens_raw={loocv_ens:.4f} ≤ 0.466). Stick with v4.")
