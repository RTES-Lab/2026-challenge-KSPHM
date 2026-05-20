"""
SR/0518 — TH v7_4_2 HI + SR RUL Pipeline
==========================================
HI source : TH/FI/07_v7/output/v7_4_2_conditional_aux_boost/
  - v7_4_2_Bearing{1-4}_HI.csv  'HI_v7_4_2' column  (Train Q=0.896)
  - v7_4_2_Test{1-6}_HI.csv     'HI_v7_4_2' column

RUL model : SR 0514 LGBM + LSTM-A (unchanged)
Baseline  : SR 0514 rul_ensemble_v3 LOOCV 0.4326
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
TH_HI_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM/User/TH/FI/07_v7"
                 "/output/v7_4_2_conditional_aux_boost")
OUT_DIR   = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0518/rul/output/th742")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants (identical to 0514) ─────────────────────────────────────────
BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH      = 10
MEAN_TRAIN_LIFE = 116.5
INTERVAL_SEC    = 600
SEEDS           = [42, 7, 123, 0, 99]

NORMAL_UNTIL = {1: 89, 2: 92, 3: 62, 4: 78}
EOL          = {1: 126, 2: 114, 3: 89, 4: 137}

# ── HI Loading ────────────────────────────────────────────────────────────
def load_train_hi():
    hi = {}
    for b in BEARINGS:
        df = pd.read_csv(TH_HI_DIR / f"v7_4_2_Bearing{b}_HI.csv")
        hi[b] = df["HI_v7_4_2"].values.astype(float)
    return hi

def load_test_hi():
    hi = {}
    for t in TEST_IDS:
        df = pd.read_csv(TH_HI_DIR / f"v7_4_2_Test{t}_HI.csv")
        hi[t] = df["HI_v7_4_2"].values.astype(float)
    return hi

# ── Competition scoring ────────────────────────────────────────────────────
def competition_score(rul_true, rul_pred):
    if rul_true <= 0:
        return np.nan
    Er = 100.0 * (rul_true - rul_pred) / rul_true
    return (np.exp(-np.log(0.5) * Er / 20.0) if Er <= 0
            else np.exp(np.log(0.5) * Er / 50.0))

# ── RUL labels ────────────────────────────────────────────────────────────
def rul_labels(n_total, bid):
    nu, eol = NORMAL_UNTIL[bid], EOL[bid]
    idx = np.arange(n_total)
    return np.where(idx <= nu, eol - nu, np.maximum(eol - idx, 0)).astype(float)

# ── LightGBM ──────────────────────────────────────────────────────────────
def make_lgbm_features(hi_array):
    features, targets = [], []
    N = len(hi_array)
    for i in range(SEQ_LENGTH, N):
        window = hi_array[i - SEQ_LENGTH: i]
        slope  = float(np.polyfit(np.arange(SEQ_LENGTH), window, 1)[0])
        feats  = list(window) + [slope, float(window.mean()), float(window.std()),
                                  float(window.max()), float(window[-1]),
                                  float(window[-1] - window[0])]
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
        x, y = make_lgbm_features(hi_dict[b])
        X_all.append(x); y_all.append(y)
    dtrain = lgb.Dataset(np.concatenate(X_all), label=np.concatenate(y_all))
    return lgb.train({"num_leaves": 15, "learning_rate": 0.05,
                      "min_child_samples": 5, "verbose": -1,
                      "objective": lgbm_asymmetric_obj},
                     dtrain, num_boost_round=200)

def predict_lgbm(model, hi_array):
    X, _ = make_lgbm_features(hi_array)
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
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)

def train_lstm(X_train, y_train, rul_scale, seed, device):
    torch.manual_seed(seed)
    y_norm = y_train / rul_scale
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_norm,  dtype=torch.float32)
    n_val = max(1, int(len(Xt) * 0.1))
    tr_dl = DataLoader(TensorDataset(Xt[:-n_val], yt[:-n_val]), batch_size=64, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xt[-n_val:], yt[-n_val:]), batch_size=64)
    model = LSTMRegressor().to(device)
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
    X_train = np.concatenate(X_all)
    y_train = np.concatenate(y_all)
    rul_scale = float(y_train.max())
    models = [train_lstm(X_train, y_train, rul_scale, s, device) for s in SEEDS]
    return models, rul_scale

def predict_lstm(models, rul_scale, hi_arr, start_obs, device):
    X, _ = make_seqs(hi_arr, np.zeros(len(hi_arr)), start_obs)
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    all_preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            all_preds.append(np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0))
    return np.median(all_preds, axis=0)

# ── Calibration search ────────────────────────────────────────────────────
def search_calibration(preds_dict, results, label=""):
    best_cf, best_score = 1.0, -np.inf
    for cf in np.arange(0.60, 1.11, 0.02):
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

# ── LOOCV ─────────────────────────────────────────────────────────────────
def run_loocv(hi_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results = {}
    sc_lgbm_list, sc_lstm_list = [], []

    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        print(f"{'='*55}")
        print(f"[LOOCV] Bearing {test_bid} held out")

        N_test = len(hi_train[test_bid])

        # LGBM
        lgbm_model  = train_lgbm(hi_train, train_bids)
        preds_lgbm  = predict_lgbm(lgbm_model, hi_train[test_bid])

        # LSTM-A (5 seeds)
        print(f"  LSTM training ({len(SEEDS)} seeds)...")
        lstm_models, rul_scale = train_lstm_ensemble(hi_train, train_bids, device)
        preds_lstm  = predict_lstm(lstm_models, rul_scale,
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

        # adaptive weight ensemble
        w_l = float(np.clip(sc_l / (sc_l + sc_a + 1e-12), 0.1, 0.7))
        w_a = 1.0 - w_l
        preds_ens = [w_l * l + w_a * a for l, a in zip(preds_lgbm, preds_lstm)]

        results[test_bid] = {
            "lgbm":    list(preds_lgbm),
            "lstm_a":  list(preds_lstm),
            "ens":     preds_ens,
            "obs_pts": list(obs_pts),
            "N":       N_test,
            "sc_lgbm": sc_l,
            "sc_lstm": sc_a,
            "w_lgbm":  w_l,
            "w_lstm":  w_a,
        }

    # ── Summary ──────────────────────────────────────────────────────────
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

    # Calibration
    print("\n  [Calibration search]")
    cf_lstm, sc_lstm_cf = search_calibration(
        {b: results[b]["lstm_a"] for b in BEARINGS}, results, "LSTM-A")
    cf_ens, sc_ens_cf = search_calibration(
        {b: results[b]["ens"]   for b in BEARINGS}, results, "Ensemble")

    # Baseline comparison
    BASELINE_LOOCV = 0.4326
    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  Config          │ LOOCV   │ best cf │ after cf         │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  SR 0514 baseline│ 0.4326  │  0.76   │  (reference)     │")
    print(f"  │  LSTM-A          │ {np.mean(sc_lstm_list):.4f}  │  {cf_lstm:.2f}   │ {sc_lstm_cf:.4f}          │")
    print(f"  │  LGBM+LSTM-A     │ {np.mean(ens_scores):.4f}  │  {cf_ens:.2f}   │ {sc_ens_cf:.4f}          │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    # Save log
    log_path = OUT_DIR / "loocv_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("SR/0518 — TH v7_4_2 HI + SR LOOCV\n")
        f.write(f"SR 0514 baseline LOOCV: {BASELINE_LOOCV}\n\n")
        f.write(f"LGBM avg:   {np.mean(sc_lgbm_list):.4f}\n")
        f.write(f"LSTM-A avg: {np.mean(sc_lstm_list):.4f}\n")
        f.write(f"Ensemble avg: {np.mean(ens_scores):.4f}\n\n")
        f.write(f"After calibration:\n")
        f.write(f"  LSTM-A: cf={cf_lstm:.2f} → {sc_lstm_cf:.4f}\n")
        f.write(f"  LGBM+LSTM-A: cf={cf_ens:.2f} → {sc_ens_cf:.4f}\n\n")
        f.write("Fold detail:\n")
        for b in BEARINGS:
            r = results[b]
            f.write(f"  B{b}: LGBM={r['sc_lgbm']:.4f}  LSTM-A={r['sc_lstm']:.4f}  "
                    f"Ens={r['sc_ens']:.4f}  "
                    f"w=[LGBM={r['w_lgbm']:.3f}, LSTM-A={r['w_lstm']:.3f}]\n")

    # LOOCV plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SR/0518 LOOCV — TH v7_4_2 HI + LGBM/LSTM-A", fontsize=12)
    for i, test_bid in enumerate(BEARINGS):
        ax   = axes.flatten()[i]
        r    = results[test_bid]
        obs  = r["obs_pts"]
        true = [r["N"] - o for o in obs]
        ax.plot(obs, true,            "k-",  lw=2,   label="True RUL")
        ax.plot(obs, r["lgbm"],       "r--", lw=1,   alpha=0.7, label="LGBM")
        ax.plot(obs, r["lstm_a"],     "g--", lw=1,   alpha=0.7, label="LSTM-A")
        ax.plot(obs, r["ens"],        "m-",  lw=2,   label="Ensemble")
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

# ── Test inference ────────────────────────────────────────────────────────
def run_test_inference(hi_train, hi_test, results, best_config, best_cf):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Test inference — "
          f"{'LSTM-A' if best_config == 'lstm_a' else 'LGBM+LSTM-A'}, cf={best_cf:.2f}")
    print(f"{'='*55}")

    # Train on all 4 bearings
    lgbm_model   = train_lgbm(hi_train, BEARINGS)
    print("  LSTM training on all 4 bearings...")
    lstm_models, rul_scale = train_lstm_ensemble(hi_train, BEARINGS, device)

    # Ensemble weights (mean of fold weights)
    if best_config == "ens":
        w_lgbm = float(np.mean([results[b]["w_lgbm"] for b in BEARINGS]))
        w_lstm = float(np.mean([results[b]["w_lstm"]  for b in BEARINGS]))
        total  = w_lgbm + w_lstm
        w_lgbm, w_lstm = w_lgbm / total, w_lstm / total
        print(f"  Weights: LGBM={w_lgbm:.3f}, LSTM-A={w_lstm:.3f}")

    summary_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"SR/0518 Test RUL — TH v7_4_2 HI  "
                 f"({'LSTM-A' if best_config == 'lstm_a' else 'LGBM+LSTM-A'}, cf={best_cf:.2f})",
                 fontsize=12)

    for i, tid in enumerate(TEST_IDS):
        hi_t  = hi_test[tid]
        N     = len(hi_t)
        obs_pts = np.arange(SEQ_LENGTH, N)

        preds_lstm = predict_lstm(lstm_models, rul_scale, hi_t, start_obs=0, device=device)
        preds_lgbm = predict_lgbm(lgbm_model, hi_t)

        if best_config == "ens":
            preds_raw   = w_lgbm * preds_lgbm + w_lstm * preds_lstm
        else:
            preds_raw   = preds_lstm
        preds_final = preds_raw * best_cf

        final_cyc = float(preds_final[-1])
        final_hr  = final_cyc * INTERVAL_SEC / 3600
        print(f"  [Test{tid}] final RUL = {final_hr:.2f} hr ({final_cyc:.1f} cycles)")

        pd.DataFrame({
            "obs_cycle":     obs_pts,
            "rul_pred_lgbm": preds_lgbm,
            "rul_pred_lstm": preds_lstm,
            "rul_pred_final": preds_final,
            "rul_pred_hours": preds_final * INTERVAL_SEC / 3600,
        }).to_csv(OUT_DIR / f"Test{tid}_RUL.csv", index=False)

        summary_rows.append({
            "test_id": tid,
            "hi_start": round(float(hi_t[0]), 3),
            "hi_end":   round(float(hi_t[-1]), 3),
            "final_rul_cycles": round(final_cyc, 2),
            "final_rul_hours":  round(final_hr, 2),
            "calib_factor":     round(best_cf, 2),
        })

        ax = axes.flatten()[i]
        ax.plot(obs_pts, preds_lgbm,   "r--", lw=1, alpha=0.6, label="LGBM")
        ax.plot(obs_pts, preds_lstm,   "g--", lw=1, alpha=0.6, label="LSTM-A")
        ax.plot(obs_pts, preds_final,  "m-",  lw=2, label="Final")
        ax.set_title(f"Test{tid}  start_HI={hi_t[0]:.3f}→{hi_t[-1]:.3f}  "
                     f"RUL={final_hr:.1f}hr")
        ax.set_xlabel("Observation cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_predictions.png", dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary.csv", index=False)

    # Comparison table vs 0514 baseline
    BASELINE = {"Test1": 5.05, "Test2": 5.08, "Test3": 4.69,
                "Test4": 3.43, "Test5": 7.46, "Test6": 5.40}
    print(f"\n  {'Test':>6} | {'0514 (hr)':>10} | {'0518 (hr)':>10} | {'hi_start':>9} | {'hi_end':>8}")
    print(f"  {'-'*55}")
    for row in summary_rows:
        tid = row["test_id"]
        b = BASELINE.get(f"Test{tid}", None)
        print(f"  {tid:>6} | {b:>10.2f} | {row['final_rul_hours']:>10.2f} | "
              f"{row['hi_start']:>9.3f} | {row['hi_end']:>8.3f}")

    with open(OUT_DIR / "loocv_log.txt", "a", encoding="utf-8") as f:
        f.write("\nTest inference:\n")
        f.write(df_sum.to_string(index=False) + "\n")

    print(f"\n[Done] {OUT_DIR}")

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading TH v7_4_2 HI...")
    hi_train = load_train_hi()
    hi_test  = load_test_hi()

    for b in BEARINGS:
        hi = hi_train[b]
        print(f"  Bearing{b}: n={len(hi)}, range=[{hi.min():.3f}, {hi.max():.3f}]")
    print()

    results, best_config, best_cf, loocv_ens = run_loocv(hi_train)
    run_test_inference(hi_train, hi_test, results, best_config, best_cf)
