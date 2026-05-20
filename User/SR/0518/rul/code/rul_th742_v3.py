"""
SR/0518 v3 — TH v7_4_2 HI + SR RUL Pipeline (LGBM + obs_fraction)
====================================================================
v2 대비 변경:
  LGBM 피처에 obs_fraction (= obs_idx / MEAN_TRAIN_LIFE) 추가
  목적: B3처럼 빠른 열화 베어링에서 "cycle 80 = 말기"를 LGBM이 인식하게 함

v2 결과 (baseline):
  LOOCV: LGBM=0.3345  LSTM-A=0.4261  Ensemble=0.4551
  B3 LGBM: 0.0916 (여전히 붕괴 — obs_fraction 없어서)
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
OUT_DIR   = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0518/rul/output/th742_v3")
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

# ── RUL labels ────────────────────────────────────────────────────────────
def rul_labels(n_total, bid):
    nu, eol = NORMAL_UNTIL[bid], EOL[bid]
    idx = np.arange(n_total)
    return np.where(idx <= nu, eol - nu, np.maximum(eol - idx, 0)).astype(float)

# ── LightGBM (v3: window-minmax + raw summary + obs_fraction) ─────────────
def make_lgbm_features(hi_array, start_obs=0):
    """
    피처 구성 (17개):
      [0:10]  window_norm  — window-minmax 정규화 (trend shape)
      [10]    slope_norm   — 정규화 window 기울기
      [11]    hi_last      — 마지막 raw HI (절대 열화 수준)
      [12]    hi_mean      — window raw 평균
      [13]    hi_max       — window raw 최댓값
      [14]    slope_raw    — raw window 기울기
      [15]    delta_raw    — window 첫-마지막 차이
      [16]    obs_fraction — (start_obs + i) / MEAN_TRAIN_LIFE  ← 핵심 추가
    """
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
                 [slope_norm,
                  float(window[-1]),
                  float(window.mean()),
                  float(window.max()),
                  slope_raw,
                  float(window[-1] - window[0]),
                  obs_frac])
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

# ── LSTM-A (unchanged) ────────────────────────────────────────────────────
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
            "lgbm":    list(preds_lgbm),
            "lstm_a":  list(preds_lstm),
            "ens":     preds_ens,
            "obs_pts": list(obs_pts),
            "N":       N_test,
            "sc_lgbm": sc_l, "sc_lstm": sc_a,
            "w_lgbm":  w_l,  "w_lstm":  w_a,
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

    print("\n  [Calibration search]")
    cf_lstm, sc_lstm_cf = search_calibration(
        {b: results[b]["lstm_a"] for b in BEARINGS}, results, "LSTM-A")
    cf_ens, sc_ens_cf = search_calibration(
        {b: results[b]["ens"]   for b in BEARINGS}, results, "Ensemble")

    V1 = {"lgbm": 0.3625, "lstm": 0.4261, "ens": 0.4529, "ens_cf": 0.4555}
    V2 = {"lgbm": 0.3345, "lstm": 0.4261, "ens": 0.4551, "ens_cf": 0.4574}
    print(f"\n  ┌─────────────────────────────────────────────────────────────────────┐")
    print(f"  │  Config      │ v1(raw) │ v2(norm) │ v3(+obsfrac) │ v3 cf │ v3 after cf │")
    print(f"  ├─────────────────────────────────────────────────────────────────────┤")
    print(f"  │  SR 0514     │  0.4326 │   0.4326 │      —       │  0.76 │   (ref)     │")
    print(f"  │  LSTM-A      │  {V1['lstm']:.4f} │   {V2['lstm']:.4f} │   {np.mean(sc_lstm_list):.4f}     │  {cf_lstm:.2f} │   {sc_lstm_cf:.4f}    │")
    print(f"  │  LGBM+LSTM-A │  {V1['ens']:.4f} │   {V2['ens']:.4f} │   {np.mean(ens_scores):.4f}     │  {cf_ens:.2f} │   {sc_ens_cf:.4f}    │")
    print(f"  └─────────────────────────────────────────────────────────────────────┘")

    with open(OUT_DIR / "loocv_log.txt", "w", encoding="utf-8") as f:
        f.write("SR/0518 v3 — TH v7_4_2 HI + LGBM(window-norm+obs_frac) + LSTM-A\n")
        f.write(f"LGBM avg:    {np.mean(sc_lgbm_list):.4f}  (v2: {V2['lgbm']:.4f})\n")
        f.write(f"LSTM-A avg:  {np.mean(sc_lstm_list):.4f}  (v2: {V2['lstm']:.4f})\n")
        f.write(f"Ensemble:    {np.mean(ens_scores):.4f}  (v2: {V2['ens']:.4f})\n")
        f.write(f"After cf:    LSTM-A cf={cf_lstm:.2f}→{sc_lstm_cf:.4f}  "
                f"Ens cf={cf_ens:.2f}→{sc_ens_cf:.4f}\n\n")
        f.write("Fold detail:\n")
        for b in BEARINGS:
            r = results[b]
            f.write(f"  B{b}: LGBM={r['sc_lgbm']:.4f}  LSTM-A={r['sc_lstm']:.4f}  "
                    f"Ens={r['sc_ens']:.4f}  "
                    f"w=[LGBM={r['w_lgbm']:.3f}, LSTM={r['w_lstm']:.3f}]\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SR/0518 v3 LOOCV — TH v7_4_2 HI + LGBM(+obs_frac) + LSTM-A", fontsize=11)
    for i, test_bid in enumerate(BEARINGS):
        ax  = axes.flatten()[i]
        r   = results[test_bid]
        obs = r["obs_pts"]
        ax.plot(obs, [r["N"] - o for o in obs], "k-",  lw=2,   label="True RUL")
        ax.plot(obs, r["lgbm"],                  "r--", lw=1,   alpha=0.7, label="LGBM")
        ax.plot(obs, r["lstm_a"],                "g--", lw=1,   alpha=0.7, label="LSTM-A")
        ax.plot(obs, r["ens"],                   "m-",  lw=2,   label="Ensemble")
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
        "v1":   {1: 8.13, 2:10.89, 3: 7.44, 4: 6.52, 5: 9.94, 6: 8.61},
        "v2":   {1: 8.58, 2:10.79, 3: 5.60, 4: 5.06, 5: 7.51, 6: 5.61},
    }

    summary_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"SR/0518 v3 Test RUL — TH v7_4_2 HI + LGBM(+obs_frac)  cf={best_cf:.2f}",
                 fontsize=11)

    for i, tid in enumerate(TEST_IDS):
        hi_t    = hi_test[tid]
        obs_pts = np.arange(SEQ_LENGTH, len(hi_t))

        preds_lstm = predict_lstm(lstm_models, rul_scale, hi_t, start_obs=0, device=device)
        preds_lgbm = predict_lgbm(lgbm_model, hi_t, start_obs=0)
        preds_raw  = (w_lgbm * preds_lgbm + w_lstm * preds_lstm
                      if best_config == "ens" else preds_lstm)
        preds_final = preds_raw * best_cf

        final_cyc = float(preds_final[-1])
        final_hr  = final_cyc * INTERVAL_SEC / 3600
        print(f"  [Test{tid}] {final_hr:.2f}hr  "
              f"(v2={REF['v2'][tid]:.2f}, v1={REF['v1'][tid]:.2f}, 0514={REF['0514'][tid]:.2f})")

        pd.DataFrame({
            "obs_cycle": obs_pts,
            "rul_lgbm":  preds_lgbm,
            "rul_lstm":  preds_lstm,
            "rul_final": preds_final,
            "rul_hours": preds_final * INTERVAL_SEC / 3600,
        }).to_csv(OUT_DIR / f"Test{tid}_RUL.csv", index=False)

        summary_rows.append({
            "test_id":   tid,
            "hi_start":  round(float(hi_t[0]), 3),
            "hi_end":    round(float(hi_t[-1]), 3),
            "rul_hours": round(final_hr, 2),
            "v2_hours":  REF["v2"][tid],
            "v1_hours":  REF["v1"][tid],
            "base0514":  REF["0514"][tid],
        })

        ax = axes.flatten()[i]
        ax.plot(obs_pts, preds_lgbm,  "r--", lw=1, alpha=0.6, label="LGBM")
        ax.plot(obs_pts, preds_lstm,  "g--", lw=1, alpha=0.6, label="LSTM-A")
        ax.plot(obs_pts, preds_final, "m-",  lw=2, label="Final")
        ax.set_title(f"Test{tid}  HI={hi_t[0]:.3f}→{hi_t[-1]:.3f}  "
                     f"RUL={final_hr:.1f}hr")
        ax.set_xlabel("Observation cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_predictions.png", dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary.csv", index=False)

    print(f"\n  {'Test':>5} | {'0514':>6} | {'v1':>6} | {'v2':>6} | {'v3':>6} | hi_end")
    print(f"  {'-'*55}")
    for row in summary_rows:
        print(f"  {row['test_id']:>5} | {row['base0514']:>6.2f} | "
              f"{row['v1_hours']:>6.2f} | {row['v2_hours']:>6.2f} | "
              f"{row['rul_hours']:>6.2f} | {row['hi_end']:.3f}")

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
