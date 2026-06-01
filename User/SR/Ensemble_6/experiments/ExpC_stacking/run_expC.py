"""
Ensemble_6 / Exp-C: Stacking with Meta-Learner
================================================
Hypothesis: A meta-learner can learn the optimal per-sample combination
of base models by using their LOOCV predictions as input features.

The key insight from baseline analysis:
  - LGBM is strong for B1, B2, B4 but completely fails B3
  - DTW is the only model that handles B3 reasonably
  - An asymmetric-loss meta-learner can learn this pattern

Architecture:
  Base models (LGBM, LSTM, GRU, TCN, DTW) trained on all-but-one bearing
  Meta features per sample t:
    [lgbm_pred, lstm_pred, gru_pred, tcn_pred, dtw_pred,   (5 base preds)
     hi_current, obs_frac, hi_slope_recent, hi_gain]        (4 context)
  Meta target: true_rul(t)
  Meta model: Ridge regression with asymmetric Huber loss
              (or simple linear meta-learner)

LOOCV nesting: For each held-out bearing, base models are trained on
the remaining 3 bearings. This gives out-of-fold predictions for the
meta-learner to train on.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BASE    = Path("/data/home/ksphm/2026-challenge-KSPHM")
SP      = BASE / "User" / "SP"
HI_DIR  = SP / "05-26" / "V1b" / "output"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH      = 10
MATCH_LEN       = 18
INTERVAL_SEC    = 600
MEAN_TRAIN_LIFE = 116.5
SEEDS           = [42]
MIN_RUL_CYCLES  = 1.0
EPS             = 1e-8

NORMAL_UNTIL = {1: 89, 2: 92, 3: 62, 4: 78}
EOL          = {1: 126, 2: 114, 3: 89, 4: 137}

def load_train_hi():
    return {b: pd.read_csv(HI_DIR / f"HI_Bearing{b}.csv")["HI"].values.astype(float)
            for b in BEARINGS}

def load_test_hi():
    return {t: pd.read_csv(HI_DIR / "test" / f"HI_Test{t}.csv")["HI"].values.astype(float)
            for t in TEST_IDS}

def minmax_norm(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + EPS)

def slope_of(x):
    x = np.asarray(x, dtype=float)
    return float(np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x) >= 2 else 0.0

def competition_score(rul_true, rul_pred):
    if rul_true <= 0: return np.nan
    er = 100.0 * (rul_true - rul_pred) / rul_true
    return (np.exp(-np.log(0.5)*er/20.0) if er <= 0
            else np.exp(np.log(0.5)*er/50.0))

def true_rul(n, obs_pts):
    return np.maximum(n - np.asarray(obs_pts, dtype=float), 1.0)

def score_curve(n, obs_pts, preds):
    y = true_rul(n, obs_pts)
    return float(np.nanmean([competition_score(t, p) for t, p in zip(y, preds)]))

def error_summary(n, obs_pts, preds):
    y = true_rul(n, obs_pts)
    p = np.asarray(preds, dtype=float)
    er = [100.0*(t-pp)/t for t, pp in zip(y, p) if t > 0]
    return {"score": score_curve(n, obs_pts, p), "mean_er": float(np.nanmean(er))}

# ── Base model training (same as baseline) ───────────────────────────────────
def make_tabular(hi_arr, start_obs=0):
    hi = np.asarray(hi_arr, dtype=float); hi0 = float(hi[0])
    x, obs_pts = [], []
    for i in range(SEQ_LENGTH, len(hi)):
        w = hi[i-SEQ_LENGTH:i]; wn = minmax_norm(w)
        obs_frac = float(np.clip((start_obs+i)/MEAN_TRAIN_LIFE, 0.0, 2.0))
        x.append(list(wn) + [slope_of(wn), float(w[-1]), float(w.mean()),
                              float(w.max()), float(w.min()), float(w.std()),
                              slope_of(w), float(w[-1]-w[0]), float(w[-1]-hi0),
                              obs_frac, float(w[-1]*obs_frac)])
        obs_pts.append(i)
    return np.asarray(x), np.asarray(obs_pts)

def train_lgbm(hi_train, train_bids):
    xs, ys = [], []
    for b in train_bids:
        x, obs = make_tabular(hi_train[b]); xs.append(x)
        ys.append(true_rul(len(hi_train[b]), obs))
    x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
    def asym_obj(y_pred, ds):
        diff = ds.get_label()-y_pred; w = np.where(diff<0, 2.8, 1.0)
        return -diff*w, np.ones_like(diff)*w
    return lgb.train(
        {"num_leaves": 15, "learning_rate": 0.04, "min_child_samples": 5,
         "feature_fraction": 0.90, "bagging_fraction": 0.90, "bagging_freq": 1,
         "verbose": -1, "objective": asym_obj},
        lgb.Dataset(x_tr, label=y_tr), num_boost_round=260)

def predict_lgbm(model, hi_arr, start_obs=0):
    x, obs = make_tabular(hi_arr, start_obs=start_obs)
    return obs, np.maximum(model.predict(x), MIN_RUL_CYCLES)

def make_seq(hi_arr, start_obs=0):
    hi = np.asarray(hi_arr, dtype=float); hi0 = float(hi[0])
    x, obs_pts = [], []
    for i in range(len(hi)-SEQ_LENGTH):
        w = hi[i:i+SEQ_LENGTH]
        obs_frac = np.clip((start_obs+i+np.arange(SEQ_LENGTH))/MEAN_TRAIN_LIFE, 0.0, 2.0)
        x.append(np.stack([minmax_norm(w), w.copy(), w-hi0, obs_frac], axis=1))
        obs_pts.append(i+SEQ_LENGTH)
    return np.asarray(x), np.asarray(obs_pts)

class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(4, 48, num_layers=2, batch_first=True, dropout=0.15)
        self.fc  = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1))
    def forward(self, x):
        y, _ = self.rnn(x); return self.fc(y[:,-1,:]).squeeze(-1)

class GRURegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(4, 48, num_layers=2, batch_first=True, dropout=0.15)
        self.fc  = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1))
    def forward(self, x):
        y, _ = self.rnn(x); return self.fc(y[:,-1,:]).squeeze(-1)

class TCNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 32, 3, padding=1), nn.ReLU(), nn.Dropout(0.10),
            nn.Conv1d(32, 32, 3, dilation=2, padding=2), nn.ReLU(), nn.Dropout(0.10),
            nn.Conv1d(32, 32, 3, dilation=4, padding=4), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(32, 24), nn.ReLU(), nn.Linear(24, 1))
    def forward(self, x):
        z = self.net(x.transpose(1,2)); return self.fc(z[:,:,-1]).squeeze(-1)

def build_model(kind):
    return {"lstm": LSTMRegressor, "gru": GRURegressor, "tcn": TCNRegressor}[kind]()

def train_torch_model(x_tr, y_tr, scale, kind, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(kind).to(device)
    ds = TensorDataset(torch.tensor(x_tr, dtype=torch.float32),
                       torch.tensor(y_tr/scale, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=min(32, len(ds)), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    best_state, best_loss, patience = None, np.inf, 0
    for _ in range(160):
        model.train(); losses = []
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(float(loss.item()))
        cur = float(np.mean(losses))
        if cur < best_loss - 1e-5:
            best_loss, patience = cur, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 18: break
    if best_state: model.load_state_dict(best_state)
    return model

def train_torch_ensemble(hi_train, train_bids, kind, device):
    xs, ys = [], []
    for b in train_bids:
        x, obs = make_seq(hi_train[b]); xs.append(x)
        ys.append(true_rul(len(hi_train[b]), obs))
    x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
    scale = float(max(y_tr.max(), 1.0))
    models = [train_torch_model(x_tr, y_tr, scale, kind, s, device) for s in SEEDS]
    return models, scale

def predict_torch(models, scale, hi_arr, start_obs, device):
    x, obs = make_seq(hi_arr, start_obs=start_obs)
    xt = torch.tensor(x, dtype=torch.float32).to(device)
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(np.maximum(m(xt).cpu().numpy()*scale, MIN_RUL_CYCLES))
    return obs, np.median(preds, axis=0)

def seg_dist(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return (0.25*abs(a[-1]-b[-1])/0.25 + 0.20*abs(a.mean()-b.mean())/0.25
            + 0.20*abs((a[-1]-a[0])-(b[-1]-b[0]))/0.25
            + 0.15*abs(slope_of(a)-slope_of(b))/0.03
            + 0.20*float(np.mean(np.abs(minmax_norm(a)-minmax_norm(b)))))

def predict_dtw_knn(hi_train, train_bids, hi_target):
    hi_target = np.asarray(hi_target, dtype=float)
    obs_pts = np.arange(SEQ_LENGTH, len(hi_target))
    preds = []
    for obs in obs_pts:
        l = min(MATCH_LEN, obs)
        seg = hi_target[obs-l:obs]
        candidates = []
        for b in train_bids:
            hi = np.asarray(hi_train[b], dtype=float)
            for end in range(l, len(hi)):
                d = seg_dist(seg, hi[end-l:end])
                pred = max(len(hi)-end, MIN_RUL_CYCLES)
                candidates.append((d, pred))
        top = sorted(candidates, key=lambda x: x[0])[:6]
        wt = np.asarray([1.0/(d+EPS) for d, p in top])
        pv = np.asarray([p for d, p in top])
        preds.append(float(np.average(pv, weights=wt)))
    return obs_pts, np.asarray(preds)

def estimate_start_obs(hi_train, hi_target, train_bids):
    target = np.asarray(hi_target, dtype=float)
    l = min(MATCH_LEN, len(target))
    seg = target[:l]
    candidates = []
    for b in train_bids:
        hi = np.asarray(hi_train[b], dtype=float)
        for s in range(0, max(1, len(hi)-l-3)):
            d = seg_dist(seg, hi[s:s+l])
            candidates.append((d, s, b))
    if not candidates: return 0
    top = sorted(candidates, key=lambda x: x[0])[:8]
    pos = np.asarray([s for d, s, b in top], dtype=float)
    wt  = np.asarray([1.0/(d+EPS) for d, s, b in top])
    est  = int(round(np.average(pos, weights=wt)))
    gain = float(target[-1] - target[0])
    est  = max(est, int(round(np.clip((gain-0.25)/0.35, 0, 1)*25)))
    return int(np.clip(est, 0, int(MEAN_TRAIN_LIFE)))

def calibrate(pred_by_bearing, results, cf_min=0.50, cf_max=1.61, cf_step=0.01):
    best_cf, best_score = 1.0, -np.inf
    for cf in np.arange(cf_min, cf_max, cf_step):
        sc = float(np.mean([score_curve(results[b]["N"], results[b]["obs_pts"],
                                        np.asarray(pred_by_bearing[b])*cf)
                            for b in BEARINGS]))
        if sc > best_score:
            best_score, best_cf = sc, float(cf)
    return best_cf, best_score

# ── Meta features ─────────────────────────────────────────────────────────────
def make_meta_features(hi_arr, obs_pts, base_preds, start_obs=0):
    """
    Build meta-feature matrix for stacking.
    Each row corresponds to one observation point.
    """
    hi = np.asarray(hi_arr, dtype=float)
    model_names = ["lgbm", "lstm", "gru", "tcn", "dtw"]
    rows = []
    for i, obs in enumerate(obs_pts):
        w = hi[max(0, obs-SEQ_LENGTH):obs]
        obs_frac = float(np.clip((start_obs + obs) / MEAN_TRAIN_LIFE, 0.0, 2.0))
        hi_current = float(hi[obs-1]) if obs > 0 else 0.0
        hi_slope   = slope_of(w) if len(w) >= 2 else 0.0
        hi_gain    = float(hi[obs-1] - hi[0]) if obs > 0 else 0.0
        row = [base_preds[m][i] for m in model_names]
        row += [hi_current, obs_frac, hi_slope, hi_gain]
        rows.append(row)
    return np.asarray(rows, dtype=float)

# ── LOOCV with stacking ───────────────────────────────────────────────────────
def run_loocv(hi_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    model_names = ["lgbm", "lstm", "gru", "tcn", "dtw"]
    results = {}

    # ── Step 1: Collect OOF predictions for meta-learning ────────────────────
    print("[Step 1: Collect OOF predictions]")
    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        print(f"\n  B{test_bid} held out:")
        n_test = len(hi_train[test_bid])
        fold = {"N": n_test}

        lgbm_m = train_lgbm(hi_train, train_bids)
        obs, pred_lgbm = predict_lgbm(lgbm_m, hi_train[test_bid])
        fold["obs_pts"] = obs
        fold["lgbm"] = pred_lgbm

        for kind in ["lstm", "gru", "tcn"]:
            print(f"    {kind.upper()}...")
            models, scale = train_torch_ensemble(hi_train, train_bids, kind, device)
            _, pp = predict_torch(models, scale, hi_train[test_bid], 0, device)
            fold[kind] = pp

        _, pred_dtw = predict_dtw_knn(hi_train, train_bids, hi_train[test_bid])
        fold["dtw"] = pred_dtw

        # Raw scores
        fold["raw_scores"] = {m: score_curve(n_test, obs, fold[m]) for m in model_names}
        results[test_bid] = fold
        print("    " + "  ".join(f"{m}={fold['raw_scores'][m]:.4f}" for m in model_names))

    # ── Step 2: Train meta-learner on OOF predictions ─────────────────────────
    print("\n[Step 2: Train meta-learner]")
    X_meta, y_meta = [], []
    for b in BEARINGS:
        r = results[b]
        base_p = {m: r[m] for m in model_names}
        X_b = make_meta_features(hi_train[b], r["obs_pts"], base_p, start_obs=0)
        y_b = true_rul(r["N"], r["obs_pts"])
        X_meta.append(X_b); y_meta.append(y_b)
    X_meta = np.concatenate(X_meta); y_meta = np.concatenate(y_meta)
    print(f"  Meta train size: {X_meta.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_meta)

    # Ridge regression (simple, interpretable)
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_scaled, y_meta)
    print(f"  Ridge coef (base models): lgbm={ridge.coef_[0]:.3f} lstm={ridge.coef_[1]:.3f} "
          f"gru={ridge.coef_[2]:.3f} tcn={ridge.coef_[3]:.3f} dtw={ridge.coef_[4]:.3f}")
    print(f"  Ridge coef (context): hi={ridge.coef_[5]:.3f} obs_frac={ridge.coef_[6]:.3f} "
          f"slope={ridge.coef_[7]:.3f} gain={ridge.coef_[8]:.3f}")

    # ── Step 3: Meta-learner LOOCV evaluation ─────────────────────────────────
    print("\n[Step 3: Stacking LOOCV evaluation]")
    for b in BEARINGS:
        r = results[b]
        base_p = {m: r[m] for m in model_names}
        X_b = make_meta_features(hi_train[b], r["obs_pts"], base_p, start_obs=0)
        X_b_scaled = scaler.transform(X_b)
        pred_meta = np.maximum(ridge.predict(X_b_scaled), MIN_RUL_CYCLES)
        results[b]["stacking"] = pred_meta
        sc = score_curve(r["N"], r["obs_pts"], pred_meta)
        print(f"  B{b} stacking score: {sc:.4f}")

    # Compare DTW vs stacking
    dtw_cf, dtw_sc  = calibrate({b: results[b]["dtw"]      for b in BEARINGS}, results)
    stk_cf, stk_sc  = calibrate({b: results[b]["stacking"] for b in BEARINGS}, results)
    print(f"\n[Calibration]")
    print(f"  dtw (baseline): cf={dtw_cf:.2f}  score={dtw_sc:.4f}")
    print(f"  stacking:       cf={stk_cf:.2f}  score={stk_sc:.4f}")

    # Choose best
    if stk_sc >= dtw_sc:
        best_key = "stacking"; best_cf = stk_cf
        print(f"\n  Selected: stacking (improvement: +{stk_sc-dtw_sc:.4f})")
    else:
        best_key = "dtw"; best_cf = dtw_cf
        print(f"\n  Selected: dtw (stacking was worse by {dtw_sc-stk_sc:.4f})")

    for b in BEARINGS:
        results[b]["final"] = np.asarray(results[b][best_key]) * best_cf
        results[b]["sc_final"] = score_curve(results[b]["N"], results[b]["obs_pts"], results[b]["final"])

    rows = []
    for b in BEARINGS:
        r = results[b]
        er = error_summary(r["N"], r["obs_pts"], r["final"])
        rows.append({"bearing": b, "score": round(r["sc_final"], 4),
                     "mean_er": round(er["mean_er"], 4)})
    overall = float(np.mean([r["score"] for r in rows]))
    print(f"\n[Final per-bearing]")
    for r in rows:
        print(f"  B{r['bearing']}: score={r['score']:.4f}  mean_er={r['mean_er']:.2f}%")
    print(f"  Overall: {overall:.4f}  (baseline: 0.5499)")

    pd.DataFrame(rows).to_csv(OUT_DIR / "train_rul_results.csv", index=False)

    # Coeff analysis
    feat_names = model_names + ["hi_current", "obs_frac", "hi_slope", "hi_gain"]
    coef_df = pd.DataFrame({"feature": feat_names, "coef": ridge.coef_[:len(feat_names)]})
    coef_df.to_csv(OUT_DIR / "meta_learner_coeff.csv", index=False)

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Exp-C Stacking LOOCV — overall={overall:.4f}", fontsize=11)
    for ax, b in zip(axes.flatten(), BEARINGS):
        r = results[b]; obs = r["obs_pts"]; true = true_rul(r["N"], obs)
        ax.plot(obs, true, "k-", lw=2, label="True")
        ax.plot(obs, r["dtw"] * dtw_cf, "r--", lw=1.5, alpha=0.7, label=f"dtw×{dtw_cf:.2f}")
        ax.plot(obs, r["final"], "b-", lw=2, label=f"{best_key}×{best_cf:.2f}")
        ax.set_title(f"B{b}  score={r['sc_final']:.3f}")
        ax.set_xlabel("Obs"); ax.set_ylabel("RUL"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_rul_predictions.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Exp-C Train Er%", fontsize=11)
    for ax, b in zip(axes.flatten(), BEARINGS):
        r = results[b]; obs = r["obs_pts"]; true = true_rul(r["N"], obs)
        er = 100.0 * (true - np.asarray(r["final"])) / true
        ax.plot(obs, er, lw=1.2); ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"B{b}  mean_er={float(er.mean()):.1f}%")
        ax.set_xlabel("Obs"); ax.set_ylabel("Er%"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_rul_er_pct.png", dpi=150)
    plt.close()

    # Coefficient visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["steelblue" if c > 0 else "tomato" for c in coef_df["coef"]]
    ax.bar(coef_df["feature"], coef_df["coef"], color=colors)
    ax.set_title("Meta-learner Ridge coefficients")
    ax.set_ylabel("Coefficient"); ax.set_xlabel("Feature")
    ax.axhline(0, color="k", lw=0.8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "meta_coefficients.png", dpi=150)
    plt.close()

    return results, ridge, scaler, best_key, best_cf, dtw_cf

def run_test(hi_train, hi_test, ridge, scaler, best_key, best_cf, dtw_cf):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Test: {best_key}  cf={best_cf:.2f}]")
    model_names = ["lgbm", "lstm", "gru", "tcn", "dtw"]

    lgbm_model = train_lgbm(hi_train, BEARINGS)
    torch_models = {}
    for kind in ["lstm", "gru", "tcn"]:
        print(f"  {kind.upper()}..."); torch_models[kind] = train_torch_ensemble(hi_train, BEARINGS, kind, device)

    summary_rows, all_rows = [], []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Exp-C Test — {best_key} cf={best_cf:.2f}", fontsize=11)

    for ax, tid in zip(axes.flatten(), TEST_IDS):
        hi = hi_test[tid]
        start_obs = estimate_start_obs(hi_train, hi, BEARINGS)
        obs, p_lgbm = predict_lgbm(lgbm_model, hi, start_obs=start_obs)
        preds = {"lgbm": p_lgbm}
        for kind in ["lstm", "gru", "tcn"]:
            models, scale = torch_models[kind]
            _, pp = predict_torch(models, scale, hi, start_obs, device)
            preds[kind] = pp
        _, p_dtw = predict_dtw_knn(hi_train, BEARINGS, hi)
        preds["dtw"] = p_dtw

        if best_key == "stacking":
            X_t = make_meta_features(hi, obs, preds, start_obs=start_obs)
            X_t_scaled = scaler.transform(X_t)
            final = np.maximum(ridge.predict(X_t_scaled), MIN_RUL_CYCLES) * best_cf
        else:
            final = np.maximum(p_dtw * best_cf, MIN_RUL_CYCLES)
        hours = final * INTERVAL_SEC / 3600.0

        for o, f, h in zip(obs, final, hours):
            all_rows.append({"test_id": tid, "obs_cycle": int(o),
                             "rul_cycles": float(f), "rul_hours": float(h)})
        summary_rows.append({"test_id": tid, "start_obs": start_obs,
                              "hi_start": round(float(hi[0]), 3),
                              "hi_end": round(float(hi[-1]), 3),
                              "rul_hours": round(float(hours[-1]), 2)})
        print(f"  T{tid}: start={start_obs}  RUL={hours[-1]:.2f}hr")

        ax.plot(obs, final, "b-", lw=2, label="final")
        for k, v in preds.items():
            ax.plot(obs, v, lw=0.8, alpha=0.45, label=k)
        ax.set_title(f"T{tid}  RUL={hours[-1]:.1f}hr")
        ax.set_xlabel("Obs"); ax.set_ylabel("RUL"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_rul_predictions.png", dpi=150)
    plt.close()
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "test_rul_results.csv", index=False)
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "test_rul_all_cycles.csv", index=False)

def save_hi_outputs(hi_train, hi_test):
    rows = []
    for b in BEARINGS:
        for i, v in enumerate(hi_train[b]): rows.append({"bearing": b, "cycle": i, "HI": float(v)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "train_hi.csv", index=False)
    rows = []
    for t in TEST_IDS:
        for i, v in enumerate(hi_test[t]): rows.append({"test_id": t, "cycle": i, "HI": float(v)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "test_hi.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, b in zip(axes.flatten(), BEARINGS):
        ax.plot(hi_train[b], lw=1.5); ax.set_title(f"B{b}"); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    fig.suptitle("Exp-C Train HI", fontsize=11); plt.tight_layout()
    plt.savefig(OUT_DIR / "train_hi.png", dpi=150); plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for ax, t in zip(axes.flatten(), TEST_IDS):
        ax.plot(hi_test[t], lw=1.5); ax.set_title(f"T{t}"); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    fig.suptitle("Exp-C Test HI", fontsize=11); plt.tight_layout()
    plt.savefig(OUT_DIR / "test_hi.png", dpi=150); plt.close()

def main():
    print("=== Ensemble_6 / Exp-C: Stacking Meta-Learner ===\n")
    hi_train = load_train_hi()
    hi_test  = load_test_hi()
    save_hi_outputs(hi_train, hi_test)
    results, ridge, scaler, best_key, best_cf, dtw_cf = run_loocv(hi_train)
    run_test(hi_train, hi_test, ridge, scaler, best_key, best_cf, dtw_cf)
    print(f"\nDone → {OUT_DIR}")

if __name__ == "__main__":
    main()
