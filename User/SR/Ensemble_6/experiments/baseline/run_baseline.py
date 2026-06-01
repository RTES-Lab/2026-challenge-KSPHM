"""
Ensemble_6 / experiments / baseline
====================================
V1b_dtw baseline reproduction.
HI is loaded from the original V1b output (already generated).
All required artifacts are saved to:  experiments/baseline/results/

Pipeline:
  1. Load V1b Train/Test HI
  2. LOOCV: LGBM + LSTM + GRU + TCN + DTW  →  calibrated ensemble selection
  3. Test inference
  4. Save all required outputs
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

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path("/data/home/ksphm/2026-challenge-KSPHM")
SP      = BASE / "User" / "SP"
HI_DIR  = SP / "05-26" / "V1b" / "output"     # original V1b HI (already generated)
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

# ── HI load ───────────────────────────────────────────────────────────────────
def load_train_hi():
    return {b: pd.read_csv(HI_DIR / f"HI_Bearing{b}.csv")["HI"].values.astype(float)
            for b in BEARINGS}

def load_test_hi():
    return {t: pd.read_csv(HI_DIR / "test" / f"HI_Test{t}.csv")["HI"].values.astype(float)
            for t in TEST_IDS}

# ── Utils ─────────────────────────────────────────────────────────────────────
def minmax_norm(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + EPS)

def slope_of(x):
    x = np.asarray(x, dtype=float)
    return float(np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x) >= 2 else 0.0

def competition_score(rul_true, rul_pred):
    if rul_true <= 0: return np.nan
    er = 100.0 * (rul_true - rul_pred) / rul_true
    return (np.exp(-np.log(0.5) * er / 20.0) if er <= 0
            else np.exp(np.log(0.5) * er / 50.0))

def true_rul(n, obs_pts):
    return np.maximum(n - np.asarray(obs_pts, dtype=float), 1.0)

def score_curve(n, obs_pts, preds):
    y = true_rul(n, obs_pts)
    return float(np.nanmean([competition_score(t, p) for t, p in zip(y, preds)]))

def rul_labels(n_total, bid):
    nu, eol = NORMAL_UNTIL[bid], EOL[bid]
    idx = np.arange(n_total)
    return np.where(idx <= nu, eol - nu, np.maximum(eol - idx, 0)).astype(float)

def error_summary(n, obs_pts, preds):
    y = true_rul(n, obs_pts)
    p = np.asarray(preds, dtype=float)
    er = np.asarray([100.0 * (t - pp) / t for t, pp in zip(y, p) if t > 0])
    return {"score": score_curve(n, obs_pts, p), "mean_er": float(np.nanmean(er))}

# ── LGBM ──────────────────────────────────────────────────────────────────────
def make_tabular(hi_arr, start_obs=0):
    hi = np.asarray(hi_arr, dtype=float)
    hi0 = float(hi[0])
    x, obs_pts = [], []
    for i in range(SEQ_LENGTH, len(hi)):
        w  = hi[i - SEQ_LENGTH:i]
        wn = minmax_norm(w)
        obs_frac = float(np.clip((start_obs + i) / MEAN_TRAIN_LIFE, 0.0, 2.0))
        x.append(list(wn) + [
            slope_of(wn), float(w[-1]), float(w.mean()), float(w.max()),
            float(w.min()), float(w.std()), slope_of(w),
            float(w[-1] - w[0]), float(w[-1] - hi0),
            obs_frac, float(w[-1] * obs_frac),
        ])
        obs_pts.append(i)
    return np.asarray(x), np.asarray(obs_pts)

def train_lgbm(hi_train, train_bids):
    xs, ys = [], []
    for b in train_bids:
        x, obs = make_tabular(hi_train[b], start_obs=0)
        xs.append(x); ys.append(true_rul(len(hi_train[b]), obs))
    x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
    def asym_obj(y_pred, ds):
        diff = ds.get_label() - y_pred
        w = np.where(diff < 0, 2.8, 1.0)
        return -diff * w, np.ones_like(diff) * w
    return lgb.train(
        {"num_leaves": 15, "learning_rate": 0.04, "min_child_samples": 5,
         "feature_fraction": 0.90, "bagging_fraction": 0.90, "bagging_freq": 1,
         "verbose": -1, "objective": asym_obj},
        lgb.Dataset(x_tr, label=y_tr), num_boost_round=260)

def predict_lgbm(model, hi_arr, start_obs=0):
    x, obs = make_tabular(hi_arr, start_obs=start_obs)
    return obs, np.maximum(model.predict(x), MIN_RUL_CYCLES)

# ── Sequence models ───────────────────────────────────────────────────────────
def make_seq(hi_arr, start_obs=0):
    hi = np.asarray(hi_arr, dtype=float)
    hi0 = float(hi[0])
    x, obs_pts = [], []
    for i in range(len(hi) - SEQ_LENGTH):
        w = hi[i:i + SEQ_LENGTH]
        wn    = minmax_norm(w)
        raw   = w.copy()
        delta = w - hi0
        obs_frac = np.clip(
            (start_obs + i + np.arange(SEQ_LENGTH)) / MEAN_TRAIN_LIFE, 0.0, 2.0)
        x.append(np.stack([wn, raw, delta, obs_frac], axis=1))
        obs_pts.append(i + SEQ_LENGTH)
    return np.asarray(x), np.asarray(obs_pts)

class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(4, 48, num_layers=2, batch_first=True, dropout=0.15)
        self.fc  = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1))
    def forward(self, x):
        y, _ = self.rnn(x)
        return self.fc(y[:, -1, :]).squeeze(-1)

class GRURegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(4, 48, num_layers=2, batch_first=True, dropout=0.15)
        self.fc  = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1))
    def forward(self, x):
        y, _ = self.rnn(x)
        return self.fc(y[:, -1, :]).squeeze(-1)

class TCNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.10),
            nn.Conv1d(32, 32, kernel_size=3, dilation=2, padding=2), nn.ReLU(), nn.Dropout(0.10),
            nn.Conv1d(32, 32, kernel_size=3, dilation=4, padding=4), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(32, 24), nn.ReLU(), nn.Linear(24, 1))
    def forward(self, x):
        z = self.net(x.transpose(1, 2))
        return self.fc(z[:, :, -1]).squeeze(-1)

def build_model(kind):
    return {"lstm": LSTMRegressor, "gru": GRURegressor, "tcn": TCNRegressor}[kind]()

def train_torch_model(x_train, y_train, scale, kind, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(kind).to(device)
    ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                       torch.tensor(y_train / scale, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=min(32, len(ds)), shuffle=True)
    opt  = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    best_state, best_loss, patience = None, np.inf, 0
    for _ in range(160):
        model.train()
        losses = []
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
        x, obs = make_seq(hi_train[b], start_obs=0)
        xs.append(x); ys.append(true_rul(len(hi_train[b]), obs))
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
            preds.append(np.maximum(m(xt).cpu().numpy() * scale, MIN_RUL_CYCLES))
    return obs, np.median(preds, axis=0)

# ── DTW/kNN ───────────────────────────────────────────────────────────────────
def seg_dist(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return (0.25 * abs(a[-1] - b[-1]) / 0.25
            + 0.20 * abs(a.mean() - b.mean()) / 0.25
            + 0.20 * abs((a[-1] - a[0]) - (b[-1] - b[0])) / 0.25
            + 0.15 * abs(slope_of(a) - slope_of(b)) / 0.03
            + 0.20 * float(np.mean(np.abs(minmax_norm(a) - minmax_norm(b)))))

def predict_dtw_knn(hi_train, train_bids, hi_target):
    hi_target = np.asarray(hi_target, dtype=float)
    obs_pts = np.arange(SEQ_LENGTH, len(hi_target))
    preds = []
    for obs in obs_pts:
        l = min(MATCH_LEN, obs)
        seg = hi_target[obs - l:obs]
        candidates = []
        for b in train_bids:
            hi = np.asarray(hi_train[b], dtype=float)
            for end in range(l, len(hi)):
                d    = seg_dist(seg, hi[end - l:end])
                pred = max(len(hi) - end, MIN_RUL_CYCLES)
                candidates.append((d, pred))
        top = sorted(candidates, key=lambda x: x[0])[:6]
        wt  = np.asarray([1.0 / (d + EPS) for d, p in top])
        pv  = np.asarray([p for d, p in top])
        preds.append(float(np.average(pv, weights=wt)))
    return obs_pts, np.asarray(preds)

def estimate_start_obs(hi_train, hi_target, train_bids):
    target = np.asarray(hi_target, dtype=float)
    l = min(MATCH_LEN, len(target))
    seg = target[:l]
    candidates = []
    for b in train_bids:
        hi = np.asarray(hi_train[b], dtype=float)
        for s in range(0, max(1, len(hi) - l - 3)):
            d = seg_dist(seg, hi[s:s + l])
            candidates.append((d, s, b))
    if not candidates: return 0
    top = sorted(candidates, key=lambda x: x[0])[:8]
    pos = np.asarray([s for d, s, b in top], dtype=float)
    wt  = np.asarray([1.0 / (d + EPS) for d, s, b in top])
    est  = int(round(np.average(pos, weights=wt)))
    gain = float(target[-1] - target[0])
    est  = max(est, int(round(np.clip((gain - 0.25) / 0.35, 0, 1) * 25)))
    return int(np.clip(est, 0, int(MEAN_TRAIN_LIFE)))

# ── Ensemble / Calibration ────────────────────────────────────────────────────
def score_weights(model_scores):
    vals  = {k: max(v, 1e-4) ** 4 for k, v in model_scores.items()}
    total = sum(vals.values())
    return {k: v / total for k, v in vals.items()}

def ensemble_predictions(preds_by_model, weights):
    total = sum(weights.values()) + EPS
    out   = None
    for k, w in weights.items():
        p   = np.asarray(preds_by_model[k], dtype=float)
        out = p * (w / total) if out is None else out + p * (w / total)
    return np.maximum(out, MIN_RUL_CYCLES)

def calibrate_model(pred_by_bearing, results):
    best_cf, best_score = 1.0, -np.inf
    for cf in np.arange(0.60, 1.61, 0.02):
        sc = float(np.mean([
            score_curve(results[b]["N"], results[b]["obs_pts"],
                        np.asarray(pred_by_bearing[b]) * cf)
            for b in BEARINGS]))
        if sc > best_score:
            best_score, best_cf = sc, float(cf)
    return best_cf, best_score

# ── LOOCV ─────────────────────────────────────────────────────────────────────
def run_loocv(hi_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    model_names = ["lgbm", "lstm", "gru", "tcn", "dtw"]
    results = {}

    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        print(f"{'='*60}\n[LOOCV] B{test_bid} held out")
        n_test = len(hi_train[test_bid])
        fold = {"N": n_test}

        lgbm_model = train_lgbm(hi_train, train_bids)
        obs, pred = predict_lgbm(lgbm_model, hi_train[test_bid])
        fold["obs_pts"] = obs
        fold["lgbm"]    = pred

        for kind in ["lstm", "gru", "tcn"]:
            print(f"  {kind.upper()} ({len(SEEDS)} seeds)...")
            models, scale = train_torch_ensemble(hi_train, train_bids, kind, device)
            obs2, pred2 = predict_torch(models, scale, hi_train[test_bid], 0, device)
            fold[kind] = pred2

        _, pred_dtw = predict_dtw_knn(hi_train, train_bids, hi_train[test_bid])
        fold["dtw"] = pred_dtw

        raw_scores = {m: score_curve(n_test, obs, fold[m]) for m in model_names}
        fold["raw_scores"] = raw_scores
        results[test_bid]  = fold
        print("  " + "  ".join(f"{m}={raw_scores[m]:.4f}" for m in model_names))

    avg_scores = {m: float(np.mean([results[b]["raw_scores"][m] for b in BEARINGS]))
                  for m in model_names}
    shared_weights = score_weights(avg_scores)
    print(f"\n[Shared weights] {{{', '.join(f'{k}:{v:.3f}' for k,v in shared_weights.items())}}}")

    for b in BEARINGS:
        results[b]["zoo_avg"] = ensemble_predictions(
            {m: results[b][m] for m in model_names}, shared_weights)

    cf_rows = []
    for m in model_names + ["zoo_avg"]:
        cf, sc = calibrate_model({b: results[b][m] for b in BEARINGS}, results)
        cf_rows.append({"model": m, "cf": cf, "score": sc})
    cf_df = pd.DataFrame(cf_rows).sort_values("score", ascending=False)
    print("\n[Calibration]\n" + cf_df.to_string(index=False))

    cal_scores = {r["model"]: r["score"] for _, r in cf_df.iterrows() if r["model"] in model_names}
    cal_cfs    = {r["model"]: r["cf"]    for _, r in cf_df.iterrows() if r["model"] in model_names}
    final_weights = score_weights(cal_scores)

    for b in BEARINGS:
        preds = {m: np.asarray(results[b][m]) * cal_cfs[m] for m in model_names}
        results[b]["zoo_cal"] = ensemble_predictions(preds, final_weights)

    zoo_cal_cf, zoo_cal_sc = calibrate_model({b: results[b]["zoo_cal"] for b in BEARINGS}, results)
    zoo_avg_cf, zoo_avg_sc = calibrate_model({b: results[b]["zoo_avg"] for b in BEARINGS}, results)

    best_row   = cf_df.iloc[0]
    best_model = str(best_row["model"])
    best_cf    = float(best_row["cf"])
    best_score = float(best_row["score"])

    candidates = [("zoo_avg", zoo_avg_cf, zoo_avg_sc),
                  ("zoo_cal", zoo_cal_cf, zoo_cal_sc),
                  (best_model, best_cf, best_score)]
    final_kind, final_cf, final_score = max(candidates, key=lambda x: x[2])

    for b in BEARINGS:
        if final_kind == "zoo_avg":
            results[b]["final"] = np.asarray(results[b]["zoo_avg"]) * zoo_avg_cf
        elif final_kind == "zoo_cal":
            results[b]["final"] = np.asarray(results[b]["zoo_cal"]) * zoo_cal_cf
        else:
            results[b]["final"] = np.asarray(results[b][best_model]) * best_cf

    print(f"\nzoo_avg: cf={zoo_avg_cf:.2f}, score={zoo_avg_sc:.4f}")
    print(f"zoo_cal: cf={zoo_cal_cf:.2f}, score={zoo_cal_sc:.4f}")
    print(f"Final: {final_kind}, cf={final_cf:.2f}, score={final_score:.4f}")

    # ── Save required outputs ─────────────────────────────────────────────────
    rows = []
    for b in BEARINGS:
        r  = results[b]; obs_pts = r["obs_pts"]
        sc = score_curve(r["N"], obs_pts, r["final"])
        er = error_summary(r["N"], obs_pts, r["final"])
        rows.append({"bearing": b, "score": round(sc, 4),
                     "mean_er": round(er["mean_er"], 4)})
        results[b]["sc_final"] = sc
    rul_df = pd.DataFrame(rows)
    overall = float(np.mean(rul_df["score"]))
    rul_df.to_csv(OUT_DIR / "train_rul_results.csv", index=False)
    print(f"\nOverall: {overall:.4f}")
    for r in rows:
        print(f"  B{r['bearing']}: score={r['score']:.4f}  mean_er={r['mean_er']:.2f}%")

    # Train RUL plot (per bearing)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Baseline (V1b_dtw) Train RUL — Final={final_kind}  cf={final_cf:.2f}", fontsize=11)
    for ax, b in zip(axes.flatten(), BEARINGS):
        r = results[b]; obs = r["obs_pts"]
        true = true_rul(r["N"], obs)
        ax.plot(obs, true, "k-", lw=2, label="True RUL")
        for m in model_names:
            ax.plot(obs, r[m], lw=0.8, alpha=0.45, label=m)
        ax.plot(obs, r["final"], "b-", lw=2, label="Final")
        ax.set_title(f"B{b}  score={r['sc_final']:.3f}  final={final_kind}")
        ax.set_xlabel("Observation cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_rul_predictions.png", dpi=150)
    plt.close()

    # Train Er% plot (per bearing)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Baseline (V1b_dtw) Train Er% per bearing", fontsize=11)
    for ax, b in zip(axes.flatten(), BEARINGS):
        r = results[b]; obs = r["obs_pts"]
        true = true_rul(r["N"], obs)
        final = np.asarray(r["final"])
        er = 100.0 * (true - final) / true
        sc = r["sc_final"]
        ax.plot(obs, er, lw=1.2)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axhline(-20, color="r", lw=0.6, ls=":")
        ax.axhline(50, color="orange", lw=0.6, ls=":")
        ax.set_title(f"B{b}  score={sc:.3f}  mean_er={float(er.mean()):.1f}%")
        ax.set_xlabel("Observation cycle"); ax.set_ylabel("Er%  (positive=under-predict)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_rul_er_pct.png", dpi=150)
    plt.close()

    return results, cal_cfs, final_weights, shared_weights, final_kind, final_cf

# ── Test inference ─────────────────────────────────────────────────────────────
def run_test(hi_train, hi_test, cal_cfs, final_weights, shared_weights, final_kind, final_cf):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\nTest inference — {final_kind}  cf={final_cf:.2f}")

    lgbm_model = train_lgbm(hi_train, BEARINGS)
    torch_models = {}
    for kind in ["lstm", "gru", "tcn"]:
        print(f"  {kind.upper()} final training...")
        torch_models[kind] = train_torch_ensemble(hi_train, BEARINGS, kind, device)

    summary_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Baseline (V1b_dtw) Test RUL — {final_kind} cf={final_cf:.2f}", fontsize=11)

    all_test_rows = []
    for ax, tid in zip(axes.flatten(), TEST_IDS):
        hi        = hi_test[tid]
        start_obs = estimate_start_obs(hi_train, hi, BEARINGS)
        print(f"  [T{tid}] start_obs={start_obs}  hi={hi[0]:.3f}→{hi[-1]:.3f}")

        obs, p_lgbm = predict_lgbm(lgbm_model, hi, start_obs=start_obs)
        preds_raw = {"lgbm": p_lgbm}
        for kind in ["lstm", "gru", "tcn"]:
            models, scale = torch_models[kind]
            _, pp = predict_torch(models, scale, hi, start_obs, device)
            preds_raw[kind] = pp
        _, p_dtw = predict_dtw_knn(hi_train, BEARINGS, hi)
        preds_raw["dtw"] = p_dtw

        if final_kind == "zoo_avg":
            final = ensemble_predictions(preds_raw, shared_weights) * final_cf
        elif final_kind == "zoo_cal":
            preds_cal = {m: preds_raw[m] * cal_cfs[m] for m in preds_raw}
            final = ensemble_predictions(preds_cal, final_weights) * final_cf
        else:
            final = preds_raw[final_kind] * final_cf
        final = np.maximum(final, MIN_RUL_CYCLES)
        hours = final * INTERVAL_SEC / 3600.0

        for i, (o, f, h) in enumerate(zip(obs, final, hours)):
            all_test_rows.append({"test_id": tid, "obs_cycle": int(o),
                                  "rul_cycles": float(f), "rul_hours": float(h)})

        summary_rows.append({"test_id": tid, "start_obs": start_obs,
                              "hi_start": round(float(hi[0]), 3),
                              "hi_end": round(float(hi[-1]), 3),
                              "rul_hours_final": round(float(hours[-1]), 2)})
        print(f"         RUL={hours[-1]:.2f}hr")

        ax.plot(obs, final, "b-", lw=2, label="final")
        for k, v in preds_raw.items():
            ax.plot(obs, v, lw=0.9, alpha=0.45, label=k)
        ax.set_title(f"T{tid}  start={start_obs}  RUL={hours[-1]:.1f}hr")
        ax.set_xlabel("Observation cycle"); ax.set_ylabel("RUL cycles")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_rul_predictions.png", dpi=150)
    plt.close()
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "test_rul_results.csv", index=False)
    pd.DataFrame(all_test_rows).to_csv(OUT_DIR / "test_rul_all_cycles.csv", index=False)

# ── Save HI outputs ───────────────────────────────────────────────────────────
def save_hi_outputs(hi_train, hi_test):
    # Train HI CSV
    rows = []
    for b in BEARINGS:
        for i, v in enumerate(hi_train[b]):
            rows.append({"bearing": b, "cycle": i, "HI": float(v)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "train_hi.csv", index=False)

    # Test HI CSV
    rows = []
    for t in TEST_IDS:
        for i, v in enumerate(hi_test[t]):
            rows.append({"test_id": t, "cycle": i, "HI": float(v)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "test_hi.csv", index=False)

    # Train HI plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Baseline (V1b) Train HI curves", fontsize=11)
    for ax, b in zip(axes.flatten(), BEARINGS):
        ax.plot(hi_train[b], lw=1.5, label=f"B{b}")
        ax.set_title(f"Bearing {b}  (len={len(hi_train[b])})")
        ax.set_xlabel("Cycle"); ax.set_ylabel("HI")
        ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_hi.png", dpi=150)
    plt.close()

    # Test HI plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle("Baseline (V1b) Test HI curves", fontsize=11)
    for ax, t in zip(axes.flatten(), TEST_IDS):
        ax.plot(hi_test[t], lw=1.5)
        ax.set_title(f"Test {t}  (len={len(hi_test[t])})")
        ax.set_xlabel("Cycle"); ax.set_ylabel("HI")
        ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_hi.png", dpi=150)
    plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== Ensemble_6 / Baseline Reproduction (V1b_dtw) ===\n")
    hi_train = load_train_hi()
    hi_test  = load_test_hi()

    for b in BEARINGS:
        hi = hi_train[b]
        print(f"  Train B{b}: {len(hi)} cycles  {hi[0]:.3f}→{hi[-1]:.3f}  max={hi.max():.3f}")
    for t in TEST_IDS:
        hi = hi_test[t]
        print(f"  Test  T{t}: {len(hi)} cycles  {hi[0]:.3f}→{hi[-1]:.3f}")

    print("\n[Saving HI outputs]")
    save_hi_outputs(hi_train, hi_test)

    print("\n[LOOCV]")
    results, cal_cfs, final_weights, shared_weights, final_kind, final_cf = run_loocv(hi_train)

    print("\n[Test inference]")
    run_test(hi_train, hi_test, cal_cfs, final_weights, shared_weights, final_kind, final_cf)

    print(f"\nDone → {OUT_DIR}")

if __name__ == "__main__":
    main()
