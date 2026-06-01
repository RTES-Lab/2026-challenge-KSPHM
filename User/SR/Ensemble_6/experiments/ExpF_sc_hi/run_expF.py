"""
Ensemble_6 / Exp-F: SC HI Exchange Experiment
===============================================
Test if SC team's HI (c035_cubic signal transform) improves RUL prediction
when plugged into the best Exp-E configuration.

SC HI source: User/SC/HI/05170000_postalign_rpmscale_hi/output3_SignalTransform/c035_cubic/
Column: HI_final (calibrated, smoother than HI_raw)

Key characteristics vs V1b HI:
  - B1,B2,B3 end at exactly HI=1.0 (vs 0.66-0.83 in V1b)
  - B4 starts at 0.564 (already degraded), ends at 0.997
  - More uniform 0→1 scale across bearings
  - Signal transform + cubic interpolation (smoother)

Strategy:
  1. Full 2D sweep (same as Exp-E) with SC HI
  2. Compare LOOCV scores vs Exp-E (V1b HI, 0.5685)
  3. Use best config for test predictions
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

BASE    = Path("/data/home/ksphm/2026-challenge-KSPHM")
SC_HI   = BASE / "User" / "SC" / "HI" / "05170000_postalign_rpmscale_hi" / "output3_SignalTransform" / "c035_cubic"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH      = 10
MATCH_LEN       = 18
INTERVAL_SEC    = 600
MEAN_TRAIN_LIFE = 116.5
TCN_SEEDS       = [42, 123, 777]
MIN_RUL_CYCLES  = 1.0
EPS             = 1e-8

NORMAL_UNTIL = {1: 89, 2: 92, 3: 62, 4: 78}
EOL          = {1: 126, 2: 114, 3: 89, 4: 137}

DTW_SWEEP    = np.arange(0.78, 0.94, 0.01).tolist()
LGBM_TCN_RATIOS = [
    (0.55, 0.45),
    (0.65, 0.35),
    (0.45, 0.55),
    (0.70, 0.30),
]

def load_train_hi():
    return {b: pd.read_csv(SC_HI / f"Bearing{b}_HI.csv")["HI_final"].values.astype(float)
            for b in BEARINGS}

def load_test_hi():
    return {t: pd.read_csv(SC_HI / f"Test{t}_HI.csv")["HI_final"].values.astype(float)
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

def train_torch_model(x_tr, y_tr, scale, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TCNRegressor().to(device)
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

def train_tcn_ensemble(hi_train, train_bids, device):
    xs, ys = [], []
    for b in train_bids:
        x, obs = make_seq(hi_train[b]); xs.append(x)
        ys.append(true_rul(len(hi_train[b]), obs))
    x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
    scale = float(max(y_tr.max(), 1.0))
    models = [train_torch_model(x_tr, y_tr, scale, s, device) for s in TCN_SEEDS]
    return models, scale

def predict_tcn(models, scale, hi_arr, start_obs, device):
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
    gain = float(target[-1]-target[0])
    est  = max(est, int(round(np.clip((gain-0.25)/0.35, 0, 1)*25)))
    return int(np.clip(est, 0, int(MEAN_TRAIN_LIFE)))

def calibrate(pred_by_bearing, results, cf_min=0.50, cf_max=1.80, cf_step=0.01):
    best_cf, best_score = 1.0, -np.inf
    for cf in np.arange(cf_min, cf_max, cf_step):
        sc = float(np.mean([score_curve(results[b]["N"], results[b]["obs_pts"],
                                        np.asarray(pred_by_bearing[b])*cf)
                            for b in BEARINGS]))
        if sc > best_score:
            best_score, best_cf = sc, float(cf)
    return best_cf, best_score

def run_loocv(hi_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    results = {}

    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        print(f"\n[LOOCV] B{test_bid} held out:")
        n_test = len(hi_train[test_bid])
        fold = {"N": n_test}

        lgbm_m = train_lgbm(hi_train, train_bids)
        obs, pred_lgbm = predict_lgbm(lgbm_m, hi_train[test_bid])
        fold["obs_pts"] = obs
        fold["lgbm"] = pred_lgbm

        print(f"  TCN ({len(TCN_SEEDS)} seeds)...")
        tcn_models, scale = train_tcn_ensemble(hi_train, train_bids, device)
        _, pred_tcn = predict_tcn(tcn_models, scale, hi_train[test_bid], 0, device)
        fold["tcn"] = pred_tcn

        _, pred_dtw = predict_dtw_knn(hi_train, train_bids, hi_train[test_bid])
        fold["dtw"] = pred_dtw

        fold["raw_scores"] = {m: score_curve(n_test, obs, fold[m]) for m in ["lgbm", "tcn", "dtw"]}
        results[test_bid] = fold
        print("  " + "  ".join(f"{m}={fold['raw_scores'][m]:.4f}" for m in ["lgbm", "tcn", "dtw"]))

    # Global calibration
    cal_cfs = {}
    for m in ["lgbm", "tcn", "dtw"]:
        cf, sc = calibrate({b: results[b][m] for b in BEARINGS}, results)
        cal_cfs[m] = cf
        print(f"  {m}: cf={cf:.2f}  score={sc:.4f}")

    for b in BEARINGS:
        for m in ["lgbm", "tcn", "dtw"]:
            results[b][f"{m}_cal"] = np.asarray(results[b][m]) * cal_cfs[m]

    print(f"\n[2D Sweep: DTW weight × LGBM/TCN ratio]")
    sweep_rows = []
    best_overall, best_params = -np.inf, None
    best_bear_preds = None

    for w_dtw in DTW_SWEEP:
        for r_lgbm, r_tcn in LGBM_TCN_RATIOS:
            w_lgbm = (1-w_dtw) * r_lgbm
            w_tcn  = (1-w_dtw) * r_tcn
            total_w = w_lgbm + w_tcn + w_dtw

            bear_preds = {}
            for b in BEARINGS:
                pred = (np.asarray(results[b]["lgbm_cal"]) * (w_lgbm/total_w)
                       + np.asarray(results[b]["tcn_cal"])  * (w_tcn/total_w)
                       + np.asarray(results[b]["dtw_cal"])  * (w_dtw/total_w))
                bear_preds[b] = np.maximum(pred, MIN_RUL_CYCLES)

            cf, overall = calibrate(bear_preds, results)

            bear_scores = {}
            for b in BEARINGS:
                final = np.asarray(bear_preds[b]) * cf
                bear_scores[b] = score_curve(results[b]["N"], results[b]["obs_pts"], final)

            sweep_rows.append({
                "w_dtw": round(w_dtw, 2), "r_lgbm": r_lgbm,
                "cf": round(cf, 2), "overall": round(overall, 4),
                "B1": round(bear_scores[1], 4), "B2": round(bear_scores[2], 4),
                "B3": round(bear_scores[3], 4), "B4": round(bear_scores[4], 4)
            })

            if overall > best_overall:
                best_overall = overall
                best_params = (w_dtw, r_lgbm, r_tcn, cf)
                best_bear_preds = {b: np.asarray(bear_preds[b]) * cf for b in BEARINGS}

    sweep_df = pd.DataFrame(sweep_rows).sort_values("overall", ascending=False)
    print("\nTop-10 configurations:")
    print(sweep_df.head(10).to_string(index=False))

    w_dtw_best, r_lgbm_best, r_tcn_best, cf_best = best_params
    print(f"\n[Best: w_dtw={w_dtw_best:.2f}  r_lgbm={r_lgbm_best}  cf={cf_best:.2f}  overall={best_overall:.4f}]")
    print(f"  Exp-E best (V1b HI, w_dtw=0.88, lgbm=0.70):  0.5685")
    print(f"  This experiment best (SC HI):                 {best_overall:.4f}")

    rows = []
    for b in BEARINGS:
        sc = score_curve(results[b]["N"], results[b]["obs_pts"], best_bear_preds[b])
        er = error_summary(results[b]["N"], results[b]["obs_pts"], best_bear_preds[b])
        rows.append({"bearing": b, "score": round(sc, 4), "mean_er": round(er["mean_er"], 4)})
        results[b]["sc_final"] = sc
        results[b]["final"] = best_bear_preds[b]
    for r in rows:
        print(f"  B{r['bearing']}: score={r['score']:.4f}  mean_er={r['mean_er']:.2f}%")
    print(f"  Overall: {float(np.mean([r['score'] for r in rows])):.4f}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "train_rul_results.csv", index=False)
    sweep_df.to_csv(OUT_DIR / "2d_sweep_results.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Exp-F SC HI LOOCV — w_dtw={w_dtw_best:.2f}  overall={best_overall:.4f}", fontsize=11)
    for ax, b in zip(axes.flatten(), BEARINGS):
        r = results[b]; obs = r["obs_pts"]; true = true_rul(r["N"], obs)
        ax.plot(obs, true, "k-", lw=2, label="True")
        ax.plot(obs, r["dtw_cal"], "r--", lw=1.5, alpha=0.7, label=f"DTW cal")
        ax.plot(obs, r["final"], "b-", lw=2, label="Best ensemble")
        ax.set_title(f"B{b}  score={r['sc_final']:.3f}")
        ax.set_xlabel("Obs cycle"); ax.set_ylabel("RUL")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_rul_predictions.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Exp-F Train Er%", fontsize=11)
    for ax, b in zip(axes.flatten(), BEARINGS):
        r = results[b]; obs = r["obs_pts"]; true = true_rul(r["N"], obs)
        er = 100.0 * (true - np.asarray(r["final"])) / true
        ax.plot(obs, er, lw=1.2); ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"B{b}  mean_er={float(er.mean()):.1f}%")
        ax.set_xlabel("Obs cycle"); ax.set_ylabel("Er%"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_rul_er_pct.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r_lgbm, r_tcn in LGBM_TCN_RATIOS:
        sub = sweep_df[sweep_df["r_lgbm"] == r_lgbm].sort_values("w_dtw")
        axes[0].plot(sub["w_dtw"], sub["overall"], marker="o", label=f"lgbm_ratio={r_lgbm}")
    axes[0].axhline(0.5685, color="r", ls="--", lw=0.8, label="Exp-E V1b HI (0.5685)")
    axes[0].axhline(0.5499, color="k", ls="--", lw=0.8, label="Baseline (0.5499)")
    axes[0].axvline(w_dtw_best, color="b", ls=":", lw=0.8)
    axes[0].set_title("Exp-F SC HI: Overall vs DTW weight")
    axes[0].set_xlabel("DTW weight"); axes[0].set_ylabel("Overall score")
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    best_sub = sweep_df[sweep_df["r_lgbm"] == r_lgbm_best].sort_values("w_dtw")
    for b in BEARINGS:
        axes[1].plot(best_sub["w_dtw"], best_sub[f"B{b}"], marker="o", label=f"B{b}")
    axes[1].set_title(f"Per-bearing (r_lgbm={r_lgbm_best})")
    axes[1].set_xlabel("DTW weight"); axes[1].set_ylabel("Score")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "2d_sweep_heatmap.png", dpi=150)
    plt.close()

    return results, cal_cfs, w_dtw_best, r_lgbm_best, r_tcn_best, cf_best

def run_test(hi_train, hi_test, cal_cfs, w_dtw, r_lgbm, r_tcn, cf_best):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Test: w_dtw={w_dtw:.2f}  r_lgbm={r_lgbm}  cf={cf_best:.2f}]")

    lgbm_model = train_lgbm(hi_train, BEARINGS)
    print(f"  TCN ({len(TCN_SEEDS)} seeds)...")
    tcn_models, scale = train_tcn_ensemble(hi_train, BEARINGS, device)

    w_lgbm = (1-w_dtw) * r_lgbm
    w_tcn  = (1-w_dtw) * r_tcn
    total_w = w_lgbm + w_tcn + w_dtw

    summary_rows, all_rows = [], []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Exp-F Test SC HI — w_dtw={w_dtw:.2f}  cf={cf_best:.2f}", fontsize=11)

    for ax, tid in zip(axes.flatten(), TEST_IDS):
        hi = hi_test[tid]
        start_obs = estimate_start_obs(hi_train, hi, BEARINGS)
        obs, p_lgbm = predict_lgbm(lgbm_model, hi, start_obs=start_obs)
        _, p_tcn = predict_tcn(tcn_models, scale, hi, start_obs, device)
        _, p_dtw = predict_dtw_knn(hi_train, BEARINGS, hi)

        p_lgbm_cal = p_lgbm * cal_cfs["lgbm"]
        p_tcn_cal  = p_tcn  * cal_cfs["tcn"]
        p_dtw_cal  = p_dtw  * cal_cfs["dtw"]

        pred = (p_lgbm_cal*(w_lgbm/total_w) + p_tcn_cal*(w_tcn/total_w)
                + p_dtw_cal*(w_dtw/total_w))
        final = np.maximum(pred * cf_best, MIN_RUL_CYCLES)
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
        ax.plot(obs, p_dtw_cal, "r--", lw=1, alpha=0.7, label="DTW_cal")
        ax.plot(obs, p_lgbm_cal, "g:", lw=1, alpha=0.7, label="LGBM_cal")
        ax.plot(obs, p_tcn_cal,  "m:", lw=1, alpha=0.7, label="TCN_cal")
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
        for i, v in enumerate(hi_train[b]):
            rows.append({"bearing": b, "cycle": i, "HI": float(v)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "train_hi.csv", index=False)
    rows = []
    for t in TEST_IDS:
        for i, v in enumerate(hi_test[t]):
            rows.append({"test_id": t, "cycle": i, "HI": float(v)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "test_hi.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, b in zip(axes.flatten(), BEARINGS):
        ax.plot(hi_train[b], lw=1.5)
        ax.set_title(f"B{b} (SC c035)"); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    fig.suptitle("Exp-F SC HI Train", fontsize=11); plt.tight_layout()
    plt.savefig(OUT_DIR / "train_hi.png", dpi=150); plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for ax, t in zip(axes.flatten(), TEST_IDS):
        ax.plot(hi_test[t], lw=1.5)
        ax.set_title(f"T{t} (SC c035)"); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    fig.suptitle("Exp-F SC HI Test", fontsize=11); plt.tight_layout()
    plt.savefig(OUT_DIR / "test_hi.png", dpi=150); plt.close()

def main():
    print("=== Ensemble_6 / Exp-F: SC HI Exchange (c035_cubic) ===\n")
    hi_train = load_train_hi()
    hi_test  = load_test_hi()
    save_hi_outputs(hi_train, hi_test)
    results, cal_cfs, w_dtw, r_lgbm, r_tcn, cf_best = run_loocv(hi_train)
    run_test(hi_train, hi_test, cal_cfs, w_dtw, r_lgbm, r_tcn, cf_best)
    print(f"\nDone → {OUT_DIR}")

if __name__ == "__main__":
    main()
