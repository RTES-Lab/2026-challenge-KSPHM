"""
Ensemble_6 / Exp-I: Dynamic obs_frac (Hybrid Time + HI Fraction)
=================================================================
[사용자 제안] obs_frac 개선: 시간 기반 + HI 기반 가중 혼합

Formula:
    obs_frac = beta*(cycle/116.5) + (1-beta)*(HI/0.75)

    - beta=1.0 → pure time fraction (Exp-E behavior)
    - beta=0.0 → pure HI fraction
    - Optimal beta found by grid search

Root cause addressed:
    B3 (89 cycles) at EOL: time_frac=89/116.5=0.76 (undercounts life)
    HI-based frac: HI_eol/0.75 ≈ 0.66/0.75 = 0.88 (closer to 1.0)
    Hybrid (beta<1) corrects the undercount for short-lived bearings.

Architecture: SP original pipeline (LGBM + TCN + DTW)
Beta grid:    11 values [0.0, 0.1, ..., 1.0]
TCN seeds:    1 for sweep (speed), 3 for final best config
Validation:   beta=1.0 results compared to Exp-E (0.5685)
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
HI_MEAN_EOL     = 0.75   # mean EOL HI: B1=0.83, B2=0.72, B3=0.66, B4=0.78 → avg 0.7475 ≈ 0.75
BETA_GRID       = [round(b, 1) for b in np.arange(0.0, 1.01, 0.1)]
TCN_SEEDS_SWEEP = [42]            # 1 seed for beta sweep (speed)
TCN_SEEDS_FINAL = [42, 123, 777]  # 3 seeds for final best-beta inference
MIN_RUL_CYCLES  = 1.0
EPS             = 1e-8

EOL = {1: 126, 2: 114, 3: 89, 4: 137}

# Same sweep grid as Exp-E
DTW_FINE_SWEEP  = np.arange(0.78, 0.93, 0.01).tolist()
LGBM_TCN_RATIOS = [(0.55, 0.45), (0.65, 0.35), (0.45, 0.55), (0.70, 0.30)]

# ── Data loading ──────────────────────────────────────────────────────────────

def load_train_hi():
    return {b: pd.read_csv(HI_DIR / f"HI_Bearing{b}.csv")["HI"].values.astype(float)
            for b in BEARINGS}

def load_test_hi():
    return {t: pd.read_csv(HI_DIR / "test" / f"HI_Test{t}.csv")["HI"].values.astype(float)
            for t in TEST_IDS}

# ── Utilities ─────────────────────────────────────────────────────────────────

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

def error_summary(n, obs_pts, preds):
    y = true_rul(n, obs_pts)
    p = np.asarray(preds, dtype=float)
    er = [100.0 * (t - pp) / t for t, pp in zip(y, p) if t > 0]
    return {"score": score_curve(n, obs_pts, p), "mean_er": float(np.nanmean(er))}

# ── Feature builders (hybrid obs_frac) ────────────────────────────────────────

def make_tabular(hi_arr, start_obs=0, beta=1.0):
    """21-feature tabular input for LGBM. obs_frac uses hybrid formula."""
    hi = np.asarray(hi_arr, dtype=float); hi0 = float(hi[0])
    x, obs_pts = [], []
    for i in range(SEQ_LENGTH, len(hi)):
        w = hi[i - SEQ_LENGTH:i]; wn = minmax_norm(w)
        time_frac = (start_obs + i) / MEAN_TRAIN_LIFE
        hi_frac   = float(w[-1]) / HI_MEAN_EOL          # current HI normalised by EOL mean
        obs_frac  = float(np.clip(beta * time_frac + (1.0 - beta) * hi_frac, 0.0, 2.0))
        x.append(list(wn) + [slope_of(wn), float(w[-1]), float(w.mean()),
                              float(w.max()), float(w.min()), float(w.std()),
                              slope_of(w), float(w[-1] - w[0]), float(w[-1] - hi0),
                              obs_frac, float(w[-1] * obs_frac)])
        obs_pts.append(i)
    return np.asarray(x), np.asarray(obs_pts)

def make_seq(hi_arr, start_obs=0, beta=1.0):
    """4-channel sequence input for TCN. obs_frac channel uses hybrid formula."""
    hi = np.asarray(hi_arr, dtype=float); hi0 = float(hi[0])
    x, obs_pts = [], []
    for i in range(len(hi) - SEQ_LENGTH):
        w = hi[i:i + SEQ_LENGTH]
        time_frac = np.clip((start_obs + i + np.arange(SEQ_LENGTH)) / MEAN_TRAIN_LIFE, 0.0, 2.0)
        hi_frac   = np.clip(w / HI_MEAN_EOL, 0.0, 2.0)
        obs_frac  = np.clip(beta * time_frac + (1.0 - beta) * hi_frac, 0.0, 2.0)
        x.append(np.stack([minmax_norm(w), w.copy(), w - hi0, obs_frac], axis=1))
        obs_pts.append(i + SEQ_LENGTH)
    return np.asarray(x), np.asarray(obs_pts)

# ── LGBM ──────────────────────────────────────────────────────────────────────

def train_lgbm(hi_train, train_bids, beta=1.0):
    xs, ys = [], []
    for b in train_bids:
        x, obs = make_tabular(hi_train[b], beta=beta); xs.append(x)
        ys.append(true_rul(len(hi_train[b]), obs))
    x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
    def asym_obj(y_pred, ds):
        diff = ds.get_label() - y_pred; w = np.where(diff < 0, 2.8, 1.0)
        return -diff * w, np.ones_like(diff) * w
    return lgb.train(
        {"num_leaves": 15, "learning_rate": 0.04, "min_child_samples": 5,
         "feature_fraction": 0.90, "bagging_fraction": 0.90, "bagging_freq": 1,
         "verbose": -1, "objective": asym_obj},
        lgb.Dataset(x_tr, label=y_tr), num_boost_round=260)

def predict_lgbm(model, hi_arr, start_obs=0, beta=1.0):
    x, obs = make_tabular(hi_arr, start_obs=start_obs, beta=beta)
    return obs, np.maximum(model.predict(x), MIN_RUL_CYCLES)

# ── TCN ───────────────────────────────────────────────────────────────────────

class TCNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 32, 3, padding=1), nn.ReLU(), nn.Dropout(0.10),
            nn.Conv1d(32, 32, 3, dilation=2, padding=2), nn.ReLU(), nn.Dropout(0.10),
            nn.Conv1d(32, 32, 3, dilation=4, padding=4), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(32, 24), nn.ReLU(), nn.Linear(24, 1))
    def forward(self, x):
        z = self.net(x.transpose(1, 2)); return self.fc(z[:, :, -1]).squeeze(-1)

def train_torch_model(x_tr, y_tr, scale, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TCNRegressor().to(device)
    ds = TensorDataset(torch.tensor(x_tr, dtype=torch.float32),
                       torch.tensor(y_tr / scale, dtype=torch.float32))
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

def train_tcn_ensemble(hi_train, train_bids, device, seeds, beta=1.0):
    xs, ys = [], []
    for b in train_bids:
        x, obs = make_seq(hi_train[b], beta=beta); xs.append(x)
        ys.append(true_rul(len(hi_train[b]), obs))
    x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
    scale = float(max(y_tr.max(), 1.0))
    models = [train_torch_model(x_tr, y_tr, scale, s, device) for s in seeds]
    return models, scale

def predict_tcn(models, scale, hi_arr, start_obs, device, beta=1.0):
    x, obs = make_seq(hi_arr, start_obs=start_obs, beta=beta)
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
    return (0.25 * abs(a[-1] - b[-1]) / 0.25 + 0.20 * abs(a.mean() - b.mean()) / 0.25
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
                d = seg_dist(seg, hi[end - l:end])
                pred = max(len(hi) - end, MIN_RUL_CYCLES)
                candidates.append((d, pred))
        top = sorted(candidates, key=lambda c: c[0])[:6]
        wt = np.asarray([1.0 / (d + EPS) for d, p in top])
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
        for s in range(0, max(1, len(hi) - l - 3)):
            d = seg_dist(seg, hi[s:s + l])
            candidates.append((d, s, b))
    if not candidates: return 0
    top = sorted(candidates, key=lambda c: c[0])[:8]
    pos = np.asarray([s for d, s, b in top], dtype=float)
    wt  = np.asarray([1.0 / (d + EPS) for d, s, b in top])
    est = int(round(np.average(pos, weights=wt)))
    gain = float(target[-1] - target[0])
    est  = max(est, int(round(np.clip((gain - 0.25) / 0.35, 0, 1) * 25)))
    return int(np.clip(est, 0, int(MEAN_TRAIN_LIFE)))

# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate(pred_by_bearing, meta, cf_min=0.50, cf_max=1.80, cf_step=0.01):
    best_cf, best_score = 1.0, -np.inf
    for cf in np.arange(cf_min, cf_max, cf_step):
        sc = float(np.mean([score_curve(meta[b]["N"], meta[b]["obs_pts"],
                                        np.asarray(pred_by_bearing[b]) * cf)
                            for b in BEARINGS]))
        if sc > best_score:
            best_score, best_cf = sc, float(cf)
    return best_cf, best_score

# ── DTW LOOCV (beta-independent, computed once) ───────────────────────────────

def run_loocv_dtw(hi_train):
    print("[DTW LOOCV] beta-independent, computed once")
    dtw_loocv, meta = {}, {}
    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        obs_pts, pred_dtw = predict_dtw_knn(hi_train, train_bids, hi_train[test_bid])
        n = len(hi_train[test_bid])
        dtw_loocv[test_bid] = pred_dtw
        meta[test_bid] = {"N": n, "obs_pts": obs_pts}
        sc = score_curve(n, obs_pts, pred_dtw)
        print(f"  B{test_bid}: raw={sc:.4f}")

    dtw_cf, dtw_sc = calibrate(dtw_loocv, meta)
    dtw_cal = {b: dtw_loocv[b] * dtw_cf for b in BEARINGS}
    print(f"  DTW calibrated: cf={dtw_cf:.2f}  score={dtw_sc:.4f}\n")
    return dtw_cal, dtw_cf, meta

# ── Beta sweep ────────────────────────────────────────────────────────────────

def run_beta_sweep(hi_train, dtw_cal, meta, device):
    print(f"[Beta Sweep] BETA_GRID={BETA_GRID}")
    print(f"  TCN seeds for sweep: {TCN_SEEDS_SWEEP} (final uses {TCN_SEEDS_FINAL})\n")

    all_rows  = []
    beta_best = []
    loocv_store = {}   # stores LOOCV preds per beta for report/reconstruct

    for beta in BETA_GRID:
        print(f"  [beta={beta:.1f}] LOOCV ...", flush=True)

        lgbm_loocv, tcn_loocv = {}, {}
        for test_bid in BEARINGS:
            train_bids = [b for b in BEARINGS if b != test_bid]

            lgbm_m = train_lgbm(hi_train, train_bids, beta=beta)
            obs, p_lgbm = predict_lgbm(lgbm_m, hi_train[test_bid], beta=beta)
            lgbm_loocv[test_bid] = p_lgbm

            tcn_models, scale = train_tcn_ensemble(
                hi_train, train_bids, device, TCN_SEEDS_SWEEP, beta=beta)
            _, p_tcn = predict_tcn(
                tcn_models, scale, hi_train[test_bid], 0, device, beta=beta)
            tcn_loocv[test_bid] = p_tcn

        lgbm_cf, _ = calibrate(lgbm_loocv, meta)
        tcn_cf, _  = calibrate(tcn_loocv,  meta)
        lgbm_cal = {b: lgbm_loocv[b] * lgbm_cf for b in BEARINGS}
        tcn_cal  = {b: tcn_loocv[b]  * tcn_cf  for b in BEARINGS}
        loocv_store[beta] = (lgbm_loocv, tcn_loocv, lgbm_cf, tcn_cf)

        best_overall = -np.inf
        best_params  = None
        best_scores  = None

        for w_dtw in DTW_FINE_SWEEP:
            for r_lgbm, r_tcn in LGBM_TCN_RATIOS:
                w_lgbm  = (1 - w_dtw) * r_lgbm
                w_tcn   = (1 - w_dtw) * r_tcn
                total_w = w_lgbm + w_tcn + w_dtw

                bear_preds = {}
                for b in BEARINGS:
                    blend = (lgbm_cal[b] * (w_lgbm / total_w)
                             + tcn_cal[b]  * (w_tcn  / total_w)
                             + dtw_cal[b]  * (w_dtw  / total_w))
                    bear_preds[b] = np.maximum(blend, MIN_RUL_CYCLES)

                cf, overall = calibrate(bear_preds, meta)
                bear_sc = {b: score_curve(meta[b]["N"], meta[b]["obs_pts"],
                                          bear_preds[b] * cf) for b in BEARINGS}

                all_rows.append({
                    "beta": beta, "w_dtw": round(w_dtw, 2), "r_lgbm": r_lgbm,
                    "lgbm_cf": round(lgbm_cf, 2), "tcn_cf": round(tcn_cf, 2),
                    "cf": round(cf, 2), "overall": round(overall, 4),
                    **{f"B{b}": round(bear_sc[b], 4) for b in BEARINGS}
                })

                if overall > best_overall:
                    best_overall = overall
                    best_params  = (w_dtw, r_lgbm, r_tcn, lgbm_cf, tcn_cf, cf)
                    best_scores  = {b: bear_sc[b] for b in BEARINGS}

        w_dtw_b, r_lgbm_b, r_tcn_b, lf, tf, gf = best_params
        print(f"    w_dtw={w_dtw_b:.2f}  r_lgbm={r_lgbm_b}  "
              f"lgbm_cf={lf:.2f}  tcn_cf={tf:.2f}  cf={gf:.2f}  "
              f"overall={best_overall:.4f}  B3={best_scores[3]:.4f}")
        beta_best.append({
            "beta": beta, "overall": best_overall,
            "w_dtw": w_dtw_b, "r_lgbm": r_lgbm_b, "r_tcn": r_tcn_b,
            "lgbm_cf": lf, "tcn_cf": tf, "cf": gf,
            **{f"B{b}": round(best_scores[b], 4) for b in BEARINGS}
        })

    return pd.DataFrame(all_rows), pd.DataFrame(beta_best), loocv_store

# ── Reporting ─────────────────────────────────────────────────────────────────

def report_results(sweep_df, beta_df, dtw_cal, meta, loocv_store):
    best_row  = beta_df.loc[beta_df["overall"].idxmax()]
    best_beta = best_row["beta"]

    print("\n[Beta Sweep Summary]")
    cols = ["beta", "overall", "w_dtw", "r_lgbm", "cf", "B1", "B2", "B3", "B4"]
    print(beta_df[cols].to_string(index=False))

    beta1_row = beta_df[beta_df["beta"] == 1.0].iloc[0]
    print(f"\n  Exp-E reference (beta=1.0, 3 TCN seeds):   0.5685")
    print(f"  beta=1.0 this sweep  (1 TCN seed):          {beta1_row['overall']:.4f}  "
          f"B3={beta1_row['B3']:.4f}")
    print(f"\n[Best: beta={best_beta}  w_dtw={best_row['w_dtw']}  "
          f"r_lgbm={best_row['r_lgbm']}  cf={best_row['cf']:.2f}  "
          f"overall={best_row['overall']:.4f}]")

    lgbm_loocv, tcn_loocv, lgbm_cf, tcn_cf = loocv_store[best_beta]
    lgbm_cal = {b: lgbm_loocv[b] * lgbm_cf for b in BEARINGS}
    tcn_cal  = {b: tcn_loocv[b]  * tcn_cf  for b in BEARINGS}
    w_dtw, r_lgbm, r_tcn = best_row["w_dtw"], best_row["r_lgbm"], best_row["r_tcn"]
    w_lgbm = (1 - w_dtw) * r_lgbm; w_tcn = (1 - w_dtw) * r_tcn
    total_w = w_lgbm + w_tcn + w_dtw
    cf = best_row["cf"]

    final_preds = {}
    rows = []
    for b in BEARINGS:
        blend = (lgbm_cal[b] * (w_lgbm / total_w)
                 + tcn_cal[b]  * (w_tcn  / total_w)
                 + dtw_cal[b]  * (w_dtw  / total_w))
        pred = np.maximum(blend, MIN_RUL_CYCLES) * cf
        final_preds[b] = pred
        sc = score_curve(meta[b]["N"], meta[b]["obs_pts"], pred)
        er = error_summary(meta[b]["N"], meta[b]["obs_pts"], pred)
        print(f"  B{b}: score={sc:.4f}  mean_er={er['mean_er']:.1f}%")
        rows.append({"bearing": b, "score": round(sc, 4), "mean_er": round(er["mean_er"], 2)})

    return best_beta, best_row, final_preds, pd.DataFrame(rows)

# ── Final test inference (best beta, 3 TCN seeds) ─────────────────────────────

def run_final_test(hi_train, hi_test, best_beta, best_row, dtw_cf, device):
    print(f"\n[Final Test: beta={best_beta}  3 TCN seeds]")
    lgbm_model = train_lgbm(hi_train, BEARINGS, beta=best_beta)
    print(f"  TCN ({len(TCN_SEEDS_FINAL)} seeds)...")
    tcn_models, scale = train_tcn_ensemble(
        hi_train, BEARINGS, device, TCN_SEEDS_FINAL, beta=best_beta)

    lgbm_cf = best_row["lgbm_cf"]
    tcn_cf  = best_row["tcn_cf"]
    w_dtw, r_lgbm, r_tcn = best_row["w_dtw"], best_row["r_lgbm"], best_row["r_tcn"]
    w_lgbm = (1 - w_dtw) * r_lgbm; w_tcn = (1 - w_dtw) * r_tcn
    total_w = w_lgbm + w_tcn + w_dtw
    cf_best = best_row["cf"]

    summary_rows, all_rows = [], []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Exp-I Test  beta={best_beta}  w_dtw={w_dtw:.2f}  cf={cf_best:.2f}", fontsize=11)

    for ax, tid in zip(axes.flatten(), TEST_IDS):
        hi = hi_test[tid]
        start_obs = estimate_start_obs(hi_train, hi, BEARINGS)

        obs, p_lgbm = predict_lgbm(lgbm_model, hi, start_obs=start_obs, beta=best_beta)
        _, p_tcn    = predict_tcn(tcn_models, scale, hi, start_obs, device, beta=best_beta)
        _, p_dtw    = predict_dtw_knn(hi_train, BEARINGS, hi)

        p_lgbm_cal = p_lgbm * lgbm_cf
        p_tcn_cal  = p_tcn  * tcn_cf
        p_dtw_cal  = p_dtw  * dtw_cf

        blend = (p_lgbm_cal * (w_lgbm / total_w)
                 + p_tcn_cal  * (w_tcn  / total_w)
                 + p_dtw_cal  * (w_dtw  / total_w))
        final = np.maximum(blend * cf_best, MIN_RUL_CYCLES)
        hours = final * INTERVAL_SEC / 3600.0

        for o, f, h in zip(obs, final, hours):
            all_rows.append({"test_id": tid, "obs_cycle": int(o),
                             "rul_cycles": float(f), "rul_hours": float(h)})
        summary_rows.append({"test_id": tid, "start_obs": start_obs,
                              "hi_start": round(float(hi[0]), 3),
                              "hi_end":   round(float(hi[-1]), 3),
                              "rul_hours": round(float(hours[-1]), 2)})
        print(f"  T{tid}: start={start_obs}  RUL={hours[-1]:.2f}hr")

        ax.plot(obs, final,       "b-",  lw=2,            label="final")
        ax.plot(obs, p_dtw_cal,   "r--", lw=1, alpha=0.7, label="DTW_cal")
        ax.plot(obs, p_lgbm_cal,  "g:",  lw=1, alpha=0.7, label="LGBM_cal")
        ax.plot(obs, p_tcn_cal,   "m:",  lw=1, alpha=0.7, label="TCN_cal")
        ax.set_title(f"T{tid}  RUL={hours[-1]:.1f}hr")
        ax.set_xlabel("Obs cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_rul_predictions.png", dpi=150)
    plt.close()
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "test_rul_results.csv", index=False)
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "test_rul_all_cycles.csv", index=False)

# ── Plots ─────────────────────────────────────────────────────────────────────

def save_plots(beta_df, final_preds, meta, best_beta):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(beta_df["beta"], beta_df["overall"], marker="o", lw=2, color="steelblue")
    axes[0].axhline(0.5685, color="r", ls="--", lw=0.8,
                    label="Exp-E (beta=1.0, 3 seeds) 0.5685")
    axes[0].axvline(best_beta, color="b", ls=":", lw=0.8, label=f"Best beta={best_beta}")
    axes[0].set_title("Overall Score vs Beta")
    axes[0].set_xlabel("Beta (time fraction weight)")
    axes[0].set_ylabel("Overall LOOCV score")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    for b in BEARINGS:
        axes[1].plot(beta_df["beta"], beta_df[f"B{b}"], marker="o", label=f"B{b}")
    axes[1].set_title("Per-Bearing Score vs Beta")
    axes[1].set_xlabel("Beta"); axes[1].set_ylabel("Score")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "beta_curve.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Exp-I LOOCV (best beta={best_beta})", fontsize=11)
    for ax, b in zip(axes.flatten(), BEARINGS):
        obs = meta[b]["obs_pts"]; n = meta[b]["N"]
        rul_true = true_rul(n, obs)
        ax.plot(obs, rul_true,       "k-", lw=2, label="True")
        ax.plot(obs, final_preds[b], "b-", lw=2, label="Pred")
        sc = score_curve(n, obs, final_preds[b])
        ax.set_title(f"B{b}  score={sc:.3f}")
        ax.set_xlabel("Obs cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_rul_predictions.png", dpi=150)
    plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Ensemble_6 / Exp-I: Dynamic obs_frac (Hybrid Time + HI Fraction) ===")
    print("[사용자 제안] obs_frac = beta*(cycle/116.5) + (1-beta)*(HI/0.75)\n")

    hi_train = load_train_hi()
    hi_test  = load_test_hi()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Step 1: DTW LOOCV (once, no obs_frac dependency)
    dtw_cal, dtw_cf, meta = run_loocv_dtw(hi_train)

    # Step 2: Beta sweep (LGBM + TCN 1-seed LOOCV per beta)
    sweep_df, beta_df, loocv_store = run_beta_sweep(hi_train, dtw_cal, meta, device)
    sweep_df.to_csv(OUT_DIR / "sweep_results.csv", index=False)
    beta_df.to_csv(OUT_DIR  / "beta_best.csv",     index=False)

    # Step 3: Report best config
    best_beta, best_row, final_preds, train_rows = report_results(
        sweep_df, beta_df, dtw_cal, meta, loocv_store)
    train_rows.to_csv(OUT_DIR / "train_rul_results.csv", index=False)

    # Step 4: Plots
    save_plots(beta_df, final_preds, meta, best_beta)

    # Step 5: Test inference (best beta, 3 TCN seeds)
    run_final_test(hi_train, hi_test, best_beta, best_row, dtw_cf, device)

    print(f"\nDone → {OUT_DIR}")


if __name__ == "__main__":
    main()
