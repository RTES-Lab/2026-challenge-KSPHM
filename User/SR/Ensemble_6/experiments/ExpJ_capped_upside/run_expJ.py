"""
Ensemble_6 / Exp-J: DTW-Base Capped Upside (SP-only)
=====================================================
DTW를 고정 기저(base)로 두고, LGBM/TCN은 DTW보다 높을 때만
상향 보정자(upside)로 기여. DTW보다 낮으면 기여 없음.

공식:
    base     = dtw_cal
    lgbm_up  = alpha_lgbm * clip(lgbm_cal - base, 0, (cap_lgbm-1)*base)
    tcn_up   = alpha_tcn  * clip(tcn_cal  - base, 0, (cap_tcn -1)*base)
    final    = (base + lgbm_up + tcn_up) * cf

기대 효과:
    B3: LGBM/TCN raw 점수 0.09~0.12 → 거의 항상 DTW보다 낮음 → 기여 0 → DTW 보호
    B4: LGBM raw 0.625 → DTW(0.529)보다 높음 → LGBM 상향 기여 → B4 개선

Exp-I 최적값 고정:
    beta=0.0  (HI-based obs_frac)

탐색:
    alpha_lgbm ∈ [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cap_lgbm   ∈ [1.1, 1.2, 1.3, 1.5, 2.0]
    alpha_tcn  ∈ [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cap_tcn    ∈ [1.1, 1.2, 1.3, 1.5, 2.0]
    cf         ∈ 0.60~1.50 (step 0.01, inner calibration)
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
from itertools import product

BASE    = Path("/data/home/ksphm/2026-challenge-KSPHM")
HI_DIR  = BASE / "User" / "SP" / "05-26" / "V1b" / "output"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH      = 10
MATCH_LEN       = 18
INTERVAL_SEC    = 600
MEAN_TRAIN_LIFE = 116.5
HI_MEAN_EOL     = 0.75
BETA            = 0.0   # fixed: pure HI-based obs_frac (Exp-I best)
TCN_SEEDS_LOOCV = [42]
TCN_SEEDS_FINAL = [42, 123, 777]
MIN_RUL_CYCLES  = 1.0
EPS             = 1e-8

EOL = {1: 126, 2: 114, 3: 89, 4: 137}

ALPHA_LGBM_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CAP_LGBM_GRID   = [1.1, 1.2, 1.3, 1.5, 2.0]
ALPHA_TCN_GRID  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CAP_TCN_GRID    = [1.1, 1.2, 1.3, 1.5, 2.0]

# ── Data ──────────────────────────────────────────────────────────────────────

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

# ── Feature builders (beta=0.0 fixed) ─────────────────────────────────────────

def make_tabular(hi_arr, start_obs=0):
    hi = np.asarray(hi_arr, dtype=float); hi0 = float(hi[0])
    x, obs_pts = [], []
    for i in range(SEQ_LENGTH, len(hi)):
        w = hi[i - SEQ_LENGTH:i]; wn = minmax_norm(w)
        hi_frac  = float(np.clip(w[-1] / HI_MEAN_EOL, 0.0, 2.0))
        obs_frac = hi_frac  # beta=0.0: pure HI fraction
        x.append(list(wn) + [slope_of(wn), float(w[-1]), float(w.mean()),
                              float(w.max()), float(w.min()), float(w.std()),
                              slope_of(w), float(w[-1] - w[0]), float(w[-1] - hi0),
                              obs_frac, float(w[-1] * obs_frac)])
        obs_pts.append(i)
    return np.asarray(x), np.asarray(obs_pts)

def make_seq(hi_arr, start_obs=0):
    hi = np.asarray(hi_arr, dtype=float); hi0 = float(hi[0])
    x, obs_pts = [], []
    for i in range(len(hi) - SEQ_LENGTH):
        w = hi[i:i + SEQ_LENGTH]
        obs_frac = np.clip(w / HI_MEAN_EOL, 0.0, 2.0)  # beta=0.0
        x.append(np.stack([minmax_norm(w), w.copy(), w - hi0, obs_frac], axis=1))
        obs_pts.append(i + SEQ_LENGTH)
    return np.asarray(x), np.asarray(obs_pts)

# ── LGBM ──────────────────────────────────────────────────────────────────────

def train_lgbm(hi_train, train_bids):
    xs, ys = [], []
    for b in train_bids:
        x, obs = make_tabular(hi_train[b]); xs.append(x)
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

def predict_lgbm(model, hi_arr, start_obs=0):
    x, obs = make_tabular(hi_arr, start_obs=start_obs)
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

def train_tcn_ensemble(hi_train, train_bids, device, seeds):
    xs, ys = [], []
    for b in train_bids:
        x, obs = make_seq(hi_train[b]); xs.append(x)
        ys.append(true_rul(len(hi_train[b]), obs))
    x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
    scale = float(max(y_tr.max(), 1.0))
    models = [train_torch_model(x_tr, y_tr, scale, s, device) for s in seeds]
    return models, scale

def predict_tcn(models, scale, hi_arr, start_obs, device):
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

# ── Capped upside blend ────────────────────────────────────────────────────────

def capped_upside_blend(dtw, lgbm, tcn, alpha_lgbm, cap_lgbm, alpha_tcn, cap_tcn):
    """
    DTW as fixed base. LGBM/TCN contribute only when above DTW.
    cap=1.2 means upside is capped at 20% of base.
    """
    base = np.asarray(dtw, dtype=float)
    ref  = np.maximum(base, MIN_RUL_CYCLES)

    lgbm_up = alpha_lgbm * np.clip(
        np.asarray(lgbm, dtype=float) - base, 0.0, (cap_lgbm - 1.0) * ref)
    tcn_up  = alpha_tcn  * np.clip(
        np.asarray(tcn,  dtype=float) - base, 0.0, (cap_tcn  - 1.0) * ref)

    return np.maximum(base + lgbm_up + tcn_up, MIN_RUL_CYCLES)

# ── LOOCV ─────────────────────────────────────────────────────────────────────

def run_loocv(hi_train, device):
    print("[LOOCV] Running DTW, LGBM, TCN for all bearings ...")
    dtw_loocv, lgbm_loocv, tcn_loocv, meta = {}, {}, {}, {}

    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]

        obs_dtw, p_dtw   = predict_dtw_knn(hi_train, train_bids, hi_train[test_bid])
        lgbm_m           = train_lgbm(hi_train, train_bids)
        obs_lgbm, p_lgbm = predict_lgbm(lgbm_m, hi_train[test_bid])
        tcn_models, scale = train_tcn_ensemble(hi_train, train_bids, device, TCN_SEEDS_LOOCV)
        _, p_tcn          = predict_tcn(tcn_models, scale, hi_train[test_bid], 0, device)

        n = len(hi_train[test_bid])
        dtw_loocv[test_bid]  = p_dtw
        lgbm_loocv[test_bid] = p_lgbm
        tcn_loocv[test_bid]  = p_tcn
        meta[test_bid]       = {"N": n, "obs_pts": obs_dtw}

        sc_dtw  = score_curve(n, obs_dtw,  p_dtw)
        sc_lgbm = score_curve(n, obs_lgbm, p_lgbm)
        sc_tcn  = score_curve(n, obs_dtw,  p_tcn)
        print(f"  B{test_bid}: dtw={sc_dtw:.4f}  lgbm={sc_lgbm:.4f}  tcn={sc_tcn:.4f}")

    dtw_cf,  _ = calibrate(dtw_loocv,  meta)
    lgbm_cf, _ = calibrate(lgbm_loocv, meta)
    tcn_cf,  _ = calibrate(tcn_loocv,  meta)

    dtw_cal  = {b: dtw_loocv[b]  * dtw_cf  for b in BEARINGS}
    lgbm_cal = {b: lgbm_loocv[b] * lgbm_cf for b in BEARINGS}
    tcn_cal  = {b: tcn_loocv[b]  * tcn_cf  for b in BEARINGS}

    print(f"\n  CFs: dtw={dtw_cf:.2f}  lgbm={lgbm_cf:.2f}  tcn={tcn_cf:.2f}")
    print(f"  LGBM>DTW per bearing (after CF):")
    for b in BEARINGS:
        n_above = int(np.sum(lgbm_cal[b] > dtw_cal[b]))
        n_total = len(dtw_cal[b])
        print(f"    B{b}: {n_above}/{n_total} = {100*n_above/n_total:.0f}%")

    return dtw_cal, lgbm_cal, tcn_cal, dtw_cf, lgbm_cf, tcn_cf, meta

# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(dtw_cal, lgbm_cal, tcn_cal, meta):
    print("\n[Sweep] alpha_lgbm × cap_lgbm × alpha_tcn × cap_tcn ...")
    configs = list(product(ALPHA_LGBM_GRID, CAP_LGBM_GRID, ALPHA_TCN_GRID, CAP_TCN_GRID))
    print(f"  Total configs: {len(configs)}")

    rows = []
    best_overall, best_params, best_bear_sc = -np.inf, None, None

    for al, cl, at, ct in configs:
        blend = {b: capped_upside_blend(
            dtw_cal[b], lgbm_cal[b], tcn_cal[b], al, cl, at, ct)
            for b in BEARINGS}
        cf, overall = calibrate(blend, meta)
        bear_sc = {b: score_curve(meta[b]["N"], meta[b]["obs_pts"],
                                  blend[b] * cf) for b in BEARINGS}
        rows.append({"alpha_lgbm": al, "cap_lgbm": cl, "alpha_tcn": at, "cap_tcn": ct,
                     "cf": round(cf, 2), "overall": round(overall, 4),
                     **{f"B{b}": round(bear_sc[b], 4) for b in BEARINGS}})
        if overall > best_overall:
            best_overall, best_params, best_bear_sc = overall, (al, cl, at, ct, cf), bear_sc

    df = pd.DataFrame(rows).sort_values("overall", ascending=False)
    al, cl, at, ct, cf = best_params

    print(f"\n[Top-10]")
    print(df.head(10).to_string(index=False))
    print(f"\n[Best] alpha_lgbm={al}  cap_lgbm={cl}  alpha_tcn={at}  cap_tcn={ct}  "
          f"cf={cf:.2f}  overall={best_overall:.4f}")
    for b in BEARINGS:
        er = error_summary(meta[b]["N"], meta[b]["obs_pts"],
                           capped_upside_blend(dtw_cal[b], lgbm_cal[b], tcn_cal[b],
                                               al, cl, at, ct) * cf)
        print(f"  B{b}: score={best_bear_sc[b]:.4f}  mean_er={er['mean_er']:.1f}%")

    return df, best_params

# ── Test inference ────────────────────────────────────────────────────────────

def run_test(hi_train, hi_test, dtw_cf, lgbm_cf, tcn_cf, best_params, device):
    al, cl, at, ct, cf = best_params
    print(f"\n[Test] alpha_lgbm={al}  cap_lgbm={cl}  alpha_tcn={at}  cap_tcn={ct}  "
          f"cf={cf:.2f}  TCN {len(TCN_SEEDS_FINAL)} seeds")

    lgbm_model = train_lgbm(hi_train, BEARINGS)
    tcn_models, scale = train_tcn_ensemble(hi_train, BEARINGS, device, TCN_SEEDS_FINAL)

    summary_rows, all_rows = [], []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Exp-J Test  al={al} cl={cl} at={at} ct={ct} cf={cf:.2f}", fontsize=10)

    for ax, tid in zip(axes.flatten(), TEST_IDS):
        hi = hi_test[tid]
        start_obs = estimate_start_obs(hi_train, hi, BEARINGS)

        obs_dtw,  p_dtw  = predict_dtw_knn(hi_train, BEARINGS, hi)
        obs_lgbm, p_lgbm = predict_lgbm(lgbm_model, hi, start_obs=start_obs)
        _,        p_tcn  = predict_tcn(tcn_models, scale, hi, start_obs, device)

        dtw_c  = p_dtw  * dtw_cf
        lgbm_c = p_lgbm * lgbm_cf
        tcn_c  = p_tcn  * tcn_cf

        blend = capped_upside_blend(dtw_c, lgbm_c, tcn_c, al, cl, at, ct)
        final = np.maximum(blend * cf, MIN_RUL_CYCLES)
        hours = final * INTERVAL_SEC / 3600.0

        for o, f, h in zip(obs_dtw, final, hours):
            all_rows.append({"test_id": tid, "obs_cycle": int(o),
                             "rul_cycles": float(f), "rul_hours": float(h)})
        summary_rows.append({"test_id": tid, "start_obs": start_obs,
                              "hi_start": round(float(hi[0]), 3),
                              "hi_end":   round(float(hi[-1]), 3),
                              "rul_hours": round(float(hours[-1]), 2)})
        print(f"  T{tid}: start={start_obs}  RUL={hours[-1]:.2f}hr")

        ax.plot(obs_dtw, final,  "b-",  lw=2,            label="final")
        ax.plot(obs_dtw, dtw_c,  "r--", lw=1, alpha=0.7, label="DTW_cal")
        ax.plot(obs_dtw, lgbm_c[:len(obs_dtw)], "g:", lw=1, alpha=0.7, label="LGBM_cal")
        ax.plot(obs_dtw, tcn_c[:len(obs_dtw)],  "m:", lw=1, alpha=0.7, label="TCN_cal")
        ax.set_title(f"T{tid}  RUL={hours[-1]:.1f}hr")
        ax.set_xlabel("Obs cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_rul_predictions.png", dpi=150)
    plt.close()
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "test_rul_results.csv", index=False)
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "test_rul_all_cycles.csv", index=False)

# ── Save plots ────────────────────────────────────────────────────────────────

def save_sweep_plot(df):
    top = df.head(20)
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [f"al={r.alpha_lgbm} cl={r.cap_lgbm}\nat={r.alpha_tcn} ct={r.cap_tcn}"
              for _, r in top.iterrows()]
    ax.bar(range(len(top)), top["overall"], color="steelblue", alpha=0.8)
    ax.axhline(0.5826, color="r", ls="--", lw=1, label="Exp-I 0.5826")
    ax.axhline(0.5685, color="gray", ls=":", lw=1, label="Exp-E 0.5685")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_title("Exp-J Top-20 Configurations")
    ax.set_ylabel("Overall LOOCV Score")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sweep_top20.png", dpi=150)
    plt.close()

def save_loocv_plot(dtw_cal, lgbm_cal, tcn_cal, best_params, meta):
    al, cl, at, ct, cf = best_params
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Exp-J LOOCV best  al={al} cl={cl} at={at} ct={ct} cf={cf:.2f}", fontsize=11)
    for ax, b in zip(axes.flatten(), BEARINGS):
        obs = meta[b]["obs_pts"]; n = meta[b]["N"]
        rul_true = true_rul(n, obs)
        blend = capped_upside_blend(dtw_cal[b], lgbm_cal[b], tcn_cal[b], al, cl, at, ct)
        pred  = blend * cf
        sc = score_curve(n, obs, pred)
        ax.plot(obs, rul_true,    "k-", lw=2, label="True")
        ax.plot(obs, pred,        "b-", lw=2, label=f"Pred (score={sc:.3f})")
        ax.plot(obs, dtw_cal[b],  "r--", lw=1, alpha=0.6, label="DTW_cal")
        ax.plot(obs, lgbm_cal[b][:len(obs)], "g:", lw=1, alpha=0.6, label="LGBM_cal")
        ax.set_title(f"B{b}  score={sc:.4f}")
        ax.set_xlabel("Obs cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_rul_predictions.png", dpi=150)
    plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Ensemble_6 / Exp-J: DTW-Base Capped Upside (SP-only) ===")
    print(f"  beta={BETA} (HI-based obs_frac, fixed from Exp-I)\n")

    hi_train = load_train_hi()
    hi_test  = load_test_hi()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    dtw_cal, lgbm_cal, tcn_cal, dtw_cf, lgbm_cf, tcn_cf, meta = run_loocv(hi_train, device)

    sweep_df, best_params = run_sweep(dtw_cal, lgbm_cal, tcn_cal, meta)
    sweep_df.to_csv(OUT_DIR / "sweep_results.csv", index=False)

    save_sweep_plot(sweep_df)
    save_loocv_plot(dtw_cal, lgbm_cal, tcn_cal, best_params, meta)

    run_test(hi_train, hi_test, dtw_cf, lgbm_cf, tcn_cf, best_params, device)

    print(f"\nDone → {OUT_DIR}")


if __name__ == "__main__":
    main()
