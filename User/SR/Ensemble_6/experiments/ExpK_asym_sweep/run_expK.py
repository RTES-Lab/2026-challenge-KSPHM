"""
Ensemble_6 / Exp-K: LGBM Asymmetric Loss Sweep
================================================
Exp-J 구조 (DTW base + LGBM/TCN capped upside, beta=0.0) 고정.
LGBM의 비대칭 손실 가중치(asym)를 sweep하여 B4 under-predict 문제 해결 시도.

현재: asym=2.8 (과대예측 2.8× 페널티 → B4 체계적 과소예측)
가설: asym을 낮추면 LGBM이 B4 RUL을 더 높게 예측 → B4 개선

구조:
    DTW LOOCV  : 1회 (asym 무관)
    TCN LOOCV  : 1회 (beta=0.0 고정, asym 무관)
    LGBM LOOCV : asym값마다 재학습
    Sweep      : asym × alpha_lgbm × cap_lgbm (TCN alpha=1.0, cap=2.0 고정)

탐색:
    ASYM_GRID      = [1.0, 1.5, 2.0, 2.5, 2.8, 3.5]
    ALPHA_LGBM_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]
    CAP_LGBM_GRID   = [1.3, 1.5, 2.0]
    ALPHA_TCN_FIXED = 1.0  (Exp-J best)
    CAP_TCN_FIXED   = 2.0  (Exp-J best)
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
HI_MEAN_EOL     = 0.75
TCN_SEEDS_LOOCV = [42]
TCN_SEEDS_FINAL = [42, 123, 777]
MIN_RUL_CYCLES  = 1.0
EPS             = 1e-8

ASYM_GRID       = [1.0, 1.5, 2.0, 2.5, 2.8, 3.5]
ALPHA_LGBM_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]
CAP_LGBM_GRID   = [1.3, 1.5, 2.0]
ALPHA_TCN_FIXED = 1.0
CAP_TCN_FIXED   = 2.0

EOL = {1: 126, 2: 114, 3: 89, 4: 137}

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
    er = [100.0 * (t - pp) / t for t, pp in zip(y, preds) if t > 0]
    return {"score": score_curve(n, obs_pts, preds), "mean_er": float(np.nanmean(er))}

# ── Features (beta=0.0 fixed) ──────────────────────────────────────────────────

def make_tabular(hi_arr, start_obs=0):
    hi = np.asarray(hi_arr, dtype=float); hi0 = float(hi[0])
    x, obs_pts = [], []
    for i in range(SEQ_LENGTH, len(hi)):
        w = hi[i - SEQ_LENGTH:i]; wn = minmax_norm(w)
        obs_frac = float(np.clip(w[-1] / HI_MEAN_EOL, 0.0, 2.0))
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
        obs_frac = np.clip(w / HI_MEAN_EOL, 0.0, 2.0)
        x.append(np.stack([minmax_norm(w), w.copy(), w - hi0, obs_frac], axis=1))
        obs_pts.append(i + SEQ_LENGTH)
    return np.asarray(x), np.asarray(obs_pts)

# ── LGBM (asym parametric) ────────────────────────────────────────────────────

def train_lgbm(hi_train, train_bids, asym=2.8):
    xs, ys = [], []
    for b in train_bids:
        x, obs = make_tabular(hi_train[b]); xs.append(x)
        ys.append(true_rul(len(hi_train[b]), obs))
    x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
    def asym_obj(y_pred, ds):
        diff = ds.get_label() - y_pred
        w = np.where(diff < 0, asym, 1.0)
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

# ── DTW ───────────────────────────────────────────────────────────────────────

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
                d = seg_dist(seg, hi[end - l:end])
                candidates.append((d, max(len(hi) - end, MIN_RUL_CYCLES)))
        top = sorted(candidates, key=lambda c: c[0])[:6]
        wt = np.asarray([1.0 / (d + EPS) for d, _ in top])
        pv = np.asarray([p for _, p in top])
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
            candidates.append((seg_dist(seg, hi[s:s + l]), s, b))
    if not candidates: return 0
    top = sorted(candidates, key=lambda c: c[0])[:8]
    pos = np.asarray([s for d, s, b in top], dtype=float)
    wt  = np.asarray([1.0 / (d + EPS) for d, s, b in top])
    est = int(round(np.average(pos, weights=wt)))
    gain = float(target[-1] - target[0])
    est  = max(est, int(round(np.clip((gain - 0.25) / 0.35, 0, 1) * 25)))
    return int(np.clip(est, 0, 130))

# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate(pred_by_bearing, meta):
    best_cf, best_score = 1.0, -np.inf
    for cf in np.arange(0.50, 1.80, 0.01):
        sc = float(np.mean([score_curve(meta[b]["N"], meta[b]["obs_pts"],
                                        np.asarray(pred_by_bearing[b]) * cf)
                            for b in BEARINGS]))
        if sc > best_score:
            best_score, best_cf = sc, float(cf)
    return best_cf, best_score

# ── Capped upside ──────────────────────────────────────────────────────────────

def capped_upside(dtw, lgbm, tcn, al, cl, at, ct):
    base = np.asarray(dtw, dtype=float)
    ref  = np.maximum(base, MIN_RUL_CYCLES)
    lgbm_up = al * np.clip(np.asarray(lgbm, dtype=float) - base, 0.0, (cl - 1.0) * ref)
    tcn_up  = at * np.clip(np.asarray(tcn,  dtype=float) - base, 0.0, (ct - 1.0) * ref)
    return np.maximum(base + lgbm_up + tcn_up, MIN_RUL_CYCLES)

# ── LOOCV (DTW+TCN once, LGBM per asym) ───────────────────────────────────────

def run_dtw_tcn_loocv(hi_train, device):
    print("[LOOCV] DTW + TCN (computed once) ...")
    dtw_raw, tcn_raw, meta = {}, {}, {}
    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        obs, p_dtw = predict_dtw_knn(hi_train, train_bids, hi_train[test_bid])
        tcn_m, sc  = train_tcn_ensemble(hi_train, train_bids, device, TCN_SEEDS_LOOCV)
        _, p_tcn   = predict_tcn(tcn_m, sc, hi_train[test_bid], 0, device)
        n = len(hi_train[test_bid])
        dtw_raw[test_bid] = p_dtw
        tcn_raw[test_bid] = p_tcn
        meta[test_bid]    = {"N": n, "obs_pts": obs}
        print(f"  B{test_bid}: dtw={score_curve(n, obs, p_dtw):.4f}  "
              f"tcn={score_curve(n, obs, p_tcn):.4f}")

    dtw_cf, _ = calibrate(dtw_raw, meta)
    tcn_cf, _ = calibrate(tcn_raw, meta)
    dtw_cal = {b: dtw_raw[b] * dtw_cf for b in BEARINGS}
    tcn_cal = {b: tcn_raw[b] * tcn_cf for b in BEARINGS}
    print(f"  dtw_cf={dtw_cf:.2f}  tcn_cf={tcn_cf:.2f}\n")
    return dtw_cal, tcn_cal, dtw_cf, tcn_cf, dtw_raw, tcn_raw, meta

def run_lgbm_loocv(hi_train, asym, meta):
    lgbm_raw = {}
    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        m = train_lgbm(hi_train, train_bids, asym=asym)
        _, p = predict_lgbm(m, hi_train[test_bid])
        lgbm_raw[test_bid] = p
    lgbm_cf, _ = calibrate(lgbm_raw, meta)
    lgbm_cal = {b: lgbm_raw[b] * lgbm_cf for b in BEARINGS}
    scores = {b: score_curve(meta[b]["N"], meta[b]["obs_pts"], lgbm_raw[b]) for b in BEARINGS}
    print(f"  lgbm_cf={lgbm_cf:.2f}  "
          + "  ".join(f"B{b}={scores[b]:.4f}" for b in BEARINGS))
    return lgbm_cal, lgbm_cf

# ── Sweep (fixed TCN params from Exp-J) ───────────────────────────────────────

def sweep_asym(dtw_cal, tcn_cal, lgbm_cal, meta):
    configs = list(product(ALPHA_LGBM_GRID, CAP_LGBM_GRID))
    rows = []
    best_overall, best_params = -np.inf, None

    for al, cl in configs:
        blend = {b: capped_upside(dtw_cal[b], lgbm_cal[b], tcn_cal[b],
                                  al, cl, ALPHA_TCN_FIXED, CAP_TCN_FIXED)
                 for b in BEARINGS}
        cf, overall = calibrate(blend, meta)
        bear_sc = {b: score_curve(meta[b]["N"], meta[b]["obs_pts"],
                                  blend[b] * cf) for b in BEARINGS}
        rows.append({"alpha_lgbm": al, "cap_lgbm": cl,
                     "cf": round(cf, 2), "overall": round(overall, 4),
                     **{f"B{b}": round(bear_sc[b], 4) for b in BEARINGS}})
        if overall > best_overall:
            best_overall = overall
            best_params  = (al, cl, cf)

    return pd.DataFrame(rows).sort_values("overall", ascending=False), best_overall, best_params

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Ensemble_6 / Exp-K: LGBM Asymmetric Loss Sweep ===")
    print(f"  Exp-J structure fixed: DTW base + LGBM/TCN capped upside (beta=0.0)")
    print(f"  TCN fixed: alpha={ALPHA_TCN_FIXED}, cap={CAP_TCN_FIXED}")
    print(f"  ASYM_GRID: {ASYM_GRID}\n")

    hi_train = load_train_hi()
    hi_test  = load_test_hi()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Step 1: DTW + TCN LOOCV (once)
    dtw_cal, tcn_cal, dtw_cf, tcn_cf, dtw_raw, tcn_raw, meta = run_dtw_tcn_loocv(hi_train, device)

    # Step 2: Per-asym LGBM LOOCV + sweep
    all_rows = []
    best_rows = []  # one row per asym (best config)

    for asym in ASYM_GRID:
        print(f"[asym={asym}] LGBM LOOCV ...")
        lgbm_cal, lgbm_cf = run_lgbm_loocv(hi_train, asym, meta)

        sweep_df, best_overall, best_params = sweep_asym(dtw_cal, tcn_cal, lgbm_cal, meta)
        al, cl, cf = best_params

        for _, row in sweep_df.iterrows():
            all_rows.append({"asym": asym, **row.to_dict()})

        best_blend = {b: capped_upside(dtw_cal[b], lgbm_cal[b], tcn_cal[b],
                                       al, cl, ALPHA_TCN_FIXED, CAP_TCN_FIXED) * cf
                     for b in BEARINGS}
        bear_sc = {b: score_curve(meta[b]["N"], meta[b]["obs_pts"], best_blend[b])
                   for b in BEARINGS}
        mean_er = {b: error_summary(meta[b]["N"], meta[b]["obs_pts"],
                                    best_blend[b])["mean_er"] for b in BEARINGS}

        best_rows.append({
            "asym": asym, "lgbm_cf": round(lgbm_cf, 2),
            "alpha_lgbm": al, "cap_lgbm": cl, "cf": round(cf, 2),
            "overall": round(best_overall, 4),
            **{f"B{b}": round(bear_sc[b], 4) for b in BEARINGS},
            **{f"er_B{b}": round(mean_er[b], 1) for b in BEARINGS}
        })
        print(f"  -> best: al={al} cl={cl} cf={cf:.2f} overall={best_overall:.4f}  "
              f"B4={bear_sc[4]:.4f}(er={mean_er[4]:.1f}%)\n")

    # Summary
    summary_df = pd.DataFrame(best_rows)
    all_df     = pd.DataFrame(all_rows)

    print("\n[Asym Sweep Summary]")
    cols = ["asym", "lgbm_cf", "alpha_lgbm", "cap_lgbm", "cf", "overall",
            "B1", "B2", "B3", "B4", "er_B3", "er_B4"]
    print(summary_df[cols].to_string(index=False))

    best_row = summary_df.loc[summary_df["overall"].idxmax()]
    best_asym = best_row["asym"]
    print(f"\n[Best] asym={best_asym}  overall={best_row['overall']:.4f}  "
          f"B4={best_row['B4']:.4f} (er={best_row['er_B4']:.1f}%)")

    summary_df.to_csv(OUT_DIR / "asym_sweep_summary.csv", index=False)
    all_df.to_csv(OUT_DIR / "asym_sweep_all.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Exp-K: LGBM Asymmetric Loss Sweep", fontsize=12)

    axes[0].plot(summary_df["asym"], summary_df["overall"], marker="o", lw=2, color="steelblue")
    axes[0].axhline(0.6064, color="r", ls="--", lw=1, label="Exp-J 0.6064")
    axes[0].axvline(best_asym, color="b", ls=":", lw=1, label=f"Best asym={best_asym}")
    axes[0].set_xlabel("LGBM Asymmetric Loss Weight"); axes[0].set_ylabel("Overall LOOCV Score")
    axes[0].set_title("Overall vs Asym"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    for b in BEARINGS:
        axes[1].plot(summary_df["asym"], summary_df[f"B{b}"], marker="o", label=f"B{b}")
    axes[1].axvline(best_asym, color="b", ls=":", lw=1)
    axes[1].set_xlabel("Asym"); axes[1].set_ylabel("Score")
    axes[1].set_title("Per-Bearing Score vs Asym")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "asym_sweep.png", dpi=150)
    plt.close()

    # Final test inference with best asym
    print(f"\n[Final Test: asym={best_asym}  3 TCN seeds]")
    lgbm_model  = train_lgbm(hi_train, BEARINGS, asym=best_asym)
    tcn_models, scale = train_tcn_ensemble(hi_train, BEARINGS, device, TCN_SEEDS_FINAL)

    al   = best_row["alpha_lgbm"]
    cl   = best_row["cap_lgbm"]
    cf   = best_row["cf"]

    summary_rows, all_rows_test = [], []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Exp-K Test  asym={best_asym} al={al} cl={cl} cf={cf:.2f}", fontsize=10)

    for ax, tid in zip(axes.flatten(), TEST_IDS):
        hi = hi_test[tid]
        start_obs = estimate_start_obs(hi_train, hi, BEARINGS)

        obs_dtw,  p_dtw  = predict_dtw_knn(hi_train, BEARINGS, hi)
        obs_lgbm, p_lgbm = predict_lgbm(lgbm_model, hi, start_obs=start_obs)
        _,        p_tcn  = predict_tcn(tcn_models, scale, hi, start_obs, device)

        dtw_c  = p_dtw  * dtw_cf
        lgbm_c = p_lgbm * lgbm_cf
        tcn_c  = p_tcn  * tcn_cf

        blend = capped_upside(dtw_c, lgbm_c[:len(obs_dtw)], tcn_c[:len(obs_dtw)],
                              al, cl, ALPHA_TCN_FIXED, CAP_TCN_FIXED)
        final = np.maximum(blend * cf, MIN_RUL_CYCLES)
        hours = final * INTERVAL_SEC / 3600.0

        for o, f, h in zip(obs_dtw, final, hours):
            all_rows_test.append({"test_id": tid, "obs_cycle": int(o),
                                  "rul_cycles": float(f), "rul_hours": float(h)})
        summary_rows.append({"test_id": tid, "start_obs": start_obs,
                              "hi_start": round(float(hi[0]), 3),
                              "hi_end":   round(float(hi[-1]), 3),
                              "rul_hours": round(float(hours[-1]), 2)})
        print(f"  T{tid}: start={start_obs}  RUL={hours[-1]:.2f}hr")

        ax.plot(obs_dtw, final,  "b-",  lw=2, label="final")
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
    pd.DataFrame(all_rows_test).to_csv(OUT_DIR / "test_rul_all_cycles.csv", index=False)

    print(f"\nDone → {OUT_DIR}")


if __name__ == "__main__":
    main()
