"""
Ensemble_6 / Exp-G: Cross-Team Capped Ensemble
===============================================
Inspired by TH's ensemble_capped_sc_ridge (leaderboard #1, 0.5747).

Key insight: TH v8_3 GRU/LSTM/TCN is better for B1/B2 but catastrophically bad for B3.
SP/V1b DTW is more reliable for B3. LGBM is excellent for B4 (no catastrophic failure).

Strategy:
  stable = w_th * rul_th + (1-w_th) * rul_sp
  upside = clip(lgbm - stable, low=0, high=(cap-1)*stable)
  final  = (stable + alpha * upside) * global_cf

The one-sided positive clip ensures:
  - B3: LGBM badly under-predicts → lgbm < stable → upside=0 → B3 unchanged
  - B1/B2/B4: LGBM better calibrated → lgbm >= stable at many points → positive boost

Sources:
  - rul_th: TH v8_3 GRU/LSTM/TCN (from TH's ensemble LOOCV CSV)
  - rul_sp: SP V10c DTW (from TH's ensemble LOOCV CSV, ≈ V1b DTW)
  - lgbm:   Our own LGBM trained on V1b HI (re-computed per LOOCV fold)
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb

BASE    = Path("/data/home/ksphm/2026-challenge-KSPHM")
HI_DIR  = BASE / "User" / "SP" / "05-26" / "V1b" / "output"
TH_DIR  = BASE / "User" / "TH" / "RUL" / "output" / "ensemble_th_v8_3_sp_v10c_dtw"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH      = 10
INTERVAL_SEC    = 600
MEAN_TRAIN_LIFE = 116.5
MIN_RUL_CYCLES  = 1.0
EPS             = 1e-8

EOL = {1: 126, 2: 114, 3: 89, 4: 137}

def load_train_hi():
    return {b: pd.read_csv(HI_DIR / f"HI_Bearing{b}.csv")["HI"].values.astype(float)
            for b in BEARINGS}

def load_test_hi():
    return {t: pd.read_csv(HI_DIR / "test" / f"HI_Test{t}.csv")["HI"].values.astype(float)
            for t in TEST_IDS}

def load_th_loocv():
    """Load TH's LOOCV predictions: rul_th (TH v8_3) and rul_sp (SP DTW)."""
    data = {}
    for b in BEARINGS:
        df = pd.read_csv(TH_DIR / f"ensemble_best_all_loocv_B{b}.csv")
        data[b] = df[["obs_cycle", "true_rul", "rul_th", "rul_sp"]].copy()
    return data

def load_th_test():
    """Load TH's test predictions: rul_th and rul_sp per test bearing."""
    data = {}
    for t in TEST_IDS:
        df = pd.read_csv(TH_DIR / f"Test{t}_RUL.csv")
        data[t] = df[["obs_cycle", "rul_th", "rul_sp"]].copy()
    return data

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

def score_curve(y, p):
    return float(np.nanmean([competition_score(t, pp) for t, pp in zip(y, p)]))

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

def true_rul_arr(n, obs_pts):
    return np.maximum(n - np.asarray(obs_pts, dtype=float), 1.0)

def train_lgbm(hi_train, train_bids):
    xs, ys = [], []
    for b in train_bids:
        x, obs = make_tabular(hi_train[b]); xs.append(x)
        ys.append(true_rul_arr(len(hi_train[b]), obs))
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

def capped_blend(rul_th, rul_sp, rul_lgbm, w_th, alpha, cap):
    """
    stable = w_th * rul_th + (1-w_th) * rul_sp
    upside = clip(lgbm - stable, 0, (cap-1)*stable)
    output = stable + alpha * upside
    """
    stable = w_th * rul_th + (1.0 - w_th) * rul_sp
    upside = np.clip(rul_lgbm - stable, 0.0, (cap - 1.0) * stable)
    return stable + alpha * upside

def run_loocv(hi_train, th_data):
    print("[LOOCV: Generating LGBM predictions per fold]\n")
    results = {}

    for b in BEARINGS:
        train_bids = [bb for bb in BEARINGS if bb != b]
        lgbm_m = train_lgbm(hi_train, train_bids)
        obs, pred_lgbm = predict_lgbm(lgbm_m, hi_train[b])

        # Read TH LOOCV for this bearing
        th_fold = th_data[b].set_index("obs_cycle")

        # Align obs_cycles (TH starts at 10, ours also starts at SEQ_LENGTH=10)
        common_obs = np.intersect1d(obs, th_fold.index.values)
        if len(common_obs) < 5:
            print(f"  WARNING: B{b} only {len(common_obs)} common obs cycles!")
            continue

        # Filter to common obs
        mask = np.isin(obs, common_obs)
        obs_c = obs[mask]
        lgbm_c = pred_lgbm[mask]
        rul_th_c = th_fold.loc[obs_c, "rul_th"].values.astype(float)
        rul_sp_c = th_fold.loc[obs_c, "rul_sp"].values.astype(float)
        true_c   = th_fold.loc[obs_c, "true_rul"].values.astype(float)

        results[b] = {
            "obs_pts": obs_c,
            "true_rul": true_c,
            "rul_th": rul_th_c,
            "rul_sp": rul_sp_c,
            "rul_lgbm": lgbm_c,
            "N": len(hi_train[b])
        }

        # Individual scores
        sc_th  = score_curve(true_c, rul_th_c)
        sc_sp  = score_curve(true_c, rul_sp_c)
        sc_lgbm = score_curve(true_c, lgbm_c)
        er_th  = float(np.nanmean((true_c - rul_th_c)  / true_c * 100.0))
        er_sp  = float(np.nanmean((true_c - rul_sp_c)  / true_c * 100.0))
        er_lgbm = float(np.nanmean((true_c - lgbm_c)   / true_c * 100.0))
        print(f"  B{b} (n={len(obs_c)}):")
        print(f"    rul_th  : score={sc_th:.4f}  mean_er={er_th:.1f}%")
        print(f"    rul_sp  : score={sc_sp:.4f}  mean_er={er_sp:.1f}%")
        print(f"    rul_lgbm: score={sc_lgbm:.4f}  mean_er={er_lgbm:.1f}%")

    # ── 3D sweep: w_th × alpha × cap × global_cf ─────────────────────────────
    print("\n[3D Sweep: w_th × alpha × cap]")
    W_TH_GRID  = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
    ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    CAP_GRID   = [1.1, 1.2, 1.3, 1.5, 1.8, 2.0]
    CF_GRID    = np.arange(0.60, 1.50, 0.01)

    best_overall, best_params, best_per_bearing = -np.inf, None, None
    sweep_rows = []

    for w_th in W_TH_GRID:
        for alpha in ALPHA_GRID:
            for cap in CAP_GRID:
                # Compute blended predictions for each bearing
                bear_preds = {}
                for b in BEARINGS:
                    r = results[b]
                    raw = capped_blend(r["rul_th"], r["rul_sp"], r["rul_lgbm"],
                                       w_th, alpha, cap)
                    raw = np.maximum(raw, MIN_RUL_CYCLES)
                    bear_preds[b] = raw

                # Global CF search
                best_cf_local, best_sc_local = 1.0, -np.inf
                for cf in CF_GRID:
                    sc = float(np.mean([
                        score_curve(results[b]["true_rul"],
                                    np.asarray(bear_preds[b]) * cf)
                        for b in BEARINGS
                    ]))
                    if sc > best_sc_local:
                        best_sc_local, best_cf_local = sc, float(cf)

                bear_scores = {}
                for b in BEARINGS:
                    final = np.asarray(bear_preds[b]) * best_cf_local
                    bear_scores[b] = score_curve(results[b]["true_rul"], final)

                row = {"w_th": w_th, "alpha": alpha, "cap": cap, "cf": round(best_cf_local,2),
                       "overall": round(best_sc_local, 4)}
                row.update({f"B{b}": round(bear_scores[b], 4) for b in BEARINGS})
                sweep_rows.append(row)

                if best_sc_local > best_overall:
                    best_overall = best_sc_local
                    best_params = (w_th, alpha, cap, best_cf_local)
                    best_per_bearing = {b: np.asarray(bear_preds[b]) * best_cf_local
                                        for b in BEARINGS}

    sweep_df = pd.DataFrame(sweep_rows).sort_values("overall", ascending=False)
    print("\nTop-10 configurations:")
    print(sweep_df.head(10).to_string(index=False))

    w_th_best, alpha_best, cap_best, cf_best = best_params
    print(f"\n[Best: w_th={w_th_best}  alpha={alpha_best}  cap={cap_best}  cf={cf_best:.2f}  overall={best_overall:.4f}]")
    print(f"  TH ensemble_capped_sc_ridge (leaderboard #1):  0.5747")
    print(f"  Our Exp-E (V1b, no cross-team):               0.5685")
    print(f"  This experiment:                               {best_overall:.4f}")

    rows = []
    for b in BEARINGS:
        sc = score_curve(results[b]["true_rul"], best_per_bearing[b])
        er = float(np.nanmean((results[b]["true_rul"] - best_per_bearing[b])
                               / results[b]["true_rul"] * 100.0))
        rows.append({"bearing": b, "score": round(sc, 4), "mean_er": round(er, 4)})
        print(f"  B{b}: score={sc:.4f}  mean_er={er:.2f}%")

    pd.DataFrame(rows).to_csv(OUT_DIR / "train_rul_results.csv", index=False)
    sweep_df.to_csv(OUT_DIR / "sweep_results.csv", index=False)

    # LOOCV predictions plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Exp-G Cross-Team LOOCV — overall={best_overall:.4f}", fontsize=11)
    for ax, b in zip(axes.flatten(), BEARINGS):
        r = results[b]; obs = r["obs_pts"]; true = r["true_rul"]
        ax.plot(obs, true, "k-", lw=2, label="True")
        ax.plot(obs, r["rul_sp"], "r--", lw=1, alpha=0.7, label="SP DTW")
        ax.plot(obs, r["rul_th"], "g--", lw=1, alpha=0.7, label="TH v8_3")
        ax.plot(obs, best_per_bearing[b], "b-", lw=2, label="Best blend")
        ax.set_title(f"B{b}  score={score_curve(true, best_per_bearing[b]):.3f}")
        ax.set_xlabel("Obs cycle"); ax.set_ylabel("RUL")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_rul_predictions.png", dpi=150)
    plt.close()

    # Error percentage
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Exp-G Train Er%", fontsize=11)
    for ax, b in zip(axes.flatten(), BEARINGS):
        r = results[b]; obs = r["obs_pts"]; true = r["true_rul"]
        er = 100.0 * (true - best_per_bearing[b]) / true
        ax.plot(obs, er, lw=1.2)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"B{b}  mean_er={float(er.mean()):.1f}%")
        ax.set_xlabel("Obs cycle"); ax.set_ylabel("Er%")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_rul_er_pct.png", dpi=150)
    plt.close()

    # Sweep plot
    fig, ax = plt.subplots(figsize=(12, 5))
    for w_th in W_TH_GRID:
        sub = sweep_df[(sweep_df["w_th"] == w_th) & (sweep_df["alpha"] == alpha_best)].sort_values("cap")
        if len(sub) > 0:
            ax.plot(sub["cap"], sub["overall"], marker="o", label=f"w_th={w_th}")
    ax.axhline(0.5747, color="r", ls="--", lw=0.8, label="TH capped (0.5747)")
    ax.axhline(0.5685, color="b", ls="--", lw=0.8, label="Exp-E (0.5685)")
    ax.set_title(f"Sweep: overall vs cap (alpha={alpha_best})")
    ax.set_xlabel("cap"); ax.set_ylabel("Overall score")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sweep_plot.png", dpi=150)
    plt.close()

    return w_th_best, alpha_best, cap_best, cf_best, best_overall

def run_test(hi_train, th_test, w_th, alpha, cap, cf):
    print(f"\n[Test: w_th={w_th}  alpha={alpha}  cap={cap}  cf={cf:.2f}]")

    # Train LGBM on all training data
    lgbm_model = train_lgbm(hi_train, BEARINGS)

    # Estimate start_obs from HI pattern (simple: compare HI to training starts)
    def estimate_start(hi_target):
        # Use DTW approach: find most similar start segment in training
        target = np.asarray(hi_target, dtype=float)
        l = min(18, len(target))
        seg = target[:l]
        best_pos, best_dist = 0, np.inf
        for b in BEARINGS:
            hi = np.asarray(hi_train[b], dtype=float)
            for s in range(0, max(1, len(hi)-l-3)):
                d = float(np.mean(np.abs(minmax_norm(seg) - minmax_norm(hi[s:s+l]))))
                if d < best_dist:
                    best_dist, best_pos = d, s
        gain = float(target[-1] - target[0])
        est = max(best_pos, int(round(np.clip((gain-0.25)/0.35, 0, 1)*25)))
        return int(np.clip(est, 0, int(MEAN_TRAIN_LIFE)))

    summary_rows, all_rows = [], []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Exp-G Test — w_th={w_th}  alpha={alpha}  cap={cap}  cf={cf:.2f}", fontsize=11)

    for ax, tid in zip(axes.flatten(), TEST_IDS):
        hi = load_test_hi()[tid]
        start_obs = estimate_start(hi)

        obs_lgbm, pred_lgbm = predict_lgbm(lgbm_model, hi, start_obs=start_obs)

        # Read TH test predictions
        th_df = th_test[tid].set_index("obs_cycle")
        common_obs = np.intersect1d(obs_lgbm, th_df.index.values)
        if len(common_obs) == 0:
            # If no overlap, use obs_lgbm range
            common_obs = obs_lgbm[:min(len(obs_lgbm), 10)]

        mask = np.isin(obs_lgbm, common_obs)
        obs_c = obs_lgbm[mask]
        lgbm_c = pred_lgbm[mask]
        rul_th_c = th_df.loc[obs_c, "rul_th"].values.astype(float) if len(common_obs) > 0 else lgbm_c
        rul_sp_c = th_df.loc[obs_c, "rul_sp"].values.astype(float) if len(common_obs) > 0 else lgbm_c

        raw = capped_blend(rul_th_c, rul_sp_c, lgbm_c, w_th, alpha, cap)
        final = np.maximum(raw, MIN_RUL_CYCLES) * cf
        hours = final * INTERVAL_SEC / 3600.0

        for o, f, h in zip(obs_c, final, hours):
            all_rows.append({"test_id": tid, "obs_cycle": int(o),
                             "rul_cycles": float(f), "rul_hours": float(h)})
        summary_rows.append({
            "test_id": tid, "start_obs": start_obs,
            "hi_start": round(float(hi[0]), 3),
            "hi_end": round(float(hi[-1]), 3),
            "rul_hours": round(float(hours[-1]), 2)
        })
        print(f"  T{tid}: start={start_obs}  RUL={hours[-1]:.2f}hr  (n_obs={len(obs_c)})")

        ax.plot(obs_c, final, "b-", lw=2, label="final")
        ax.plot(obs_c, rul_sp_c * cf, "r--", lw=1, alpha=0.7, label="SP_dtw×cf")
        ax.plot(obs_c, rul_th_c * cf, "g--", lw=1, alpha=0.7, label="TH×cf")
        ax.plot(obs_c, lgbm_c  * cf, "m:", lw=1, alpha=0.7, label="LGBM×cf")
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
        ax.plot(hi_train[b], lw=1.5); ax.set_title(f"B{b}"); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    fig.suptitle("Exp-G Train HI (V1b)", fontsize=11); plt.tight_layout()
    plt.savefig(OUT_DIR / "train_hi.png", dpi=150); plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for ax, t in zip(axes.flatten(), TEST_IDS):
        ax.plot(hi_test[t], lw=1.5); ax.set_title(f"T{t}"); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    fig.suptitle("Exp-G Test HI (V1b)", fontsize=11); plt.tight_layout()
    plt.savefig(OUT_DIR / "test_hi.png", dpi=150); plt.close()

def main():
    print("=== Ensemble_6 / Exp-G: Cross-Team Capped Ensemble ===\n")
    hi_train = load_train_hi()
    hi_test  = load_test_hi()
    th_data  = load_th_loocv()
    th_test  = load_th_test()
    save_hi_outputs(hi_train, hi_test)
    w_th, alpha, cap, cf, overall = run_loocv(hi_train, th_data)
    run_test(hi_train, th_test, w_th, alpha, cap, cf)
    print(f"\nDone → {OUT_DIR}")

if __name__ == "__main__":
    main()
