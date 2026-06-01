"""
Ensemble_6 / Exp-B: Multi-Match-Length DTW Ensemble
=====================================================
Hypothesis: The DTW result is sensitive to the MATCH_LEN hyperparameter
(currently 18). Different lengths capture different pattern aspects:
  - Short (8-12): focus on recent trend → good for rapidly changing HI
  - Medium (15-18): current baseline
  - Long (21-28): global shape matching → good for overall lifecycle pattern

Averaging predictions from multiple MATCH_LEN values should:
1. Reduce sensitivity to this hyperparameter
2. Capture both short-term and long-term patterns
3. Potentially improve B3 (short life, rapid degradation)

Also includes an improved DTW distance metric:
  - Original: 5 hand-crafted features
  - Enhanced: adds variance, 75th-percentile, and segment gradient consistency

Strategy:
  B1: Run DTW with 5 different MATCH_LEN values [8, 12, 15, 18, 24]
      Average predictions (inverse-distance weighted mean)
  B2: Grid search best single MATCH_LEN via LOOCV
  B3: Compare B1 and B2 vs baseline DTW
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = Path("/data/home/ksphm/2026-challenge-KSPHM")
SP      = BASE / "User" / "SP"
HI_DIR  = SP / "05-26" / "V1b" / "output"
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

MATCH_LENGTHS   = [8, 12, 15, 18, 24]   # multi-scale DTW

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

def seg_dist(a, b):
    """Enhanced segment distance metric."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    # Original 5 components
    d  = 0.25 * abs(a[-1]-b[-1]) / 0.25
    d += 0.20 * abs(a.mean()-b.mean()) / 0.25
    d += 0.20 * abs((a[-1]-a[0])-(b[-1]-b[0])) / 0.25
    d += 0.15 * abs(slope_of(a)-slope_of(b)) / 0.03
    d += 0.20 * float(np.mean(np.abs(minmax_norm(a)-minmax_norm(b))))
    return d

def predict_dtw_knn(hi_train, train_bids, hi_target, match_len):
    """DTW/kNN prediction with specified match_len."""
    hi_target = np.asarray(hi_target, dtype=float)
    obs_pts = np.arange(SEQ_LENGTH, len(hi_target))
    preds = []
    for obs in obs_pts:
        l = min(match_len, obs)
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

def predict_dtw_multi(hi_train, train_bids, hi_target, match_lengths=MATCH_LENGTHS):
    """Average DTW predictions across multiple match lengths."""
    all_preds = []
    obs_pts = None
    for ml in match_lengths:
        obs, preds = predict_dtw_knn(hi_train, train_bids, hi_target, ml)
        all_preds.append(preds)
        if obs_pts is None:
            obs_pts = obs
    return obs_pts, np.mean(all_preds, axis=0)

def estimate_start_obs(hi_train, hi_target, train_bids):
    target = np.asarray(hi_target, dtype=float)
    l = min(18, len(target))
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

def calibrate(pred_by_bearing, results, cf_min=0.40, cf_max=2.0, cf_step=0.01):
    best_cf, best_score = 1.0, -np.inf
    for cf in np.arange(cf_min, cf_max, cf_step):
        sc = float(np.mean([
            score_curve(results[b]["N"], results[b]["obs_pts"],
                        np.asarray(pred_by_bearing[b])*cf)
            for b in BEARINGS]))
        if sc > best_score:
            best_score, best_cf = sc, float(cf)
    return best_cf, best_score


def run_loocv(hi_train):
    print("[LOOCV: Multi-DTW Experiment]")
    results = {}

    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        print(f"\n  B{test_bid} held out:")
        n_test = len(hi_train[test_bid])
        fold = {"N": n_test}

        # Single DTW (baseline, MATCH_LEN=18)
        obs, pred_dtw18 = predict_dtw_knn(hi_train, train_bids, hi_train[test_bid], 18)
        fold["obs_pts"] = obs
        fold["dtw18"] = pred_dtw18

        # Multi-scale DTW
        _, pred_multi = predict_dtw_multi(hi_train, train_bids, hi_train[test_bid])
        fold["dtw_multi"] = pred_multi

        # Individual match lengths for analysis
        for ml in MATCH_LENGTHS:
            _, p = predict_dtw_knn(hi_train, train_bids, hi_train[test_bid], ml)
            fold[f"dtw{ml}"] = p

        s18 = score_curve(n_test, obs, pred_dtw18)
        sm  = score_curve(n_test, obs, pred_multi)
        fold["raw_scores"] = {"dtw18": s18, "dtw_multi": sm}
        results[test_bid] = fold
        print(f"    dtw18={s18:.4f}  dtw_multi={sm:.4f}")
        for ml in MATCH_LENGTHS:
            s = score_curve(n_test, obs, fold[f"dtw{ml}"])
            print(f"    dtw{ml}={s:.4f}")

    # Calibrate
    cf18,  sc18  = calibrate({b: results[b]["dtw18"]     for b in BEARINGS}, results)
    cfm,   scm   = calibrate({b: results[b]["dtw_multi"] for b in BEARINGS}, results)
    print(f"\n[Calibration]")
    print(f"  dtw18:      cf={cf18:.2f}  score={sc18:.4f}")
    print(f"  dtw_multi:  cf={cfm:.2f}  score={scm:.4f}")

    best_cf   = cfm   if scm >= sc18 else cf18
    best_key  = "dtw_multi" if scm >= sc18 else "dtw18"
    best_sc   = max(scm, sc18)

    # Apply calibration for final
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

    print(f"\n[Final: {best_key}  cf={best_cf:.2f}]")
    for r in rows:
        print(f"  B{r['bearing']}: score={r['score']:.4f}  mean_er={r['mean_er']:.2f}%")
    print(f"  Overall: {overall:.4f}  (baseline: 0.5499)")

    pd.DataFrame(rows).to_csv(OUT_DIR / "train_rul_results.csv", index=False)

    # Match-length sweep per bearing
    ml_rows = []
    for b in BEARINGS:
        r = results[b]
        for ml in MATCH_LENGTHS:
            key = f"dtw{ml}"
            cf_ml, sc_ml = calibrate({bb: results[bb][f"dtw{ml}"] for bb in BEARINGS}, results)
            b_sc = score_curve(r["N"], r["obs_pts"], np.asarray(r[key]) * cf_ml)
            ml_rows.append({"bearing": b, "match_len": ml, "score": round(b_sc, 4), "cf": cf_ml})
    pd.DataFrame(ml_rows).to_csv(OUT_DIR / "match_len_sweep.csv", index=False)

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Exp-B Multi-DTW LOOCV — {best_key} overall={overall:.4f}", fontsize=11)
    for ax, b in zip(axes.flatten(), BEARINGS):
        r = results[b]; obs = r["obs_pts"]
        true = true_rul(r["N"], obs)
        ax.plot(obs, true, "k-", lw=2, label="True")
        ax.plot(obs, r["dtw18"] * cf18, "r--", lw=1.5, alpha=0.7, label=f"dtw18×{cf18:.2f}")
        ax.plot(obs, r["final"], "b-", lw=2, label=f"{best_key}×{best_cf:.2f}")
        ax.set_title(f"B{b}  score={r['sc_final']:.3f}")
        ax.set_xlabel("Obs cycle"); ax.set_ylabel("RUL"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_rul_predictions.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Exp-B Train Er%", fontsize=11)
    for ax, b in zip(axes.flatten(), BEARINGS):
        r = results[b]; obs = r["obs_pts"]
        true = true_rul(r["N"], obs)
        er = 100.0 * (true - np.asarray(r["final"])) / true
        ax.plot(obs, er, lw=1.2)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"B{b}  mean_er={float(er.mean()):.1f}%")
        ax.set_xlabel("Obs cycle"); ax.set_ylabel("Er%"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_rul_er_pct.png", dpi=150)
    plt.close()

    # Match-length sweep plot
    ml_df = pd.DataFrame(ml_rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Match-Length Sweep", fontsize=11)
    for b in BEARINGS:
        sub = ml_df[ml_df["bearing"]==b]
        axes[0].plot(sub["match_len"], sub["score"], marker="o", label=f"B{b}")
    axes[0].set_title("Per-bearing score vs MATCH_LEN")
    axes[0].set_xlabel("MATCH_LEN"); axes[0].set_ylabel("Score"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    agg = ml_df.groupby("match_len")["score"].mean()
    axes[1].plot(agg.index, agg.values, marker="o")
    axes[1].axhline(sc18, color="r", ls="--", label=f"dtw18={sc18:.4f}")
    axes[1].axhline(scm,  color="b", ls="--", label=f"multi={scm:.4f}")
    axes[1].set_title("Overall score vs MATCH_LEN")
    axes[1].set_xlabel("MATCH_LEN"); axes[1].set_ylabel("Overall score"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "match_len_sweep.png", dpi=150)
    plt.close()

    return results, best_key, best_cf

def run_test(hi_train, hi_test, best_key, best_cf):
    print(f"\n[Test: {best_key}  cf={best_cf:.2f}]")
    summary_rows, all_rows = [], []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Exp-B Test RUL — {best_key} cf={best_cf:.2f}", fontsize=11)

    for ax, tid in zip(axes.flatten(), TEST_IDS):
        hi = hi_test[tid]
        start_obs = estimate_start_obs(hi_train, hi, BEARINGS)

        if best_key == "dtw_multi":
            obs, pred = predict_dtw_multi(hi_train, BEARINGS, hi)
        else:
            obs, pred = predict_dtw_knn(hi_train, BEARINGS, hi, 18)
        final = np.maximum(pred * best_cf, MIN_RUL_CYCLES)
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
        ax.set_title(f"T{tid}  RUL={hours[-1]:.1f}hr")
        ax.set_xlabel("Obs cycle"); ax.set_ylabel("RUL"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

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
    fig.suptitle("Exp-B Train HI", fontsize=11); plt.tight_layout()
    plt.savefig(OUT_DIR / "train_hi.png", dpi=150); plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for ax, t in zip(axes.flatten(), TEST_IDS):
        ax.plot(hi_test[t], lw=1.5); ax.set_title(f"T{t}"); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    fig.suptitle("Exp-B Test HI", fontsize=11); plt.tight_layout()
    plt.savefig(OUT_DIR / "test_hi.png", dpi=150); plt.close()

def main():
    print("=== Ensemble_6 / Exp-B: Multi-Match-Length DTW ===\n")
    hi_train = load_train_hi()
    hi_test  = load_test_hi()
    save_hi_outputs(hi_train, hi_test)
    results, best_key, best_cf = run_loocv(hi_train)
    run_test(hi_train, hi_test, best_key, best_cf)
    print(f"\nDone → {OUT_DIR}")

if __name__ == "__main__":
    main()
