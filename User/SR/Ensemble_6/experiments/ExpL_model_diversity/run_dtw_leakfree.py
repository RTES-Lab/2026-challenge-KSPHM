"""
DTW Leak-free Per-Bearing Analysis
- Global (leaked) CF=0.68 → per-bearing calibrated scores
- Leak-free: CF found on training folds only → applied to held-out bearing
- Shows per-bearing score difference
"""
import numpy as np, pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

BASE   = Path("/data/home/ksphm/2026-challenge-KSPHM")
HI_DIR = BASE / "User/SP/05-26/V1b/output"
OUT    = Path(__file__).parent / "results"

BEARINGS = [1, 2, 3, 4]
SEQ = 10; MATCH_LEN = 18; EPS = 1e-8; MIN_RUL = 1.0

def load_hi():
    return {b: pd.read_csv(HI_DIR / f"HI_Bearing{b}.csv")["HI"].values.astype(float)
            for b in BEARINGS}

def mnorm(x):
    x = np.asarray(x, float)
    return (x - x.min()) / (x.max() - x.min() + EPS)

def slp(x):
    x = np.asarray(x, float)
    return float(np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x) >= 2 else 0.0

def true_rul(n, obs):
    return np.maximum(n - np.asarray(obs, float), 1.0)

def sc_curve(n, obs, p):
    y = true_rul(n, obs); p = np.asarray(p, float)
    mask = y > 0; y = y[mask]; p = p[mask]
    if len(y) == 0: return 0.0
    er = 100.0 * (y - p) / y
    return float(np.nanmean(np.where(er <= 0,
        np.exp(-np.log(0.5) * er / 20.0),
        np.exp(np.log(0.5) * er / 50.0))))

def seg_dist(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (0.25 * abs(a[-1] - b[-1]) / 0.25
            + 0.20 * abs(a.mean() - b.mean()) / 0.25
            + 0.20 * abs((a[-1] - a[0]) - (b[-1] - b[0])) / 0.25
            + 0.15 * abs(slp(a) - slp(b)) / 0.03
            + 0.20 * float(np.mean(np.abs(mnorm(a) - mnorm(b)))))

def pred_dtw(hi_tr, bids, hi_t):
    hi_t = np.asarray(hi_t, float)
    obs = np.arange(SEQ, len(hi_t)); ps = []
    for o in obs:
        l = min(MATCH_LEN, o); s = hi_t[o - l:o]; cs = []
        for b in bids:
            hi = np.asarray(hi_tr[b], float)
            for e in range(l, len(hi)):
                cs.append((seg_dist(s, hi[e - l:e]), max(len(hi) - e, MIN_RUL)))
        top = sorted(cs, key=lambda c: c[0])[:6]
        wt = np.asarray([1 / (d + EPS) for d, p in top])
        pv = np.asarray([p for d, p in top])
        ps.append(float(np.average(pv, weights=wt)))
    return obs, np.asarray(ps)

def calibrate(pbb, meta, bids, lo=0.50, hi=1.80, step=0.01):
    bc, bs = 1.0, -np.inf
    for cf in np.arange(lo, hi, step):
        sc = float(np.mean([sc_curve(meta[b]["N"], meta[b]["obs"],
                                     np.asarray(pbb[b]) * cf) for b in bids]))
        if sc > bs: bs, bc = sc, float(cf)
    return bc, bs

def main():
    hi_tr = load_hi()
    raw = {}
    meta = {}

    print("Running DTW LOOCV...")
    for tb in BEARINGS:
        trb = [b for b in BEARINGS if b != tb]
        n = len(hi_tr[tb])
        obs, p = pred_dtw(hi_tr, trb, hi_tr[tb])
        raw[tb] = p
        meta[tb] = {"N": n, "obs": obs}
        print(f"  B{tb} done (n={n})")

    # ── Global (leaked) calibration ──
    cf_global, sc_global = calibrate(raw, meta, BEARINGS)
    print(f"\n[Global (Leaked) CF] cf={cf_global:.2f}  overall={sc_global:.4f}")
    print("  Per-bearing scores with global CF:")
    for b in BEARINGS:
        sc = sc_curve(meta[b]["N"], meta[b]["obs"], raw[b] * cf_global)
        er = float(np.nanmean(100 * (true_rul(meta[b]["N"], meta[b]["obs"]) - raw[b] * cf_global)
                              / true_rul(meta[b]["N"], meta[b]["obs"])))
        print(f"    B{b}: score={sc:.4f}  mean_er={er:+.1f}%")

    # ── Leak-free calibration ──
    local_cfs = {}
    cal_scores = {}
    print(f"\n[Leak-Free] Per-fold CF (trained on 3 bearings, applied to held-out):")
    for tb in BEARINGS:
        trb = [b for b in BEARINGS if b != tb]
        cf_tb, _ = calibrate(raw, meta, trb)
        local_cfs[tb] = cf_tb
        sc_tb = sc_curve(meta[tb]["N"], meta[tb]["obs"], raw[tb] * cf_tb)
        er_tb = float(np.nanmean(100 * (true_rul(meta[tb]["N"], meta[tb]["obs"]) - raw[tb] * cf_tb)
                                 / true_rul(meta[tb]["N"], meta[tb]["obs"])))
        cal_scores[tb] = sc_tb
        print(f"  B{tb}: leak-free cf={cf_tb:.2f}  score={sc_tb:.4f}  mean_er={er_tb:+.1f}%")

    lf_overall = float(np.mean(list(cal_scores.values())))
    print(f"\n  Leak-free Overall: {lf_overall:.4f}")

    # ── Summary table ──
    print("\n[Summary] Global(Leaked) vs Leak-Free per bearing:")
    print(f"  {'Bearing':>8} | {'Leaked CF':>9} | {'Leaked Sc':>9} | {'LF CF':>6} | {'LF Score':>8} | {'Delta':>6}")
    print("  " + "-" * 65)
    for b in BEARINGS:
        sc_g = sc_curve(meta[b]["N"], meta[b]["obs"], raw[b] * cf_global)
        sc_l = cal_scores[b]
        print(f"  B{b}       | {cf_global:>9.2f} | {sc_g:>9.4f} | {local_cfs[b]:>6.2f} | {sc_l:>8.4f} | {sc_l-sc_g:>+6.4f}")
    sc_g_all = float(np.mean([sc_curve(meta[b]["N"], meta[b]["obs"], raw[b] * cf_global)
                               for b in BEARINGS]))
    print(f"  {'Overall':>8} | {cf_global:>9.2f} | {sc_g_all:>9.4f} | {'—':>6} | {lf_overall:>8.4f} | {lf_overall-sc_g_all:>+6.4f}")

    # Save
    rows = []
    for b in BEARINGS:
        sc_g = sc_curve(meta[b]["N"], meta[b]["obs"], raw[b] * cf_global)
        rows.append({
            "bearing": b,
            "global_cf": cf_global,
            "global_score": round(sc_g, 4),
            "lf_cf": round(local_cfs[b], 2),
            "lf_score": round(cal_scores[b], 4),
            "delta": round(cal_scores[b] - sc_g, 4),
        })
    rows.append({
        "bearing": "overall",
        "global_cf": cf_global,
        "global_score": round(sc_g_all, 4),
        "lf_cf": None,
        "lf_score": round(lf_overall, 4),
        "delta": round(lf_overall - sc_g_all, 4),
    })
    pd.DataFrame(rows).to_csv(OUT / "dtw_leakfree_analysis.csv", index=False)
    print(f"\nSaved → {OUT}/dtw_leakfree_analysis.csv")

if __name__ == "__main__":
    main()
