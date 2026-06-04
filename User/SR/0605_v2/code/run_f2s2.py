"""
F2S2 (bug-fixed) — Train LOO + Test RUL
=========================================
Outputs:
  output/train_hi.png          — train bearing state HI (LOO params)
  output/train_rul_loo.png     — train LOO RUL curves + scores
  output/test_hi.png           — test bearing state HI
  output/test_rul.png          — test bearing RUL curves
  output/train_rul_results.csv
  output/test_rul_results.csv
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f2s2_fixed as F

OUT      = HERE.parent / "output"
OUT.mkdir(exist_ok=True)
FEATURES = F.ROOT / "User/TH/common_source"


# ── data loaders ──────────────────────────────────────────────────────────────

def load_train() -> tuple[dict, dict]:
    y, prof = {}, {}
    for b in F.BEARINGS:
        df    = pd.read_csv(FEATURES / f"Bearing{b}_features.csv")
        y[b]  = df["ch3_rms"].values.astype(float)
        prof[b] = F.regime_profile_train(b, len(y[b]))
        low_cnt = int(np.sum(prof[b] == 0))
        hi_cnt  = int(np.sum(prof[b] == 1))
        print(f"  B{b}: K={len(y[b])}  low={low_cnt}  high={hi_cnt}")
    return y, prof


def load_test() -> tuple[dict, dict]:
    y, prof = {}, {}
    for t in F.TEST_IDS:
        print(f"  Test{t}: extracting TDMS...", end=" ", flush=True)
        rms, reg = F.regime_profile_test(t)
        y[t]    = rms
        prof[t] = reg
        low_cnt = int(np.sum(reg == 0))
        hi_cnt  = int(np.sum(reg == 1))
        print(f"K={len(rms)}  low={low_cnt}  high={hi_cnt}")
    return y, prof


# ── helpers ───────────────────────────────────────────────────────────────────

def fit_params(train_bids: list[int], y: dict, prof: dict) -> dict:
    profiles = [prof[b] for b in train_bids]
    series   = [y[b]    for b in train_bids]
    st       = F.fit_state_transition(profiles)
    trans    = F.fit_signal_transform(series, profiles)
    yb_list  = [F.transform_to_baseline(y[b], prof[b], trans) for b in train_bids]
    meas     = F.fit_measurement(yb_list, st["eta"], st["r"], profiles)
    return {**st, **{k: meas[k] for k in ("a_B", "b_B", "c", "sigma2")}, "trans": trans}


def find_fpt(yb: np.ndarray, frac0: float = 0.10) -> int:
    sm  = F.loess_smooth(yb, 0.3)
    n0  = max(5, int(len(sm) * frac0))
    mu  = sm[:n0].mean()
    sd  = sm[:n0].std() + 1e-9
    above = np.where(sm > mu + 3 * sd)[0]
    return int(above[0]) if len(above) else n0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ────────────────────── load data ─────────────────────────────────────────
    print("Loading train features...")
    ty, tprof = load_train()
    print("Extracting test features from TDMS...")
    tsy, tsprof = load_test()

    # ────────────────────── TRAIN LOO ─────────────────────────────────────────
    print("\n=== Train LOO ===")
    tr_states:     dict[int, np.ndarray] = {}
    tr_rul_curves: dict[int, tuple]      = {}
    tr_rul_rows:   list[dict]            = []
    tr_params_loo: dict[int, dict]       = {}

    for tb in F.BEARINGS:
        train_bids = [b for b in F.BEARINGS if b != tb]
        params     = fit_params(train_bids, ty, tprof)
        tr_params_loo[tb] = params

        x = F.particle_filter(ty[tb], tprof[tb], params, seed=tb)
        tr_states[tb] = x
        K    = len(ty[tb])
        preds = np.array([F.predict_rul(x[k], tprof[tb], k, params)
                          for k in range(K)])
        true  = np.maximum(K - np.arange(K), 1.0)
        sc_all = [F.competition_score(t, p) for t, p in zip(true, preds)]

        yb_tb  = F.transform_to_baseline(ty[tb], tprof[tb], params["trans"])
        fpt    = find_fpt(yb_tb)
        sc_fpt  = float(np.nanmean(sc_all[fpt:]))
        sc_full = float(np.nanmean(sc_all))
        er_fpt  = float(np.nanmean(
            [100 * (t - p) / t for t, p in zip(true[fpt:], preds[fpt:])]))

        tr_rul_rows.append({
            "bearing": tb, "score_fpt": round(sc_fpt, 4),
            "score_full": round(sc_full, 4), "mean_er%": round(er_fpt, 2),
            "fpt": fpt, "r_high": round(float(params["r"][1]), 3),
            "eta": round(float(params["eta"]), 5), "c": round(float(params["c"]), 3)})
        tr_rul_curves[tb] = (preds, true, fpt)

        print(f"  B{tb}: r_high={params['r'][1]:.3f}  eta={params['eta']:.5f}  "
              f"c={params['c']:.2f}  |  score(fpt)={sc_fpt:.4f}  score(full)={sc_full:.4f}")

    pd.DataFrame(tr_rul_rows).to_csv(OUT / "train_rul_results.csv", index=False)
    avg_fpt  = float(np.mean([r["score_fpt"]  for r in tr_rul_rows]))
    avg_full = float(np.mean([r["score_full"] for r in tr_rul_rows]))
    print(f"  → LOO avg score(fpt)={avg_fpt:.4f}  avg(full)={avg_full:.4f}")

    # ────────────────────── TEST ───────────────────────────────────────────────
    print("\n=== Test bearings ===")
    # Fit on all 4 train bearings
    full_params = fit_params(F.BEARINGS, ty, tprof)
    print(f"  Full-train params: r_high={full_params['r'][1]:.3f}  "
          f"eta={full_params['eta']:.5f}  c={full_params['c']:.2f}")

    ts_states:     dict[int, np.ndarray] = {}
    ts_rul_curves: dict[int, np.ndarray] = {}
    ts_rul_rows:   list[dict]            = []

    for t in F.TEST_IDS:
        x = F.particle_filter(tsy[t], tsprof[t], full_params, seed=t + 100)
        ts_states[t] = x
        K    = len(tsy[t])
        preds = np.array([F.predict_rul(x[k], tsprof[t], k, full_params)
                          for k in range(K)])
        ts_rul_curves[t] = preds
        final_rul = float(preds[-1])
        ts_rul_rows.append({
            "test_id": t, "n_cycles": K,
            "final_state": round(float(x[-1]), 4),
            "final_rul_cycles": round(final_rul, 1)})
        print(f"  Test{t}: K={K}  state[-1]={x[-1]:.3f}  RUL={final_rul:.1f} cycles")

    pd.DataFrame(ts_rul_rows).to_csv(OUT / "test_rul_results.csv", index=False)

    # ═══════════════════════════════ PLOTS ═══════════════════════════════════

    # ── Plot 1: Train HI ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(24, 8))
    for j, b in enumerate(F.BEARINGS):
        params = tr_params_loo[b]
        yb  = F.transform_to_baseline(ty[b], tprof[b], params["trans"])
        cyc = np.arange(len(ty[b]))

        ax = axes[0, j]
        for c, col, lab in [(0, "tab:blue", "low"), (1, "tab:red", "high")]:
            m = tprof[b] == c
            ax.scatter(cyc[m], ty[b][m], s=9, c=col, label=f"raw {lab}", alpha=0.7)
        ax.plot(cyc, yb, "k-", lw=1.3, alpha=0.85, label="baseline-transformed")
        ax.set_title(f"Bearing{b}  ch3_rms  (raw signal)", fontsize=10)
        ax.set_xlabel("cycle"); ax.set_ylabel("ch3_rms")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax2 = axes[1, j]
        ax2.fill_between(cyc, 0, 1, where=tprof[b] == 1,
                         color="tab:red", alpha=0.07, label="high regime")
        ax2.plot(cyc, tr_states[b], "g-", lw=2, label="F2S2 state x")
        ax2.axhline(1.0, ls="--", c="gray", lw=1, label="D=1 (failure)")
        mon  = F.monotonicity(tr_states[b])
        tred = F.trendability(tr_states[b])
        ax2.set_title(f"Bearing{b}  HI  Mon={mon:.2f}  Tred={tred:.2f}", fontsize=10)
        ax2.set_xlabel("cycle"); ax2.set_ylabel("state x")
        ax2.set_ylim(-0.05, 1.1)
        ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    fig.suptitle(
        "F2S2 (fixed) — Train Bearing HI  [raw y_k → baseline transform → PF state x]",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "train_hi.png", dpi=140); plt.close()
    print("\n  saved train_hi.png")

    # ── Plot 2: Train LOO RUL ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, b in zip(axes.flatten(), F.BEARINGS):
        preds, true, fpt = tr_rul_curves[b]
        cyc = np.arange(len(preds))
        ax.fill_between(cyc, 0, true.max() * 1.1,
                        where=tprof[b] == 1, color="tab:red", alpha=0.05, label="high")
        ax.plot(cyc, true,  "k-",  lw=2,   label="True RUL")
        ax.plot(cyc, preds, "b-",  lw=1.5, label="F2S2 pred")
        ax.axvline(fpt, ls=":", c="r", lw=1.5, label=f"FPT={fpt}")
        r = next(rr for rr in tr_rul_rows if rr["bearing"] == b)
        ax.set_title(
            f"Bearing{b}  score(fpt)={r['score_fpt']:.3f}  "
            f"score(full)={r['score_full']:.3f}  er={r['mean_er%']:+.1f}%",
            fontsize=10)
        ax.set_xlabel("cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        f"F2S2 (fixed) — Train LOO RUL  "
        f"avg score(fpt)={avg_fpt:.4f}  avg score(full)={avg_full:.4f}",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "train_rul_loo.png", dpi=140); plt.close()
    print("  saved train_rul_loo.png")

    # ── Plot 3: Test HI ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 6, figsize=(30, 8))
    for j, t in enumerate(F.TEST_IDS):
        cyc = np.arange(len(tsy[t]))
        yb  = F.transform_to_baseline(tsy[t], tsprof[t], full_params["trans"])

        ax = axes[0, j]
        for c, col, lab in [(0, "tab:blue", "low"), (1, "tab:red", "high")]:
            m = tsprof[t] == c
            ax.scatter(cyc[m], tsy[t][m], s=9, c=col, label=f"raw {lab}", alpha=0.7)
        ax.plot(cyc, yb, "k-", lw=1.3, alpha=0.85, label="baseline")
        ax.set_title(f"Test{t}  ch3_rms", fontsize=9)
        ax.set_xlabel("cycle"); ax.set_ylabel("ch3_rms")
        ax.legend(fontsize=6); ax.grid(alpha=0.3)

        ax2 = axes[1, j]
        ax2.fill_between(cyc, 0, 1, where=tsprof[t] == 1,
                         color="tab:red", alpha=0.07, label="high")
        ax2.plot(cyc, ts_states[t], "g-", lw=2, label="F2S2 state x")
        ax2.axhline(1.0, ls="--", c="gray", lw=1, label="D=1")
        r   = next(rr for rr in ts_rul_rows if rr["test_id"] == t)
        mon  = F.monotonicity(ts_states[t])
        tred = F.trendability(ts_states[t])
        ax2.set_title(
            f"Test{t}  Mon={mon:.2f}  Tred={tred:.2f}\n"
            f"state[-1]={r['final_state']:.3f}",
            fontsize=9)
        ax2.set_xlabel("cycle"); ax2.set_ylabel("state x")
        ax2.set_ylim(-0.05, 1.1)
        ax2.legend(fontsize=6); ax2.grid(alpha=0.3)

    fig.suptitle(
        "F2S2 (fixed) — Test Bearing HI  [params: all 4 train bearings]",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "test_hi.png", dpi=140); plt.close()
    print("  saved test_hi.png")

    # ── Plot 4: Test RUL ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, t in zip(axes.flatten(), F.TEST_IDS):
        preds = ts_rul_curves[t]
        cyc   = np.arange(len(preds))
        ax.fill_between(cyc, 0, preds.max() * 1.1,
                        where=tsprof[t] == 1, color="tab:red", alpha=0.05, label="high")
        ax.plot(cyc, preds, "b-", lw=2, label="F2S2 pred RUL")
        ax.axhline(preds[-1], ls=":", c="orange", lw=1.5,
                   label=f"final RUL = {preds[-1]:.1f} cyc")
        r = next(rr for rr in ts_rul_rows if rr["test_id"] == t)
        ax.set_title(
            f"Test{t}  K={r['n_cycles']}  state[-1]={r['final_state']:.3f}\n"
            f"Predicted RUL = {r['final_rul_cycles']:.1f} cycles",
            fontsize=10)
        ax.set_xlabel("cycle"); ax.set_ylabel("predicted RUL (cycles)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("F2S2 (fixed) — Test Bearing RUL prediction", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "test_rul.png", dpi=140); plt.close()
    print("  saved test_rul.png")

    print(f"\nAll outputs → {OUT}")
    return tr_rul_rows, ts_rul_rows


if __name__ == "__main__":
    main()
