"""
RUL Sliding Window v4 — Hybrid start_frac (Level + Slope weighted) (0604)
=========================================================================
v2 level: T5/T6 말기 잘 잡음, B1/T1 초기 과소평가
v3 slope: B1 초기 증가 잘 잡음, T5/T6 평탄을 초기로 오해

v4 hybrid:
  w = clip(mean(hi_window) / HI_THRESHOLD, 0, 1)
  start_frac = w * level_sf + (1-w) * slope_sf

  HI mean 높으면 → level 신뢰 (T5/T6: mean≈0.34 → w≈1.0)
  HI mean 낮으면 → slope 신뢰 (T1:    mean≈0.04 → w≈0.2)
  중간이면        → 혼합        (T2:    mean≈0.12 → w≈0.6)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

BASE     = Path("/data/home/ksphm/2026-challenge-KSPHM")
HI_TRAIN = BASE / "User/SR/0603_v3/output/train"
HI_TEST  = BASE / "User/SR/0603_v3/output/test"
OUT_DIR  = BASE / "User/SR/0604/output/rul_sliding_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
WIN_SIZE        = 50
SEQ_LEN         = 10
STRIDE          = 1
MEAN_TRAIN_LIFE = 116.5
INTERVAL_SEC    = 600
SEEDS           = [42, 7, 123, 0, 99]
N_FEAT          = 3
HI_THRESHOLD    = 0.20   # 이 이상이면 level, 이하면 slope 주도

NORMAL_UNTIL = {1: 89, 2: 92, 3: 62, 4: 78}
EOL          = {1: 126, 2: 114, 3: 89, 4: 137}


# ══════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════
def load_train_hi():
    data = {}
    for bid in BEARINGS:
        df = pd.read_csv(HI_TRAIN / f"Bearing{bid}_HI.csv")
        data[bid] = {"hi":     df["HI"].values.astype(float),
                     "regime": df["regime"].values.astype(int)}
    return data

def load_test_hi():
    data = {}
    for tid in TEST_IDS:
        df = pd.read_csv(HI_TEST / f"Test{tid}_HI.csv")
        data[tid] = {"hi":     df["HI"].values.astype(float),
                     "regime": df["regime"].values.astype(int)}
    return data

def rul_labels(n_total, bid):
    nu, eol = NORMAL_UNTIL[bid], EOL[bid]
    idx = np.arange(n_total)
    return np.where(idx <= nu, eol - nu,
                    np.maximum(eol - idx, 0)).astype(float)

def compute_hi_stats(train_data, bids):
    all_hi = np.concatenate([train_data[b]["hi"] for b in bids])
    return float(all_hi.mean()), float(all_hi.std() + 1e-8)


# ══════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════
def comp_score(rul_true, rul_pred):
    if rul_true <= 0: return np.nan
    Er = 100.0 * (rul_true - rul_pred) / rul_true
    return (np.exp(-np.log(0.5) * Er / 20.0) if Er <= 0
            else np.exp(np.log(0.5) * Er / 50.0))

def avg_score(true_ruls, preds):
    valid = [comp_score(t, p) for t, p in zip(true_ruls, preds)
             if not np.isnan(comp_score(t, p))]
    return float(np.mean(valid)) if valid else np.nan


# ══════════════════════════════════════════════════════════════════
# Start fraction estimation (level / slope / hybrid)
# ══════════════════════════════════════════════════════════════════
def estimate_sf_level(hi_window, train_data, ref_bids):
    """HI 절댓값 기반 (v2): train 베어링에서 처음 hi_start에 도달하는 시점."""
    hi_start = float(hi_window[0])
    fracs = []
    for bid in ref_bids:
        hi = train_data[bid]["hi"]
        if hi[0] >= hi_start:
            continue
        exceed = np.where(hi >= hi_start)[0]
        fracs.append(float(len(hi)) / MEAN_TRAIN_LIFE if len(exceed) == 0
                     else float(exceed[0]) / MEAN_TRAIN_LIFE)
    return float(np.mean(fracs)) if fracs else 0.0


def estimate_sf_slope(hi_window, train_data, ref_bids, win_size=WIN_SIZE):
    """HI slope 기반 (v3): inverse-distance weighted average."""
    t_win = np.arange(win_size, dtype=float)
    obs_slope = float(np.polyfit(t_win, hi_window, 1)[0])
    fracs, weights = [], []
    for bid in ref_bids:
        hi = train_data[bid]["hi"]
        N  = len(hi)
        if N < win_size: continue
        for t_start in range(0, N - win_size + 1):
            slope = float(np.polyfit(t_win, hi[t_start:t_start+win_size], 1)[0])
            fracs.append(t_start / MEAN_TRAIN_LIFE)
            weights.append(1.0 / (abs(slope - obs_slope) + 1e-5))
    if not fracs: return 0.0
    w = np.array(weights); w /= w.sum()
    return float(np.dot(w, fracs))


def estimate_sf_hybrid(hi_window, train_data, ref_bids,
                        hi_threshold=HI_THRESHOLD):
    """
    HI 평균에 따른 level/slope 가중 혼합:
      w = clip(mean(hi_window) / hi_threshold, 0, 1)
      sf = w * level_sf + (1-w) * slope_sf
    """
    hi_mean = float(np.mean(hi_window))
    w       = float(np.clip(hi_mean / hi_threshold, 0.0, 1.0))
    sf_lv   = estimate_sf_level(hi_window, train_data, ref_bids)
    sf_sl   = estimate_sf_slope(hi_window, train_data, ref_bids)
    return w * sf_lv + (1.0 - w) * sf_sl, w, sf_lv, sf_sl


# ══════════════════════════════════════════════════════════════════
# LSTM
# ══════════════════════════════════════════════════════════════════
class LSTMRegressor(nn.Module):
    def __init__(self, n_feat=N_FEAT, hidden=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=n_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                   nn.Linear(32, 1))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)


def build_train_seqs(train_data, bids, hi_mean, hi_std):
    X_list, y_list = [], []
    for b in bids:
        hi_b  = train_data[b]["hi"]
        reg_b = train_data[b]["regime"]
        rul_b = rul_labels(len(hi_b), b)
        for i in range(len(hi_b) - SEQ_LEN):
            win_norm = (hi_b[i:i+SEQ_LEN] - hi_mean) / hi_std
            obs_frac = np.clip((i + np.arange(SEQ_LEN)) / MEAN_TRAIN_LIFE,
                                0.0, 2.0)
            rg = reg_b[i:i+SEQ_LEN].astype(float)
            X_list.append(np.stack([win_norm, obs_frac, rg], axis=1))
            y_list.append(float(rul_b[i + SEQ_LEN]))
    return np.array(X_list), np.array(y_list)


def train_lstm_ensemble(X_train, y_train, rul_scale, device):
    models = []
    for s in SEEDS:
        torch.manual_seed(s)
        y_norm = y_train / rul_scale
        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_norm,  dtype=torch.float32)
        n_val  = max(1, int(len(Xt) * 0.1))
        tr_dl  = DataLoader(TensorDataset(Xt[:-n_val], yt[:-n_val]),
                             batch_size=64, shuffle=True)
        val_dl = DataLoader(TensorDataset(Xt[-n_val:], yt[-n_val:]),
                             batch_size=64)
        model  = LSTMRegressor().to(device)
        opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit   = nn.MSELoss()
        best_val, patience, best_state = np.inf, 0, None
        for _ in range(200):
            model.train()
            for xb, yb in tr_dl:
                opt.zero_grad()
                crit(model(xb.to(device)), yb.to(device)).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([
                    crit(model(xb.to(device)), yb.to(device)).item()
                    for xb, yb in val_dl]))
            if vl < best_val:
                best_val, patience = vl, 0
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 20: break
        model.load_state_dict(best_state)
        models.append(model)
    return models


def predict_lstm(models, hi_arr, reg_arr, start_frac,
                  hi_mean, hi_std, rul_scale, device):
    start_obs = int(start_frac * MEAN_TRAIN_LIFE)
    X = []
    for j in range(len(hi_arr) - SEQ_LEN):
        win_norm = (hi_arr[j:j+SEQ_LEN] - hi_mean) / hi_std
        obs_frac = np.clip(
            (start_obs + j + np.arange(SEQ_LEN)) / MEAN_TRAIN_LIFE, 0.0, 2.0)
        rg = reg_arr[j:j+SEQ_LEN].astype(float)
        X.append(np.stack([win_norm, obs_frac, rg], axis=1))
    if not X: return np.array([0.0])
    Xt = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    all_p = []
    for m in models:
        m.eval()
        with torch.no_grad():
            all_p.append(np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0))
    return np.median(all_p, axis=0)


# ══════════════════════════════════════════════════════════════════
# Plot helpers
# ══════════════════════════════════════════════════════════════════
def monotonicity(s):
    d = np.diff(s)
    return abs(np.sum(d > 0) - np.sum(d < 0)) / max(len(d), 1)

def trendability(s):
    rho, _ = spearmanr(np.arange(len(s)), s)
    return abs(rho) if not np.isnan(rho) else 0.0

def _plot_hi(ax, hi, regime, title, q):
    t = np.arange(len(hi)) * INTERVAL_SEC / 3600
    for lbl, col, name in [(0, "#4C72B0", "Low"), (1, "#DD8452", "High")]:
        idx = regime == lbl
        ax.scatter(t[idx], hi[idx], s=10, color=col, alpha=0.5,
                   label=name, zorder=3)
    ax.plot(t, hi, color="gray", lw=0.7, alpha=0.4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{title}  Q={q:.3f}", fontsize=10)
    ax.set_xlabel("Time [hr]"); ax.set_ylabel("HI")
    ax.legend(fontsize=7); ax.grid(True, ls="--", alpha=0.3)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def run():
    print("=" * 70)
    print("  RUL Sliding Window v4 — Hybrid start_frac (Level + Slope)")
    print(f"  WIN={WIN_SIZE}  HI_THRESHOLD={HI_THRESHOLD}")
    print("=" * 70)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    train_data = load_train_hi()
    test_data  = load_test_hi()

    # ── [1] Train HI ──────────────────────────────────────────────
    print("\n[1/4] Train HI")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Train HI (FDR LOO, 0603_v3)", fontsize=13, fontweight="bold")
    axes = axes.flatten()
    for i, bid in enumerate(BEARINGS):
        hi  = train_data[bid]["hi"]
        reg = train_data[bid]["regime"]
        q   = (monotonicity(hi) + trendability(hi)) / 2
        _plot_hi(axes[i], hi, reg, f"Bearing{bid}", q)
        fig_i, ax_i = plt.subplots(figsize=(9, 4))
        _plot_hi(ax_i, hi, reg, f"Bearing{bid}", q)
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Bearing{bid}_HI.png", dpi=150)
        plt.close(fig_i)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Train_HI_all.png", dpi=150)
    plt.close()
    print(f"  → {OUT_DIR}/Train_HI_all.png")

    # ── [2] Sliding LOOCV ─────────────────────────────────────────
    print("\n[2/4] Sliding LOOCV (hybrid start_frac)")
    # reference scores
    ref = {"level": [0.3870, 0.5767, 0.4681, 0.3727],
           "slope": [0.4451, 0.5394, 0.4682, 0.3744]}
    fold_results = {}

    for i, test_bid in enumerate(BEARINGS):
        train_bids = [b for b in BEARINGS if b != test_bid]
        hi_t  = train_data[test_bid]["hi"]
        reg_t = train_data[test_bid]["regime"]
        N     = len(hi_t)
        rul_t = rul_labels(N, test_bid)

        hi_mean, hi_std = compute_hi_stats(train_data, train_bids)
        rul_scale       = float(max(EOL[b] for b in train_bids))

        print(f"\n  Fold B{test_bid}  (train={train_bids}  N={N})")
        print("    LSTM 학습...")
        X_tr, y_tr = build_train_seqs(train_data, train_bids, hi_mean, hi_std)
        models     = train_lstm_ensemble(X_tr, y_tr, rul_scale, device)

        t_starts, true_ruls, preds = [], [], []
        sf_list, w_list = [], []

        for t_start in range(0, N - WIN_SIZE + 1, STRIDE):
            hi_win  = hi_t[t_start: t_start + WIN_SIZE]
            reg_win = reg_t[t_start: t_start + WIN_SIZE]
            true_rul = float(rul_t[t_start + WIN_SIZE - 1])

            sf, w, sf_lv, sf_sl = estimate_sf_hybrid(
                hi_win, train_data, train_bids)

            p = predict_lstm(models, hi_win, reg_win,
                              sf, hi_mean, hi_std, rul_scale, device)
            t_starts.append(t_start); true_ruls.append(true_rul)
            preds.append(float(p[-1]))
            sf_list.append(sf); w_list.append(w)

        sc = avg_score(true_ruls, preds)
        print(f"    Score: hybrid={sc:.4f}  "
              f"level={ref['level'][i]:.4f}  slope={ref['slope'][i]:.4f}")
        fold_results[test_bid] = {
            "N": N, "t_starts": t_starts, "true_ruls": true_ruls,
            "preds": preds, "sf_list": sf_list, "w_list": w_list, "sc": sc,
        }

    # ── [3] Train RUL plots ───────────────────────────────────────
    print("\n[3/4] Train RUL 그림")
    sc_all  = [fold_results[b]["sc"] for b in BEARINGS]
    mean_sc = float(np.mean(sc_all))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Train Sliding RUL — Hybrid start_frac  avg={mean_sc:.3f}"
        f"  (level=0.451 / slope=0.457)",
        fontsize=11, fontweight="bold")
    axes = axes.flatten()

    for i, bid in enumerate(BEARINGS):
        r  = fold_results[bid]
        ts = r["t_starts"]
        ax = axes[i]
        ax.plot(ts, r["true_ruls"], "k-",  lw=1.5, label="True RUL")
        ax.plot(ts, r["preds"],     "m--", lw=1.5, alpha=0.8,
                label=f"Hybrid {r['sc']:.3f}")
        ax.set_title(
            f"Bearing{bid}  hybrid={r['sc']:.3f}  "
            f"lv={ref['level'][i]:.3f}  sl={ref['slope'][i]:.3f}",
            fontsize=9)
        ax.set_xlabel("Window Start (cycle)"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # individual — weight 시각화 포함
        fig_i, axes_i = plt.subplots(2, 1, figsize=(10, 7))
        axes_i[0].plot(ts, r["true_ruls"], "k-",  lw=1.5, label="True RUL")
        axes_i[0].plot(ts, r["preds"],     "m--", lw=1.5, alpha=0.9,
                       label=f"Hybrid  {r['sc']:.3f}")
        axes_i[0].set_title(
            f"Bearing{bid} — Hybrid start_frac  "
            f"hybrid={r['sc']:.3f} / level={ref['level'][i]:.3f} / "
            f"slope={ref['slope'][i]:.3f}", fontsize=9)
        axes_i[0].set_xlabel("Window Start (cycle)")
        axes_i[0].set_ylabel("RUL (cycles)")
        axes_i[0].legend(fontsize=8); axes_i[0].grid(True, alpha=0.3)

        # w(level weight) 추이
        axes_i[1].plot(ts, r["w_list"], "b-", lw=1.5)
        axes_i[1].axhline(0.5, color="gray", ls="--", lw=1)
        axes_i[1].set_ylim(-0.05, 1.05)
        axes_i[1].set_xlabel("Window Start (cycle)")
        axes_i[1].set_ylabel("w (level weight)")
        axes_i[1].set_title(
            "Level weight w  (w=1: level only, w=0: slope only)")
        axes_i[1].grid(True, alpha=0.3)
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Bearing{bid}_sliding.png", dpi=150)
        plt.close(fig_i)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "Train_RUL_sliding_all.png", dpi=150)
    plt.close()

    print(f"\n  Summary (hybrid vs level vs slope):")
    for i, bid in enumerate(BEARINGS):
        sc = fold_results[bid]["sc"]
        best = max(sc, ref['level'][i], ref['slope'][i])
        tag = ("Hybrid" if sc == best else
               "Level"  if ref['level'][i] == best else "Slope")
        print(f"    B{bid}: hybrid={sc:.4f}  level={ref['level'][i]:.4f}  "
              f"slope={ref['slope'][i]:.4f}  best={tag}")
    print(f"    Mean: hybrid={mean_sc:.4f}  level=0.4511  slope=0.4568")

    # ── [4] Test HI + RUL ─────────────────────────────────────────
    print("\n[4/4] Test HI + RUL")

    # Test HI
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Test HI (FDR, 0603_v3)", fontsize=13, fontweight="bold")
    axes = axes.flatten()
    for i, tid in enumerate(TEST_IDS):
        hi  = test_data[tid]["hi"]
        reg = test_data[tid]["regime"]
        q   = (monotonicity(hi) + trendability(hi)) / 2
        _plot_hi(axes[i], hi, reg, f"Test{tid}", q)
        fig_i, ax_i = plt.subplots(figsize=(9, 4))
        _plot_hi(ax_i, hi, reg, f"Test{tid}", q)
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Test{tid}_HI.png", dpi=150)
        plt.close(fig_i)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Test_HI_all.png", dpi=150)
    plt.close()

    # Full-train LSTM
    print("  Full train LSTM 학습...")
    hi_mean_all, hi_std_all = compute_hi_stats(train_data, BEARINGS)
    rul_scale_all = float(max(EOL.values()))
    X_all, y_all  = build_train_seqs(train_data, BEARINGS, hi_mean_all, hi_std_all)
    lstm_all      = train_lstm_ensemble(X_all, y_all, rul_scale_all, device)

    # v2/v3 비교용 reference
    v2_hr = {1:3.78, 2:2.85, 3:2.87, 4:3.85, 5:2.83, 6:2.80}
    v3_hr = {1:3.66, 2:3.77, 3:3.79, 4:3.18, 5:4.23, 6:4.52}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Test RUL — Hybrid start_frac (0604 v4)",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()
    summary_rows = []

    for i, tid in enumerate(TEST_IDS):
        hi_t  = test_data[tid]["hi"]
        reg_t = test_data[tid]["regime"]
        n     = len(hi_t)

        sf, w, sf_lv, sf_sl = estimate_sf_hybrid(hi_t, train_data, BEARINGS)
        preds    = predict_lstm(lstm_all, hi_t, reg_t,
                                 sf, hi_mean_all, hi_std_all, rul_scale_all, device)
        obs_pts  = np.arange(SEQ_LEN, n)
        final_cyc = float(np.median(preds[-10:]))
        final_hr  = final_cyc * INTERVAL_SEC / 3600

        print(f"  Test{tid}: hi_mean={np.mean(hi_t):.3f}  w={w:.2f}"
              f"  sf={sf:.3f}({int(sf*100)}%)"
              f"  [lv={sf_lv:.3f} sl={sf_sl:.3f}]"
              f"  RUL={final_hr:.2f}hr"
              f"  (v2={v2_hr[tid]:.2f}hr  v3={v3_hr[tid]:.2f}hr)")

        ax = axes[i]
        ax.plot(obs_pts, preds, "m-", lw=1.5, label="Hybrid")
        ax.axhline(final_cyc, color="m", ls="--", lw=1.5,
                   label=f"Final: {final_hr:.1f}hr")
        ax.axhline(v2_hr[tid] * 3600 / INTERVAL_SEC, color="b",
                   ls=":", lw=1, alpha=0.5, label=f"v2(lv): {v2_hr[tid]:.1f}hr")
        ax.axhline(v3_hr[tid] * 3600 / INTERVAL_SEC, color="g",
                   ls=":", lw=1, alpha=0.5, label=f"v3(sl): {v3_hr[tid]:.1f}hr")
        ax.set_title(
            f"Test{tid}  w={w:.2f}  sf={int(sf*100)}%  "
            f"RUL={final_hr:.1f}hr", fontsize=9)
        ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

        fig_i, ax_i = plt.subplots(figsize=(9, 4))
        ax_i.plot(obs_pts, preds, "m-", lw=1.5, label="Hybrid")
        ax_i.axhline(final_cyc, color="m", ls="--", lw=1.5,
                     label=f"Final: {final_hr:.2f}hr")
        ax_i.set_title(
            f"Test{tid}  w={w:.2f}(level wt)  sf={int(sf*100)}%  "
            f"RUL={final_hr:.2f}hr", fontsize=9)
        ax_i.set_xlabel("Obs Cycle"); ax_i.set_ylabel("RUL (cycles)")
        ax_i.legend(fontsize=7); ax_i.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Test{tid}_RUL.png", dpi=150)
        plt.close(fig_i)

        pd.DataFrame({"obs_cycle": obs_pts, "rul_pred_cycles": preds,
                      "rul_hours": preds * INTERVAL_SEC / 3600}
                     ).to_csv(OUT_DIR / f"Test{tid}_RUL.csv", index=False)
        summary_rows.append({
            "test_id": tid,
            "hi_mean": round(float(np.mean(hi_t)), 4),
            "w_level": round(w, 3),
            "start_frac": round(sf, 3),
            "sf_level": round(sf_lv, 3),
            "sf_slope": round(sf_sl, 3),
            "final_rul_hours": round(final_hr, 2),
            "v2_level_hours":  v2_hr[tid],
            "v3_slope_hours":  v3_hr[tid],
        })

    plt.tight_layout()
    plt.savefig(OUT_DIR / "Test_RUL_all.png", dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary.csv", index=False)

    print(f"\n{'='*70}")
    print(f"  Sliding LOOCV avg:  hybrid={mean_sc:.4f}  "
          f"level=0.4511  slope=0.4568")
    print(f"\n  Test RUL summary:")
    print(df_sum[["test_id","hi_mean","w_level","start_frac",
                   "final_rul_hours","v2_level_hours","v3_slope_hours"]
                ].to_string(index=False))


if __name__ == "__main__":
    run()
