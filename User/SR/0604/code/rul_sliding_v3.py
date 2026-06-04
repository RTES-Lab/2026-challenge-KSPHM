"""
RUL Sliding Window v3 — Slope-based start_frac (0604)
======================================================
v2 대비 변경: estimate_start_frac (HI level) → estimate_start_frac_slope

문제:
  HI 절댓값은 베어링 간 비교 불가 (B3 EOL HI=0.14 = B1 30% HI)
  → level 기반 위치 추정 오류 → LSTM obs_frac 피처 오염

수정:
  50사이클 창의 선형 slope를 train 베어링의 모든 창 slope와 비교
  → inverse-distance weighted average로 lifecycle 위치 추정
  → HI 절댓값 스케일 불일치 문제 우회

나머지는 v2와 동일:
  - HI: 0603_v3 FDR LOO HI 그대로 사용
  - LSTM only (LGBM 제거)
  - sliding window LOOCV (WIN=50, STRIDE=1)
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
OUT_DIR  = BASE / "User/SR/0604/output/rul_sliding_v3"
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
# Slope-based lifecycle position estimation
# ══════════════════════════════════════════════════════════════════
def estimate_start_frac_slope(hi_window, train_data, ref_bids,
                               win_size=WIN_SIZE):
    """
    HI 창의 선형 기울기를 train 베어링의 모든 동일 크기 창 기울기와 비교.
    Inverse-distance weighted average로 lifecycle 시작 위치 추정.

    level 기반 대비 장점:
      - HI 절댓값 스케일 불일치 우회
      - 기울기는 "얼마나 빠르게 열화되는가"를 반영 → 위치 추정에 더 적합

    ref_bids = train_bids (held-out 베어링 제외) → leakage 없음
    """
    t_win = np.arange(win_size, dtype=float)
    obs_slope = float(np.polyfit(t_win, hi_window, 1)[0])

    fracs, weights = [], []
    for bid in ref_bids:
        hi = train_data[bid]["hi"]
        N  = len(hi)
        if N < win_size:
            continue
        for t_start in range(0, N - win_size + 1):
            win = hi[t_start: t_start + win_size]
            slope = float(np.polyfit(t_win, win, 1)[0])
            dist  = abs(slope - obs_slope)
            fracs.append(t_start / MEAN_TRAIN_LIFE)
            weights.append(1.0 / (dist + 1e-5))

    if not fracs:
        return 0.0
    w = np.array(weights)
    w /= w.sum()
    return float(np.dot(w, fracs))


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
        n = len(hi_b)
        for i in range(n - SEQ_LEN):
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
    n = len(hi_arr)
    X = []
    for j in range(n - SEQ_LEN):
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
    print("  RUL Sliding Window v3 — Slope-based start_frac")
    print(f"  WIN={WIN_SIZE}  STRIDE={STRIDE}  SEQ_LEN={SEQ_LEN}")
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
    print("\n[2/4] Sliding LOOCV (slope-based start_frac)")
    fold_results = {}

    for test_bid in BEARINGS:
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

        t_starts, true_ruls, preds, sf_list = [], [], [], []

        for t_start in range(0, N - WIN_SIZE + 1, STRIDE):
            hi_win  = hi_t[t_start: t_start + WIN_SIZE]
            reg_win = reg_t[t_start: t_start + WIN_SIZE]
            true_rul = float(rul_t[t_start + WIN_SIZE - 1])

            # Slope-based start_frac — train_bids만 사용 (leakage 없음)
            sf = estimate_start_frac_slope(hi_win, train_data, train_bids)

            p  = predict_lstm(models, hi_win, reg_win,
                               sf, hi_mean, hi_std, rul_scale, device)
            t_starts.append(t_start)
            true_ruls.append(true_rul)
            preds.append(float(p[-1]))
            sf_list.append(sf)

        sc = avg_score(true_ruls, preds)
        print(f"    Score: {sc:.4f}  (v2 level-based: "
              f"{[0.3870,0.5767,0.4681,0.3727][BEARINGS.index(test_bid)]:.4f})")
        fold_results[test_bid] = {
            "N": N, "t_starts": t_starts, "true_ruls": true_ruls,
            "preds": preds, "sf_list": sf_list, "sc": sc,
        }

    # ── [3] Train RUL plots ───────────────────────────────────────
    print("\n[3/4] Train RUL 그림")
    sc_all  = [fold_results[b]["sc"] for b in BEARINGS]
    mean_sc = float(np.mean(sc_all))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Train Sliding RUL — LSTM slope start_frac  avg={mean_sc:.3f}"
        f"  (v2 level: 0.451)",
        fontsize=12, fontweight="bold")
    axes = axes.flatten()

    for i, bid in enumerate(BEARINGS):
        r  = fold_results[bid]
        ts = r["t_starts"]
        v2_sc = [0.3870, 0.5767, 0.4681, 0.3727][i]

        ax = axes[i]
        ax.plot(ts, r["true_ruls"], "k-",  lw=1.5, label="True RUL")
        ax.plot(ts, r["preds"],     "b--", lw=1.5, alpha=0.8,
                label=f"LSTM slope {r['sc']:.3f}  (v2:{v2_sc:.3f})")
        ax.set_title(f"Bearing{bid}  slope={r['sc']:.3f}  v2={v2_sc:.3f}",
                     fontsize=10)
        ax.set_xlabel("Window Start (cycle)"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # individual — RUL + lifecycle estimation 비교
        fig_i, axes_i = plt.subplots(2, 1, figsize=(10, 7))
        axes_i[0].plot(ts, r["true_ruls"], "k-",  lw=1.5, label="True RUL")
        axes_i[0].plot(ts, r["preds"],     "b--", lw=1.5, alpha=0.8,
                       label=f"LSTM  {r['sc']:.3f}")
        axes_i[0].set_title(
            f"Bearing{bid} — Sliding RUL (slope start_frac)", fontsize=10)
        axes_i[0].set_xlabel("Window Start (cycle)")
        axes_i[0].set_ylabel("RUL (cycles)")
        axes_i[0].legend(fontsize=8); axes_i[0].grid(True, alpha=0.3)

        # Lifecycle 추정 vs 실제
        actual_end_frac = [(t + WIN_SIZE - 1) / r["N"] for t in ts]
        axes_i[1].plot(ts, [sf * 100 for sf in r["sf_list"]],
                       "g-", lw=1.5, label="Estimated start %")
        axes_i[1].plot(ts, [f * 100 for f in actual_end_frac],
                       "k--", lw=1, label="Actual window end %")
        axes_i[1].set_xlabel("Window Start (cycle)")
        axes_i[1].set_ylabel("Lifecycle %")
        axes_i[1].set_title("Lifecycle Position: Slope-based Estimate vs Actual")
        axes_i[1].legend(fontsize=8); axes_i[1].grid(True, alpha=0.3)
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Bearing{bid}_sliding.png", dpi=150)
        plt.close(fig_i)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "Train_RUL_sliding_all.png", dpi=150)
    plt.close()

    print(f"\n  Summary (slope vs v2 level):")
    v2_scores = [0.3870, 0.5767, 0.4681, 0.3727]
    for bid, v2 in zip(BEARINGS, v2_scores):
        sc = fold_results[bid]["sc"]
        print(f"    B{bid}: slope={sc:.4f}  level={v2:.4f}  "
              f"{'↑' if sc > v2 else '↓'} {sc-v2:+.4f}")
    print(f"    Mean: slope={mean_sc:.4f}  level=0.4511  "
          f"{'↑' if mean_sc > 0.4511 else '↓'} {mean_sc-0.4511:+.4f}")

    # ── [4] Test HI + Test RUL ────────────────────────────────────
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

    # Test RUL
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Test RUL — LSTM slope start_frac (0604 v3)",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()
    summary_rows = []

    v2_preds = {1:3.78, 2:2.85, 3:2.87, 4:3.85, 5:2.83, 6:2.80}

    for i, tid in enumerate(TEST_IDS):
        hi_t  = test_data[tid]["hi"]
        reg_t = test_data[tid]["regime"]
        n     = len(hi_t)

        sf    = estimate_start_frac_slope(hi_t, train_data, BEARINGS)
        preds = predict_lstm(lstm_all, hi_t, reg_t,
                              sf, hi_mean_all, hi_std_all, rul_scale_all, device)
        obs_pts   = np.arange(SEQ_LEN, n)
        final_cyc = float(np.median(preds[-10:]))   # 마지막 10개 중앙값
        final_hr  = final_cyc * INTERVAL_SEC / 3600

        print(f"  Test{tid}: slope sf={sf:.3f} ({int(sf*100)}%)  "
              f"RUL={final_hr:.2f}hr  (v2 level: {v2_preds[tid]:.2f}hr)")

        ax = axes[i]
        ax.plot(obs_pts, preds, "b-", lw=1.5, label="LSTM")
        ax.axhline(final_cyc, color="m", ls="--", lw=1.5,
                   label=f"Final: {final_hr:.1f}hr")
        ax.axhline(v2_preds[tid] * 3600 / INTERVAL_SEC, color="r",
                   ls=":", lw=1, alpha=0.6, label=f"v2: {v2_preds[tid]:.1f}hr")
        ax.set_title(f"Test{tid}  start≈{int(sf*100)}%  "
                     f"RUL={final_hr:.1f}hr  (v2={v2_preds[tid]:.1f}hr)",
                     fontsize=9)
        ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        fig_i, ax_i = plt.subplots(figsize=(9, 4))
        ax_i.plot(obs_pts, preds, "b-", lw=1.5, label="LSTM slope")
        ax_i.axhline(final_cyc, color="m", ls="--", lw=1.5,
                     label=f"Final: {final_hr:.2f}hr")
        ax_i.set_title(f"Test{tid} — LSTM slope  start≈{int(sf*100)}%",
                       fontsize=10)
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
            "hi_start": round(float(hi_t[0]), 4),
            "start_frac_slope": round(sf, 3),
            "final_rul_cycles": round(final_cyc, 2),
            "final_rul_hours":  round(final_hr, 2),
            "v2_level_hours":   v2_preds[tid],
        })

    plt.tight_layout()
    plt.savefig(OUT_DIR / "Test_RUL_all.png", dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary.csv", index=False)

    print(f"\n{'='*70}")
    print(f"  Sliding LOOCV: slope={mean_sc:.4f}  level=0.4511  "
          f"{'↑' if mean_sc > 0.4511 else '↓'}")
    print(f"\n  Test RUL:")
    print(df_sum[["test_id","start_frac_slope","final_rul_hours",
                   "v2_level_hours"]].to_string(index=False))


if __name__ == "__main__":
    run()
