"""
RUL Sliding Window v2 — LSTM Only (0604)
=========================================
v1 결과:
  - LGBM: sliding 시나리오에서 start_frac 추정 오차로 완전히 붕괴
  - LSTM: start_frac 추정에 덜 의존, sliding에서도 robust
  - LSTM 단독 sliding LOOCV: 0.451 (50/50 앙상블 0.285 대비 대폭 개선)

v2 변경:
  - LSTM 단독 사용 (LGBM 제거)
  - 필수 그림 전부 생성:
      1. Train HI × 4  (0603_v3 FDR HI)
      2. Train RUL × 4  (sliding LOOCV LSTM 예측 vs 실제)
      3. Test  HI × 6  (0603_v3 FDR HI)
      4. Test  RUL × 6  (LSTM 단독 예측)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
BASE     = Path("/data/home/ksphm/2026-challenge-KSPHM")
HI_TRAIN = BASE / "User/SR/0603_v3/output/train"
HI_TEST  = BASE / "User/SR/0603_v3/output/test"
OUT_DIR  = BASE / "User/SR/0604/output/rul_sliding"
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
        data[bid] = {"hi": df["HI"].values.astype(float),
                     "regime": df["regime"].values.astype(int)}
    return data

def load_test_hi():
    data = {}
    for tid in TEST_IDS:
        df = pd.read_csv(HI_TEST / f"Test{tid}_HI.csv")
        data[tid] = {"hi": df["HI"].values.astype(float),
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
    if rul_true <= 0:
        return np.nan
    Er = 100.0 * (rul_true - rul_pred) / rul_true
    return (np.exp(-np.log(0.5) * Er / 20.0) if Er <= 0
            else np.exp(np.log(0.5) * Er / 50.0))

def avg_score(true_ruls, preds):
    scores = [comp_score(t, p) for t, p in zip(true_ruls, preds)]
    valid  = [s for s in scores if not np.isnan(s)]
    return float(np.mean(valid)) if valid else np.nan


# ══════════════════════════════════════════════════════════════════
# Start position estimation
# ══════════════════════════════════════════════════════════════════
def estimate_start_frac(hi_start, train_data, ref_bids):
    fracs = []
    for bid in ref_bids:
        hi = train_data[bid]["hi"]
        if hi[0] >= hi_start:
            continue
        exceed = np.where(hi >= hi_start)[0]
        fracs.append(float(len(hi)) / MEAN_TRAIN_LIFE if len(exceed) == 0
                     else float(exceed[0]) / MEAN_TRAIN_LIFE)
    return float(np.mean(fracs)) if fracs else 0.0


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


def _make_seq_dataset(train_data, bids, hi_mean, hi_std, start_obs_map=None):
    """start_obs_map: {bid: start_obs} — None이면 모두 0"""
    X_list, y_list = [], []
    for b in bids:
        hi_b  = train_data[b]["hi"]
        reg_b = train_data[b]["regime"]
        rul_b = rul_labels(len(hi_b), b)
        so    = 0 if start_obs_map is None else start_obs_map.get(b, 0)
        n = len(hi_b)
        for i in range(n - SEQ_LEN):
            win_norm = (hi_b[i:i+SEQ_LEN] - hi_mean) / hi_std
            obs_frac = np.clip((so + i + np.arange(SEQ_LEN)) / MEAN_TRAIN_LIFE,
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
        model = LSTMRegressor().to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit  = nn.MSELoss()
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
                if patience >= 20:
                    break
        model.load_state_dict(best_state)
        models.append(model)
    return models


def predict_lstm(models, hi_arr, reg_arr, start_frac, hi_mean, hi_std,
                  rul_scale, device):
    """전체 시계열에 대한 LSTM 예측 — 각 위치에서의 예측값 배열 반환."""
    start_obs = int(start_frac * MEAN_TRAIN_LIFE)
    n = len(hi_arr)
    X = []
    for j in range(n - SEQ_LEN):
        win_norm = (hi_arr[j:j+SEQ_LEN] - hi_mean) / hi_std
        obs_frac = np.clip(
            (start_obs + j + np.arange(SEQ_LEN)) / MEAN_TRAIN_LIFE, 0.0, 2.0)
        rg = reg_arr[j:j+SEQ_LEN].astype(float)
        X.append(np.stack([win_norm, obs_frac, rg], axis=1))
    if not X:
        return np.array([0.0])
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
def _plot_hi(ax, hi, regime, title, q, interval=INTERVAL_SEC):
    t = np.arange(len(hi)) * interval / 3600
    for lbl, col, name in [(0, "#4C72B0", "Low"), (1, "#DD8452", "High")]:
        idx = regime == lbl
        ax.scatter(t[idx], hi[idx], s=10, color=col, alpha=0.5,
                   label=name, zorder=3)
    ax.plot(t, hi, color="gray", lw=0.7, alpha=0.4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{title}  Q={q:.3f}", fontsize=10)
    ax.set_xlabel("Time [hr]"); ax.set_ylabel("HI")
    ax.legend(fontsize=7); ax.grid(True, ls="--", alpha=0.3)

def monotonicity(s):
    d = np.diff(s)
    return abs(np.sum(d > 0) - np.sum(d < 0)) / max(len(d), 1)

def trendability(s):
    from scipy.stats import spearmanr
    rho, _ = spearmanr(np.arange(len(s)), s)
    return abs(rho) if not np.isnan(rho) else 0.0


# ══════════════════════════════════════════════════════════════════
# Phase 1: Train HI plots
# ══════════════════════════════════════════════════════════════════
def plot_train_hi(train_data):
    print("\n[1/4] Train HI 그림 생성...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Train Bearings — HI (FDR LOO, 0603_v3)",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for i, bid in enumerate(BEARINGS):
        hi  = train_data[bid]["hi"]
        reg = train_data[bid]["regime"]
        q   = (monotonicity(hi) + trendability(hi)) / 2
        _plot_hi(axes[i], hi, reg, f"Bearing{bid}", q)

        # individual
        fig_i, ax_i = plt.subplots(figsize=(9, 4))
        _plot_hi(ax_i, hi, reg, f"Bearing{bid}", q)
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Bearing{bid}_HI.png", dpi=150)
        plt.close(fig_i)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "Train_HI_all.png", dpi=150)
    plt.close()
    print(f"   → {OUT_DIR}/Train_HI_all.png")


# ══════════════════════════════════════════════════════════════════
# Phase 2+3: Sliding LOOCV (LSTM only) + Train RUL plots
# ══════════════════════════════════════════════════════════════════
def run_sliding_loocv_lstm(train_data, device):
    print("\n[2/4] Sliding Window LOOCV — LSTM only")
    print(f"      WIN={WIN_SIZE}  STRIDE={STRIDE}  SEEDS={len(SEEDS)}")

    fold_results = {}

    for test_bid in BEARINGS:
        train_bids  = [b for b in BEARINGS if b != test_bid]
        hi_t        = train_data[test_bid]["hi"]
        reg_t       = train_data[test_bid]["regime"]
        N           = len(hi_t)
        rul_t       = rul_labels(N, test_bid)
        hi_mean, hi_std = compute_hi_stats(train_data, train_bids)
        rul_scale   = float(max(EOL[b] for b in train_bids))

        print(f"\n  Fold B{test_bid}  (train={train_bids}  N={N}  "
              f"rul_scale={rul_scale:.0f})")

        # LSTM 학습
        X_tr, y_tr = _make_seq_dataset(train_data, train_bids, hi_mean, hi_std)
        models = train_lstm_ensemble(X_tr, y_tr, rul_scale, device)
        print(f"    LSTM 학습 완료 ({len(SEEDS)} seeds)")

        # 슬라이딩 창 평가
        t_starts, true_ruls, preds = [], [], []

        for t_start in range(0, N - WIN_SIZE + 1, STRIDE):
            hi_win  = hi_t[t_start: t_start + WIN_SIZE]
            reg_win = reg_t[t_start: t_start + WIN_SIZE]
            true_rul = float(rul_t[t_start + WIN_SIZE - 1])
            sf = estimate_start_frac(float(hi_win[0]), train_data, train_bids)
            p  = predict_lstm(models, hi_win, reg_win,
                               sf, hi_mean, hi_std, rul_scale, device)
            t_starts.append(t_start)
            true_ruls.append(true_rul)
            preds.append(float(p[-1]))

        sc = avg_score(true_ruls, preds)
        print(f"    LSTM sliding score: {sc:.4f}")
        fold_results[test_bid] = {
            "N": N, "t_starts": t_starts,
            "true_ruls": true_ruls, "preds": preds,
            "sc": sc, "rul_t": rul_t,
            "hi_t": hi_t, "reg_t": reg_t,
        }

    # ── [3/4] Train RUL plots ──────────────────────────────────────
    print("\n[3/4] Train RUL 그림 생성...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Train Bearings — Sliding Window RUL (LSTM only, WIN=50)",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()

    sc_all = []
    for i, bid in enumerate(BEARINGS):
        r   = fold_results[bid]
        ts  = r["t_starts"]
        sc_all.append(r["sc"])

        ax = axes[i]
        ax.plot(ts, r["true_ruls"], "k-",  lw=1.5, label="True RUL")
        ax.plot(ts, r["preds"],     "b--", lw=1.5, alpha=0.8,
                label=f"LSTM  {r['sc']:.3f}")
        ax.set_title(f"Bearing{bid}  (LSTM={r['sc']:.3f})", fontsize=10)
        ax.set_xlabel("Window Start (cycle)"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # individual
        fig_i, ax_i = plt.subplots(figsize=(10, 5))
        ax_i.plot(ts, r["true_ruls"], "k-",  lw=1.5, label="True RUL")
        ax_i.plot(ts, r["preds"],     "b--", lw=1.5, alpha=0.8,
                  label=f"LSTM  {r['sc']:.3f}")

        # start_frac 보조 축
        ax2 = ax_i.twinx()
        sfs = [estimate_start_frac(float(r["hi_t"][t]),
                                    {b: {"hi": fold_results.get(b, {}).get("hi_t",
                                          train_data[b]["hi"])}
                                     for b in BEARINGS if b != bid},
                                    [b for b in BEARINGS if b != bid])
               for t in ts]
        ax2.plot(ts, [sf * 100 for sf in sfs], "g:", lw=1, alpha=0.6,
                 label="est. start %")
        ax2.set_ylabel("Estimated lifecycle %", color="g")
        ax2.tick_params(axis="y", labelcolor="g")

        ax_i.set_title(f"Bearing{bid} — Sliding RUL (LSTM  sc={r['sc']:.3f})",
                       fontsize=10)
        ax_i.set_xlabel("Window Start (cycle)"); ax_i.set_ylabel("RUL (cycles)")
        lines1, labs1 = ax_i.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax_i.legend(lines1 + lines2, labs1 + labs2, fontsize=7)
        ax_i.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Bearing{bid}_sliding_lstm.png", dpi=150)
        plt.close(fig_i)

    mean_sc = float(np.mean(sc_all))
    fig.suptitle(f"Train Bearings — Sliding RUL (LSTM only)  avg={mean_sc:.3f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Train_RUL_sliding_all.png", dpi=150)
    plt.close()

    print(f"\n  Sliding LOOCV LSTM summary:")
    for bid in BEARINGS:
        print(f"    B{bid}: {fold_results[bid]['sc']:.4f}")
    print(f"    Mean: {mean_sc:.4f}")
    print(f"   → {OUT_DIR}/Train_RUL_sliding_all.png")

    return fold_results, mean_sc


# ══════════════════════════════════════════════════════════════════
# Phase 4+5: Test HI + Test RUL (LSTM only)
# ══════════════════════════════════════════════════════════════════
def run_test(train_data, test_data, device):
    print("\n[4/4] Test HI + RUL (LSTM only, full train)")

    # ── Test HI plots ─────────────────────────────────────────────
    print("  Test HI 그림 생성...")
    fig_hi, axes_hi = plt.subplots(2, 3, figsize=(18, 10))
    fig_hi.suptitle("Test Bearings — HI (FDR, 0603_v3)",
                    fontsize=13, fontweight="bold")
    axes_hi = axes_hi.flatten()

    for i, tid in enumerate(TEST_IDS):
        hi  = test_data[tid]["hi"]
        reg = test_data[tid]["regime"]
        q   = (monotonicity(hi) + trendability(hi)) / 2
        _plot_hi(axes_hi[i], hi, reg, f"Test{tid}", q)

        fig_i, ax_i = plt.subplots(figsize=(9, 4))
        _plot_hi(ax_i, hi, reg, f"Test{tid}", q)
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Test{tid}_HI.png", dpi=150)
        plt.close(fig_i)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "Test_HI_all.png", dpi=150)
    plt.close()
    print(f"   → {OUT_DIR}/Test_HI_all.png")

    # ── LSTM 학습 (전체 train) ────────────────────────────────────
    hi_mean, hi_std = compute_hi_stats(train_data, BEARINGS)
    rul_scale       = float(max(EOL.values()))
    print(f"\n  LSTM 학습 (full train, mean={hi_mean:.4f}  "
          f"std={hi_std:.4f}  rul_scale={rul_scale:.0f})...")
    X_tr, y_tr = _make_seq_dataset(train_data, BEARINGS, hi_mean, hi_std)
    models = train_lstm_ensemble(X_tr, y_tr, rul_scale, device)
    print(f"  완료 ({len(SEEDS)} seeds)")

    # ── Test RUL 예측 + 그림 ─────────────────────────────────────
    fig_rul, axes_rul = plt.subplots(2, 3, figsize=(18, 10))
    fig_rul.suptitle("Test Bearings — RUL (LSTM only, WIN=50)",
                     fontsize=13, fontweight="bold")
    axes_rul = axes_rul.flatten()

    summary_rows = []
    for i, tid in enumerate(TEST_IDS):
        hi_t = test_data[tid]["hi"]
        reg_t= test_data[tid]["regime"]
        n    = len(hi_t)
        sf   = estimate_start_frac(float(hi_t[0]), train_data, BEARINGS)

        preds = predict_lstm(models, hi_t, reg_t, sf,
                              hi_mean, hi_std, rul_scale, device)
        obs_pts   = np.arange(SEQ_LEN, n)
        final_cyc = float(preds[-1])
        final_hr  = final_cyc * INTERVAL_SEC / 3600

        print(f"  Test{tid}: hi_start={hi_t[0]:.3f}  "
              f"start_frac={sf:.3f} ({int(sf*100)}%)  "
              f"RUL={final_hr:.2f}hr ({final_cyc:.1f}cyc)")

        # combined plot
        ax = axes_rul[i]
        ax.plot(obs_pts, preds, "b-", lw=2, label=f"LSTM  {final_hr:.1f}hr")
        ax.axhline(final_cyc, color="m", ls="--", lw=1, alpha=0.5)
        ax.set_title(f"Test{tid}  start≈{int(sf*100)}%  RUL={final_hr:.1f}hr",
                     fontsize=10)
        ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # individual
        fig_i, ax_i = plt.subplots(figsize=(9, 4))
        ax_i.plot(obs_pts, preds, "b-", lw=2, label=f"LSTM")
        ax_i.axhline(final_cyc, color="m", ls="--", lw=1, alpha=0.6,
                     label=f"Final: {final_hr:.2f}hr ({final_cyc:.1f}cyc)")
        ax_i.set_title(f"Test{tid} — RUL (LSTM only)  "
                       f"start≈{int(sf*100)}%  RUL={final_hr:.2f}hr",
                       fontsize=10)
        ax_i.set_xlabel("Obs Cycle"); ax_i.set_ylabel("RUL (cycles)")
        ax_i.legend(fontsize=8); ax_i.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Test{tid}_RUL.png", dpi=150)
        plt.close(fig_i)

        pd.DataFrame({"obs_cycle": obs_pts, "rul_pred_cycles": preds,
                      "rul_pred_hours": preds * INTERVAL_SEC / 3600}
                     ).to_csv(OUT_DIR / f"Test{tid}_RUL.csv", index=False)

        summary_rows.append({
            "test_id": tid, "hi_start": round(float(hi_t[0]), 4),
            "start_frac": round(sf, 3),
            "final_rul_cycles": round(final_cyc, 2),
            "final_rul_hours": round(final_hr, 2),
        })

    plt.tight_layout()
    plt.savefig(OUT_DIR / "Test_RUL_all.png", dpi=150)
    plt.close()
    print(f"   → {OUT_DIR}/Test_RUL_all.png")

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary_lstm.csv", index=False)
    return df_sum


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  RUL Sliding Window v2 — LSTM Only (0604)")
    print("=" * 70)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    train_data = load_train_hi()
    test_data  = load_test_hi()

    # 1. Train HI
    plot_train_hi(train_data)

    # 2+3. Sliding LOOCV + Train RUL
    fold_results, mean_sc = run_sliding_loocv_lstm(train_data, device)

    # 4+5. Test HI + Test RUL
    df_sum = run_test(train_data, test_data, device)

    print(f"\n{'='*70}")
    print(f"  완료")
    print(f"  Sliding LOOCV LSTM avg: {mean_sc:.4f}")
    print(f"\n  Test RUL 예측:")
    print(df_sum.to_string(index=False))
    print(f"\n  출력: {OUT_DIR}/")
