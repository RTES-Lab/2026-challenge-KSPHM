"""
RUL no-obs_frac v1 — obs_frac removed from LSTM inputs
=======================================================
Change from kurtosis_feat_v1:
  N_FEAT = 3: [HI_norm, regime, kurt_norm]  (obs_frac dropped)

Why:
  - LOOCV used actual t_start (information leakage: test conditions differ)
  - Even with correct obs_frac, LSTM showed flat ~18-cycle predictions
  - Removing obs_frac makes LOOCV and test conditions identical
  - No start_frac estimation needed at inference time
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
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
BASE       = Path("/data/home/ksphm/2026-challenge-KSPHM")
HI_TRAIN   = BASE / "User/SR/0604/output/train"
HI_TEST    = BASE / "User/SR/0604/output/test"
FEAT_TRAIN = BASE / "User/SR/0604/output/train"
FEAT_TEST  = BASE / "User/SR/0604/output/test"
OUT_DIR    = BASE / "User/SR/0605/output_cq/rul_nof"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS     = [1, 2, 3, 4]
TEST_IDS     = [1, 2, 3, 4, 5, 6]
WIN_SIZE     = 50
SEQ_LEN      = 10
STRIDE       = 1
INTERVAL_SEC = 600
SEEDS        = [42, 7, 123, 0, 99]
N_FEAT       = 3    # HI_norm, regime, kurt_norm

EOL          = {1: 126, 2: 114, 3: 89, 4: 137}
NORMAL_UNTIL = {1: 89,  2: 92,  3: 62, 4: 78}
KURT_CLIP    = 10.0


# ══════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════
def load_train_hi():
    data = {}
    for bid in BEARINGS:
        hi_df   = pd.read_csv(HI_TRAIN  / f"Bearing{bid}_HI.csv")
        feat_df = pd.read_csv(FEAT_TRAIN / f"Bearing{bid}_features_raw.csv")
        n = min(len(hi_df), len(feat_df))
        kurt = np.log1p(np.maximum(
            feat_df["ch1_kurt_log"].values[:n],
            feat_df["ch2_kurt_log"].values[:n],
        ))
        data[bid] = {
            "hi":     hi_df["HI"].values[:n].astype(float),
            "regime": hi_df["regime"].values[:n].astype(int),
            "kurt":   kurt,
        }
    return data


def load_test_hi():
    data = {}
    for tid in TEST_IDS:
        hi_df   = pd.read_csv(HI_TEST  / f"Test{tid}_HI.csv")
        feat_df = pd.read_csv(FEAT_TEST / f"Test{tid}_features_raw.csv")
        n = min(len(hi_df), len(feat_df))
        kurt = np.log1p(np.maximum(
            feat_df["ch1_kurt_log"].values[:n],
            feat_df["ch2_kurt_log"].values[:n],
        ))
        data[tid] = {
            "hi":     hi_df["HI"].values[:n].astype(float),
            "regime": hi_df["regime"].values[:n].astype(int),
            "kurt":   kurt,
        }
    return data


def rul_labels(n_total, bid):
    nu, eol = NORMAL_UNTIL[bid], EOL[bid]
    idx = np.arange(n_total)
    return np.where(idx <= nu, eol - nu,
                    np.maximum(eol - idx, 0)).astype(float)


def compute_stats(train_data, bids):
    all_hi   = np.concatenate([train_data[b]["hi"]   for b in bids])
    all_kurt = np.concatenate([train_data[b]["kurt"]  for b in bids])
    hi_mean,   hi_std   = float(all_hi.mean()),   float(all_hi.std()   + 1e-8)
    kurt_mean, kurt_std = float(all_kurt.mean()), float(all_kurt.std() + 1e-8)
    return hi_mean, hi_std, kurt_mean, kurt_std


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
# LSTM model
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


def _make_seqs(train_data, bids, hi_mean, hi_std, kurt_mean, kurt_std):
    X_list, y_list = [], []
    for b in bids:
        hi_b   = train_data[b]["hi"]
        reg_b  = train_data[b]["regime"]
        kurt_b = train_data[b]["kurt"]
        rul_b  = rul_labels(len(hi_b), b)
        n = len(hi_b)
        for i in range(n - SEQ_LEN):
            win_hi = (hi_b[i:i+SEQ_LEN] - hi_mean) / hi_std
            rg     = reg_b[i:i+SEQ_LEN].astype(float)
            kn     = np.clip((kurt_b[i:i+SEQ_LEN] - kurt_mean) / kurt_std,
                              -3.0, KURT_CLIP)
            X_list.append(np.stack([win_hi, rg, kn], axis=1))
            y_list.append(float(rul_b[i + SEQ_LEN]))
    return np.array(X_list), np.array(y_list)


def train_ensemble(X_train, y_train, rul_scale, device):
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


def predict_ensemble(models, hi_arr, reg_arr, kurt_arr,
                     hi_mean, hi_std, kurt_mean, kurt_std, rul_scale, device):
    n = len(hi_arr)
    X = []
    for j in range(n - SEQ_LEN):
        win_hi = (hi_arr[j:j+SEQ_LEN] - hi_mean) / hi_std
        rg     = reg_arr[j:j+SEQ_LEN].astype(float)
        kn     = np.clip((kurt_arr[j:j+SEQ_LEN] - kurt_mean) / kurt_std,
                          -3.0, KURT_CLIP)
        X.append(np.stack([win_hi, rg, kn], axis=1))
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
# Main
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  RUL no-obs_frac v1 — [HI_norm, regime, kurt_norm]")
    print("  obs_frac removed: LOOCV conditions == test conditions")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    train_data = load_train_hi()
    test_data  = load_test_hi()

    # ── LOOCV ─────────────────────────────────────────────────────
    print(f"\n[1] Sliding LOOCV  WIN={WIN_SIZE}  SEQ={SEQ_LEN}  SEEDS={len(SEEDS)}")
    fold_scores = {}
    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        hi_t   = train_data[test_bid]["hi"]
        reg_t  = train_data[test_bid]["regime"]
        kurt_t = train_data[test_bid]["kurt"]
        N      = len(hi_t)
        rul_t  = rul_labels(N, test_bid)
        hi_mean, hi_std, kurt_mean, kurt_std = compute_stats(train_data, train_bids)
        rul_scale = float(max(EOL[b] for b in train_bids))

        print(f"\n  Fold B{test_bid}  train={train_bids}  N={N}")
        X_tr, y_tr = _make_seqs(train_data, train_bids,
                                  hi_mean, hi_std, kurt_mean, kurt_std)
        models = train_ensemble(X_tr, y_tr, rul_scale, device)
        print(f"    LSTM done ({len(SEEDS)} seeds)")

        t_starts, true_ruls, preds = [], [], []
        for t_start in range(0, N - WIN_SIZE + 1, STRIDE):
            hw = hi_t[t_start:t_start+WIN_SIZE]
            rw = reg_t[t_start:t_start+WIN_SIZE]
            kw = kurt_t[t_start:t_start+WIN_SIZE]
            p  = predict_ensemble(models, hw, rw, kw,
                                   hi_mean, hi_std, kurt_mean, kurt_std, rul_scale, device)
            t_starts.append(t_start)
            true_ruls.append(float(rul_t[t_start + WIN_SIZE - 1]))
            preds.append(float(p[-1]))

        sc = avg_score(true_ruls, preds)
        fold_scores[test_bid] = sc
        print(f"    sliding score: {sc:.4f}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_starts, true_ruls, "k-", lw=1.5, label="True RUL")
        ax.plot(t_starts, preds, "b--", lw=1.5, alpha=0.8, label=f"LSTM  {sc:.3f}")
        ax.set_title(f"Bearing{test_bid} LOOCV — no_obs_frac  sc={sc:.3f}", fontsize=10)
        ax.set_xlabel("Window Start (cycle)"); ax.set_ylabel("RUL (cycles)")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"Bearing{test_bid}_loocv.png", dpi=150)
        plt.close()

    mean_sc = float(np.mean(list(fold_scores.values())))
    print(f"\n  LOOCV summary:")
    for bid in BEARINGS:
        print(f"    B{bid}: {fold_scores[bid]:.4f}")
    print(f"    Mean: {mean_sc:.4f}  (kurtosis_feat: 0.4910  baseline: 0.4600)")

    # ── Full-train for test inference ─────────────────────────────
    print("\n[2] Test RUL (full train, no start_frac needed)...")
    hi_mean, hi_std, kurt_mean, kurt_std = compute_stats(train_data, BEARINGS)
    rul_scale = float(max(EOL.values()))
    X_tr, y_tr = _make_seqs(train_data, BEARINGS,
                              hi_mean, hi_std, kurt_mean, kurt_std)
    models = train_ensemble(X_tr, y_tr, rul_scale, device)
    print(f"  Done ({len(SEEDS)} seeds)")

    summary_rows = []
    fig_rul, axes_rul = plt.subplots(2, 3, figsize=(18, 10))
    axes_rul = axes_rul.flatten()

    for i, tid in enumerate(TEST_IDS):
        hi_t   = test_data[tid]["hi"]
        reg_t  = test_data[tid]["regime"]
        kurt_t = test_data[tid]["kurt"]
        n      = len(hi_t)

        preds     = predict_ensemble(models, hi_t, reg_t, kurt_t,
                                      hi_mean, hi_std, kurt_mean, kurt_std, rul_scale, device)
        obs_pts   = np.arange(SEQ_LEN, n)
        final_cyc = float(preds[-1])
        final_hr  = final_cyc * INTERVAL_SEC / 3600

        print(f"  Test{tid}: hi_start={hi_t[0]:.4f}  "
              f"RUL={final_hr:.2f}hr ({final_cyc:.1f}cyc)  "
              f"kurt_max={kurt_t.max():.3f}")

        ax = axes_rul[i]
        ax.plot(obs_pts, preds, "b-", lw=2, label=f"LSTM {final_hr:.1f}hr")
        ax.set_title(f"Test{tid}  hi={hi_t[0]:.3f}  RUL={final_hr:.1f}hr", fontsize=10)
        ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(); ax.grid(alpha=0.3)

        fig_i, ax_i = plt.subplots(figsize=(9, 4))
        ax_i.plot(obs_pts, preds, "b-", lw=2, label="LSTM prediction")
        ax_i.axhline(final_cyc, color="m", ls="--", lw=1,
                     label=f"Final: {final_hr:.2f}hr ({final_cyc:.1f}cyc)")
        ax_i.set_title(f"Test{tid} — no_obs_frac RUL", fontsize=10)
        ax_i.set_xlabel("Obs Cycle"); ax_i.set_ylabel("RUL (cycles)")
        ax_i.legend(); ax_i.grid(alpha=0.3)
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Test{tid}_RUL.png", dpi=150)
        plt.close(fig_i)

        pd.DataFrame({
            "obs_cycle":       obs_pts,
            "rul_pred_cycles": preds,
            "rul_pred_hours":  preds * INTERVAL_SEC / 3600,
        }).to_csv(OUT_DIR / f"Test{tid}_RUL.csv", index=False)

        summary_rows.append({
            "test_id":          tid,
            "hi_start":         round(float(hi_t[0]), 4),
            "final_rul_cycles": round(final_cyc, 2),
            "final_rul_hours":  round(final_hr, 2),
            "kurt_max":         round(float(kurt_t.max()), 3),
            "final_rul_seconds": round(final_hr * 3600),
        })

    fig_rul.suptitle(
        f"Test RUL — no_obs_frac (0604 HI + kurt)  LOOCV={mean_sc:.3f}",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Test_RUL_all.png", dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary.csv", index=False)

    print(f"\n{'='*70}")
    print(f"  Sliding LOOCV avg: {mean_sc:.4f}")
    print(f"  (kurtosis_feat: 0.4910  |  0604 baseline: 0.4600)")
    print(f"\n  Test RUL predictions:")
    print(df_sum[["test_id", "hi_start", "final_rul_hours", "kurt_max"]].to_string(index=False))
    print(f"\n  Output: {OUT_DIR}/")
