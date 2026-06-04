"""
RUL kurtosis_feat v1 — 0604 HI + kurtosis as separate LSTM feature
====================================================================
Hypothesis: adding kurtosis (ch1_kurt_log) as a 4th LSTM input lets
the model detect B3-type impulsive failure without changing the HI.

N_FEAT = 4: [HI_norm, obs_frac, regime, kurt_norm]
  kurt_norm = max(ch1_kurt_log, ch2_kurt_log), normalized by fleet stats

LOOCV: actual start_frac (same as 0604 baseline conditions).
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
BASE      = Path("/data/home/ksphm/2026-challenge-KSPHM")
HI_TRAIN  = BASE / "User/SR/0604/output/train"
HI_TEST   = BASE / "User/SR/0604/output/test"   # 0604 test HI (consistent scale)
FEAT_TRAIN = BASE / "User/SR/0604/output/train"
FEAT_TEST  = BASE / "User/SR/0604/output/test"
OUT_DIR   = BASE / "User/SR/0605/output_cq/rul_kf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
WIN_SIZE        = 50
SEQ_LEN         = 10
STRIDE          = 1
MEAN_TRAIN_LIFE = 116.5
INTERVAL_SEC    = 600
SEEDS           = [42, 7, 123, 0, 99]
N_FEAT          = 4    # HI, obs_frac, regime, kurt_norm

EOL          = {1: 126, 2: 114, 3: 89, 4: 137}
NORMAL_UNTIL = {1: 89,  2: 92,  3: 62, 4: 78}

KURT_CLIP = 10.0   # clip kurt_norm after fleet-normalization


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
    hi_mean, hi_std   = float(all_hi.mean()),   float(all_hi.std()   + 1e-8)
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

def monotonicity(s):
    d = np.diff(s)
    return abs(np.sum(d > 0) - np.sum(d < 0)) / max(len(d), 1)

def trendability(s):
    rho, _ = spearmanr(np.arange(len(s)), s)
    return abs(rho) if not np.isnan(rho) else 0.0


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


def _make_seqs(train_data, bids, hi_mean, hi_std, kurt_mean, kurt_std,
               start_obs_map=None):
    X_list, y_list = [], []
    for b in bids:
        hi_b   = train_data[b]["hi"]
        reg_b  = train_data[b]["regime"]
        kurt_b = train_data[b]["kurt"]
        rul_b  = rul_labels(len(hi_b), b)
        so     = 0 if start_obs_map is None else start_obs_map.get(b, 0)
        n = len(hi_b)
        for i in range(n - SEQ_LEN):
            win_hi   = (hi_b[i:i+SEQ_LEN] - hi_mean) / hi_std
            obs_frac = np.clip((so + i + np.arange(SEQ_LEN)) / MEAN_TRAIN_LIFE,
                                0.0, 2.0)
            rg = reg_b[i:i+SEQ_LEN].astype(float)
            kn = np.clip((kurt_b[i:i+SEQ_LEN] - kurt_mean) / kurt_std,
                          -3.0, KURT_CLIP)
            X_list.append(np.stack([win_hi, obs_frac, rg, kn], axis=1))
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


def predict_ensemble(models, hi_arr, reg_arr, kurt_arr, start_frac,
                      hi_mean, hi_std, kurt_mean, kurt_std, rul_scale, device):
    start_obs = int(start_frac * MEAN_TRAIN_LIFE)
    n = len(hi_arr)
    X = []
    for j in range(n - SEQ_LEN):
        win_hi   = (hi_arr[j:j+SEQ_LEN] - hi_mean) / hi_std
        obs_frac = np.clip(
            (start_obs + j + np.arange(SEQ_LEN)) / MEAN_TRAIN_LIFE, 0.0, 2.0)
        rg = reg_arr[j:j+SEQ_LEN].astype(float)
        kn = np.clip((kurt_arr[j:j+SEQ_LEN] - kurt_mean) / kurt_std,
                      -3.0, KURT_CLIP)
        X.append(np.stack([win_hi, obs_frac, rg, kn], axis=1))
    if not X:
        return np.array([0.0])
    Xt = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    all_p = []
    for m in models:
        m.eval()
        with torch.no_grad():
            all_p.append(np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0))
    return np.median(all_p, axis=0)


def estimate_start_frac_hi(hi_start, train_data, ref_bids):
    fracs = []
    for bid in ref_bids:
        hi = train_data[bid]["hi"]
        exceed = np.where(hi >= hi_start)[0]
        if len(exceed) == 0:
            fracs.append(float(len(hi)) / MEAN_TRAIN_LIFE)
        elif hi[0] >= hi_start:
            continue
        else:
            fracs.append(float(exceed[0]) / MEAN_TRAIN_LIFE)
    if not fracs:
        return float(hi_start)
    return float(np.mean(fracs))


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  RUL kurtosis_feat v1 — 0604 HI + kurt as 4th LSTM feature")
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
        hi_t    = train_data[test_bid]["hi"]
        reg_t   = train_data[test_bid]["regime"]
        kurt_t  = train_data[test_bid]["kurt"]
        N       = len(hi_t)
        rul_t   = rul_labels(N, test_bid)
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
            sf = float(t_start) / MEAN_TRAIN_LIFE   # actual position (same as 0604)
            p  = predict_ensemble(models, hw, rw, kw, sf,
                                   hi_mean, hi_std, kurt_mean, kurt_std, rul_scale, device)
            t_starts.append(t_start)
            true_ruls.append(float(rul_t[t_start + WIN_SIZE - 1]))
            preds.append(float(p[-1]))

        sc = avg_score(true_ruls, preds)
        fold_scores[test_bid] = sc
        print(f"    sliding score: {sc:.4f}")

        # Per-fold plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_starts, true_ruls, "k-", lw=1.5, label="True RUL")
        ax.plot(t_starts, preds, "b--", lw=1.5, alpha=0.8, label=f"LSTM  {sc:.3f}")
        ax.set_title(f"Bearing{test_bid} LOOCV — kurtosis_feat  sc={sc:.3f}", fontsize=10)
        ax.set_xlabel("Window Start (cycle)"); ax.set_ylabel("RUL (cycles)")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"Bearing{test_bid}_loocv.png", dpi=150)
        plt.close()

    mean_sc = float(np.mean(list(fold_scores.values())))
    print(f"\n  LOOCV summary:")
    for bid in BEARINGS:
        print(f"    B{bid}: {fold_scores[bid]:.4f}")
    print(f"    Mean: {mean_sc:.4f}  (0604 baseline: 0.4600)")

    # ── Full-train for test inference ─────────────────────────────
    print("\n[2] Test RUL (full train)...")
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

        sf = estimate_start_frac_hi(float(hi_t[0]), train_data, BEARINGS)
        preds  = predict_ensemble(models, hi_t, reg_t, kurt_t, sf,
                                   hi_mean, hi_std, kurt_mean, kurt_std, rul_scale, device)
        obs_pts   = np.arange(SEQ_LEN, n)
        final_cyc = float(preds[-1])
        final_hr  = final_cyc * INTERVAL_SEC / 3600

        print(f"  Test{tid}: hi_start={hi_t[0]:.3f}  sf={sf:.3f}  "
              f"RUL={final_hr:.2f}hr ({final_cyc:.1f}cyc)  "
              f"kurt_max={kurt_t.max():.2f}")

        ax = axes_rul[i]
        ax.plot(obs_pts, preds, "b-", lw=2, label=f"LSTM {final_hr:.1f}hr")
        ax.set_title(f"Test{tid}  sf={int(sf*100)}%  RUL={final_hr:.1f}hr", fontsize=10)
        ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(); ax.grid(alpha=0.3)

        fig_i, ax_i = plt.subplots(figsize=(9, 4))
        ax_i.plot(obs_pts, preds, "b-", lw=2, label="LSTM prediction")
        ax_i.axhline(final_cyc, color="m", ls="--", lw=1,
                     label=f"Final: {final_hr:.2f}hr ({final_cyc:.1f}cyc)")
        ax_i.set_title(f"Test{tid} — kurtosis_feat RUL  sf={int(sf*100)}%", fontsize=10)
        ax_i.set_xlabel("Obs Cycle"); ax_i.set_ylabel("RUL (cycles)")
        ax_i.legend(); ax_i.grid(alpha=0.3)
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Test{tid}_RUL.png", dpi=150)
        plt.close(fig_i)

        pd.DataFrame({
            "obs_cycle": obs_pts,
            "rul_pred_cycles": preds,
            "rul_pred_hours": preds * INTERVAL_SEC / 3600,
        }).to_csv(OUT_DIR / f"Test{tid}_RUL.csv", index=False)

        summary_rows.append({
            "test_id":          tid,
            "hi_start":         round(float(hi_t[0]), 4),
            "start_frac":       round(sf, 3),
            "start_frac_pct":   int(sf * 100),
            "final_rul_cycles": round(final_cyc, 2),
            "final_rul_hours":  round(final_hr, 2),
            "kurt_max":         round(float(kurt_t.max()), 3),
        })

    fig_rul.suptitle(f"Test RUL — kurtosis_feat (0604 HI + kurt feat)  LOOCV={mean_sc:.3f}",
                     fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Test_RUL_all.png", dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary.csv", index=False)

    print(f"\n{'='*70}")
    print(f"  Sliding LOOCV avg: {mean_sc:.4f}  (0604 baseline: 0.4600)")
    print(f"\n  Test RUL predictions:")
    print(df_sum[["test_id","hi_start","start_frac_pct",
                  "final_rul_hours","kurt_max"]].to_string(index=False))
    print(f"\n  Output: {OUT_DIR}/")
