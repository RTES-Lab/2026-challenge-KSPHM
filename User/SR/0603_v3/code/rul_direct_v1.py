"""
RUL Direct v1 — Raw features -> RUL (bypasses HI entirely)
===========================================================
HI normalization problem fix:
  - No HI as intermediate representation
  - 7 raw features normalized via LOO baseline z-score (per regime)
  - LGBM + LSTM predict RUL directly from normalized features
  - No EOL-anchor-free HI scale inconsistency

Start position estimation:
  - Compute mean degradation level = mean of clipped z-scores at test start
  - Find when train bearings first reach this level -> start_frac
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
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
SR_BASE       = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0603_v3")
RAW_TRAIN_DIR = SR_BASE / "output/train"
RAW_TEST_DIR  = SR_BASE / "output/test"
OUT_DIR       = SR_BASE / "output/rul_direct"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH      = 10
MEAN_TRAIN_LIFE = 116.5
INTERVAL_SEC    = 600
SEEDS           = [42, 7, 123, 0, 99]
BASELINE_RATIO  = 0.10
FLAT_IDS        = {2, 5, 6}
CONSERVATIVE_FACTOR = 0.875

NORMAL_UNTIL = {1: 89, 2: 92, 3: 62, 4: 78}
EOL          = {1: 126, 2: 114, 3: 89, 4: 137}

ALL_FEATS = ["ch3_high_band", "ch4_high_band", "ch3_total_power",
             "ch3_energy", "ch3_rms", "ch3_std", "ch3_p2p"]
N_RAW     = len(ALL_FEATS)    # 7
N_LSTM_IN = N_RAW + 2         # + obs_frac + regime = 9


# ══════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════
def load_train_raw() -> dict:
    """Load raw feature CSVs + regime from existing HI CSV (reuse classification)."""
    data = {}
    for bid in BEARINGS:
        feat_df   = pd.read_csv(RAW_TRAIN_DIR / f"Bearing{bid}_features_raw.csv")
        regime_df = pd.read_csv(RAW_TRAIN_DIR / f"Bearing{bid}_HI.csv")
        n = min(len(feat_df), len(regime_df))
        data[bid] = {
            "feat":   feat_df[ALL_FEATS].values[:n].astype(float),
            "regime": regime_df["regime"].values[:n].astype(int),
        }
    return data


def load_test_raw() -> dict:
    data = {}
    for tid in TEST_IDS:
        feat_df   = pd.read_csv(RAW_TEST_DIR / f"Test{tid}_features_raw.csv")
        regime_df = pd.read_csv(RAW_TEST_DIR / f"Test{tid}_HI.csv")
        n = min(len(feat_df), len(regime_df))
        data[tid] = {
            "feat":   feat_df[ALL_FEATS].values[:n].astype(float),
            "regime": regime_df["regime"].values[:n].astype(int),
        }
    return data


def rul_labels(n_total: int, bid: int) -> np.ndarray:
    nu, eol = NORMAL_UNTIL[bid], EOL[bid]
    idx = np.arange(n_total)
    return np.where(idx <= nu, eol - nu, np.maximum(eol - idx, 0)).astype(float)


# ══════════════════════════════════════════════════════════════════
# Per-regime LOO Normalization
# ══════════════════════════════════════════════════════════════════
def compute_norm_stats(raw_data: dict, train_bids: list) -> dict:
    """
    Pooled baseline from train_bids: first BASELINE_RATIO% cycles per regime.
    Returns {regime: {mean: (N_RAW,), std: (N_RAW,)}}
    """
    stats = {}
    for regime in [0, 1]:
        acc = []
        for bid in train_bids:
            idx_r  = np.where(raw_data[bid]["regime"] == regime)[0]
            n_base = max(3, int(len(idx_r) * BASELINE_RATIO))
            acc.append(raw_data[bid]["feat"][idx_r[:n_base]])
        if acc:
            all_vals = np.concatenate(acc, axis=0)
            stats[regime] = {
                "mean": all_vals.mean(axis=0),
                "std":  all_vals.std(axis=0) + 1e-8,
            }
        else:
            stats[regime] = {"mean": np.zeros(N_RAW), "std": np.ones(N_RAW)}
    return stats


def normalize_feat(feat: np.ndarray, regime_arr: np.ndarray,
                   norm_stats: dict) -> np.ndarray:
    """Z-score each cycle using that cycle's regime baseline stats."""
    norm = np.zeros_like(feat, dtype=float)
    for regime in [0, 1]:
        idx_r = np.where(regime_arr == regime)[0]
        if len(idx_r) == 0:
            continue
        m = norm_stats[regime]["mean"]
        s = norm_stats[regime]["std"]
        norm[idx_r] = (feat[idx_r] - m) / s
    return norm


# ══════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════
def comp_score(rul_true: float, rul_pred: float) -> float:
    if rul_true <= 0:
        return np.nan
    Er = 100.0 * (rul_true - rul_pred) / rul_true
    return (np.exp(-np.log(0.5) * Er / 20.0) if Er <= 0
            else np.exp(np.log(0.5) * Er / 50.0))


def avg_score(N: int, obs_pts, preds) -> float:
    return float(np.nanmean([comp_score(N - obs, p)
                              for obs, p in zip(obs_pts, preds)]))


# ══════════════════════════════════════════════════════════════════
# Start Position Estimation
# ══════════════════════════════════════════════════════════════════
def estimate_start_frac_raw(norm_feat_start: np.ndarray,
                             raw_data: dict, train_bids: list,
                             norm_stats: dict) -> float:
    """
    test start degradation level = mean of positive z-scores at first cycle.
    Find when each train bearing first reaches this level, average the fractions.
    """
    deg_start = float(np.clip(norm_feat_start, 0, None).mean())
    fracs = []
    for bid in train_bids:
        feat  = raw_data[bid]["feat"]
        cond  = raw_data[bid]["regime"]
        n     = len(feat)
        norm  = normalize_feat(feat, cond, norm_stats)
        deg   = np.clip(norm, 0, None).mean(axis=1)
        if deg[0] >= deg_start:
            continue  # already above threshold at start -> skip
        exceed = np.where(deg >= deg_start)[0]
        if len(exceed) == 0:
            fracs.append(float(n) / MEAN_TRAIN_LIFE)
        else:
            fracs.append(float(exceed[0]) / MEAN_TRAIN_LIFE)
    return float(np.mean(fracs)) if fracs else 0.0


# ══════════════════════════════════════════════════════════════════
# LGBM
# ══════════════════════════════════════════════════════════════════
def make_lgbm_features_raw(norm_feat: np.ndarray, regime_arr: np.ndarray,
                            seq_len: int = SEQ_LENGTH,
                            start_frac: float = 0.0):
    """
    Features: last + win_mean + win_std + win_slope + win_delta (each N_RAW)
              + elapsed_frac + regime_cur + regime_frac
    Total: 5*N_RAW + 3 = 38 features
    """
    feats, targets = [], []
    N = len(norm_feat)
    for i in range(seq_len, N):
        win   = norm_feat[i - seq_len: i]         # (seq_len, N_RAW)
        last  = win[-1]
        wmean = win.mean(axis=0)
        wstd  = win.std(axis=0)
        wslop = (win[-1] - win[0]) / (seq_len - 1 + 1e-8)
        wdelt = win[-1] - win[0]
        elapsed_frac = float(np.clip(start_frac + i / MEAN_TRAIN_LIFE, 0.0, 3.0))
        regime_cur   = float(regime_arr[i])
        regime_frac  = float(regime_arr[i - seq_len: i].mean())
        feats.append(np.concatenate([
            last, wmean, wstd, wslop, wdelt,
            [elapsed_frac, regime_cur, regime_frac]
        ]))
        targets.append(float(N - i))
    return np.array(feats), np.array(targets)


def lgbm_asym_obj(y_pred, dataset):
    y_true = dataset.get_label()
    diff   = y_true - y_pred
    w      = np.where(diff < 0, 2.5, 1.0)
    return -diff * w, np.ones_like(diff) * w


def train_lgbm_raw(raw_data: dict, norm_stats: dict,
                   train_bids: list) -> lgb.Booster:
    X_list, y_list = [], []
    for b in train_bids:
        norm = normalize_feat(raw_data[b]["feat"], raw_data[b]["regime"], norm_stats)
        x, y = make_lgbm_features_raw(norm, raw_data[b]["regime"], start_frac=0.0)
        X_list.append(x); y_list.append(y)
    dtrain = lgb.Dataset(np.concatenate(X_list), label=np.concatenate(y_list))
    return lgb.train(
        {"num_leaves": 15, "learning_rate": 0.05,
         "min_child_samples": 5, "verbose": -1,
         "objective": lgbm_asym_obj},
        dtrain, num_boost_round=200,
    )


def predict_lgbm_raw(model: lgb.Booster, norm_feat: np.ndarray,
                     regime_arr: np.ndarray,
                     start_frac: float = 0.0) -> np.ndarray:
    X, _ = make_lgbm_features_raw(norm_feat, regime_arr, start_frac=start_frac)
    return np.maximum(model.predict(X), 0.0)


# ══════════════════════════════════════════════════════════════════
# LSTM
# ══════════════════════════════════════════════════════════════════
def make_seqs_raw(norm_feat: np.ndarray, regime_arr: np.ndarray,
                  rul_arr: np.ndarray, seq_len: int, start_obs: int = 0):
    """Input shape per timestep: [7 norm feats | obs_frac | regime] = 9."""
    N = len(norm_feat)
    X, y = [], []
    for i in range(N - seq_len):
        feat_seq = norm_feat[i: i + seq_len]               # (seq_len, N_RAW)
        obs_frac = np.clip(
            (start_obs + i + np.arange(seq_len)) / MEAN_TRAIN_LIFE,
            0.0, 2.0).reshape(-1, 1)
        reg_seq = regime_arr[i: i + seq_len].astype(float).reshape(-1, 1)
        X.append(np.concatenate([feat_seq, obs_frac, reg_seq], axis=1))
        y.append(float(rul_arr[i + seq_len]))
    return np.array(X), np.array(y)


class LSTMRegressor(nn.Module):
    def __init__(self, n_feat: int = N_LSTM_IN, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=n_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)


def train_lstm_raw(X_train: np.ndarray, y_train: np.ndarray,
                   rul_scale: float, seed: int, device) -> LSTMRegressor:
    torch.manual_seed(seed)
    y_norm = y_train / rul_scale
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_norm,  dtype=torch.float32)
    n_val  = max(1, int(len(Xt) * 0.1))
    tr_dl  = DataLoader(TensorDataset(Xt[:-n_val], yt[:-n_val]),
                        batch_size=64, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xt[-n_val:], yt[-n_val:]), batch_size=64)
    model  = LSTMRegressor(n_feat=N_LSTM_IN).to(device)
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
                for xb, yb in val_dl
            ]))
        if vl < best_val:
            best_val, patience = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    model.load_state_dict(best_state)
    return model


def build_lstm_preds_raw(raw_data: dict, norm_stats: dict,
                          train_bids: list, test_bid: int,
                          device) -> tuple:
    X_list, y_list = [], []
    for b in train_bids:
        norm  = normalize_feat(raw_data[b]["feat"], raw_data[b]["regime"], norm_stats)
        rul_b = rul_labels(len(norm), b)
        X, y  = make_seqs_raw(norm, raw_data[b]["regime"], rul_b, SEQ_LENGTH, start_obs=0)
        X_list.append(X); y_list.append(y)
    X_train   = np.concatenate(X_list)
    y_train   = np.concatenate(y_list)
    rul_scale = float(max(EOL[b] for b in train_bids))

    norm_t = normalize_feat(raw_data[test_bid]["feat"],
                             raw_data[test_bid]["regime"], norm_stats)
    rul_t  = rul_labels(len(norm_t), test_bid)
    X_test, _ = make_seqs_raw(norm_t, raw_data[test_bid]["regime"],
                               rul_t, SEQ_LENGTH, start_obs=0)
    Xt = torch.tensor(X_test, dtype=torch.float32).to(device)

    all_preds = []
    for s in SEEDS:
        m = train_lstm_raw(X_train, y_train, rul_scale, s, device)
        m.eval()
        with torch.no_grad():
            all_preds.append(
                np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0))
    return np.median(all_preds, axis=0), rul_scale


# ══════════════════════════════════════════════════════════════════
# Ensemble + CF
# ══════════════════════════════════════════════════════════════════
def fold_ensemble_weights(results: dict, test_bid: int) -> tuple:
    other = [b for b in BEARINGS if b != test_bid]
    sc_l  = float(np.mean([results[b]["sc_lgbm"] for b in other]))
    sc_a  = float(np.mean([results[b]["sc_lstm"]  for b in other]))
    total = sc_l + sc_a + 1e-12
    w_l   = float(np.clip(sc_l / total, 0.1, 0.7))
    return w_l, 1.0 - w_l


def find_fold_cf(results: dict, test_bid: int, pred_key: str = "ens_raw") -> tuple:
    other = [b for b in BEARINGS if b != test_bid]
    best_cf, best_sc = 1.0, -np.inf
    for cf in np.arange(0.60, 1.41, 0.01):
        sc = float(np.mean([
            avg_score(results[b]["N"], results[b]["obs_pts"],
                      [p * cf for p in results[b][pred_key]])
            for b in other
        ]))
        if sc > best_sc:
            best_sc, best_cf = sc, float(cf)
    return best_cf, best_sc


# ══════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════
def _draw_rul_ax(ax, obs, true_rul, r, title):
    ax.plot(obs, true_rul,        "k-",  lw=1.5, label="True RUL")
    ax.plot(obs, r["preds_lgbm"], "r--", lw=1,   alpha=0.6,
            label=f"LGBM {r['sc_lgbm']:.3f}")
    ax.plot(obs, r["preds_lstm"], "b--", lw=1,   alpha=0.6,
            label=f"LSTM {r['sc_lstm']:.3f}")
    ax.plot(obs, r["preds_cal"],  "m-",  lw=2,
            label=f"Ens+CF {r['sc_cal']:.3f}")
    ax.set_title(
        f"{title}  [w_lgbm={r['w_lgbm']:.2f}  cf={r['fold_cf']:.2f}]", fontsize=9)
    ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)


def _draw_test_ax(ax, obs, preds_lgbm, preds_lstm, preds_final, cf, sf, title):
    ax.plot(obs, preds_lgbm,  "r--", lw=1, alpha=0.6, label="LGBM")
    ax.plot(obs, preds_lstm,  "b--", lw=1, alpha=0.6, label="LSTM")
    ax.plot(obs, preds_final, "m-",  lw=2, label=f"Final (cf={cf:.2f})")
    final_hr = float(preds_final[-1]) * INTERVAL_SEC / 3600
    ax.set_title(f"{title}  RUL={final_hr:.1f}hr  (start~{int(sf*100)}%)", fontsize=9)
    ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)


# ══════════════════════════════════════════════════════════════════
# LOOCV
# ══════════════════════════════════════════════════════════════════
def run_loocv():
    print("=" * 65)
    print("  RUL Direct v1 — raw z-score features, LGBM + LSTM")
    print("=" * 65)

    raw_data = load_train_raw()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    results = {}

    # Step 1: raw predictions per fold
    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        N          = raw_data[test_bid]["feat"].shape[0]
        obs_pts    = np.arange(SEQ_LENGTH, N)

        norm_stats = compute_norm_stats(raw_data, train_bids)
        norm_test  = normalize_feat(raw_data[test_bid]["feat"],
                                    raw_data[test_bid]["regime"], norm_stats)

        print(f"[Fold B{test_bid}]  train: {train_bids}")

        # LGBM (start_frac=0: LOOCV knows lifecycle start)
        lgbm_model = train_lgbm_raw(raw_data, norm_stats, train_bids)
        preds_lgbm = predict_lgbm_raw(lgbm_model, norm_test,
                                       raw_data[test_bid]["regime"], start_frac=0.0)
        sc_lgbm    = avg_score(N, obs_pts, preds_lgbm)

        # LSTM (start_obs=0)
        preds_lstm, _ = build_lstm_preds_raw(
            raw_data, norm_stats, train_bids, test_bid, device)
        sc_lstm = avg_score(N, obs_pts, preds_lstm)

        print(f"  LGBM={sc_lgbm:.4f}  LSTM={sc_lstm:.4f}")
        results[test_bid] = {
            "N": N, "obs_pts": list(obs_pts),
            "preds_lgbm": list(preds_lgbm),
            "preds_lstm": list(preds_lstm),
            "sc_lgbm": sc_lgbm, "sc_lstm": sc_lstm,
        }

    # Step 2: ensemble weights (LOO-aware)
    for test_bid in BEARINGS:
        w_lgbm, w_lstm = fold_ensemble_weights(results, test_bid)
        preds_ens = [w_lgbm * l + w_lstm * a
                     for l, a in zip(results[test_bid]["preds_lgbm"],
                                     results[test_bid]["preds_lstm"])]
        sc_ens = avg_score(results[test_bid]["N"], results[test_bid]["obs_pts"], preds_ens)
        results[test_bid].update({
            "ens_raw": preds_ens, "w_lgbm": w_lgbm, "w_lstm": w_lstm,
            "sc_ens_raw": sc_ens,
        })

    # Step 3: CF search
    print("\n[Fold-wise CF search (other 3 bearings, 0.60~1.40)]")
    for test_bid in BEARINGS:
        cf, cf_ref = find_fold_cf(results, test_bid, pred_key="ens_raw")
        preds_cal  = [p * cf for p in results[test_bid]["ens_raw"]]
        sc_cal     = avg_score(results[test_bid]["N"],
                               results[test_bid]["obs_pts"], preds_cal)
        results[test_bid].update({
            "fold_cf": cf, "preds_cal": preds_cal,
            "sc_cal": sc_cal, "cf_ref_score": cf_ref,
        })
        print(f"  B{test_bid}: cf={cf:.2f} (ref={cf_ref:.4f}) -> sc={sc_cal:.4f}")

    # Summary
    print(f"\n{'='*65}")
    print("  LOOCV Summary")
    print(f"{'='*65}")
    print(f"  {'Bearing':>8} | {'LGBM':>8} | {'LSTM':>8} | "
          f"{'Ens(raw)':>10} | {'CF':>5} | {'Ens+CF':>8}")
    print(f"  {'-'*62}")
    sc_lists = {k: [] for k in ["lgbm", "lstm", "ens", "cal"]}
    for test_bid in BEARINGS:
        r = results[test_bid]
        print(f"  {test_bid:>8} | {r['sc_lgbm']:>8.4f} | {r['sc_lstm']:>8.4f} | "
              f"{r['sc_ens_raw']:>10.4f} | {r['fold_cf']:>5.2f} | {r['sc_cal']:>8.4f}")
        sc_lists["lgbm"].append(r["sc_lgbm"])
        sc_lists["lstm"].append(r["sc_lstm"])
        sc_lists["ens"].append(r["sc_ens_raw"])
        sc_lists["cal"].append(r["sc_cal"])

    mean_ens = np.mean(sc_lists["ens"])
    mean_cal = np.mean(sc_lists["cal"])
    print(f"  {'avg':>8} | {np.mean(sc_lists['lgbm']):>8.4f} | "
          f"{np.mean(sc_lists['lstm']):>8.4f} | "
          f"{mean_ens:>10.4f} | {'—':>5} | {mean_cal:>8.4f}")
    print(f"\n  * LOOCV Ens_raw: {mean_ens:.4f}")
    print(f"  * LOOCV Ens+CF:  {mean_cal:.4f}")
    print(f"  (v4 baseline: Ens_raw=0.466, Ens+CF=0.371)")

    # Log
    with open(OUT_DIR / "loocv_log_direct_v1.txt", "w") as fh:
        fh.write("RUL Direct v1 — raw z-score features + LGBM + LSTM\n")
        fh.write(f"LGBM  mean: {np.mean(sc_lists['lgbm']):.4f}\n")
        fh.write(f"LSTM  mean: {np.mean(sc_lists['lstm']):.4f}\n")
        fh.write(f"Ens (raw):  {mean_ens:.4f}\n")
        fh.write(f"Ens + CF :  {mean_cal:.4f}\n\n")
        for test_bid in BEARINGS:
            r = results[test_bid]
            fh.write(f"  B{test_bid}: LGBM={r['sc_lgbm']:.4f}  LSTM={r['sc_lstm']:.4f}  "
                     f"Ens={r['sc_ens_raw']:.4f}  CF={r['fold_cf']:.2f}  "
                     f"w=[LGBM={r['w_lgbm']:.3f},LSTM={r['w_lstm']:.3f}]  "
                     f"final={r['sc_cal']:.4f}\n")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RUL Direct v1 — LOOCV (raw z-score features)", fontsize=12)
    axes = axes.flatten()
    for i, test_bid in enumerate(BEARINGS):
        r        = results[test_bid]
        obs      = r["obs_pts"]
        true_rul = [r["N"] - o for o in obs]
        _draw_rul_ax(axes[i], obs, true_rul, r, f"Bearing{test_bid}")

        fig_i, ax_i = plt.subplots(figsize=(9, 5))
        _draw_rul_ax(ax_i, obs, true_rul, r, f"Bearing{test_bid}")
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Bearing{test_bid}_RUL_direct_v1.png", dpi=150)
        plt.close(fig_i)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "loocv_predictions_direct_v1.png", dpi=150)
    plt.close()

    return results, mean_ens


# ══════════════════════════════════════════════════════════════════
# Test Inference
# ══════════════════════════════════════════════════════════════════
def run_test_inference(loocv_results: dict):
    print(f"\n{'='*65}")
    print("  Test inference — full train + start position estimation")
    print(f"{'='*65}")

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data = load_train_raw()
    test_raw = load_test_raw()

    norm_stats = compute_norm_stats(raw_data, BEARINGS)
    w_lgbm  = float(np.mean([loocv_results[b]["w_lgbm"]  for b in BEARINGS]))
    w_lstm  = 1.0 - w_lgbm
    cf_test = float(np.mean([loocv_results[b]["fold_cf"] for b in BEARINGS]))
    print(f"  Test weights: LGBM={w_lgbm:.3f}, LSTM={w_lstm:.3f}")
    print(f"  Test CF:      {cf_test:.2f}")

    print("  LGBM training (full train)...")
    lgbm_model = train_lgbm_raw(raw_data, norm_stats, BEARINGS)

    print("  LSTM training (full train, 5 seeds)...")
    X_all, y_all = [], []
    for b in BEARINGS:
        norm  = normalize_feat(raw_data[b]["feat"], raw_data[b]["regime"], norm_stats)
        rul_b = rul_labels(len(norm), b)
        X, y  = make_seqs_raw(norm, raw_data[b]["regime"], rul_b, SEQ_LENGTH, start_obs=0)
        X_all.append(X); y_all.append(y)
    X_train   = np.concatenate(X_all)
    y_train   = np.concatenate(y_all)
    rul_scale = float(max(EOL.values()))

    lstm_models = []
    for s in SEEDS:
        m = train_lstm_raw(X_train, y_train, rul_scale, s, device)
        lstm_models.append(m)
        print(f"    seed={s} done")

    summary_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"RUL Direct v1 — Test (cf={cf_test:.2f})", fontsize=12)
    axes = axes.flatten()

    for i, tid in enumerate(TEST_IDS):
        feat_t  = test_raw[tid]["feat"]
        cond_t  = test_raw[tid]["regime"]
        N       = len(feat_t)
        obs_pts = np.arange(SEQ_LENGTH, N)

        norm_t = normalize_feat(feat_t, cond_t, norm_stats)

        start_frac = estimate_start_frac_raw(norm_t[0], raw_data, BEARINGS, norm_stats)
        start_obs  = int(start_frac * MEAN_TRAIN_LIFE)
        print(f"\n  [Test{tid}] start_frac={start_frac:.3f}  "
              f"start_obs~{start_obs}cyc")

        # LGBM
        preds_lgbm = predict_lgbm_raw(lgbm_model, norm_t, cond_t,
                                       start_frac=start_frac)

        # LSTM
        X_test = []
        for j in range(N - SEQ_LENGTH):
            feat_seq = norm_t[j: j + SEQ_LENGTH]
            obs_frac = np.clip(
                (start_obs + j + np.arange(SEQ_LENGTH)) / MEAN_TRAIN_LIFE,
                0.0, 2.0).reshape(-1, 1)
            reg_seq = cond_t[j: j + SEQ_LENGTH].astype(float).reshape(-1, 1)
            X_test.append(np.concatenate([feat_seq, obs_frac, reg_seq], axis=1))
        Xt = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)

        all_lstm = []
        for m in lstm_models:
            m.eval()
            with torch.no_grad():
                all_lstm.append(
                    np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0))
        preds_lstm = np.median(all_lstm, axis=0)

        # Ensemble + CF
        preds_ens   = w_lgbm * preds_lgbm + w_lstm * preds_lstm
        preds_final = preds_ens * cf_test
        if tid in FLAT_IDS:
            preds_final = preds_final * CONSERVATIVE_FACTOR

        final_cyc = float(preds_final[-1])
        final_hr  = final_cyc * INTERVAL_SEC / 3600
        bias_note = f"  *conservative" if tid in FLAT_IDS else ""
        print(f"    RUL={final_hr:.2f}hr ({final_cyc:.1f}cyc)  "
              f"LGBM_last={preds_lgbm[-1]:.1f}  LSTM_last={preds_lstm[-1]:.1f}"
              f"{bias_note}")

        pd.DataFrame({
            "obs_cycle":    obs_pts,
            "preds_lgbm":  preds_lgbm,
            "preds_lstm":  preds_lstm,
            "preds_final": preds_final,
            "rul_hours":   preds_final * INTERVAL_SEC / 3600,
        }).to_csv(OUT_DIR / f"Test{tid}_RUL_direct_v1.csv", index=False)

        summary_rows.append({
            "test_id":          tid,
            "start_frac":       round(start_frac, 3),
            "start_obs_est":    start_obs,
            "final_rul_cycles": round(final_cyc, 2),
            "final_rul_hours":  round(final_hr, 2),
            "cf":               round(cf_test, 2),
            "w_lgbm":           round(w_lgbm, 3),
            "w_lstm":           round(w_lstm, 3),
        })

        _draw_test_ax(axes[i], obs_pts, preds_lgbm, preds_lstm,
                      preds_final, cf_test, start_frac, f"Test{tid}")

        fig_i, ax_i = plt.subplots(figsize=(9, 5))
        _draw_test_ax(ax_i, obs_pts, preds_lgbm, preds_lstm,
                      preds_final, cf_test, start_frac, f"Test{tid}")
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Test{tid}_RUL_direct_v1.png", dpi=150)
        plt.close(fig_i)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_predictions_direct_v1.png", dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary_direct_v1.csv", index=False)
    print(f"\n  Test summary:")
    print(df_sum.to_string(index=False))
    print(f"\n[Done] {OUT_DIR}")


if __name__ == "__main__":
    loocv_results, loocv_score = run_loocv()
    run_test_inference(loocv_results)
