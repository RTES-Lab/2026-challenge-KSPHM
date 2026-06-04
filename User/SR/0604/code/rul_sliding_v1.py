"""
RUL Sliding Window LOOCV v1 (0604)
===================================
현재 LOOCV 문제:
  - 전체 수명(126/114/89/137사이클)을 한 번에 평가
  - 실제 test는 50사이클 창만 보유, 임의 지점에서 시작
  → validation이 test 시나리오를 반영하지 못함

개선:
  - 50사이클 슬라이딩 창으로 held-out 베어링 평가
  - 각 창의 시작 HI에서 lifecycle 위치 추정 (estimate_start_frac)
  - 창 끝 단일 예측 vs 실제 잔여 수명으로 채점
  → validation이 실제 test 시나리오와 동일한 구조

HI 소스: 0603_v3/output/train/BearingX_HI.csv (FDR, 검증된 최고 HI)
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
BASE     = Path("/data/home/ksphm/2026-challenge-KSPHM")
HI_TRAIN = BASE / "User/SR/0603_v3/output/train"
HI_TEST  = BASE / "User/SR/0603_v3/output/test"
OUT_DIR  = BASE / "User/SR/0604/output/rul_sliding"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
WIN_SIZE        = 50        # 실제 test 창 크기
SEQ_LEN         = 10        # LGBM/LSTM look-back
STRIDE          = 1         # 창 이동 간격
MEAN_TRAIN_LIFE = 116.5
INTERVAL_SEC    = 600
SEEDS           = [42, 7, 123, 0, 99]
N_FEAT          = 3         # [hi_norm, obs_frac, regime]

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
    return np.where(idx <= nu, eol - nu, np.maximum(eol - idx, 0)).astype(float)

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
# Lifecycle position estimation
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
# LGBM
# ══════════════════════════════════════════════════════════════════
def make_lgbm_features(hi_arr, regime_arr, start_frac=0.0):
    feats, targets = [], []
    N = len(hi_arr)
    t = np.arange(SEQ_LEN, dtype=float)
    hi_obs_start = float(hi_arr[0])
    for i in range(SEQ_LEN, N):
        win          = hi_arr[i - SEQ_LEN: i]
        slope        = float(np.polyfit(t, win, 1)[0])
        elapsed      = np.clip(start_frac + float(i) / MEAN_TRAIN_LIFE, 0.0, 3.0)
        hi_slope_5   = float((win[-1] - win[-6]) / 5)
        hi_slope_10  = float((win[-1] - win[0]) / (SEQ_LEN - 1))
        hi_delta     = float(win[-1] - hi_obs_start)
        feats.append([*win, slope, float(win.mean()), float(win.std()),
                      float(win.max()), float(win[-1]), float(win[-1]-win[0]),
                      float(regime_arr[i]), float(regime_arr[i-SEQ_LEN:i].mean()),
                      elapsed, hi_slope_5, hi_slope_10, hi_delta])
        targets.append(float(N - i))
    return np.array(feats), np.array(targets)

def lgbm_asym_obj(y_pred, dataset):
    y_true = dataset.get_label()
    diff   = y_true - y_pred
    w      = np.where(diff < 0, 2.5, 1.0)
    return -diff * w, np.ones_like(diff) * w

def train_lgbm(train_data, train_bids):
    X_list, y_list = [], []
    for b in train_bids:
        x, y = make_lgbm_features(train_data[b]["hi"], train_data[b]["regime"],
                                    start_frac=0.0)
        X_list.append(x); y_list.append(y)
    dtrain = lgb.Dataset(np.concatenate(X_list), label=np.concatenate(y_list))
    return lgb.train(
        {"num_leaves": 15, "learning_rate": 0.05,
         "min_child_samples": 5, "verbose": -1,
         "objective": lgbm_asym_obj},
        dtrain, num_boost_round=200)

def predict_lgbm_window(model, hi_window, reg_window, start_frac):
    """50사이클 창에 대해 LGBM 예측 → 창 끝 예측값 반환."""
    X, _ = make_lgbm_features(hi_window, reg_window, start_frac=start_frac)
    if len(X) == 0:
        return 0.0
    preds = np.maximum(model.predict(X), 0.0)
    return float(preds[-1])   # 창 끝 예측


# ══════════════════════════════════════════════════════════════════
# LSTM
# ══════════════════════════════════════════════════════════════════
class LSTMRegressor(nn.Module):
    def __init__(self, n_feat=N_FEAT, hidden=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=n_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)

def train_lstm_fold(train_data, train_bids, hi_mean, hi_std, rul_scale, device):
    X_list, y_list = [], []
    for b in train_bids:
        hi_b  = train_data[b]["hi"]
        reg_b = train_data[b]["regime"]
        rul_b = rul_labels(len(hi_b), b)
        n = len(hi_b)
        for i in range(n - SEQ_LEN):
            win      = hi_b[i:i+SEQ_LEN]
            win_norm = (win - hi_mean) / hi_std
            obs_frac = np.clip((i + np.arange(SEQ_LEN)) / MEAN_TRAIN_LIFE, 0.0, 2.0)
            rg       = reg_b[i:i+SEQ_LEN].astype(float)
            X_list.append(np.stack([win_norm, obs_frac, rg], axis=1))
            y_list.append(float(rul_b[i + SEQ_LEN]))
    X_train = np.array(X_list)
    y_train = np.array(y_list)

    models = []
    for s in SEEDS:
        torch.manual_seed(s)
        y_norm = y_train / rul_scale
        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_norm,  dtype=torch.float32)
        n_val = max(1, int(len(Xt) * 0.1))
        tr_dl = DataLoader(TensorDataset(Xt[:-n_val], yt[:-n_val]),
                           batch_size=64, shuffle=True)
        val_dl= DataLoader(TensorDataset(Xt[-n_val:], yt[-n_val:]), batch_size=64)
        model = LSTMRegressor().to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit  = nn.MSELoss()
        best_val, patience, best_state = np.inf, 0, None
        for _ in range(200):
            model.train()
            for xb, yb in tr_dl:
                opt.zero_grad(); crit(model(xb.to(device)), yb.to(device)).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([crit(model(xb.to(device)), yb.to(device)).item()
                                    for xb, yb in val_dl]))
            if vl < best_val:
                best_val, patience = vl, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 20: break
        model.load_state_dict(best_state)
        models.append(model)
    return models

def predict_lstm_window(models, hi_window, reg_window, start_frac,
                         hi_mean, hi_std, rul_scale, device):
    """50사이클 창 끝의 LSTM 예측값 반환."""
    start_obs = int(start_frac * MEAN_TRAIN_LIFE)
    n = len(hi_window)
    X_win = []
    for j in range(n - SEQ_LEN):
        win_norm = (hi_window[j:j+SEQ_LEN] - hi_mean) / hi_std
        obs_frac = np.clip((start_obs + j + np.arange(SEQ_LEN)) / MEAN_TRAIN_LIFE,
                            0.0, 2.0)
        rg = reg_window[j:j+SEQ_LEN].astype(float)
        X_win.append(np.stack([win_norm, obs_frac, rg], axis=1))
    if not X_win:
        return 0.0
    Xt = torch.tensor(np.array(X_win), dtype=torch.float32).to(device)
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            p = np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0)
        preds.append(p)
    return float(np.median(preds, axis=0)[-1])   # 창 끝 예측


# ══════════════════════════════════════════════════════════════════
# Sliding Window LOOCV
# ══════════════════════════════════════════════════════════════════
def run_sliding_loocv():
    print("=" * 70)
    print("  RUL Sliding Window LOOCV v1 (0604)")
    print(f"  Window={WIN_SIZE}cyc  Stride={STRIDE}  SEQ_LEN={SEQ_LEN}")
    print("=" * 70)

    train_data = load_train_hi()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    all_results = {}   # bid → {per-window scores, aggregates}

    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        hi_t  = train_data[test_bid]["hi"]
        reg_t = train_data[test_bid]["regime"]
        N     = len(hi_t)
        rul_t = rul_labels(N, test_bid)

        hi_mean, hi_std = compute_hi_stats(train_data, train_bids)
        rul_scale       = float(max(EOL[b] for b in train_bids))

        print(f"[Fold B{test_bid}]  train={train_bids}  N={N}"
              f"  windows={(N - WIN_SIZE) // STRIDE + 1}")

        # 모델 학습 (fold당 1회)
        print("  LGBM 학습...")
        lgbm_model = train_lgbm(train_data, train_bids)
        print("  LSTM 학습...")
        lstm_models = train_lstm_fold(train_data, train_bids,
                                       hi_mean, hi_std, rul_scale, device)

        # 슬라이딩 창 평가
        window_starts   = []
        true_ruls       = []
        preds_lgbm_list = []
        preds_lstm_list = []
        preds_ens_list  = []
        start_fracs     = []

        t_starts = range(0, N - WIN_SIZE + 1, STRIDE)
        for t_start in t_starts:
            hi_win  = hi_t[t_start : t_start + WIN_SIZE]
            reg_win = reg_t[t_start : t_start + WIN_SIZE]

            # 실제 RUL: 창 끝(t_start+WIN_SIZE-1) 시점의 잔여 수명
            true_rul = float(rul_t[t_start + WIN_SIZE - 1])

            # 시작 위치 추정
            sf = estimate_start_frac(float(hi_win[0]), train_data, train_bids)

            # 예측
            pred_l = predict_lgbm_window(lgbm_model, hi_win, reg_win, sf)
            pred_s = predict_lstm_window(lstm_models, hi_win, reg_win,
                                          sf, hi_mean, hi_std, rul_scale, device)
            pred_e = 0.5 * pred_l + 0.5 * pred_s   # 단순 평균 앙상블

            window_starts.append(t_start)
            true_ruls.append(true_rul)
            preds_lgbm_list.append(pred_l)
            preds_lstm_list.append(pred_s)
            preds_ens_list.append(pred_e)
            start_fracs.append(sf)

        sc_lgbm = avg_score(true_ruls, preds_lgbm_list)
        sc_lstm = avg_score(true_ruls, preds_lstm_list)
        sc_ens  = avg_score(true_ruls, preds_ens_list)

        print(f"  LGBM={sc_lgbm:.4f}  LSTM={sc_lstm:.4f}  Ens={sc_ens:.4f}")

        all_results[test_bid] = {
            "N":              N,
            "window_starts":  window_starts,
            "true_ruls":      true_ruls,
            "preds_lgbm":     preds_lgbm_list,
            "preds_lstm":     preds_lstm_list,
            "preds_ens":      preds_ens_list,
            "start_fracs":    start_fracs,
            "sc_lgbm":        sc_lgbm,
            "sc_lstm":        sc_lstm,
            "sc_ens":         sc_ens,
        }

        # ── 개별 plot ────────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes[0].plot(window_starts, true_ruls, "k-", lw=1.5, label="True RUL")
        axes[0].plot(window_starts, preds_lgbm_list, "r--", lw=1,
                     alpha=0.7, label=f"LGBM {sc_lgbm:.3f}")
        axes[0].plot(window_starts, preds_lstm_list, "b--", lw=1,
                     alpha=0.7, label=f"LSTM {sc_lstm:.3f}")
        axes[0].plot(window_starts, preds_ens_list, "m-", lw=2,
                     label=f"Ensemble {sc_ens:.3f}")
        axes[0].set_xlabel("Window Start (cycle)")
        axes[0].set_ylabel("RUL (cycles)")
        axes[0].set_title(f"Bearing{test_bid} — Sliding Window RUL (WIN={WIN_SIZE})")
        axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

        axes[1].plot(window_starts, [sf * 100 for sf in start_fracs],
                     "g-", lw=1.5, label="Estimated start %")
        actual_fracs = [(t + WIN_SIZE - 1) / N * 100 for t in window_starts]
        axes[1].plot(window_starts, actual_fracs, "k--", lw=1, label="Actual window end %")
        axes[1].set_xlabel("Window Start (cycle)")
        axes[1].set_ylabel("Lifecycle %")
        axes[1].set_title("Start Position Estimation")
        axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"Bearing{test_bid}_sliding.png", dpi=150)
        plt.close()

    # ── 전체 요약 ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Sliding Window LOOCV Summary")
    print(f"{'='*70}")
    print(f"  {'Bearing':>8} | {'LGBM':>8} | {'LSTM':>8} | {'Ensemble':>10} | {'N windows':>10}")
    print(f"  {'-'*55}")

    sc_lgbm_all, sc_lstm_all, sc_ens_all = [], [], []
    for bid in BEARINGS:
        r = all_results[bid]
        n_win = len(r["window_starts"])
        print(f"  {bid:>8} | {r['sc_lgbm']:>8.4f} | {r['sc_lstm']:>8.4f} | "
              f"{r['sc_ens']:>10.4f} | {n_win:>10}")
        sc_lgbm_all.append(r["sc_lgbm"])
        sc_lstm_all.append(r["sc_lstm"])
        sc_ens_all.append(r["sc_ens"])

    mean_lgbm = float(np.nanmean(sc_lgbm_all))
    mean_lstm  = float(np.nanmean(sc_lstm_all))
    mean_ens   = float(np.nanmean(sc_ens_all))
    print(f"  {'avg':>8} | {mean_lgbm:>8.4f} | {mean_lstm:>8.4f} | {mean_ens:>10.4f}")
    print(f"\n  ★ Sliding LOOCV Ensemble: {mean_ens:.4f}")
    print(f"    (v4 전체 수명 LOOCV Ens_raw: 0.466 — 직접 비교 불가, 다른 평가 구조)")

    # v4와의 베어링별 LGBM 비교 (LGBM은 모델 동일, 평가 방식만 다름)
    v4_lgbm = {1: 0.490, 2: 0.550, 3: 0.136, 4: 0.622}
    print(f"\n  LGBM 베어링별 비교 (v4=전체수명 / sliding=50창):")
    for bid in BEARINGS:
        print(f"    B{bid}: v4={v4_lgbm[bid]:.3f}  sliding={all_results[bid]['sc_lgbm']:.3f}")

    # 통합 log
    with open(OUT_DIR / "loocv_sliding_log.txt", "w") as f:
        f.write(f"Sliding Window LOOCV (WIN={WIN_SIZE}, STRIDE={STRIDE})\n\n")
        f.write(f"{'Bearing':>8} | {'LGBM':>8} | {'LSTM':>8} | {'Ens':>8}\n")
        for bid in BEARINGS:
            r = all_results[bid]
            f.write(f"{bid:>8} | {r['sc_lgbm']:>8.4f} | {r['sc_lstm']:>8.4f} | "
                    f"{r['sc_ens']:>8.4f}\n")
        f.write(f"{'avg':>8} | {mean_lgbm:>8.4f} | {mean_lstm:>8.4f} | {mean_ens:>8.4f}\n")

    return all_results, mean_ens


# ══════════════════════════════════════════════════════════════════
# Test inference (50사이클 창 = test 시나리오 그대로)
# ══════════════════════════════════════════════════════════════════
def run_test_inference():
    print(f"\n{'='*70}")
    print("  Test Inference (전체 Train 4개, 50사이클 창)")
    print(f"{'='*70}")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = load_train_hi()
    test_data  = load_test_hi()

    hi_mean, hi_std = compute_hi_stats(train_data, BEARINGS)
    rul_scale       = float(max(EOL.values()))
    print(f"  Full train HI: mean={hi_mean:.4f}  std={hi_std:.4f}")

    print("  LGBM 학습 (full)...")
    lgbm_model  = train_lgbm(train_data, BEARINGS)
    print("  LSTM 학습 (full)...")
    lstm_models = train_lstm_fold(train_data, BEARINGS,
                                   hi_mean, hi_std, rul_scale, device)

    summary_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Test RUL — Sliding Window (0604)", fontsize=12)
    axes = axes.flatten()

    for i, tid in enumerate(TEST_IDS):
        hi_t    = test_data[tid]["hi"]
        reg_t   = test_data[tid]["regime"]
        hi_start = float(hi_t[0])

        sf = estimate_start_frac(hi_start, train_data, BEARINGS)
        print(f"\n  [Test{tid}] hi_start={hi_start:.3f}  start_frac={sf:.3f}"
              f"  (≈{int(sf*100)}% lifecycle)")

        # 전체 창(50사이클 전체)에 대해 LGBM/LSTM
        n = len(hi_t)
        X_lgbm, _ = make_lgbm_features(hi_t, reg_t, start_frac=sf)
        preds_lgbm = np.maximum(lgbm_model.predict(X_lgbm), 0.0)

        # LSTM — 전체 50사이클 창
        start_obs = int(sf * MEAN_TRAIN_LIFE)
        X_lstm = []
        for j in range(n - SEQ_LEN):
            win_norm = (hi_t[j:j+SEQ_LEN] - hi_mean) / hi_std
            obs_frac = np.clip((start_obs + j + np.arange(SEQ_LEN)) / MEAN_TRAIN_LIFE,
                                0.0, 2.0)
            rg = reg_t[j:j+SEQ_LEN].astype(float)
            X_lstm.append(np.stack([win_norm, obs_frac, rg], axis=1))
        Xt = torch.tensor(np.array(X_lstm), dtype=torch.float32).to(device)
        all_lstm = []
        for m in lstm_models:
            m.eval()
            with torch.no_grad():
                all_lstm.append(np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0))
        preds_lstm = np.median(all_lstm, axis=0)

        preds_ens = 0.5 * preds_lgbm + 0.5 * preds_lstm

        final_cyc = float(preds_ens[-1])
        final_hr  = final_cyc * INTERVAL_SEC / 3600
        print(f"    RUL={final_hr:.2f}hr ({final_cyc:.1f}cyc)")

        obs_pts = np.arange(SEQ_LEN, n)
        ax = axes[i]
        ax.plot(obs_pts, preds_lgbm, "r--", lw=1, alpha=0.7, label="LGBM")
        ax.plot(obs_pts, preds_lstm, "b--", lw=1, alpha=0.7, label="LSTM")
        ax.plot(obs_pts, preds_ens,  "m-",  lw=2,
                label=f"Ens  RUL={final_hr:.1f}hr")
        ax.set_title(f"Test{tid}  (start≈{int(sf*100)}%)", fontsize=10)
        ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        pd.DataFrame({
            "obs_cycle": obs_pts, "preds_lgbm": preds_lgbm,
            "preds_lstm": preds_lstm, "preds_ens": preds_ens,
            "rul_hours": preds_ens * INTERVAL_SEC / 3600,
        }).to_csv(OUT_DIR / f"Test{tid}_RUL.csv", index=False)

        summary_rows.append({
            "test_id": tid, "hi_start": round(hi_start, 4),
            "start_frac": round(sf, 3),
            "final_rul_cycles": round(final_cyc, 2),
            "final_rul_hours":  round(final_hr, 2),
        })

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_predictions.png", dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary.csv", index=False)
    print(f"\n  Test summary:")
    print(df_sum.to_string(index=False))


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    results, score = run_sliding_loocv()
    run_test_inference()
