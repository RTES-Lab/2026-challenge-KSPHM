"""
RUL Regime v1: LOO-correct RUL with Per-regime HI + Regime Flag
================================================================
HI 소스: SR/0603_v2/output/train/BearingX_HI.csv (HI, regime 컬럼)
          SR/0603_v2/output/test/TestX_HI.csv

Leakage 방지 체크:
  [HI]     - hi_loo_regime_v1.py에서 LOO로 생성 (이 코드에선 그대로 로드)
  [LGBM]   - 각 fold마다 test_bid 제외한 3개 bearing으로만 학습
  [LSTM]   - 각 fold마다 test_bid 제외한 3개 bearing으로만 학습
  [CF]     - fold별로 "나머지 3개 bearing의 LOO 예측"으로만 탐색 후 test_bid에 적용
  [앙상블] - 앙상블 가중치를 "나머지 3개 bearing의 LOO 점수 평균"으로 결정
  [정규화] - LSTM rul_scale은 train_bids의 레이블 max만 사용
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

# ── 경로 ──────────────────────────────────────────────────────────────────
SR_BASE  = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0603_regime_f2s2")
HI_TRAIN = SR_BASE / "output/train"
HI_TEST  = SR_BASE / "output/test"
OUT_DIR  = SR_BASE / "output/rul"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH      = 10
MEAN_TRAIN_LIFE = 116.5      # (126+114+89+137)/4, 고정값
INTERVAL_SEC    = 600
SEEDS           = [42, 7, 123, 0, 99]

# Ground-truth: 대회 제공 정보 (leakage 아님 — 평가 기준점)
NORMAL_UNTIL = {1: 89, 2: 92, 3: 62, 4: 78}
EOL          = {1: 126, 2: 114, 3: 89, 4: 137}

N_FEAT = 3   # [HI_norm, obs_frac, regime]


# ══════════════════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════════════════
def load_train_hi():
    data = {}
    for bid in BEARINGS:
        df = pd.read_csv(HI_TRAIN / f"Bearing{bid}_HI.csv")
        data[bid] = {
            "hi":     df["HI"].values.astype(float),
            "regime": df["regime"].values.astype(int),
        }
    return data


def load_test_hi():
    data = {}
    for tid in TEST_IDS:
        df = pd.read_csv(HI_TEST / f"Test{tid}_HI.csv")
        data[tid] = {
            "hi":     df["HI"].values.astype(float),
            "regime": df["regime"].values.astype(int),
        }
    return data


# ══════════════════════════════════════════════════════════════════
# RUL 레이블 (ground truth, leakage 없음)
# ══════════════════════════════════════════════════════════════════
def rul_labels(n_total: int, bid: int) -> np.ndarray:
    nu, eol = NORMAL_UNTIL[bid], EOL[bid]
    idx = np.arange(n_total)
    return np.where(idx <= nu, eol - nu, np.maximum(eol - idx, 0)).astype(float)


# ══════════════════════════════════════════════════════════════════
# 대회 채점 함수
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
# LGBM 피처 (+ regime 피처 추가)
# ══════════════════════════════════════════════════════════════════
def make_lgbm_features(hi_arr: np.ndarray, regime_arr: np.ndarray,
                        seq_len: int = SEQ_LENGTH):
    """
    피처: HI window (seq_len개) + slope, mean, std, max, last, delta,
          regime_current (현재 레짐), regime_frac (window 내 고속 비율)
    레이블: 잔여 window 수
    """
    feats, targets = [], []
    N = len(hi_arr)
    t = np.arange(seq_len, dtype=float)
    for i in range(seq_len, N):
        win    = hi_arr[i - seq_len: i]
        reg_w  = regime_arr[i - seq_len: i]
        slope  = float(np.polyfit(t, win, 1)[0])
        feats.append([
            *win,                       # HI 이력 10개
            slope,                      # HI 기울기
            float(win.mean()),
            float(win.std()),
            float(win.max()),
            float(win[-1]),
            float(win[-1] - win[0]),    # window 내 변화량
            float(regime_arr[i]),       # 현재 레짐 (0/1)
            float(reg_w.mean()),        # window 내 고속 비율
        ])
        targets.append(float(N - i))
    return np.array(feats), np.array(targets)


def lgbm_asym_obj(y_pred, dataset):
    y_true = dataset.get_label()
    diff   = y_true - y_pred
    w      = np.where(diff < 0, 2.5, 1.0)
    return -diff * w, np.ones_like(diff) * w


def train_lgbm(train_data: dict, train_bids: list) -> lgb.Booster:
    X_list, y_list = [], []
    for b in train_bids:
        x, y = make_lgbm_features(train_data[b]["hi"], train_data[b]["regime"])
        X_list.append(x)
        y_list.append(y)
    dtrain = lgb.Dataset(np.concatenate(X_list), label=np.concatenate(y_list))
    return lgb.train(
        {"num_leaves": 15, "learning_rate": 0.05,
         "min_child_samples": 5, "verbose": -1,
         "objective": lgbm_asym_obj},
        dtrain, num_boost_round=200,
    )


def predict_lgbm(model: lgb.Booster, hi_arr: np.ndarray,
                  regime_arr: np.ndarray) -> np.ndarray:
    X, _ = make_lgbm_features(hi_arr, regime_arr)
    return np.maximum(model.predict(X), 0.0)


# ══════════════════════════════════════════════════════════════════
# LSTM (N_FEAT=3: HI_norm, obs_frac, regime)
# ══════════════════════════════════════════════════════════════════
def make_seqs(hi_arr: np.ndarray, regime_arr: np.ndarray,
              rul_arr: np.ndarray, seq_len: int, start_obs: int = 0):
    """
    window-minmax 정규화된 HI + obs_frac + regime → 3-채널 시퀀스.
    정규화에 사용되는 min/max는 해당 window 내부 값만 사용 (leakage 없음).
    obs_frac은 MEAN_TRAIN_LIFE(고정 상수)로만 나눔 (leakage 없음).
    """
    n = len(hi_arr)
    X, y = [], []
    for i in range(n - seq_len):
        win      = hi_arr[i: i + seq_len].copy()
        w_min, w_max = win.min(), win.max()
        win_norm = (win - w_min) / (w_max - w_min + 1e-8)
        obs_frac = np.clip(
            (start_obs + i + np.arange(seq_len)) / MEAN_TRAIN_LIFE, 0.0, 2.0)
        reg_seq  = regime_arr[i: i + seq_len].astype(float)
        X.append(np.stack([win_norm, obs_frac, reg_seq], axis=1))
        y.append(float(rul_arr[i + seq_len]))
    return np.array(X), np.array(y)


class LSTMRegressor(nn.Module):
    def __init__(self, n_feat: int = N_FEAT, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=n_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               rul_scale: float, seed: int, device) -> LSTMRegressor:
    torch.manual_seed(seed)
    y_norm = y_train / rul_scale
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_norm,  dtype=torch.float32)
    n_val = max(1, int(len(Xt) * 0.1))
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


def build_lstm_preds(train_data: dict, train_bids: list, test_bid: int,
                     device) -> tuple:
    """
    train_bids로만 LSTM 학습, test_bid의 예측 반환.
    rul_scale: train_bids 레이블 max (test_bid 데이터 미사용).
    """
    X_list, y_list = [], []
    for b in train_bids:
        hi_b, reg_b = train_data[b]["hi"], train_data[b]["regime"]
        rul_b = rul_labels(len(hi_b), b)
        X, y  = make_seqs(hi_b, reg_b, rul_b, SEQ_LENGTH)
        X_list.append(X); y_list.append(y)
    X_train = np.concatenate(X_list)
    y_train = np.concatenate(y_list)
    rul_scale = float(y_train.max())   # train_bids 기반 → leakage 없음

    hi_t  = train_data[test_bid]["hi"]
    reg_t = train_data[test_bid]["regime"]
    rul_t = rul_labels(len(hi_t), test_bid)
    X_test, _ = make_seqs(hi_t, reg_t, rul_t, SEQ_LENGTH)
    Xt = torch.tensor(X_test, dtype=torch.float32).to(device)

    all_preds = []
    for s in SEEDS:
        m = train_lstm(X_train, y_train, rul_scale, s, device)
        m.eval()
        with torch.no_grad():
            all_preds.append(
                np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0))
    return np.median(all_preds, axis=0), rul_scale


# ══════════════════════════════════════════════════════════════════
# Leakage-free CF 탐색
# ══════════════════════════════════════════════════════════════════
def find_fold_cf(results: dict, test_bid: int,
                 pred_key: str = "ens_raw") -> tuple:
    """
    test_bid의 CF를 '나머지 3개 bearing의 LOO 예측 점수'로만 탐색.
    test_bid의 실제 성능은 CF 탐색에 전혀 사용하지 않음.
    """
    other_bids = [b for b in BEARINGS if b != test_bid]
    best_cf, best_sc = 1.0, -np.inf
    for cf in np.arange(0.65, 1.06, 0.01):
        sc_list = []
        for bid in other_bids:
            N   = results[bid]["N"]
            obs = results[bid]["obs_pts"]
            preds = [p * cf for p in results[bid][pred_key]]
            sc_list.append(avg_score(N, obs, preds))
        mean_sc = float(np.mean(sc_list))
        if mean_sc > best_sc:
            best_sc, best_cf = mean_sc, float(cf)
    return best_cf, best_sc


# ══════════════════════════════════════════════════════════════════
# LOO 앙상블 가중치 (leakage-free)
# ══════════════════════════════════════════════════════════════════
def fold_ensemble_weights(results: dict, test_bid: int) -> tuple:
    """
    test_bid의 앙상블 가중치를 '나머지 3개 bearing의 LOO 점수 평균'으로 결정.
    """
    other_bids = [b for b in BEARINGS if b != test_bid]
    sc_lgbm = float(np.mean([results[b]["sc_lgbm"] for b in other_bids]))
    sc_lstm  = float(np.mean([results[b]["sc_lstm"]  for b in other_bids]))
    total    = sc_lgbm + sc_lstm + 1e-12
    # clamp [0.1, 0.7] 동일하게 유지
    w_lgbm = float(np.clip(sc_lgbm / total, 0.1, 0.7))
    w_lstm  = 1.0 - w_lgbm
    return w_lgbm, w_lstm


# ══════════════════════════════════════════════════════════════════
# LOOCV 메인
# ══════════════════════════════════════════════════════════════════
def run_loocv():
    print("=" * 65)
    print("  RUL Regime v1 — LOO-correct LGBM + LSTM (regime features)")
    print("=" * 65)

    train_data = load_train_hi()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    results = {}

    # ── Step 1: 각 fold에서 raw 예측 수집 ─────────────────────────
    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        hi_t  = train_data[test_bid]["hi"]
        reg_t = train_data[test_bid]["regime"]
        N     = len(hi_t)
        obs_pts = np.arange(SEQ_LENGTH, N)

        print(f"[Fold B{test_bid}]  train: {train_bids}")

        # LGBM (train_bids만으로 학습)
        lgbm_model  = train_lgbm(train_data, train_bids)
        preds_lgbm  = predict_lgbm(lgbm_model, hi_t, reg_t)
        sc_lgbm     = avg_score(N, obs_pts, preds_lgbm)

        # LSTM (train_bids만으로 학습, rul_scale도 train_bids 기반)
        preds_lstm, _  = build_lstm_preds(train_data, train_bids, test_bid, device)
        sc_lstm         = avg_score(N, obs_pts, preds_lstm)

        print(f"  LGBM={sc_lgbm:.4f}  LSTM={sc_lstm:.4f}")

        results[test_bid] = {
            "N":          N,
            "obs_pts":    list(obs_pts),
            "preds_lgbm": list(preds_lgbm),
            "preds_lstm": list(preds_lstm),
            "sc_lgbm":    sc_lgbm,
            "sc_lstm":    sc_lstm,
        }

    # ── Step 2: leakage-free 앙상블 가중치 + raw 앙상블 ───────────
    for test_bid in BEARINGS:
        w_lgbm, w_lstm = fold_ensemble_weights(results, test_bid)
        preds_ens = [
            w_lgbm * l + w_lstm * a
            for l, a in zip(results[test_bid]["preds_lgbm"],
                             results[test_bid]["preds_lstm"])
        ]
        results[test_bid]["ens_raw"]  = preds_ens
        results[test_bid]["w_lgbm"]   = w_lgbm
        results[test_bid]["w_lstm"]   = w_lstm
        results[test_bid]["sc_ens_raw"] = avg_score(
            results[test_bid]["N"], results[test_bid]["obs_pts"], preds_ens)

    # ── Step 3: leakage-free CF 탐색 (fold별) ─────────────────────
    print("\n[Fold-wise CF 탐색 (나머지 3개 bearing 기준)]")
    for test_bid in BEARINGS:
        cf, cf_ref_sc = find_fold_cf(results, test_bid, pred_key="ens_raw")
        preds_cal = [p * cf for p in results[test_bid]["ens_raw"]]
        sc_cal    = avg_score(
            results[test_bid]["N"], results[test_bid]["obs_pts"], preds_cal)
        results[test_bid]["fold_cf"]      = cf
        results[test_bid]["preds_cal"]    = preds_cal
        results[test_bid]["sc_cal"]       = sc_cal
        results[test_bid]["cf_ref_score"] = cf_ref_sc
        print(f"  B{test_bid}: cf={cf:.2f} (ref_sc={cf_ref_sc:.4f}) → test_sc={sc_cal:.4f}")

    # ── Step 4: LOOCV 요약 출력 ────────────────────────────────────
    print(f"\n{'='*65}")
    print("  LOOCV 결과 요약")
    print(f"{'='*65}")
    print(f"  {'Bearing':>8} | {'LGBM':>8} | {'LSTM':>8} | {'Ens(raw)':>10} | {'CF':>5} | {'Ens(CF)':>9}")
    print(f"  {'-'*62}")

    sc_lgbm_list, sc_lstm_list, sc_ens_list, sc_cal_list = [], [], [], []
    for test_bid in BEARINGS:
        r = results[test_bid]
        print(f"  {test_bid:>8} | {r['sc_lgbm']:>8.4f} | {r['sc_lstm']:>8.4f} | "
              f"{r['sc_ens_raw']:>10.4f} | {r['fold_cf']:>5.2f} | {r['sc_cal']:>9.4f}")
        sc_lgbm_list.append(r['sc_lgbm'])
        sc_lstm_list.append(r['sc_lstm'])
        sc_ens_list.append(r['sc_ens_raw'])
        sc_cal_list.append(r['sc_cal'])

    print(f"  {'평균':>8} | {np.mean(sc_lgbm_list):>8.4f} | {np.mean(sc_lstm_list):>8.4f} | "
          f"{np.mean(sc_ens_list):>10.4f} | {'—':>5} | {np.mean(sc_cal_list):>9.4f}")
    print(f"\n  ★ LOOCV 최종 (앙상블 + fold CF): {np.mean(sc_cal_list):.4f}")

    # ── 로그 저장 ──────────────────────────────────────────────────
    with open(OUT_DIR / "loocv_log.txt", "w") as f:
        f.write("RUL Regime v1 — LOO-correct LGBM + LSTM\n")
        f.write(f"LGBM  mean: {np.mean(sc_lgbm_list):.4f}\n")
        f.write(f"LSTM  mean: {np.mean(sc_lstm_list):.4f}\n")
        f.write(f"Ens (raw): {np.mean(sc_ens_list):.4f}\n")
        f.write(f"Ens + CF : {np.mean(sc_cal_list):.4f}\n\n")
        f.write("Fold 상세:\n")
        for test_bid in BEARINGS:
            r = results[test_bid]
            f.write(f"  B{test_bid}: LGBM={r['sc_lgbm']:.4f}  LSTM={r['sc_lstm']:.4f}  "
                    f"Ens={r['sc_ens_raw']:.4f}  CF={r['fold_cf']:.2f}  "
                    f"w=[LGBM={r['w_lgbm']:.3f},LSTM={r['w_lstm']:.3f}]  "
                    f"final={r['sc_cal']:.4f}\n")

    # ── 시각화 ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RUL Regime v1 — LOOCV (LGBM + LSTM + regime)", fontsize=12)
    axes = axes.flatten()
    for i, test_bid in enumerate(BEARINGS):
        ax  = axes[i]
        r   = results[test_bid]
        obs = r["obs_pts"]
        N   = r["N"]
        true_rul = [N - o for o in obs]
        ax.plot(obs, true_rul,          "k-",  lw=1.5, label="True RUL")
        ax.plot(obs, r["preds_lgbm"],   "r--", lw=1,   alpha=0.6, label=f"LGBM {r['sc_lgbm']:.3f}")
        ax.plot(obs, r["preds_lstm"],   "b--", lw=1,   alpha=0.6, label=f"LSTM {r['sc_lstm']:.3f}")
        ax.plot(obs, r["preds_cal"],    "m-",  lw=2,   label=f"Ens+CF {r['sc_cal']:.3f}")
        ax.set_title(f"Bearing{test_bid}  [w_lgbm={r['w_lgbm']:.2f}  cf={r['fold_cf']:.2f}]")
        ax.set_xlabel("Obs Cycle")
        ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loocv_predictions.png", dpi=150)
    plt.close()

    return results, np.mean(sc_cal_list)


# ══════════════════════════════════════════════════════════════════
# Test 추론
# ══════════════════════════════════════════════════════════════════
def run_test_inference(loocv_results: dict):
    print(f"\n{'='*65}")
    print("  Test 추론 — 전체 Train 4개로 학습")
    print(f"{'='*65}")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = load_train_hi()
    test_data  = load_test_hi()

    # Test용 앙상블 가중치: 4개 fold의 w 평균
    w_lgbm = float(np.mean([loocv_results[b]["w_lgbm"] for b in BEARINGS]))
    w_lstm  = 1.0 - w_lgbm
    # Test용 CF: 4개 fold의 fold_cf 평균
    cf_test = float(np.mean([loocv_results[b]["fold_cf"] for b in BEARINGS]))
    print(f"  Test 가중치: LGBM={w_lgbm:.3f}, LSTM={w_lstm:.3f}")
    print(f"  Test CF:     {cf_test:.2f}")

    # Train 전체로 LGBM 학습
    print("  LGBM 학습 (전체 Train)...")
    lgbm_model = train_lgbm(train_data, BEARINGS)

    # Train 전체로 LSTM 학습
    print("  LSTM 학습 (전체 Train, 5 seeds)...")
    X_all, y_all = [], []
    for b in BEARINGS:
        hi_b  = train_data[b]["hi"]
        reg_b = train_data[b]["regime"]
        rul_b = rul_labels(len(hi_b), b)
        X, y  = make_seqs(hi_b, reg_b, rul_b, SEQ_LENGTH)
        X_all.append(X); y_all.append(y)
    X_train  = np.concatenate(X_all)
    y_train  = np.concatenate(y_all)
    rul_scale = float(y_train.max())

    lstm_models = []
    for s in SEEDS:
        m = train_lstm(X_train, y_train, rul_scale, s, device)
        lstm_models.append(m)
        print(f"    seed={s} done")

    # Test 추론
    summary_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"RUL Regime v1 — Test (cf={cf_test:.2f})", fontsize=12)
    axes = axes.flatten()

    for i, tid in enumerate(TEST_IDS):
        hi_t  = test_data[tid]["hi"]
        reg_t = test_data[tid]["regime"]
        N     = len(hi_t)
        obs_pts = np.arange(SEQ_LENGTH, N)

        # LGBM
        preds_lgbm = predict_lgbm(lgbm_model, hi_t, reg_t)

        # LSTM
        X_test = []
        for j in range(N - SEQ_LENGTH):
            win  = hi_t[j: j + SEQ_LENGTH].copy()
            wmin, wmax = win.min(), win.max()
            wn   = (win - wmin) / (wmax - wmin + 1e-8)
            of   = np.clip((j + np.arange(SEQ_LENGTH)) / MEAN_TRAIN_LIFE, 0.0, 2.0)
            rg   = reg_t[j: j + SEQ_LENGTH].astype(float)
            X_test.append(np.stack([wn, of, rg], axis=1))
        Xt = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)
        all_lstm_preds = []
        for m in lstm_models:
            m.eval()
            with torch.no_grad():
                all_lstm_preds.append(
                    np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0))
        preds_lstm = np.median(all_lstm_preds, axis=0)

        # 앙상블 + CF
        preds_final = (w_lgbm * preds_lgbm + w_lstm * preds_lstm) * cf_test

        final_rul_cyc = float(preds_final[-1])
        final_rul_hr  = final_rul_cyc * INTERVAL_SEC / 3600
        print(f"  [Test{tid}] RUL={final_rul_hr:.2f}hr ({final_rul_cyc:.1f}cyc)  "
              f"regime_frac_high={reg_t.mean():.2f}")

        pd.DataFrame({
            "obs_cycle":    obs_pts,
            "preds_lgbm":  preds_lgbm,
            "preds_lstm":  preds_lstm,
            "preds_final": preds_final,
            "rul_hours":   preds_final * INTERVAL_SEC / 3600,
        }).to_csv(OUT_DIR / f"Test{tid}_RUL.csv", index=False)

        summary_rows.append({
            "test_id":         tid,
            "final_rul_cycles": round(final_rul_cyc, 2),
            "final_rul_hours":  round(final_rul_hr, 2),
            "cf":               round(cf_test, 2),
            "w_lgbm":           round(w_lgbm, 3),
            "w_lstm":           round(w_lstm, 3),
        })

        ax = axes[i]
        ax.plot(obs_pts, preds_lgbm,  "r--", lw=1, alpha=0.6, label="LGBM")
        ax.plot(obs_pts, preds_lstm,  "b--", lw=1, alpha=0.6, label="LSTM")
        ax.plot(obs_pts, preds_final, "m-",  lw=2, label=f"Final (cf={cf_test:.2f})")
        ax.set_title(f"Test{tid}  RUL={final_rul_hr:.1f}hr")
        ax.set_xlabel("Obs Cycle")
        ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_predictions.png", dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary.csv", index=False)
    print(f"\n  Test 요약:")
    print(df_sum.to_string(index=False))
    print(f"\n[완료] {OUT_DIR}")


# ── 메인 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loocv_results, loocv_score = run_loocv()
    run_test_inference(loocv_results)
