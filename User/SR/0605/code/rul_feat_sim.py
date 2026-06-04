"""
RUL Feat-Sim v1: rul_regime_v5 + feature-similarity start_frac 추정
================================================================
핵심 변경: estimate_start_frac(hi_start) → estimate_start_frac_feat_sim(raw features)

기존 문제:
  - hi_start 기반 매칭: B3(최대 HI=0.14)는 hi_start≥0.326인 Test5/6에서 제외
    B4(초기 HI=0.55)는 hi[0]>=hi_start 조건으로 제외
  → B1, B2 두 개 베어링만 참조 → 편향된 start_frac

새 접근:
  - Test 베어링의 초기 N관측 raw feature vector를 훈련 베어링 궤적과 비교
  - 레짐별 z-score 정규화 후 Euclidean 최근접이웃 → 수명 비율 추정
  - B3/B4 제외 없이 4개 베어링 모두 참조

나머지(LGBM+LSTM 앙상블, CF, 가중치)는 v5와 동일.
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
SR_BASE  = BASE / "User/SR/0603_v3"
HI_TRAIN = SR_BASE / "output/train_v4"
HI_TEST  = SR_BASE / "output/test_v4"
OUT_DIR  = BASE / "User/SR/0605/output/rul_feat_sim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature cache (0604: includes ch1/ch2 impulse, raw energy → apply log1p)
FEAT_TRAIN_DIR = BASE / "User/SR/0604/output/train"
FEAT_TEST_DIR  = BASE / "User/SR/0604/output/test"
LOG_FEATS = ["ch3_total_power", "ch3_energy", "ch3_rms"]
SIM_FEATS = ["ch3_high_band", "ch4_high_band",
             "ch3_total_power", "ch3_energy", "ch3_rms",
             "ch1_kurt_log", "ch2_kurt_log", "ch1_crest", "ch2_crest",
             "ch3_std", "ch3_p2p"]
SIM_N_OBS = 10   # first N observations for test feature vector

BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH      = 10
MEAN_TRAIN_LIFE = 116.5
INTERVAL_SEC    = 600
SEEDS           = [42, 7, 123, 0, 99]

NORMAL_UNTIL = {1: 89, 2: 92, 3: 62, 4: 78}
EOL          = {1: 126, 2: 114, 3: 89, 4: 137}

N_FEAT = 3  # [hi_norm, obs_frac, regime]


# ══════════════════════════════════════════════════════════════════
# Data Loading
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


def rul_labels(n_total: int, bid: int) -> np.ndarray:
    nu, eol = NORMAL_UNTIL[bid], EOL[bid]
    idx = np.arange(n_total)
    return np.where(idx <= nu, eol - nu, np.maximum(eol - idx, 0)).astype(float)


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
# 시작 위치 추정 (이미 열화된 test 데이터 대응)
# ══════════════════════════════════════════════════════════════════
def estimate_start_frac(hi_start: float, train_data: dict,
                         ref_bids: list) -> float:
    """기존 HI 기반 추정 (레거시, 비교용)."""
    fracs = []
    for bid in ref_bids:
        hi = train_data[bid]["hi"]
        n  = len(hi)
        if hi[0] >= hi_start:
            continue
        exceed = np.where(hi >= hi_start)[0]
        if len(exceed) == 0:
            fracs.append(float(n) / MEAN_TRAIN_LIFE)
        else:
            fracs.append(float(exceed[0]) / MEAN_TRAIN_LIFE)
    return float(np.mean(fracs)) if fracs else 0.0


# ══════════════════════════════════════════════════════════════════
# Feature-similarity start_frac (신규)
# ══════════════════════════════════════════════════════════════════
def _load_feat_df(path: Path) -> pd.DataFrame:
    """0604 cache 로드 후 energy features에 log1p 적용."""
    df = pd.read_csv(path)
    for f in LOG_FEATS:
        if f in df.columns:
            df[f] = np.log1p(df[f])
    return df


def load_train_feat_cache() -> dict:
    return {bid: _load_feat_df(FEAT_TRAIN_DIR / f"Bearing{bid}_features_raw.csv")
            for bid in BEARINGS}


def load_test_feat_cache() -> dict:
    return {tid: _load_feat_df(FEAT_TEST_DIR / f"Test{tid}_features_raw.csv")
            for tid in TEST_IDS}


def estimate_start_frac_feat_sim(test_feat_df: pd.DataFrame,
                                  test_regime: np.ndarray,
                                  train_feat_dfs: dict,
                                  train_hi_data: dict,
                                  n_obs: int = SIM_N_OBS) -> float:
    """
    Test 베어링 초기 n_obs 관측의 raw feature vector를
    훈련 베어링 궤적에서 최근접이웃 탐색 → 수명 비율 추정.

    레짐별 z-score 정규화 적용:
      - B4처럼 baseline이 다른 베어링도 올바르게 비교 가능
      - B3(max HI=0.14), B4(hi_start=0.55)의 HI 기반 제외 문제 없음

    Returns: start_frac (0~1+)
    """
    fracs = []

    for regime in [0, 1]:
        # Test 초기 관측 중 해당 레짐 인덱스
        all_idx = np.where(test_regime == regime)[0]
        use_idx = all_idx[:n_obs]
        if len(use_idx) < 2:
            continue

        test_vec = test_feat_df[SIM_FEATS].values[use_idx].mean(axis=0)

        # 훈련 전체(해당 레짐)로 정규화 통계 계산
        train_regime_feats = []
        for bid, df in train_feat_dfs.items():
            cond = train_hi_data[bid]["regime"]
            idx_r = np.where(cond == regime)[0]
            if len(idx_r) > 0:
                train_regime_feats.append(df[SIM_FEATS].values[idx_r])
        if not train_regime_feats:
            continue
        all_r = np.concatenate(train_regime_feats, axis=0)
        feat_mean = all_r.mean(axis=0)
        feat_std  = all_r.std(axis=0) + 1e-8

        test_norm = (test_vec - feat_mean) / feat_std

        for bid, df in train_feat_dfs.items():
            cond  = train_hi_data[bid]["regime"]
            idx_r = np.where(cond == regime)[0]
            if len(idx_r) < 2:
                continue
            mat   = df[SIM_FEATS].values[idx_r]
            mat_n = (mat - feat_mean) / feat_std

            dists     = np.linalg.norm(mat_n - test_norm, axis=1)
            best_local = int(np.argmin(dists))
            best_cycle = idx_r[best_local]          # 실제 훈련 사이클 번호
            frac = best_cycle / float(EOL[bid])     # 해당 베어링 기준 수명 비율
            fracs.append(frac)

    return float(np.mean(fracs)) if fracs else 0.0


# ══════════════════════════════════════════════════════════════════
# HI global statistics (LOO or full)
# ══════════════════════════════════════════════════════════════════
def compute_hi_stats(train_data: dict, bids: list) -> tuple:
    all_hi = np.concatenate([train_data[b]["hi"] for b in bids])
    return float(all_hi.mean()), float(all_hi.std() + 1e-8)


# ══════════════════════════════════════════════════════════════════
# LGBM
# ══════════════════════════════════════════════════════════════════
def make_lgbm_features(hi_arr: np.ndarray, regime_arr: np.ndarray,
                        seq_len: int = SEQ_LENGTH,
                        start_frac: float = 0.0):
    """
    피처: HI window (seq_len개) + slope, mean, std, max, last, delta,
          regime_current, regime_frac, elapsed_frac,
          hi_slope_5  (최근 5사이클 기울기),
          hi_slope_10 (window 전체 기울기, endpoint diff),
          hi_delta    (관측 시작 대비 누적 상승)
    start_frac: 관측 시작 시점의 수명 비율
    """
    feats, targets = [], []
    N = len(hi_arr)
    t = np.arange(seq_len, dtype=float)
    hi_obs_start = float(hi_arr[0])   # 관측 시작 시점 HI (hi_delta 기준)
    for i in range(seq_len, N):
        win   = hi_arr[i - seq_len: i]
        reg_w = regime_arr[i - seq_len: i]
        slope        = float(np.polyfit(t, win, 1)[0])
        elapsed_frac = np.clip(start_frac + float(i) / MEAN_TRAIN_LIFE, 0.0, 3.0)
        hi_slope_5   = float((win[-1] - win[-6]) / 5)           # 최근 5사이클
        hi_slope_10  = float((win[-1] - win[0]) / (seq_len - 1)) # window 양 끝점
        hi_delta     = float(win[-1] - hi_obs_start)             # 누적 상승량
        feats.append([
            *win,
            slope,
            float(win.mean()),
            float(win.std()),
            float(win.max()),
            float(win[-1]),
            float(win[-1] - win[0]),
            float(regime_arr[i]),
            float(reg_w.mean()),
            elapsed_frac,
            hi_slope_5,
            hi_slope_10,
            hi_delta,
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
        # Train bearings: start_frac=0 (수명 처음부터 관측)
        x, y = make_lgbm_features(train_data[b]["hi"], train_data[b]["regime"],
                                    start_frac=0.0)
        X_list.append(x); y_list.append(y)
    dtrain = lgb.Dataset(np.concatenate(X_list), label=np.concatenate(y_list))
    return lgb.train(
        {"num_leaves": 15, "learning_rate": 0.05,
         "min_child_samples": 5, "verbose": -1,
         "objective": lgbm_asym_obj},
        dtrain, num_boost_round=200,
    )


def predict_lgbm(model: lgb.Booster, hi_arr: np.ndarray,
                  regime_arr: np.ndarray,
                  start_frac: float = 0.0) -> np.ndarray:
    X, _ = make_lgbm_features(hi_arr, regime_arr, start_frac=start_frac)
    return np.maximum(model.predict(X), 0.0)


# ══════════════════════════════════════════════════════════════════
# LSTM
# ══════════════════════════════════════════════════════════════════
def make_seqs(hi_arr: np.ndarray, regime_arr: np.ndarray,
              rul_arr: np.ndarray, seq_len: int,
              hi_mean: float, hi_std: float,
              start_obs: int = 0):
    """
    HI 정규화: window-local minmax 대신 LOO train 기반 global standardization.
    → 절댓값(열화 수준) 정보 보존.
    start_obs: 관측 시작 사이클 (Train: 0, Test: estimate_start_frac × MEAN_TRAIN_LIFE)
    """
    n = len(hi_arr)
    X, y = [], []
    for i in range(n - seq_len):
        win      = hi_arr[i: i + seq_len].copy()
        win_norm = (win - hi_mean) / hi_std          # global standardization
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


def build_lstm_preds(train_data: dict, train_bids: list,
                      test_bid: int, device,
                      hi_mean: float, hi_std: float) -> tuple:
    """
    train_bids로만 LSTM 학습.
    LOOCV test bearing: start_obs=0 (수명 처음부터 관측 — 실제 위치 알고 있음).
    hi_mean, hi_std: LOO train 통계.
    rul_scale: train_bids의 EOL 최댓값 (leakage-free).
               test bearing의 EOL은 사용하지 않음.
               test inference 시 동일한 scale로 복원하므로 문제 없음.
    """
    X_list, y_list = [], []
    for b in train_bids:
        hi_b  = train_data[b]["hi"]
        reg_b = train_data[b]["regime"]
        rul_b = rul_labels(len(hi_b), b)
        X, y  = make_seqs(hi_b, reg_b, rul_b, SEQ_LENGTH,
                           hi_mean, hi_std, start_obs=0)
        X_list.append(X); y_list.append(y)
    X_train   = np.concatenate(X_list)
    y_train   = np.concatenate(y_list)
    rul_scale = float(max(EOL[b] for b in train_bids))  # train_bids EOL 최댓값만 사용

    hi_t  = train_data[test_bid]["hi"]
    reg_t = train_data[test_bid]["regime"]
    rul_t = rul_labels(len(hi_t), test_bid)
    # LOOCV: 전체 수명 데이터 있으므로 start_obs=0
    X_test, _ = make_seqs(hi_t, reg_t, rul_t, SEQ_LENGTH,
                           hi_mean, hi_std, start_obs=0)
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
# CF search (leakage-free)
# ══════════════════════════════════════════════════════════════════
def find_fold_cf(results: dict, test_bid: int,
                 pred_key: str = "ens_raw") -> tuple:
    other_bids = [b for b in BEARINGS if b != test_bid]
    best_cf, best_sc = 1.0, -np.inf
    for cf in np.arange(0.60, 1.41, 0.01):
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


def fold_ensemble_weights(results: dict, test_bid: int) -> tuple:
    other_bids = [b for b in BEARINGS if b != test_bid]
    sc_lgbm = float(np.mean([results[b]["sc_lgbm"] for b in other_bids]))
    sc_lstm  = float(np.mean([results[b]["sc_lstm"]  for b in other_bids]))
    total    = sc_lgbm + sc_lstm + 1e-12
    w_lgbm   = float(np.clip(sc_lgbm / total, 0.1, 0.7))
    return w_lgbm, 1.0 - w_lgbm


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
        f"{title}  [w_lgbm={r['w_lgbm']:.2f}  cf={r['fold_cf']:.2f}]",
        fontsize=9)
    ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)


def _draw_test_ax(ax, obs, preds_lgbm, preds_lstm, preds_final,
                   cf, start_frac, title):
    ax.plot(obs, preds_lgbm,  "r--", lw=1, alpha=0.6, label="LGBM")
    ax.plot(obs, preds_lstm,  "b--", lw=1, alpha=0.6, label="LSTM")
    ax.plot(obs, preds_final, "m-",  lw=2,
            label=f"Final (cf={cf:.2f})")
    final_hr = float(preds_final[-1]) * INTERVAL_SEC / 3600
    sf_pct   = int(start_frac * 100)
    ax.set_title(f"{title}  RUL={final_hr:.1f}hr  (start≈{sf_pct}%)", fontsize=9)
    ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)


# ══════════════════════════════════════════════════════════════════
# LOOCV
# ══════════════════════════════════════════════════════════════════
def run_loocv():
    print("=" * 65)
    print("  RUL Regime v5 — LGBM(+slope5/10/delta) + LSTM(global norm)")
    print("=" * 65)

    train_data = load_train_hi()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    results = {}

    # Step 1: raw predictions per fold
    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        hi_t  = train_data[test_bid]["hi"]
        reg_t = train_data[test_bid]["regime"]
        N     = len(hi_t)
        obs_pts = np.arange(SEQ_LENGTH, N)

        # LOO HI statistics (for LSTM global normalization)
        hi_mean, hi_std = compute_hi_stats(train_data, train_bids)

        print(f"[Fold B{test_bid}]  train: {train_bids}"
              f"  hi_mean={hi_mean:.4f}  hi_std={hi_std:.4f}")

        # LGBM: start_frac=0 (LOOCV — 수명 처음부터 관측)
        lgbm_model = train_lgbm(train_data, train_bids)
        preds_lgbm = predict_lgbm(lgbm_model, hi_t, reg_t, start_frac=0.0)
        sc_lgbm    = avg_score(N, obs_pts, preds_lgbm)

        # LSTM: start_obs=0 (LOOCV — 수명 처음부터 관측)
        preds_lstm, _ = build_lstm_preds(
            train_data, train_bids, test_bid, device, hi_mean, hi_std)
        sc_lstm = avg_score(N, obs_pts, preds_lstm)

        print(f"  LGBM={sc_lgbm:.4f}  LSTM={sc_lstm:.4f}")
        results[test_bid] = {
            "N": N, "obs_pts": list(obs_pts),
            "preds_lgbm": list(preds_lgbm),
            "preds_lstm": list(preds_lstm),
            "sc_lgbm": sc_lgbm, "sc_lstm": sc_lstm,
        }

    # Step 2: ensemble weights
    for test_bid in BEARINGS:
        w_lgbm, w_lstm = fold_ensemble_weights(results, test_bid)
        preds_ens = [
            w_lgbm * l + w_lstm * a
            for l, a in zip(results[test_bid]["preds_lgbm"],
                             results[test_bid]["preds_lstm"])
        ]
        results[test_bid].update({
            "ens_raw":    preds_ens,
            "w_lgbm":     w_lgbm,
            "w_lstm":     w_lstm,
            "sc_ens_raw": avg_score(
                results[test_bid]["N"], results[test_bid]["obs_pts"], preds_ens),
        })

    # Step 3: CF search
    print("\n[Fold-wise CF search (other 3 bearings only, range 0.60~1.40)]")
    for test_bid in BEARINGS:
        cf, cf_ref_sc = find_fold_cf(results, test_bid, pred_key="ens_raw")
        preds_cal     = [p * cf for p in results[test_bid]["ens_raw"]]
        sc_cal        = avg_score(
            results[test_bid]["N"], results[test_bid]["obs_pts"], preds_cal)
        results[test_bid].update({
            "fold_cf": cf, "preds_cal": preds_cal,
            "sc_cal": sc_cal, "cf_ref_score": cf_ref_sc,
        })
        print(f"  B{test_bid}: cf={cf:.2f} (ref={cf_ref_sc:.4f}) → sc={sc_cal:.4f}")

    # Step 4: summary
    print(f"\n{'='*65}")
    print("  LOOCV Summary")
    print(f"{'='*65}")
    print(f"  {'Bearing':>8} | {'LGBM':>8} | {'LSTM':>8} | {'Ens(raw)':>10} | {'CF':>5} | {'Ens+CF':>8}")
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

    mean_cal = np.mean(sc_lists["cal"])
    print(f"  {'avg':>8} | {np.mean(sc_lists['lgbm']):>8.4f} | "
          f"{np.mean(sc_lists['lstm']):>8.4f} | "
          f"{np.mean(sc_lists['ens']):>10.4f} | {'—':>5} | {mean_cal:>8.4f}")
    print(f"\n  ★ LOOCV final: {mean_cal:.4f}")

    # Log
    with open(OUT_DIR / "loocv_log_v5.txt", "w") as f:
        f.write("RUL Regime v5 — LGBM(+slope5/10/delta) + LSTM(global norm)\n")
        f.write(f"LGBM  mean: {np.mean(sc_lists['lgbm']):.4f}\n")
        f.write(f"LSTM  mean: {np.mean(sc_lists['lstm']):.4f}\n")
        f.write(f"Ens (raw):  {np.mean(sc_lists['ens']):.4f}\n")
        f.write(f"Ens + CF :  {mean_cal:.4f}\n\n")
        for test_bid in BEARINGS:
            r = results[test_bid]
            f.write(f"  B{test_bid}: LGBM={r['sc_lgbm']:.4f}  LSTM={r['sc_lstm']:.4f}  "
                    f"Ens={r['sc_ens_raw']:.4f}  CF={r['fold_cf']:.2f}  "
                    f"w=[LGBM={r['w_lgbm']:.3f},LSTM={r['w_lstm']:.3f}]  "
                    f"final={r['sc_cal']:.4f}\n")

    # Plots — combined + individual
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RUL Regime v5 — LOOCV (+slope5/10/delta)", fontsize=12)
    axes = axes.flatten()
    for i, test_bid in enumerate(BEARINGS):
        r   = results[test_bid]
        obs = r["obs_pts"]
        N   = r["N"]
        true_rul = [N - o for o in obs]
        _draw_rul_ax(axes[i], obs, true_rul, r, f"Bearing{test_bid}")

        fig_i, ax_i = plt.subplots(figsize=(9, 5))
        _draw_rul_ax(ax_i, obs, true_rul, r, f"Bearing{test_bid}")
        plt.tight_layout()
        fig_i.savefig(OUT_DIR / f"Bearing{test_bid}_RUL_v5.png", dpi=150)
        plt.close(fig_i)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "loocv_predictions_v5.png", dpi=150)
    plt.close()

    return results, mean_cal


# ══════════════════════════════════════════════════════════════════
# Test Inference
# ══════════════════════════════════════════════════════════════════
def run_test_inference(loocv_results: dict):
    print(f"\n{'='*65}")
    print("  Test inference — 전체 Train 4개 + feature-sim start_frac")
    print(f"{'='*65}")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = load_train_hi()
    test_data  = load_test_hi()

    # ── Feature cache for similarity-based start_frac ─────────────
    print("  Loading feature caches for start_frac estimation...")
    train_feat_dfs = load_train_feat_cache()
    test_feat_dfs  = load_test_feat_cache()

    # Full train statistics
    hi_mean, hi_std = compute_hi_stats(train_data, BEARINGS)
    print(f"  Full train HI: mean={hi_mean:.4f}  std={hi_std:.4f}")

    w_lgbm  = float(np.mean([loocv_results[b]["w_lgbm"]  for b in BEARINGS]))
    w_lstm  = 1.0 - w_lgbm
    cf_test = float(np.mean([loocv_results[b]["fold_cf"] for b in BEARINGS]))
    print(f"  Test weights: LGBM={w_lgbm:.3f}, LSTM={w_lstm:.3f}")
    print(f"  Test CF:      {cf_test:.2f}")

    print("  LGBM training (full train)...")
    lgbm_model = train_lgbm(train_data, BEARINGS)

    print("  LSTM training (full train, 5 seeds)...")
    X_all, y_all = [], []
    for b in BEARINGS:
        hi_b  = train_data[b]["hi"]
        reg_b = train_data[b]["regime"]
        rul_b = rul_labels(len(hi_b), b)
        X, y  = make_seqs(hi_b, reg_b, rul_b, SEQ_LENGTH,
                           hi_mean, hi_std, start_obs=0)
        X_all.append(X); y_all.append(y)
    X_train   = np.concatenate(X_all)
    y_train   = np.concatenate(y_all)
    rul_scale = float(max(EOL.values()))  # 전체 train EOL 최댓값 = 137 (leakage 없음)

    lstm_models = []
    for s in SEEDS:
        m = train_lstm(X_train, y_train, rul_scale, s, device)
        lstm_models.append(m)
        print(f"    seed={s} done")

    summary_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"RUL Regime v5 — Test (cf={cf_test:.2f})", fontsize=12)
    axes = axes.flatten()

    for i, tid in enumerate(TEST_IDS):
        hi_t    = test_data[tid]["hi"]
        reg_t   = test_data[tid]["regime"]
        N       = len(hi_t)
        obs_pts = np.arange(SEQ_LENGTH, N)

        # ── 시작 위치 추정 (feature similarity) ──────────────────────
        hi_start       = float(hi_t[0])
        hi_start_frac  = estimate_start_frac(hi_start, train_data, BEARINGS)  # 비교용

        test_feat_df   = test_feat_dfs[tid]
        start_frac     = estimate_start_frac_feat_sim(
                             test_feat_df, reg_t, train_feat_dfs, train_data)
        start_obs      = int(start_frac * MEAN_TRAIN_LIFE)
        print(f"\n  [Test{tid}] hi_start={hi_start:.3f}  "
              f"→ feat_sim={start_frac:.3f}  hi_based={hi_start_frac:.3f}  "
              f"start_obs≈{start_obs}cyc")

        # ── LGBM ─────────────────────────────────────────────────
        preds_lgbm = predict_lgbm(lgbm_model, hi_t, reg_t, start_frac=start_frac)

        # ── LSTM ──────────────────────────────────────────────────
        X_test = []
        for j in range(N - SEQ_LENGTH):
            win      = hi_t[j: j + SEQ_LENGTH].copy()
            win_norm = (win - hi_mean) / hi_std
            obs_frac = np.clip(
                (start_obs + j + np.arange(SEQ_LENGTH)) / MEAN_TRAIN_LIFE,
                0.0, 2.0)
            rg = reg_t[j: j + SEQ_LENGTH].astype(float)
            X_test.append(np.stack([win_norm, obs_frac, rg], axis=1))
        Xt = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)

        all_lstm = []
        for m in lstm_models:
            m.eval()
            with torch.no_grad():
                all_lstm.append(
                    np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0))
        preds_lstm  = np.median(all_lstm, axis=0)

        # ── 앙상블 + CF ───────────────────────────────────────────
        preds_ens   = w_lgbm * preds_lgbm + w_lstm * preds_lstm
        preds_final = preds_ens * cf_test

        final_cyc = float(preds_final[-1])
        final_hr  = final_cyc * INTERVAL_SEC / 3600
        print(f"    RUL={final_hr:.2f}hr ({final_cyc:.1f}cyc)  "
              f"LGBM_last={preds_lgbm[-1]:.1f}  LSTM_last={preds_lstm[-1]:.1f}")

        pd.DataFrame({
            "obs_cycle":    obs_pts,
            "preds_lgbm":  preds_lgbm,
            "preds_lstm":  preds_lstm,
            "preds_final": preds_final,
            "rul_hours":   preds_final * INTERVAL_SEC / 3600,
        }).to_csv(OUT_DIR / f"Test{tid}_RUL.csv", index=False)

        summary_rows.append({
            "test_id":          tid,
            "hi_start":         round(hi_start, 4),
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
        fig_i.savefig(OUT_DIR / f"Test{tid}_RUL_v5.png", dpi=150)
        plt.close(fig_i)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_predictions_v5.png", dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary_v5.csv", index=False)
    print(f"\n  Test summary:")
    print(df_sum.to_string(index=False))
    print(f"\n[Done] {OUT_DIR}")


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    loocv_results, loocv_score = run_loocv()
    run_test_inference(loocv_results)
