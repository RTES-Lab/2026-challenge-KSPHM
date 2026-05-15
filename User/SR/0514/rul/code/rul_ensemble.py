"""
앙상블 RUL 예측 — v4 FDR HI 인라인 계산 버전
========================
모델 구성:
  - LSTM-A : HI-A (주파수도메인) + AsymmetricRULLoss + 5 seeds median
  - LSTM-B : HI-B (시간도메인)   + AsymmetricRULLoss + 5 seeds median
  - LGBM   : HI-A flat 피처 (최근 10개 HI + 기울기/mean/std/max) + 비대칭 custom obj

HI 계산 방식 (v4 FDR 인라인):
  - LOOCV fold별: exclude_bid 제외 3개 베어링 기준으로 baseline/p5/p95 계산 → 4개 전체에 적용
  - Test 추론: 전체 4개 베어링 기준으로 baseline/p5/p95 계산 → train HI 생성
  - Test HI: 사전 계산된 test_v4 파일 사용 (hi_test_v4.py의 global 4-bearing 기준과 동일)

앙상블:
  - LOOCV fold별 모델 점수 → 가중치 결정 (클램핑 [0.1, 0.5])
  - 최종 calibration factor (0.7~1.0 그리드 서치)
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
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ── 경로 ──────────────────────────────────────────────────────────────────
BASE           = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0514")
TRAIN_FEAT_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SC/HI/04142304_signal_transform_v2/output")
HI_A_TEST      = BASE / "hi/output/test_v4"
HI_B_TEST      = BASE / "hi/output/test_hib_v4"
OUT_DIR        = BASE / "rul/output/ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH      = 10
EPOCHS          = 150
BATCH_SIZE      = 32
MEAN_TRAIN_LIFE = 116.5   # (126+114+89+137)/4
INTERVAL_SEC    = 600
SEEDS           = [42, 7, 123, 0, 99]

# HI v4 FDR 파라미터 (hi_train.py 그리드서치 최적값)
BR_A, ALPHA_A = 0.25, 0.1
BR_B, ALPHA_B = 0.20, 0.1

# ── HI-A 피처 (주파수도메인) ──────────────────────────────────────────────
FEATURE_Q = {
    "ch3_high_band":   0.4315732105779938,
    "ch4_high_band":   0.41934581236265145,
    "ch3_std":         0.4143663846438889,
    "ch3_total_power": 0.41207524516168137,
    "ch3_energy":      0.41108403369865365,
    "ch3_rms":         0.41108403369865365,
    "ch3_p2p":         0.3665441916808586,
}
FEATURE_GROUPS = {
    "highfreq":  ["ch3_high_band", "ch4_high_band"],
    "energy":    ["ch3_total_power", "ch3_energy", "ch3_rms"],
    "variation": ["ch3_std", "ch3_p2p"],
}
ALL_FEATS = list(FEATURE_Q.keys())

# ── HI-B 피처 (시간도메인) ────────────────────────────────────────────────
FEATURE_Q_B = {
    "ch3_kurtosis": 0.40,
    "ch3_crest_f":  0.38,
    "ch3_rms":      0.36,
    "ch3_p2p":      0.35,
    "ch4_kurtosis": 0.37,
    "ch4_rms":      0.34,
}
FEATURE_GROUPS_B = {
    "impulse":   ["ch3_kurtosis", "ch3_crest_f", "ch4_kurtosis"],
    "amplitude": ["ch3_rms", "ch3_p2p", "ch4_rms"],
}
ALL_FEATS_B = list(FEATURE_Q_B.keys())


# =========================================================
# V4 FDR HI 인라인 계산 헬퍼 함수
# =========================================================
def _moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, np.ones(window) / window, mode="valid")[:len(x)]


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def _train_anchored_scale(x: np.ndarray, p5: float, p95: float) -> np.ndarray:
    denom = p95 - p5
    if abs(denom) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return np.clip((x - p5) / denom, 0.0, 1.0)


def _fdr_ratios(feat_matrix: np.ndarray, feature_names: list,
                cond: np.ndarray, baseline: dict, eps: float = 1e-8) -> np.ndarray:
    ratios = np.zeros_like(feat_matrix, dtype=float)
    for regime in [0, 1]:
        idx = np.where(cond == regime)[0]
        if len(idx) == 0:
            continue
        bvec = np.array([baseline[(regime, f)] for f in feature_names])
        bvec = np.where(np.abs(bvec) < eps, eps, bvec)
        ratios[idx] = (feat_matrix[idx] - bvec) / (np.abs(bvec) + eps)
    return ratios


def _fdr_baseline(dfs: dict, br: float, exclude_bid: int, feat_list: list) -> dict:
    """exclude_bid를 제외한 베어링들의 레짐별 건강구간 평균 μ."""
    feat_vals = {(r, f): [] for r in [0, 1] for f in feat_list}
    for bid, df in dfs.items():
        if bid == exclude_bid:
            continue
        cond = df["cond"].values
        n_base = max(3, int(len(df) * br))
        for regime in [0, 1]:
            idx_regime = np.where(cond == regime)[0]
            base_idx = idx_regime[:n_base]
            if len(base_idx) == 0:
                continue
            for f in feat_list:
                if f in df.columns:
                    feat_vals[(regime, f)].extend(df[f].values[base_idx].tolist())
    baseline = {}
    for regime in [0, 1]:
        for f in feat_list:
            vals = feat_vals[(regime, f)]
            baseline[(regime, f)] = float(np.mean(vals)) if vals else 1.0
    return baseline


def _group_stats(dfs: dict, br: float, exclude_bid: int,
                 feat_groups: dict, feat_q: dict, feat_list: list,
                 baseline: dict) -> dict:
    """exclude_bid 제외 학습 베어링에서 그룹별 direction/p5/p95 계산."""
    all_scores = {g: [] for g in feat_groups}
    dir_votes  = {g: [] for g in feat_groups}

    for bid, df in dfs.items():
        if bid == exclude_bid:
            continue
        cond = df["cond"].values
        for gname, feats in feat_groups.items():
            available = [f for f in feats if f in df.columns]
            if not available:
                continue
            mat     = df[available].values
            ratios  = _fdr_ratios(mat, available, cond, baseline)
            weights = np.array([feat_q[f] for f in available], dtype=float)
            weights /= weights.sum() + 1e-12
            score   = (ratios * weights.reshape(1, -1)).sum(axis=1)
            rho, _  = spearmanr(np.arange(len(score)), score)
            dir_votes[gname].append(+1 if (not np.isnan(rho) and rho >= 0) else -1)
            all_scores[gname].extend(score.tolist())

    stats = {}
    for gname in feat_groups:
        direction = +1 if sum(dir_votes[gname]) >= 0 else -1
        arr = np.array(all_scores[gname]) * direction
        stats[gname] = {
            "direction": direction,
            "p5":  float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }
    return stats


def _v4fdr_hi(df: pd.DataFrame, baseline: dict, group_stats: dict,
              feat_groups: dict, feat_q: dict, ema_alpha: float) -> np.ndarray:
    """단일 베어링의 v4 FDR HI 계산."""
    cond = df["cond"].values
    sub_his, group_weights = {}, {}
    for gname, feats in feat_groups.items():
        available = [f for f in feats if f in df.columns]
        if not available:
            continue
        mat     = df[available].values
        ratios  = _fdr_ratios(mat, available, cond, baseline)
        weights = np.array([feat_q[f] for f in available], dtype=float)
        weights /= weights.sum() + 1e-12
        score   = (ratios * weights.reshape(1, -1)).sum(axis=1)
        score   = score * group_stats[gname]["direction"]
        score   = _train_anchored_scale(score, group_stats[gname]["p5"], group_stats[gname]["p95"])
        score   = _ema(score, alpha=ema_alpha)
        sub_his[gname]       = np.clip(score, 0.0, 1.0)
        group_weights[gname] = np.mean([feat_q[f] for f in available])

    if not sub_his:
        return np.zeros(len(df))

    sub_mat = np.column_stack([sub_his[g] for g in sub_his])
    w = np.array([group_weights[g] for g in sub_his], dtype=float)
    w /= w.sum() + 1e-12
    final_hi = (sub_mat * w.reshape(1, -1)).sum(axis=1)
    return np.clip(_moving_avg(final_hi, 7), 0.0, 1.0)


def load_train_features() -> dict:
    """Train 베어링 feature CSV + cond 로드."""
    dfs = {}
    for bid in BEARINGS:
        df = pd.read_csv(TRAIN_FEAT_DIR / f"Bearing{bid}_features_transformed.csv")
        cond = pd.read_csv(TRAIN_FEAT_DIR / f"Bearing{bid}_SSM_result.csv")["cond"].values
        df["cond"] = cond
        dfs[bid] = df
    return dfs


def compute_fold_hi(dfs: dict, exclude_bid: int):
    """
    LOOCV fold용 HI 계산.
    exclude_bid 제외 3개 베어링 기준 baseline/stats → 4개 전체에 동일 기준 적용.
    exclude_bid=0 → 없는 ID이므로 4개 전체 포함 (Test 추론용).
    """
    baseline_a    = _fdr_baseline(dfs, BR_A, exclude_bid, ALL_FEATS)
    group_stats_a = _group_stats(dfs, BR_A, exclude_bid,
                                  FEATURE_GROUPS, FEATURE_Q, ALL_FEATS, baseline_a)
    baseline_b    = _fdr_baseline(dfs, BR_B, exclude_bid, ALL_FEATS_B)
    group_stats_b = _group_stats(dfs, BR_B, exclude_bid,
                                  FEATURE_GROUPS_B, FEATURE_Q_B, ALL_FEATS_B, baseline_b)

    hi_a = {b: _v4fdr_hi(dfs[b], baseline_a, group_stats_a,
                           FEATURE_GROUPS, FEATURE_Q, ALPHA_A) for b in BEARINGS}
    hi_b = {b: _v4fdr_hi(dfs[b], baseline_b, group_stats_b,
                           FEATURE_GROUPS_B, FEATURE_Q_B, ALPHA_B) for b in BEARINGS}
    return hi_a, hi_b


# ── 대회 채점 함수 ─────────────────────────────────────────────────────────
def competition_score(rul_true: float, rul_pred: float) -> float:
    if rul_true <= 0:
        return np.nan
    Er = 100.0 * (rul_true - rul_pred) / rul_true
    if Er <= 0:
        return np.exp(-np.log(0.5) * Er / 20.0)
    else:
        return np.exp(np.log(0.5) * Er / 50.0)


# ── 비대칭 손실함수 ────────────────────────────────────────────────────────
class AsymmetricRULLoss(nn.Module):
    def forward(self, y_pred, y_true):
        y_pred_abs = y_pred * MEAN_TRAIN_LIFE
        y_true_abs = y_true * MEAN_TRAIN_LIFE
        Er = 100.0 * (y_true_abs - y_pred_abs) / y_true_abs.clamp(min=1.0)
        ln_half = torch.log(torch.tensor(0.5, dtype=y_pred.dtype, device=y_pred.device))
        score_over  = torch.exp(-ln_half * Er / 20.0)
        score_under = torch.exp( ln_half * Er / 50.0)
        return -torch.where(Er <= 0, score_over, score_under).mean()


# ── LSTM 모델 ──────────────────────────────────────────────────────────────
class LSTM_RUL(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1  = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc2(self.relu(self.fc1(out[:, -1, :])))


def create_sequences(hi_array, seq_length=SEQ_LENGTH):
    X, Y = [], []
    for i in range(len(hi_array) - seq_length):
        X.append(hi_array[i: i + seq_length])
        Y.append((len(hi_array) - (i + seq_length)) / MEAN_TRAIN_LIFE)
    return np.array(X)[..., np.newaxis], np.array(Y)


def train_lstm(hi_dict, train_bids, seed):
    X_list, Y_list = [], []
    for b in train_bids:
        x, y = create_sequences(hi_dict[b])
        X_list.append(x)
        Y_list.append(y)
    X_t = torch.tensor(np.concatenate(X_list), dtype=torch.float32)
    Y_t = torch.tensor(np.concatenate(Y_list), dtype=torch.float32).view(-1, 1)

    torch.manual_seed(seed)
    model = LSTM_RUL()
    criterion = AsymmetricRULLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loader = DataLoader(TensorDataset(X_t, Y_t), batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()
    return model


def predict_lstm(model, hi_array, n_total):
    model.eval()
    preds = []
    for obs in range(SEQ_LENGTH, len(hi_array)):
        seq = hi_array[obs - SEQ_LENGTH: obs]
        inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            preds.append(max(0.0, model(inp).item() * n_total))
    return np.array(preds)


def predict_lstm_ensemble(hi_dict, train_bids, test_hi, n_total):
    all_preds = []
    for s in SEEDS:
        model = train_lstm(hi_dict, train_bids, s)
        all_preds.append(predict_lstm(model, test_hi, n_total))
    return np.median(all_preds, axis=0)


# ── LightGBM ──────────────────────────────────────────────────────────────
def make_lgbm_features(hi_array, seq_length=SEQ_LENGTH):
    features, targets = [], []
    N = len(hi_array)
    for i in range(seq_length, N):
        window = hi_array[i - seq_length: i]
        slope  = float(np.polyfit(np.arange(seq_length), window, 1)[0])
        feats  = list(window) + [
            slope,
            float(window.mean()),
            float(window.std()),
            float(window.max()),
            float(window[-1]),
            float(window[-1] - window[0]),
        ]
        features.append(feats)
        targets.append(float(N - i))
    return np.array(features), np.array(targets)


def lgbm_asymmetric_obj(y_pred, dataset):
    y_true = dataset.get_label()
    diff = y_true - y_pred
    weight = np.where(diff < 0, 2.5, 1.0)
    grad = -diff * weight
    hess = np.ones_like(grad) * weight
    return grad, hess


def train_lgbm(hi_dict, train_bids):
    X_list, Y_list = [], []
    for b in train_bids:
        x, y = make_lgbm_features(hi_dict[b])
        X_list.append(x)
        Y_list.append(y)
    X_train = np.concatenate(X_list)
    Y_train = np.concatenate(Y_list)
    dtrain = lgb.Dataset(X_train, label=Y_train)
    params = {
        "num_leaves": 15,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": 5,
        "verbose": -1,
        "objective": lgbm_asymmetric_obj,
    }
    return lgb.train(params, dtrain, num_boost_round=200)


def predict_lgbm(model, hi_array):
    X, _ = make_lgbm_features(hi_array)
    return np.maximum(model.predict(X), 0.0)


# ── 가중치 결정 ────────────────────────────────────────────────────────────
def clamp_weights(scores: dict, lo=0.1, hi_w=0.5):
    keys = list(scores.keys())
    vals = np.array([scores[k] for k in keys], dtype=float)
    vals = np.clip(vals, 1e-6, None)
    w = vals / vals.sum()
    w = np.clip(w, lo, hi_w)
    w = w / w.sum()
    return {k: float(w[i]) for i, k in enumerate(keys)}


# ── LOOCV ─────────────────────────────────────────────────────────────────
def run_loocv():
    print("  Train feature 로드 중...")
    dfs = load_train_features()

    results = {b: {"lstm_a": [], "lstm_b": [], "lgbm": [], "ensemble": []} for b in BEARINGS}
    model_fold_scores = {"lstm_a": [], "lstm_b": [], "lgbm": []}
    fold_weights = {}

    for test_bid in BEARINGS:
        print(f"\n{'='*60}")
        print(f"[LOOCV] Test Bearing {test_bid}")
        train_bids = [b for b in BEARINGS if b != test_bid]

        # fold-consistent v4 HI 계산 (test_bid 제외 3개 기준 → 4개 전체 적용)
        print(f"  HI v4 계산 (Bearing{test_bid} 제외 기준)...")
        hi_a, hi_b = compute_fold_hi(dfs, exclude_bid=test_bid)

        N_test = len(hi_a[test_bid])

        # 1. LSTM-A
        print("  학습: LSTM-A (HI-A, 5 seeds)...")
        preds_a = predict_lstm_ensemble(hi_a, train_bids, hi_a[test_bid], N_test)

        # 2. LSTM-B
        print("  학습: LSTM-B (HI-B, 5 seeds)...")
        preds_b = predict_lstm_ensemble(hi_b, train_bids, hi_b[test_bid], N_test)

        # 3. LGBM
        print("  학습: LGBM (HI-A flat)...")
        lgbm_model = train_lgbm(hi_a, train_bids)
        preds_lgbm = predict_lgbm(lgbm_model, hi_a[test_bid])

        # 4. 각 모델 점수 계산
        obs_pts = np.arange(SEQ_LENGTH, N_test)
        def avg_score(preds):
            scores = [competition_score(N_test - obs, p)
                      for obs, p in zip(obs_pts, preds)]
            return float(np.nanmean(scores))

        sc_a    = avg_score(preds_a)
        sc_b    = avg_score(preds_b)
        sc_lgbm = avg_score(preds_lgbm)

        print(f"  → LSTM-A: {sc_a:.4f}, LSTM-B: {sc_b:.4f}, LGBM: {sc_lgbm:.4f}")

        model_fold_scores["lstm_a"].append(sc_a)
        model_fold_scores["lstm_b"].append(sc_b)
        model_fold_scores["lgbm"].append(sc_lgbm)

        # 5. 가중 평균
        w = clamp_weights({"lstm_a": sc_a, "lstm_b": sc_b, "lgbm": sc_lgbm})
        fold_weights[test_bid] = w
        print(f"  → 가중치: LSTM-A={w['lstm_a']:.3f}, LSTM-B={w['lstm_b']:.3f}, LGBM={w['lgbm']:.3f}")

        preds_ens = w["lstm_a"] * preds_a + w["lstm_b"] * preds_b + w["lgbm"] * preds_lgbm
        sc_ens = avg_score(preds_ens)
        print(f"  → 앙상블: {sc_ens:.4f}")

        results[test_bid]["lstm_a"]   = list(preds_a)
        results[test_bid]["lstm_b"]   = list(preds_b)
        results[test_bid]["lgbm"]     = list(preds_lgbm)
        results[test_bid]["ensemble"] = list(preds_ens)
        results[test_bid]["obs_pts"]  = list(obs_pts)
        results[test_bid]["N"]        = N_test

    # 6. LOOCV 요약
    print(f"\n{'='*60}")
    print("  LOOCV 요약")
    print(f"{'='*60}")
    overall = {k: float(np.mean(v)) for k, v in model_fold_scores.items()}
    ens_scores = []
    for test_bid in BEARINGS:
        obs_pts = results[test_bid]["obs_pts"]
        N_test  = results[test_bid]["N"]
        preds   = results[test_bid]["ensemble"]
        s = [competition_score(N_test - obs, p) for obs, p in zip(obs_pts, preds)]
        ens_scores.append(float(np.nanmean(s)))
        print(f"  Bearing {test_bid}: LSTM-A={model_fold_scores['lstm_a'][test_bid-1]:.4f}  "
              f"LSTM-B={model_fold_scores['lstm_b'][test_bid-1]:.4f}  "
              f"LGBM={model_fold_scores['lgbm'][test_bid-1]:.4f}  "
              f"Ensemble={ens_scores[-1]:.4f}")

    print(f"\n  평균  : LSTM-A={overall['lstm_a']:.4f}  "
          f"LSTM-B={overall['lstm_b']:.4f}  "
          f"LGBM={overall['lgbm']:.4f}  "
          f"Ensemble={np.mean(ens_scores):.4f}")

    # 7. Calibration factor 그리드 서치
    print("\n  [Calibration] factor 그리드 서치 (0.7~1.0)...")
    best_cf, best_cf_score = 1.0, -np.inf
    for cf in np.arange(0.70, 1.01, 0.02):
        cf_scores = []
        for test_bid in BEARINGS:
            obs_pts = results[test_bid]["obs_pts"]
            N_test  = results[test_bid]["N"]
            preds   = [p * cf for p in results[test_bid]["ensemble"]]
            s = [competition_score(N_test - obs, p) for obs, p in zip(obs_pts, preds)]
            cf_scores.append(float(np.nanmean(s)))
        mean_cf = float(np.mean(cf_scores))
        print(f"    cf={cf:.2f} → {mean_cf:.4f}")
        if mean_cf > best_cf_score:
            best_cf_score, best_cf = mean_cf, float(cf)

    print(f"\n  → Best calibration factor: {best_cf:.2f} (score={best_cf_score:.4f})")

    # 결과 저장
    with open(OUT_DIR / "loocv_log.txt", "w", encoding="utf-8") as f:
        f.write(f"LOOCV 결과 (v4 FDR 인라인 HI)\n")
        f.write(f"LSTM-A (HI-A 5seeds median): {overall['lstm_a']:.4f}\n")
        f.write(f"LSTM-B (HI-B 5seeds median): {overall['lstm_b']:.4f}\n")
        f.write(f"LGBM   (HI-A flat feat):     {overall['lgbm']:.4f}\n")
        f.write(f"Ensemble (weighted avg):      {np.mean(ens_scores):.4f}\n")
        f.write(f"Best calibration factor:      {best_cf:.2f}\n")

    # CSV 저장 (train LOOCV)
    TRAIN_OUT = BASE / "rul/output/train"
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for test_bid in BEARINGS:
        obs_pts = results[test_bid]["obs_pts"]
        N_test  = results[test_bid]["N"]
        for obs, pred in zip(obs_pts, results[test_bid]["ensemble"]):
            rows.append({
                "test_bearing": test_bid,
                "obs_cycle":    obs,
                "rul_true":     N_test - obs,
                "rul_pred":     pred,
            })
    pd.DataFrame(rows).to_csv(TRAIN_OUT / "LSTM_RUL_results.csv", index=False)

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Ensemble LOOCV – Predicted vs True RUL (v4 FDR inline)", fontsize=13)
    axes = axes.flatten()
    for i, test_bid in enumerate(BEARINGS):
        ax = axes[i]
        obs_pts = results[test_bid]["obs_pts"]
        N_test  = results[test_bid]["N"]
        true_rul = [N_test - obs for obs in obs_pts]
        ax.plot(obs_pts, true_rul, "k-", lw=1.5, label="True RUL")
        ax.plot(obs_pts, results[test_bid]["lstm_a"],   "b--", lw=1, alpha=0.7, label="LSTM-A")
        ax.plot(obs_pts, results[test_bid]["lstm_b"],   "g--", lw=1, alpha=0.7, label="LSTM-B")
        ax.plot(obs_pts, results[test_bid]["lgbm"],     "r--", lw=1, alpha=0.7, label="LGBM")
        ax.plot(obs_pts, results[test_bid]["ensemble"], "m-",  lw=2, label="Ensemble")
        ax.set_title(f"Bearing {test_bid}")
        ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ensemble_loocv_predictions.png", dpi=150)
    plt.close()

    return overall, best_cf, fold_weights


# ── Test 추론 ──────────────────────────────────────────────────────────────
def run_test_inference(best_cf: float, fold_weights: dict):
    print(f"\n{'='*60}")
    print("  Test 추론 (전체 Train 4개 베어링으로 재학습)")
    print(f"{'='*60}")

    # Train HI: 전체 4개 기준 인라인 계산
    print("  Train HI v4 계산 (전체 4개 베어링 기준)...")
    dfs = load_train_features()
    hi_a, hi_b = compute_fold_hi(dfs, exclude_bid=0)  # exclude_bid=0 → 전체 포함

    # Test HI: 사전 계산 파일 사용
    hi_a_test_exists = all((HI_A_TEST / f"Test{t}_best.csv").exists() for t in TEST_IDS)
    hi_b_test_exists = all((HI_B_TEST / f"Test{t}_best.csv").exists() for t in TEST_IDS)

    if not hi_a_test_exists:
        print("[경고] HI-A Test 파일 없음. hi_test_v4.py를 먼저 실행하세요.")
        return

    # 전체 모델 학습
    print("  LSTM-A (5 seeds)...")
    lstm_a_models = [train_lstm(hi_a, BEARINGS, s) for s in SEEDS]
    print("  LSTM-B (5 seeds)...")
    lstm_b_models = [train_lstm(hi_b, BEARINGS, s) for s in SEEDS]
    print("  LGBM...")
    lgbm_model = train_lgbm(hi_a, BEARINGS)

    # LOOCV fold 가중치 평균
    w_a    = float(np.mean([fold_weights[b]["lstm_a"] for b in BEARINGS]))
    w_b    = float(np.mean([fold_weights[b]["lstm_b"] for b in BEARINGS]))
    w_lgbm = float(np.mean([fold_weights[b]["lgbm"]   for b in BEARINGS]))
    total  = w_a + w_b + w_lgbm
    w_a, w_b, w_lgbm = w_a/total, w_b/total, w_lgbm/total
    print(f"\n  Test 가중치: LSTM-A={w_a:.3f}, LSTM-B={w_b:.3f}, LGBM={w_lgbm:.3f}")
    print(f"  Calibration factor: {best_cf:.2f}")

    summary_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Ensemble RUL – Test Bearings (v4 FDR inline)", fontsize=13)
    axes = axes.flatten()

    for i, tid in enumerate(TEST_IDS):
        hi_a_t = pd.read_csv(HI_A_TEST / f"Test{tid}_best.csv")["HI"].values
        hi_b_t = (pd.read_csv(HI_B_TEST / f"Test{tid}_best.csv")["HI"].values
                  if hi_b_test_exists else hi_a_t)
        N = len(hi_a_t)
        obs_pts = np.arange(SEQ_LENGTH, N)

        pa_all = [predict_lstm(m, hi_a_t, MEAN_TRAIN_LIFE) for m in lstm_a_models]
        preds_a = np.median(pa_all, axis=0)

        pb_all = [predict_lstm(m, hi_b_t, MEAN_TRAIN_LIFE) for m in lstm_b_models]
        preds_b = np.median(pb_all, axis=0)

        preds_lgbm = predict_lgbm(lgbm_model, hi_a_t)

        preds_ens = (w_a * preds_a + w_b * preds_b + w_lgbm * preds_lgbm) * best_cf

        final_rul_cyc = float(preds_ens[-1])
        final_rul_hr  = final_rul_cyc * INTERVAL_SEC / 3600
        print(f"  [Test{tid}] {N}슬롯({N*INTERVAL_SEC/3600:.1f}hr) | "
              f"최종 RUL={final_rul_hr:.2f}hr ({final_rul_cyc:.1f}cycles)")

        df_out = pd.DataFrame({
            "obs_cycle":         obs_pts,
            "rul_pred_lstm_a":   preds_a,
            "rul_pred_lstm_b":   preds_b,
            "rul_pred_lgbm":     preds_lgbm,
            "rul_pred_ensemble": preds_ens,
            "rul_pred_hours":    preds_ens * INTERVAL_SEC / 3600,
        })
        df_out.to_csv(OUT_DIR / f"Test{tid}_ensemble_RUL.csv", index=False)

        summary_rows.append({
            "test_id":          tid,
            "observed_slots":   N,
            "observed_hours":   round(N * INTERVAL_SEC / 3600, 2),
            "final_rul_cycles": round(final_rul_cyc, 2),
            "final_rul_hours":  round(final_rul_hr,  2),
            "w_lstm_a":         round(w_a, 3),
            "w_lstm_b":         round(w_b, 3),
            "w_lgbm":           round(w_lgbm, 3),
            "calib_factor":     round(best_cf, 2),
        })

        ax = axes[i]
        ax.plot(obs_pts, preds_a,    "b--", lw=1, alpha=0.6, label="LSTM-A")
        ax.plot(obs_pts, preds_b,    "g--", lw=1, alpha=0.6, label="LSTM-B")
        ax.plot(obs_pts, preds_lgbm, "r--", lw=1, alpha=0.6, label="LGBM")
        ax.plot(obs_pts, preds_ens,  "m-",  lw=2, label="Ensemble")
        ax.axvline(N, color="gray", ls="--", lw=1)
        ax.set_title(f"Test{tid}  Final={final_rul_hr:.1f}hr")
        ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "ensemble_test_predictions.png", dpi=150)
    plt.close()

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / "Test_ensemble_summary.csv", index=False)
    print(f"\n  최종 요약:")
    print(df_summary.to_string(index=False))
    print(f"\n[완료] {OUT_DIR}")


# ── 메인 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    overall, best_cf, fold_weights = run_loocv()
    run_test_inference(best_cf, fold_weights)
