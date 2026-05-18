"""
앙상블 RUL 예측 v3 — LGBM + LSTM-A (2-모델) vs LSTM-A 단독 비교
================================================================
v2에서 LSTM-A 단독 LOOCV 0.4287 > 3-모델 앙상블 0.4267 관찰.
LSTM-C 제거 후 두 구성 비교:
  A) LSTM-A 단독 + calibration
  B) LGBM + LSTM-A 2-모델 + calibration
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
TEST_FEAT_DIR  = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SC/HI/05072245_signal_transform_v5_test/output")
HI_A_TEST      = BASE / "hi/output/test_v4"
OUT_DIR        = BASE / "rul/output/ensemble_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH      = 10
MEAN_TRAIN_LIFE = 116.5
INTERVAL_SEC    = 600
SEEDS           = [42, 7, 123, 0, 99]

BR_A, ALPHA_A = 0.25, 0.1

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


# =========================================================
# V4 FDR HI 인라인 계산
# =========================================================
def _moving_avg(x, window):
    if window <= 1:
        return x.copy()
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, np.ones(window) / window, mode="valid")[:len(x)]

def _ema(x, alpha):
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def _train_anchored_scale(x, p5, p95):
    denom = p95 - p5
    if abs(denom) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return np.clip((x - p5) / denom, 0.0, 1.0)

def _fdr_ratios(feat_matrix, feature_names, cond, baseline, eps=1e-8):
    ratios = np.zeros_like(feat_matrix, dtype=float)
    for regime in [0, 1]:
        idx = np.where(cond == regime)[0]
        if len(idx) == 0:
            continue
        bvec = np.array([baseline[(regime, f)] for f in feature_names])
        bvec = np.where(np.abs(bvec) < eps, eps, bvec)
        ratios[idx] = (feat_matrix[idx] - bvec) / (np.abs(bvec) + eps)
    return ratios

def _fdr_baseline(dfs, br, exclude_bid, feat_list):
    feat_vals = {(r, f): [] for r in [0, 1] for f in feat_list}
    for bid, df in dfs.items():
        if bid == exclude_bid:
            continue
        cond = df["cond"].values
        n_base = max(3, int(len(df) * br))
        for regime in [0, 1]:
            idx_r = np.where(cond == regime)[0]
            base_idx = idx_r[:n_base]
            if len(base_idx) == 0:
                continue
            for f in feat_list:
                if f in df.columns:
                    feat_vals[(regime, f)].extend(df[f].values[base_idx].tolist())
    return {(r, f): float(np.mean(feat_vals[(r, f)])) if feat_vals[(r, f)] else 1.0
            for r in [0, 1] for f in feat_list}

def _group_stats(dfs, br, exclude_bid, feat_groups, feat_q, feat_list, baseline):
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
        stats[gname] = {"direction": direction,
                        "p5":  float(np.percentile(arr, 5)),
                        "p95": float(np.percentile(arr, 95))}
    return stats

def _v4fdr_hi(df, baseline, group_stats, feat_groups, feat_q, ema_alpha):
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
    return np.clip(_moving_avg((sub_mat * w.reshape(1, -1)).sum(axis=1), 7), 0.0, 1.0)

def load_train_features():
    dfs = {}
    for bid in BEARINGS:
        df = pd.read_csv(TRAIN_FEAT_DIR / f"Bearing{bid}_features_transformed.csv")
        cond = pd.read_csv(TRAIN_FEAT_DIR / f"Bearing{bid}_SSM_result.csv")["cond"].values
        df["cond"] = cond
        dfs[bid] = df
    return dfs

def compute_fold_hi_a(dfs, exclude_bid):
    baseline    = _fdr_baseline(dfs, BR_A, exclude_bid, ALL_FEATS)
    group_stats = _group_stats(dfs, BR_A, exclude_bid,
                                FEATURE_GROUPS, FEATURE_Q, ALL_FEATS, baseline)
    return {b: _v4fdr_hi(dfs[b], baseline, group_stats,
                          FEATURE_GROUPS, FEATURE_Q, ALPHA_A) for b in BEARINGS}


# ── 대회 채점 함수 ─────────────────────────────────────────────────────────
def competition_score(rul_true, rul_pred):
    if rul_true <= 0:
        return np.nan
    Er = 100.0 * (rul_true - rul_pred) / rul_true
    return (np.exp(-np.log(0.5) * Er / 20.0) if Er <= 0
            else np.exp(np.log(0.5) * Er / 50.0))


# ── LightGBM ──────────────────────────────────────────────────────────────
def make_lgbm_features(hi_array, seq_length=SEQ_LENGTH):
    features, targets = [], []
    N = len(hi_array)
    for i in range(seq_length, N):
        window = hi_array[i - seq_length: i]
        slope  = float(np.polyfit(np.arange(seq_length), window, 1)[0])
        feats  = list(window) + [slope, float(window.mean()), float(window.std()),
                                  float(window.max()), float(window[-1]),
                                  float(window[-1] - window[0])]
        features.append(feats)
        targets.append(float(N - i))
    return np.array(features), np.array(targets)

def lgbm_asymmetric_obj(y_pred, dataset):
    y_true = dataset.get_label()
    diff = y_true - y_pred
    weight = np.where(diff < 0, 2.5, 1.0)
    return -diff * weight, np.ones_like(diff) * weight

def train_lgbm(hi_dict, train_bids):
    X_list, Y_list = [], []
    for b in train_bids:
        x, y = make_lgbm_features(hi_dict[b])
        X_list.append(x); Y_list.append(y)
    dtrain = lgb.Dataset(np.concatenate(X_list), label=np.concatenate(Y_list))
    return lgb.train({"num_leaves": 15, "learning_rate": 0.05,
                      "min_child_samples": 5, "verbose": -1,
                      "objective": lgbm_asymmetric_obj},
                     dtrain, num_boost_round=200)

def predict_lgbm(model, hi_array):
    X, _ = make_lgbm_features(hi_array)
    return np.maximum(model.predict(X), 0.0)


# =========================================================
# LSTM-A: window-minmax HI + obs_fraction
# =========================================================
N_FEAT_A = 2
NORMAL_UNTIL_A = {1: 89, 2: 92, 3: 62, 4: 78}
EOL_A          = {1: 126, 2: 114, 3: 89, 4: 137}

def rul_labels_a(n_total, bid):
    nu, eol = NORMAL_UNTIL_A[bid], EOL_A[bid]
    idx = np.arange(n_total)
    return np.where(idx <= nu, eol - nu, np.maximum(eol - idx, 0)).astype(float)

def make_seqs_a(hi_arr, rul_arr, seq_len, start_obs=0):
    n = len(hi_arr)
    X, y = [], []
    for i in range(n - seq_len):
        window = hi_arr[i:i + seq_len].copy()
        w_min, w_max = window.min(), window.max()
        window_norm = (window - w_min) / (w_max - w_min + 1e-8)
        obs_frac = np.clip((start_obs + i + np.arange(seq_len)) / MEAN_TRAIN_LIFE, 0.0, 2.0)
        X.append(np.stack([window_norm, obs_frac], axis=1))
        y.append(float(rul_arr[i + seq_len]))
    return np.array(X), np.array(y)

class LSTMRegressorA(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_FEAT_A, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)

def train_model_a(X_train, y_train, rul_scale, seed, device):
    torch.manual_seed(seed)
    y_norm = y_train / rul_scale
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_norm,  dtype=torch.float32)
    n_val = max(1, int(len(Xt) * 0.1))
    tr_dl = DataLoader(TensorDataset(Xt[:-n_val], yt[:-n_val]), batch_size=64, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xt[-n_val:], yt[-n_val:]), batch_size=64)
    model = LSTMRegressorA().to(device)
    opt, crit = torch.optim.Adam(model.parameters(), lr=1e-3), nn.MSELoss()
    best_val, patience, best_state = np.inf, 0, None
    for _ in range(200):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([crit(model(xb.to(device)), yb.to(device)).item()
                          for xb, yb in val_dl])
        if vl < best_val:
            best_val, patience = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    model.load_state_dict(best_state)
    return model

def predict_lstm_a_ensemble(hi_dict, train_bids, test_bid, device):
    X_list, y_list = [], []
    for b in train_bids:
        X, y = make_seqs_a(hi_dict[b], rul_labels_a(len(hi_dict[b]), b), SEQ_LENGTH)
        X_list.append(X); y_list.append(y)
    X_train = np.concatenate(X_list)
    y_train = np.concatenate(y_list)
    rul_scale = float(y_train.max())
    X_test, _ = make_seqs_a(hi_dict[test_bid],
                             rul_labels_a(len(hi_dict[test_bid]), test_bid), SEQ_LENGTH)
    Xt = torch.tensor(X_test, dtype=torch.float32)
    all_preds = []
    for s in SEEDS:
        m = train_model_a(X_train, y_train, rul_scale, s, device)
        m.eval()
        with torch.no_grad():
            all_preds.append(np.maximum(m(Xt.to(device)).cpu().numpy() * rul_scale, 0.0))
    return np.median(all_preds, axis=0), rul_scale

def build_lstm_a_for_test(hi_dict, device):
    X_list, y_list = [], []
    for b in BEARINGS:
        X, y = make_seqs_a(hi_dict[b], rul_labels_a(len(hi_dict[b]), b), SEQ_LENGTH)
        X_list.append(X); y_list.append(y)
    X_train = np.concatenate(X_list)
    y_train = np.concatenate(y_list)
    rul_scale = float(y_train.max())
    models = []
    for s in SEEDS:
        m = train_model_a(X_train, y_train, rul_scale, s, device)
        models.append(m)
        print(f"    LSTM-A seed={s} 완료")
    return models, rul_scale

def predict_lstm_a_from_models(models, rul_scale, hi_arr, start_obs, device):
    n = len(hi_arr)
    X = []
    for i in range(n - SEQ_LENGTH):
        window = hi_arr[i:i + SEQ_LENGTH].copy()
        w_min, w_max = window.min(), window.max()
        window_norm = (window - w_min) / (w_max - w_min + 1e-8)
        obs_frac = np.clip((start_obs + i + np.arange(SEQ_LENGTH)) / MEAN_TRAIN_LIFE, 0.0, 2.0)
        X.append(np.stack([window_norm, obs_frac], axis=1))
    Xt = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    all_preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            all_preds.append(np.maximum(m(Xt).cpu().numpy() * rul_scale, 0.0))
    return np.median(all_preds, axis=0)


# ── Calibration 그리드 서치 ────────────────────────────────────────────────
def search_calibration(preds_dict, results, label=""):
    """results[bid]['obs_pts'], ['N'], preds_dict[bid] 사용."""
    best_cf, best_score = 1.0, -np.inf
    cf_log = []
    for cf in np.arange(0.70, 1.01, 0.02):
        sc_list = []
        for bid in BEARINGS:
            obs_pts = results[bid]["obs_pts"]
            N = results[bid]["N"]
            preds = [p * cf for p in preds_dict[bid]]
            sc = np.nanmean([competition_score(N - obs, p)
                             for obs, p in zip(obs_pts, preds)])
            sc_list.append(sc)
        mean_sc = float(np.mean(sc_list))
        cf_log.append((cf, mean_sc))
        if mean_sc > best_score:
            best_score, best_cf = mean_sc, float(cf)
    print(f"  [{label}] Best cf={best_cf:.2f} → {best_score:.4f}")
    return best_cf, best_score


# ── LOOCV ─────────────────────────────────────────────────────────────────
def run_loocv():
    print("  Train feature 로드 중...")
    dfs = load_train_features()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    results = {b: {} for b in BEARINGS}
    fold_scores = {"lgbm": [], "lstm_a": []}

    for test_bid in BEARINGS:
        print(f"\n{'='*60}")
        print(f"[LOOCV] Test Bearing {test_bid}")
        train_bids = [b for b in BEARINGS if b != test_bid]

        hi_a = compute_fold_hi_a(dfs, exclude_bid=test_bid)
        N_test = len(hi_a[test_bid])

        print("  LGBM...")
        lgbm_model = train_lgbm(hi_a, train_bids)
        preds_lgbm = predict_lgbm(lgbm_model, hi_a[test_bid])

        print("  LSTM-A (5 seeds)...")
        preds_a, _ = predict_lstm_a_ensemble(hi_a, train_bids, test_bid, device)

        obs_pts = np.arange(SEQ_LENGTH, N_test)

        def avg_score(preds):
            return float(np.nanmean([competition_score(N_test - obs, p)
                                     for obs, p in zip(obs_pts, preds)]))

        sc_lgbm = avg_score(preds_lgbm)
        sc_a    = avg_score(preds_a)
        print(f"  → LGBM: {sc_lgbm:.4f}, LSTM-A: {sc_a:.4f}")

        fold_scores["lgbm"].append(sc_lgbm)
        fold_scores["lstm_a"].append(sc_a)

        results[test_bid] = {
            "lgbm":   list(preds_lgbm),
            "lstm_a": list(preds_a),
            "obs_pts": list(obs_pts),
            "N":       N_test,
            "sc_lgbm": sc_lgbm,
            "sc_a":    sc_a,
        }

    # ── 구성별 LOOCV 요약 ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  LOOCV 요약")
    print(f"{'='*60}")
    print(f"  {'Bearing':>8} | {'LGBM':>8} | {'LSTM-A':>8} | {'2-model':>8}")
    print(f"  {'-'*45}")

    # 2-모델 점수 계산 (clamp 가중치)
    ens2_scores = []
    for test_bid in BEARINGS:
        obs_pts = results[test_bid]["obs_pts"]
        N = results[test_bid]["N"]
        sc_l = results[test_bid]["sc_lgbm"]
        sc_a = results[test_bid]["sc_a"]
        # clamp [0.1, 0.7]
        w_l = np.clip(sc_l / (sc_l + sc_a + 1e-12), 0.1, 0.7)
        w_a = 1.0 - w_l
        preds_ens = [w_l * l + w_a * a
                     for l, a in zip(results[test_bid]["lgbm"], results[test_bid]["lstm_a"])]
        sc_ens = float(np.nanmean([competition_score(N - obs, p)
                                   for obs, p in zip(obs_pts, preds_ens)]))
        ens2_scores.append(sc_ens)
        results[test_bid]["ens2"]   = preds_ens
        results[test_bid]["sc_ens2"] = sc_ens
        results[test_bid]["w_lgbm"]  = w_l
        results[test_bid]["w_a"]     = w_a
        i = test_bid - 1
        print(f"  {test_bid:>8} | {fold_scores['lgbm'][i]:>8.4f} | "
              f"{fold_scores['lstm_a'][i]:>8.4f} | {sc_ens:>8.4f}")

    print(f"  {'평균':>8} | {np.mean(fold_scores['lgbm']):>8.4f} | "
          f"{np.mean(fold_scores['lstm_a']):>8.4f} | {np.mean(ens2_scores):>8.4f}")

    # ── Calibration 탐색 ─────────────────────────────────────────────────
    print("\n  [Calibration 탐색]")
    cf_a,   sc_a_cf   = search_calibration({b: results[b]["lstm_a"] for b in BEARINGS},
                                            results, "LSTM-A 단독")
    cf_ens2, sc_ens2_cf = search_calibration({b: results[b]["ens2"]   for b in BEARINGS},
                                              results, "LGBM+LSTM-A")

    print(f"\n  ┌──────────────────────────────────────────────┐")
    print(f"  │  구성           │  LOOCV   │  Best cf  │ cf후  │")
    print(f"  ├──────────────────────────────────────────────┤")
    print(f"  │  LSTM-A 단독    │  {np.mean(fold_scores['lstm_a']):.4f}  │  {cf_a:.2f}    │ {sc_a_cf:.4f}│")
    print(f"  │  LGBM+LSTM-A   │  {np.mean(ens2_scores):.4f}  │  {cf_ens2:.2f}    │ {sc_ens2_cf:.4f}│")
    print(f"  └──────────────────────────────────────────────┘")

    # 로그 저장
    with open(OUT_DIR / "loocv_log.txt", "w", encoding="utf-8") as f:
        f.write("LOOCV 결과 — LSTM-A 단독 vs LGBM+LSTM-A\n")
        f.write(f"LGBM   평균: {np.mean(fold_scores['lgbm']):.4f}\n")
        f.write(f"LSTM-A 평균: {np.mean(fold_scores['lstm_a']):.4f}\n")
        f.write(f"LGBM+LSTM-A (uncalibrated): {np.mean(ens2_scores):.4f}\n")
        f.write(f"\nCalibration 후:\n")
        f.write(f"  LSTM-A 단독: cf={cf_a:.2f} → {sc_a_cf:.4f}\n")
        f.write(f"  LGBM+LSTM-A: cf={cf_ens2:.2f} → {sc_ens2_cf:.4f}\n")
        f.write(f"\nFold 상세:\n")
        for test_bid in BEARINGS:
            r = results[test_bid]
            f.write(f"  B{test_bid}: LGBM={r['sc_lgbm']:.4f}  LSTM-A={r['sc_a']:.4f}  "
                    f"2-model={r['sc_ens2']:.4f}  "
                    f"w=[LGBM={r['w_lgbm']:.3f},A={r['w_a']:.3f}]\n")

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("v3 LOOCV — LGBM + LSTM-A", fontsize=12)
    axes = axes.flatten()
    for i, test_bid in enumerate(BEARINGS):
        ax = axes[i]
        obs_pts = results[test_bid]["obs_pts"]
        N = results[test_bid]["N"]
        true_rul = [N - obs for obs in obs_pts]
        ax.plot(obs_pts, true_rul,                    "k-",  lw=1.5, label="True RUL")
        ax.plot(obs_pts, results[test_bid]["lgbm"],   "r--", lw=1,   alpha=0.7, label="LGBM")
        ax.plot(obs_pts, results[test_bid]["lstm_a"], "g--", lw=1,   alpha=0.7, label="LSTM-A")
        ax.plot(obs_pts, results[test_bid]["ens2"],   "m-",  lw=2,   label="LGBM+LSTM-A")
        ax.set_title(f"B{test_bid}  LGBM={results[test_bid]['sc_lgbm']:.3f}  "
                     f"A={results[test_bid]['sc_a']:.3f}  "
                     f"Ens={results[test_bid]['sc_ens2']:.3f}")
        ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "v3_loocv_predictions.png", dpi=150)
    plt.close()

    # 최적 구성 결정
    best_config = "lstm_a" if sc_a_cf >= sc_ens2_cf else "ens2"
    best_cf     = cf_a if best_config == "lstm_a" else cf_ens2
    print(f"\n  → 최적 구성: {'LSTM-A 단독' if best_config == 'lstm_a' else 'LGBM+LSTM-A'} "
          f"(cf={best_cf:.2f})")

    return results, fold_scores, best_config, best_cf


# ── Test 추론 ──────────────────────────────────────────────────────────────
def run_test_inference(results, fold_scores, best_config, best_cf):
    print(f"\n{'='*60}")
    print(f"  Test 추론 — {('LSTM-A 단독' if best_config == 'lstm_a' else 'LGBM+LSTM-A')}, cf={best_cf:.2f}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not all((HI_A_TEST / f"Test{t}_best.csv").exists() for t in TEST_IDS):
        print("[경고] HI-A Test 파일 없음.")
        return

    dfs = load_train_features()
    hi_a = compute_fold_hi_a(dfs, exclude_bid=0)

    need_lgbm = (best_config == "ens2")
    if need_lgbm:
        print("  LGBM 학습...")
        lgbm_model = train_lgbm(hi_a, BEARINGS)

    print("  LSTM-A 학습 (5 seeds)...")
    lstm_a_models, lstm_a_scale = build_lstm_a_for_test(hi_a, device)

    # 가중치 (LGBM+LSTM-A 구성 시)
    if best_config == "ens2":
        w_lgbm_list = [results[b]["w_lgbm"] for b in BEARINGS]
        w_a_list    = [results[b]["w_a"]    for b in BEARINGS]
        w_lgbm = float(np.mean(w_lgbm_list))
        w_a    = float(np.mean(w_a_list))
        total  = w_lgbm + w_a
        w_lgbm, w_a = w_lgbm / total, w_a / total
        print(f"  Test 가중치: LGBM={w_lgbm:.3f}, LSTM-A={w_a:.3f}")

    summary_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    label = "LSTM-A 단독" if best_config == "lstm_a" else "LGBM+LSTM-A"
    fig.suptitle(f"v3 Test RUL — {label} (cf={best_cf:.2f})", fontsize=12)
    axes = axes.flatten()

    for i, tid in enumerate(TEST_IDS):
        hi_a_t = pd.read_csv(HI_A_TEST / f"Test{tid}_best.csv")["HI"].values
        N = len(hi_a_t)
        obs_pts = np.arange(SEQ_LENGTH, N)

        preds_a = predict_lstm_a_from_models(lstm_a_models, lstm_a_scale, hi_a_t, 0, device)

        if best_config == "ens2":
            preds_lgbm = predict_lgbm(lgbm_model, hi_a_t)
            preds_final = (w_lgbm * preds_lgbm + w_a * preds_a) * best_cf
        else:
            preds_lgbm  = None
            preds_final = preds_a * best_cf

        final_rul_cyc = float(preds_final[-1])
        final_rul_hr  = final_rul_cyc * INTERVAL_SEC / 3600
        print(f"  [Test{tid}] 최종 RUL={final_rul_hr:.2f}hr ({final_rul_cyc:.1f}cycles)")

        cols = {"obs_cycle": obs_pts, "rul_pred_lstm_a": preds_a,
                "rul_pred_final": preds_final,
                "rul_pred_hours": preds_final * INTERVAL_SEC / 3600}
        if preds_lgbm is not None:
            cols["rul_pred_lgbm"] = preds_lgbm
        pd.DataFrame(cols).to_csv(OUT_DIR / f"Test{tid}_v3_RUL.csv", index=False)

        summary_rows.append({
            "test_id": tid, "config": label,
            "final_rul_cycles": round(final_rul_cyc, 2),
            "final_rul_hours":  round(final_rul_hr,  2),
            "calib_factor":     round(best_cf, 2),
        })

        ax = axes[i]
        if preds_lgbm is not None:
            ax.plot(obs_pts, preds_lgbm, "r--", lw=1, alpha=0.6, label="LGBM")
        ax.plot(obs_pts, preds_a,     "g--", lw=1, alpha=0.6, label="LSTM-A")
        ax.plot(obs_pts, preds_final, "m-",  lw=2, label=label)
        ax.set_title(f"Test{tid}  Final={final_rul_hr:.1f}hr")
        ax.set_xlabel("Obs Cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "v3_test_predictions.png", dpi=150)
    plt.close()

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / "Test_v3_summary.csv", index=False)
    print(f"\n  최종 요약:")
    print(df_summary.to_string(index=False))
    print(f"\n[완료] {OUT_DIR}")


# ── 메인 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results, fold_scores, best_config, best_cf = run_loocv()
    run_test_inference(results, fold_scores, best_config, best_cf)
