"""
LSTM-C: 직접 RUL 예측 (Piecewise Linear RUL, Raw Features)
==========================================================
핵심 목적: v4 HI의 Bearing3/4 절대 범위 역전 문제 우회
  - Bearing3 HI 범위 (0.016~0.152) vs Bearing4 (0.244~0.814) 비중첩 →
    HI=0.15가 "Bearing3 사망 직전" vs HI=0.25가 "Bearing4 초기" 모순
  - 해결: HI 없이 원시 피처 직접 사용

입력: 8개 raw 피처 (train/test 피처 파일 공통 컬럼)
라벨: piecewise linear RUL
  - 건강 구간 (idx <= normal_until): RUL = EOL - normal_until (plateau 상수)
  - 열화 구간 (idx > normal_until):  RUL = max(EOL - idx, 0) (선형 감소)
정규화: fold별 y_train.max() (가변 scale)
  - fold3에서 B4 포함(rul_scale=59)이 B3 예측에 최적
  - 고정 scale(116.5)은 B3 성능이 0.71→0.51로 저하되어 fold-specific 유지
평가: LOOCV × 4, competition score
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

# ── 경로 ──────────────────────────────────────────────────────────────
TRAIN_FEAT_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SC/HI/04142304_signal_transform_v2/output")
TEST_FEAT_DIR  = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SC/HI/05072245_signal_transform_v5_test/output")
OUT_DIR        = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0514/rul/output/direct")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 피처: train/test 공통 컬럼만 사용 ────────────────────────────────
INPUT_COLS = [
    "ch3_high_band", "ch4_high_band",           # 고주파 에너지
    "ch3_std", "ch3_total_power", "ch3_energy",  # CH3 에너지
    "ch3_rms", "ch3_p2p",                        # CH3 진폭
    "ch4_rms",                                   # CH4 에너지
]
N_FEAT = len(INPUT_COLS)  # 8

# ── Piecewise RUL 설정 ─────────────────────────────────────────────────
# normal_until: v3 HI > 0.3 최초 돌파 시점 (열화 개시 추정)
NORMAL_UNTIL = {1: 89, 2: 92, 3: 62, 4: 78}
EOL          = {1: 126, 2: 114, 3: 89, 4: 137}
# 각 베어링 plateau RUL: Bearing1=37, Bearing2=22, Bearing3=27, Bearing4=59

BEARINGS     = [1, 2, 3, 4]
TEST_IDS     = [1, 2, 3, 4, 5, 6]
SEQ_LEN      = 10
BATCH_SIZE   = 64
EPOCHS       = 200
PATIENCE     = 20
LR           = 1e-3
SEEDS        = [42, 7, 123, 0, 99]
INTERVAL_SEC = 600

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── 대회 채점 함수 ─────────────────────────────────────────────────────
def competition_score(rul_true: float, rul_pred: float) -> float:
    if rul_true <= 0:
        return np.nan
    Er = 100.0 * (rul_true - rul_pred) / rul_true
    if Er <= 0:
        return np.exp(-np.log(0.5) * Er / 20.0)
    else:
        return np.exp(np.log(0.5) * Er / 50.0)


# ── RUL 라벨 (piecewise linear) ───────────────────────────────────────
def rul_labels(n_total: int, normal_until: int, eol: int) -> np.ndarray:
    idx = np.arange(n_total)
    rul = np.where(
        idx <= normal_until,
        eol - normal_until,
        np.maximum(eol - idx, 0)
    )
    return rul.astype(float)


# ── 시퀀스 생성 ────────────────────────────────────────────────────────
def make_sequences(feat_arr: np.ndarray, rul_arr: np.ndarray, seq_len: int):
    n = len(feat_arr)
    X = np.array([feat_arr[i:i + seq_len] for i in range(n - seq_len)])
    y = rul_arr[seq_len:]
    return X, y


# ── 데이터 로드 ────────────────────────────────────────────────────────
def load_train_bearing(bid: int):
    df = pd.read_csv(TRAIN_FEAT_DIR / f"Bearing{bid}_features_transformed.csv")
    feat = df[INPUT_COLS].values
    rul  = rul_labels(len(df), NORMAL_UNTIL[bid], EOL[bid])
    return feat, rul


def load_test_bearing(tid: int) -> np.ndarray:
    df = pd.read_csv(TEST_FEAT_DIR / f"Test{tid}_features.csv")
    return df[INPUT_COLS].values


# ── LSTM 모델 ──────────────────────────────────────────────────────────
class LSTMRegressor(nn.Module):
    def __init__(self, n_feat: int = N_FEAT):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)


# ── 모델 학습 ──────────────────────────────────────────────────────────
def train_model(X_train: np.ndarray, y_train: np.ndarray,
                rul_scale: float, seed: int) -> LSTMRegressor:
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_norm = y_train / rul_scale
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_norm,  dtype=torch.float32)

    n_val = max(1, int(len(Xt) * 0.1))
    Xt_tr, Xt_val = Xt[:-n_val], Xt[-n_val:]
    yt_tr, yt_val = yt[:-n_val], yt[-n_val:]

    tr_dl  = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xt_val, yt_val), batch_size=BATCH_SIZE)

    model     = LSTMRegressor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val, patience_cnt, best_state = np.inf, 0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vl = np.mean([criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
                          for xb, yb in val_dl])
        if vl < best_val:
            best_val, patience_cnt = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


# ── 예측 (5 seeds median 앙상블) ───────────────────────────────────────
def predict_ensemble(X_test: np.ndarray, models: list, rul_scale: float) -> np.ndarray:
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    all_preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred_norm = model(Xt).cpu().numpy()
        all_preds.append(np.maximum(pred_norm * rul_scale, 0.0))
    return np.median(all_preds, axis=0)


# ── LOOCV ──────────────────────────────────────────────────────────────
def run_loocv():
    print("=" * 60)
    print("  LSTM-C LOOCV (Piecewise RUL, Raw Features)")
    print("=" * 60)

    train_data = {b: load_train_bearing(b) for b in BEARINGS}

    all_results = []
    fold_scores = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LSTM-C Direct RUL Prediction (Piecewise Linear RUL)", fontsize=13)
    axes = axes.flatten()

    for test_bid in BEARINGS:
        print(f"\n{'='*50}\n[LOOCV] Test Bearing {test_bid}")
        train_bids = [b for b in BEARINGS if b != test_bid]

        # 스케일러 fit (train 3개 베어링)
        all_train_feat = np.concatenate([train_data[b][0] for b in train_bids])
        scaler = StandardScaler()
        scaler.fit(all_train_feat)

        # 학습 데이터 구성
        X_list, y_list = [], []
        for b in train_bids:
            feat_s = scaler.transform(train_data[b][0])
            X, y   = make_sequences(feat_s, train_data[b][1], SEQ_LEN)
            X_list.append(X)
            y_list.append(y)
        X_train = np.concatenate(X_list)
        y_train = np.concatenate(y_list)
        rul_scale = float(y_train.max())
        print(f"  학습 샘플: {X_train.shape}  RUL scale={rul_scale:.1f} cycles")

        # 5 seeds 모델 학습
        models = []
        for seed in SEEDS:
            m = train_model(X_train, y_train, rul_scale, seed)
            models.append(m)
            print(f"    seed={seed} 완료")

        # 테스트 베어링 예측
        feat_test = scaler.transform(train_data[test_bid][0])
        rul_test  = train_data[test_bid][1]
        X_test, y_test = make_sequences(feat_test, rul_test, SEQ_LEN)
        y_pred = predict_ensemble(X_test, models, rul_scale)

        # Competition score 계산
        obs_pts = np.arange(SEQ_LEN, len(train_data[test_bid][0]) + 1)
        scores  = [competition_score(float(yt), float(yp))
                   for yt, yp in zip(y_test, y_pred)]
        avg_score = float(np.nanmean(scores))
        rmse      = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
        fold_scores[test_bid] = avg_score

        print(f"  → Score: {avg_score:.4f}  RMSE: {rmse:.2f} cycles")

        # 결과 저장
        for obs, yt, yp, sc in zip(obs_pts, y_test, y_pred, scores):
            all_results.append({
                "test_bearing": test_bid,
                "obs_idx":      int(obs),
                "rul_true":     float(yt),
                "rul_pred":     float(yp),
                "score":        float(sc) if not np.isnan(sc) else np.nan,
            })

        # 시각화
        ax = axes[test_bid - 1]
        ax.plot(obs_pts, y_test,     "k-",  lw=1.5, label="True RUL")
        ax.plot(obs_pts, y_pred,     "b-",  lw=1.2, alpha=0.85, label="Pred RUL (5-seed median)")
        ax.axvline(NORMAL_UNTIL[test_bid], color="r", ls=":", lw=1,
                   label=f"normal_until={NORMAL_UNTIL[test_bid]}")
        ax.set_title(f"Bearing {test_bid}  Score={avg_score:.4f}  RMSE={rmse:.1f}cycles",
                     fontsize=10)
        ax.set_xlabel("file_index")
        ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "loocv_direct_rul.png", dpi=150, bbox_inches="tight")
    plt.close()

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(OUT_DIR / "loocv_direct_results.csv", index=False)

    print(f"\n{'='*60}")
    print("  LOOCV 요약")
    for b in BEARINGS:
        print(f"  Bearing {b}: Score = {fold_scores[b]:.4f}")
    overall = float(np.mean(list(fold_scores.values())))
    print(f"  전체 평균 Score: {overall:.4f}")
    print(f"{'='*60}")

    return fold_scores, overall


# ── Test 추론 ──────────────────────────────────────────────────────────
def run_test_inference():
    print(f"\n{'='*60}")
    print("  Test 추론 (전체 Train 4개 베어링으로 재학습)")
    print(f"{'='*60}")

    train_data = {b: load_train_bearing(b) for b in BEARINGS}

    # 스케일러 fit (전체 4개 베어링)
    all_feat = np.concatenate([train_data[b][0] for b in BEARINGS])
    scaler = StandardScaler()
    scaler.fit(all_feat)

    # 학습 데이터 구성
    X_list, y_list = [], []
    for b in BEARINGS:
        feat_s = scaler.transform(train_data[b][0])
        X, y   = make_sequences(feat_s, train_data[b][1], SEQ_LEN)
        X_list.append(X)
        y_list.append(y)
    X_train = np.concatenate(X_list)
    y_train = np.concatenate(y_list)
    rul_scale = float(y_train.max())
    print(f"  학습 샘플: {X_train.shape}  RUL scale={rul_scale:.1f} cycles")

    # 5 seeds 모델 학습
    models = []
    for seed in SEEDS:
        m = train_model(X_train, y_train, rul_scale, seed)
        models.append(m)
        print(f"  seed={seed} 완료")

    # Test 추론
    summary = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("LSTM-C Direct RUL – Test Bearings (Piecewise Linear RUL)", fontsize=13)
    axes = axes.flatten()

    for i, tid in enumerate(TEST_IDS):
        feat_raw = load_test_bearing(tid)
        feat_s   = scaler.transform(feat_raw)
        N        = len(feat_s)

        # 시퀀스 생성 (RUL 라벨 없음)
        X_test = np.array([feat_s[j:j + SEQ_LEN] for j in range(N - SEQ_LEN)])
        obs_pts = np.arange(SEQ_LEN, N)

        y_pred = predict_ensemble(X_test, models, rul_scale)

        final_rul_cyc = float(y_pred[-1])
        final_rul_hr  = final_rul_cyc * INTERVAL_SEC / 3600

        print(f"  [Test{tid}] {N}슬롯({N*INTERVAL_SEC/3600:.1f}hr) | "
              f"최종 RUL={final_rul_hr:.2f}hr ({final_rul_cyc:.1f}cycles)")

        # 결과 저장
        df_out = pd.DataFrame({
            "obs_idx":        obs_pts,
            "rul_pred_cycles": y_pred,
            "rul_pred_hours":  y_pred * INTERVAL_SEC / 3600,
        })
        df_out.to_csv(OUT_DIR / f"Test{tid}_direct_RUL.csv", index=False)

        summary.append({
            "test_id":          tid,
            "observed_slots":   N,
            "observed_hours":   round(N * INTERVAL_SEC / 3600, 2),
            "final_rul_cycles": round(final_rul_cyc, 2),
            "final_rul_hours":  round(final_rul_hr, 2),
        })

        ax = axes[i]
        ax.plot(obs_pts, y_pred, "b-", lw=1.5, label="Pred RUL")
        ax.axvline(N, color="gray", ls="--", lw=1)
        ax.set_title(f"Test{tid}  Final={final_rul_hr:.1f}hr ({final_rul_cyc:.1f}cycles)",
                     fontsize=10)
        ax.set_xlabel("file_index")
        ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_direct_rul.png", dpi=150, bbox_inches="tight")
    plt.close()

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(OUT_DIR / "test_direct_summary.csv", index=False)
    print(f"\n  최종 요약:")
    print(df_summary.to_string(index=False))
    print(f"\n[완료] {OUT_DIR}")


# ── 메인 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fold_scores, overall = run_loocv()
    run_test_inference()
