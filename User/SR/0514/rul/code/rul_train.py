import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# ── 경로 설정 ─────────────────────────────────────────────────────
HI_DIR  = "/data/home/ksphm/2026-challenge-KSPHM/User/SR/0514/hi/output/train"
OUT_DIR = "/data/home/ksphm/2026-challenge-KSPHM/User/SR/0514/rul/output/train"
RUL_LOG = "/data/home/ksphm/2026-challenge-KSPHM/User/SR/0514/rul/output/train/rul_log.txt"

os.makedirs(OUT_DIR, exist_ok=True)

BEARINGS = [1, 2, 3, 4]

# ── 평가 지표 (대회 공식 Score 함수) ──────────────────────────────
def competition_score(rul_true: float, rul_pred: float) -> float:
    if rul_true <= 0: return np.nan
    Er = 100.0 * (rul_true - rul_pred) / rul_true
    if Er <= 0:
        return np.exp(-np.log(0.5) * Er / 20.0)
    else:
        return np.exp(np.log(0.5) * Er / 50.0)

# ── 상수: Train 베어링 평균 수명 ──────────────────────────────────
# Bearing 1~4: 126, 114, 89, 137 cycles → 평균 116.5
MEAN_TRAIN_LIFE = 116.5

# ── 데이터 로드 및 전처리 (시퀀스 생성, RUL 정규화) ───────────────
def load_hi(bid):
    df = pd.read_csv(f"{HI_DIR}/Bearing{bid}_best.csv")
    return df["HI"].values

def create_sequences(hi_array, seq_length=10):
    N = len(hi_array)
    X, Y = [], []
    for i in range(N - seq_length):
        X.append(hi_array[i : i + seq_length])
        # RUL을 MEAN_TRAIN_LIFE(고정 상수)로 정규화 → 베어링 간 타겟 범위 균일화
        # Bearing3(89)/116.5=0.76 ~ Bearing4(137)/116.5=1.18: 모두 [0, ~1.2] 범위
        Y.append((N - (i + seq_length)) / MEAN_TRAIN_LIFE)
    return np.array(X)[..., np.newaxis], np.array(Y)

# ── 비대칭 손실함수 (절대 cycles 기준으로 Er 계산) ──────────────────
class AsymmetricRULLoss(nn.Module):
    """LSTM은 RUL/MEAN_TRAIN_LIFE를 예측. 손실은 절대 cycles로 역변환 후 계산.
    → 수명 말기 분모 폭발 방지 + 대회 채점 방식 유지.
    """
    def forward(self, y_pred, y_true):
        # 절대 cycles로 역변환
        y_pred_abs = y_pred * MEAN_TRAIN_LIFE
        y_true_abs = y_true * MEAN_TRAIN_LIFE
        Er = 100.0 * (y_true_abs - y_pred_abs) / y_true_abs.clamp(min=1.0)
        ln_half = torch.log(torch.tensor(0.5, dtype=y_pred.dtype, device=y_pred.device))
        score_over  = torch.exp(-ln_half * Er / 20.0)
        score_under = torch.exp( ln_half * Er / 50.0)
        score = torch.where(Er <= 0, score_over, score_under)
        return -score.mean()


# ── LSTM 모델 정의 ────────────────────────────────────────────────
class LSTM_RUL(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2):
        super(LSTM_RUL, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 마지막 타임스텝의 출력
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ── LOOCV 학습 및 평가 ──────────────────────────────────────────
def run_loocv():
    seq_length = 10
    epochs = 150
    batch_size = 32

    all_hi = {bid: load_hi(bid) for bid in BEARINGS}
    results_all = []

    # Figure 1: True RUL vs Predicted RUL (전체 구간)
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle("LSTM RUL Prediction – True vs Predicted RUL (Full Range)", fontsize=14)
    axes1 = axes1.flatten()

    # Figure 2: Predicted Failure Cycle over Observation Cycle
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle("LSTM Predicted Failure Cycle vs Actual Failure (Full Observation Range)", fontsize=14)
    axes2 = axes2.flatten()

    for test_bid in BEARINGS:
        print(f"\n[Test Bearing {test_bid}]")
        train_bids = [b for b in BEARINGS if b != test_bid]

        # 1. Train Data 준비
        train_x_list, train_y_list = [], []
        for b in train_bids:
            x, y = create_sequences(all_hi[b], seq_length)
            train_x_list.append(x)
            train_y_list.append(y)

        X_train = np.concatenate(train_x_list, axis=0)
        Y_train = np.concatenate(train_y_list, axis=0)

        # 2. Tensor 변환 및 모델 초기화
        X_t = torch.tensor(X_train, dtype=torch.float32)
        Y_t = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)

        dataset = TensorDataset(X_t, Y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        torch.manual_seed(42)
        model = LSTM_RUL()
        criterion = AsymmetricRULLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        # 3. 모델 학습
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # 4. 전체 구간 Observation (seq_length ~ N_test-1)
        model.eval()
        hi_test = all_hi[test_bid]
        N_test = len(hi_test)

        # 전체 구간으로 확장
        observe_pts = np.arange(seq_length, N_test)

        row_results = []

        for obs in observe_pts:
            rul_true = N_test - obs
            input_seq = hi_test[obs - seq_length : obs]
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

            with torch.no_grad():
                rul_pred_norm = model(input_tensor).item()

            # 역정규화: 정규화된 예측 × 실제 베어링 총 수명
            rul_pred = max(0.0, rul_pred_norm * N_test)
            score = competition_score(rul_true, rul_pred)

            row_results.append({
                "test_bearing": test_bid,
                "obs_cycle": obs,
                "rul_true": rul_true,
                "rul_pred": rul_pred,
                "pred_failure_cycle": obs + rul_pred,
                "actual_failure_cycle": N_test,
                "score": score
            })

        results_all.extend(row_results)

        obs_cycles     = [r["obs_cycle"] for r in row_results]
        rul_true_list  = [r["rul_true"] for r in row_results]
        rul_pred_list  = [r["rul_pred"] for r in row_results]
        pred_fail_list = [r["pred_failure_cycle"] for r in row_results]

        # ── Figure 1: True vs Predicted RUL ───────────────────────
        ax1 = axes1[test_bid - 1]
        ax1.plot(obs_cycles, rul_true_list, "k-", lw=1.5, label="True RUL")
        ax1.plot(obs_cycles, rul_pred_list, "b-", lw=1.5, alpha=0.8, label="Predicted RUL")
        ax1.axvline(N_test, color="r", linestyle="--", lw=1, label=f"Failure (cycle {N_test})")
        ax1.set_title(f"Bearing {test_bid}")
        ax1.set_xlabel("Observation Cycle")
        ax1.set_ylabel("RUL (cycles)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.4)

        # ── Figure 2: Predicted Failure Cycle ─────────────────────
        ax2 = axes2[test_bid - 1]

        # HI curve (배경, 2nd y-axis)
        ax2_hi = ax2.twinx()
        ax2_hi.plot(np.arange(N_test), hi_test, color="gray", lw=1, alpha=0.35, label="HI")
        ax2_hi.set_ylabel("HI", color="gray", fontsize=9)
        ax2_hi.tick_params(axis="y", labelcolor="gray")

        # 예측 failure 시점
        ax2.plot(obs_cycles, pred_fail_list, "b-", lw=1.5, label="Predicted Failure Cycle")
        # 실제 failure 시점 (수평선)
        ax2.axhline(N_test, color="r", linestyle="--", lw=1.5, label=f"Actual Failure (cycle {N_test})")
        # 현재 관측 기준선 (y=x, 즉 obs 시점 자체)
        ax2.plot(obs_cycles, obs_cycles, color="gray", linestyle=":", lw=1, label="Current Obs (y=x)")

        ax2.set_title(f"Bearing {test_bid}")
        ax2.set_xlabel("Observation Cycle")
        ax2.set_ylabel("Predicted Failure Cycle")
        ax2.set_xlim(0, N_test + 5)
        ax2.set_ylim(0, max(N_test * 1.5, max(pred_fail_list) * 1.1))

        # 두 legend 통합
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_hi.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax2.grid(True, alpha=0.4)

        avg_score = np.nanmean([r["score"] for r in row_results])
        print(f"  → {len(observe_pts)} obs points | Avg Score: {avg_score:.4f}")

    fig1.tight_layout()
    fig1.savefig(f"{OUT_DIR}/LSTM_RUL_Predictions.png", dpi=150)
    plt.close(fig1)

    fig2.tight_layout()
    fig2.savefig(f"{OUT_DIR}/LSTM_Failure_Prediction_Timeline.png", dpi=150)
    plt.close(fig2)

    df_res = pd.DataFrame(results_all)
    df_res.to_csv(f"{OUT_DIR}/LSTM_RUL_results.csv", index=False)

    # ── RUL.txt 업데이트 ──────────────────────────────────────
    summary_text = "\n▶ Experiment: 04262315_lstm_rul (LSTM on v5_v3_best HI, Full Obs Range)\n"
    summary_text += "-" * 50 + "\n"

    for bid in BEARINGS:
        sub = df_res[df_res["test_bearing"] == bid]
        avg_score = sub["score"].mean()
        summary_text += f"  Bearing {bid} 평균 Score: {avg_score:.4f}\n"

    overall_avg = df_res["score"].mean()
    summary_text += "-" * 50 + "\n"
    summary_text += f"  전체 평균 Score: {overall_avg:.4f}\n"

    with open(RUL_LOG, "a", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print("Done!")

if __name__ == "__main__":
    run_loocv()
