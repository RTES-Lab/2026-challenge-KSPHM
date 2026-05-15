"""
Test 데이터 RUL 예측 (04262315_lstm_rul 동일 아키텍처)
=======================================================
- Train Bearing 1~4의 HI (v5/v3_best)로 LSTM 학습
- Test 1~6의 HI (v5/v3_best)로 RUL 추론
- Test는 수명 미공개 → 매 사이클 RUL 예측값 출력 및 시각화

아키텍처:  2-Layer LSTM (hidden=32) + FC
입력:      과거 10 사이클 HI 시퀀스
출력:      예당 시점의 예측 RUL (cycles)
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
import warnings
warnings.filterwarnings("ignore")

# ── 경로 ───────────────────────────────────────────────────────────────
TRAIN_HI_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0514/hi/output/train")
TEST_HI_DIR  = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0514/hi/output/test")
OUT_DIR      = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/0514/rul/output/test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_BEARINGS = [1, 2, 3, 4]
TEST_IDS       = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH     = 10
EPOCHS         = 150
BATCH_SIZE     = 32
INTERVAL_SEC   = 600

# Train 베어링 평균 수명: 역정규화에 사용
MEAN_TRAIN_LIFE = 116.5  # (126+114+89+137)/4

# ── 비대칭 손실함수 (절대 cycles 기준으로 Er 계산) ──────────────
class AsymmetricRULLoss(nn.Module):
    def forward(self, y_pred, y_true):
        y_pred_abs = y_pred * MEAN_TRAIN_LIFE
        y_true_abs = y_true * MEAN_TRAIN_LIFE
        Er = 100.0 * (y_true_abs - y_pred_abs) / y_true_abs.clamp(min=1.0)
        ln_half = torch.log(torch.tensor(0.5, dtype=y_pred.dtype, device=y_pred.device))
        score_over  = torch.exp(-ln_half * Er / 20.0)
        score_under = torch.exp( ln_half * Er / 50.0)
        score = torch.where(Er <= 0, score_over, score_under)
        return -score.mean()


# ── LSTM 모델 (04262315_lstm_rul 동일) ────────────────────────────
class LSTM_RUL(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1  = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        return self.fc2(out)


def create_sequences(hi_array, seq_length=SEQ_LENGTH):
    N = len(hi_array)
    X, Y = [], []
    for i in range(N - seq_length):
        X.append(hi_array[i: i + seq_length])
        Y.append((N - (i + seq_length)) / MEAN_TRAIN_LIFE)
    return np.array(X)[..., np.newaxis], np.array(Y)


# ── 모델 학습 (전체 Train 4개 베어링으로 단일 모델 훈련) ──────────
def train_model():
    print("[모델 학습] Train Bearing 1~4 전체 사용")
    train_x_list, train_y_list = [], []
    for bid in TRAIN_BEARINGS:
        hi = pd.read_csv(TRAIN_HI_DIR / f"Bearing{bid}_best.csv")["HI"].values
        x, y = create_sequences(hi)
        train_x_list.append(x)
        train_y_list.append(y)
        print(f"  Bearing{bid}: {len(hi)} 슬롯, {len(y)} 시퀀스")

    X_train = np.concatenate(train_x_list, axis=0)
    Y_train = np.concatenate(train_y_list, axis=0)
    print(f"  총 학습 샘플: {len(X_train)}")

    X_t = torch.tensor(X_train, dtype=torch.float32)
    Y_t = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(TensorDataset(X_t, Y_t),
                        batch_size=BATCH_SIZE, shuffle=True)

    torch.manual_seed(42)
    model = LSTM_RUL()
    criterion = AsymmetricRULLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(EPOCHS):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch+1}/{EPOCHS}] loss={loss.item():.4f}")

    print("  학습 완료")
    return model


# ── Test 추론 ─────────────────────────────────────────────────────
def infer_test(model, test_id: int):
    hi_path = TEST_HI_DIR / f"Test{test_id}_best.csv"
    hi = pd.read_csv(hi_path)["HI"].values
    N  = len(hi)
    print(f"  [Test{test_id}] {N} 슬롯")

    model.eval()
    observe_pts = np.arange(SEQ_LENGTH, N)
    rul_preds   = []

    for obs in observe_pts:
        seq = hi[obs - SEQ_LENGTH: obs]
        inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred_norm = model(inp).item()
        # 역정규화: 정규화된 예측 × MEAN_TRAIN_LIFE → 절대 cycles
        rul_preds.append(max(0.0, pred_norm * MEAN_TRAIN_LIFE))

    return hi, observe_pts, np.array(rul_preds)


# ── 메인 ──────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Test RUL 예측 (LSTM, Train 4 베어링으로 학습)")
    print("=" * 70)

    # 1. 학습
    model = train_model()

    # 2. 시각화 준비
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle("LSTM RUL Prediction – Test Bearings", fontsize=14)
    axes1 = axes1.flatten()

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle("Predicted Failure Cycle – Test Bearings", fontsize=14)
    axes2 = axes2.flatten()

    all_rows = []

    for i, tid in enumerate(TEST_IDS):
        hi, obs_pts, rul_preds = infer_test(model, tid)
        N = len(hi)

        # 예측 고장 시점 (obs + rul_pred)
        pred_fail = obs_pts + rul_preds

        # 결과 저장
        rows = []
        for obs, rp, pf in zip(obs_pts, rul_preds, pred_fail):
            rows.append({
                "test_id":           tid,
                "obs_cycle":         int(obs),
                "rul_pred_cycles":   float(rp),
                "rul_pred_hours":    round(float(rp) * INTERVAL_SEC / 3600, 3),
                "pred_failure_cycle": float(pf),
                "pred_failure_hours": round(float(pf) * INTERVAL_SEC / 3600, 3),
                "current_time_hours": round(float(obs) * INTERVAL_SEC / 3600, 3),
            })
        df_test = pd.DataFrame(rows)
        df_test.to_csv(OUT_DIR / f"Test{tid}_RUL_results.csv", index=False)
        all_rows.extend(rows)

        # 최종 예측값 (마지막 관측 시점 기준)
        final_rul_cyc = rul_preds[-1]
        final_rul_hr  = final_rul_cyc * INTERVAL_SEC / 3600

        print(f"  [Test{tid}] 현재 {N}슬롯({N*INTERVAL_SEC/3600:.1f}hr) | "
              f"최종 예측 RUL={final_rul_hr:.2f}hr ({final_rul_cyc:.1f} cycles)")

        # ── Figure 1: Predicted RUL 추이 ──────────────────────────
        ax1 = axes1[i]
        ax1.plot(obs_pts, rul_preds, "b-", lw=1.5, label="Predicted RUL")
        ax1.axvline(N, color="gray", ls="--", lw=1, label=f"Current end (={N})")
        ax1.set_title(f"Test{tid}  (50 slices observed)")
        ax1.set_xlabel("Observation Cycle")
        ax1.set_ylabel("Predicted RUL (cycles)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.4)

        # ── Figure 2: Predicted Failure Cycle ─────────────────────
        ax2 = axes2[i]
        ax2_hi = ax2.twinx()
        ax2_hi.plot(np.arange(N), hi, color="gray", lw=1, alpha=0.35, label="HI")
        ax2_hi.set_ylabel("HI", color="gray", fontsize=9)
        ax2_hi.tick_params(axis="y", labelcolor="gray")

        ax2.plot(obs_pts, pred_fail, "b-", lw=1.5, label="Pred Failure Cycle")
        ax2.plot(obs_pts, obs_pts, color="gray", ls=":", lw=1, label="Current Obs (y=x)")
        ax2.set_title(f"Test{tid}  Final RUL≈{final_rul_hr:.1f}hr")
        ax2.set_xlabel("Observation Cycle")
        ax2.set_ylabel("Predicted Failure Cycle")

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_hi.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax2.grid(True, alpha=0.4)

    fig1.tight_layout()
    fig1.savefig(OUT_DIR / "LSTM_RUL_Test_Predictions.png", dpi=150)
    plt.close(fig1)

    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "LSTM_Failure_Timeline_Test.png", dpi=150)
    plt.close(fig2)

    # 전체 결과 저장
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT_DIR / "All_Test_RUL_results.csv", index=False)

    # 최종 요약 (각 Test의 마지막 사이클 기준 RUL)
    print("\n" + "=" * 70)
    print("  최종 RUL 예측 요약 (마지막 관측 시점 기준)")
    print("=" * 70)
    summary = []
    for tid in TEST_IDS:
        sub = df_all[df_all["test_id"] == tid]
        last = sub.iloc[-1]
        summary.append({
            "test_id": tid,
            "observed_slots": int(sub["obs_cycle"].max()),
            "observed_hours": round(float(sub["obs_cycle"].max()) * INTERVAL_SEC / 3600, 2),
            "final_rul_cycles": round(last["rul_pred_cycles"], 2),
            "final_rul_hours":  round(last["rul_pred_hours"], 2),
            "pred_failure_hours": round(last["pred_failure_hours"], 2),
        })
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(OUT_DIR / "Test_RUL_summary.csv", index=False)
    print(df_summary.to_string(index=False))
    print(f"\n[완료] {OUT_DIR}")


if __name__ == "__main__":
    main()
