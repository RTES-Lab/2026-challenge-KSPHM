# SR/0514 작업 진행 기록

## 폴더 구조

```
0514/
├── hi/
│   ├── code/
│   │   ├── hi_train.py   ← Train 베어링 HI 생성 (그리드 탐색)
│   │   └── hi_test.py    ← Test 베어링 HI 생성
│   └── output/
│       ├── train/        ← Bearing1~4 HI CSV
│       └── test/         ← Test1~6 HI CSV
└── rul/
    ├── code/
    │   ├── rul_train.py  ← LOOCV로 LSTM 성능 검증
    │   └── rul_test.py   ← Test 1~6 RUL 추론
    └── output/
        ├── train/        ← LOOCV 결과 및 로그
        └── test/         ← Test RUL 예측 결과
```

---

## 이 코드가 어디서 왔나

### 배경

현재 리더보드 1위 파이프라인 (SC 폴더 기준):

| 단계 | SC 폴더 | 설명 | 성능 |
|------|---------|------|------|
| HI 생성 | `04262255_signal_transform_v5` | Signal Transformation + FDR 가중합 | Q-Score 0.8098 |
| RUL 예측 | `04262315_lstm_rul` | 2-Layer LSTM (hidden=32, window=10) | LOOCV 0.4446 |

### 0514에서 해결하려는 문제

SC 실험에서 발견된 두 가지 약점:

1. **Test1, Test4**: K-Means 레짐 분류 실패 (일치율 64%) → FFT 기반 RPM 추정으로 교체
2. **Test2, Test5, Test6**: 관측 시작 시점에 이미 열화 진행 중 → HI 기준점을 Test 자체가 아닌 Train에서 가져와야 함

---

## HI 파이프라인 (`hi/`)

### 입력 피처

| 코드 | 피처 소스 |
|------|----------|
| `hi_train.py` | `SC/HI/04142304_signal_transform_v2/output/Bearing{N}_features_transformed.csv` |
| `hi_test.py` | `SC/HI/05072245_signal_transform_v5_test/output/Test{N}_features.csv` |

사용 피처 7개 (FEATURE_Q): `ch3_high_band, ch4_high_band, ch3_std, ch3_total_power, ch3_energy, ch3_rms, ch3_p2p`

### HI 생성 방식 (FDR)

1. 피처값에서 건강 구간 기준값(baseline μ)을 빼고 baseline으로 나눔 → 상대적 열화 비율
2. 7개 피처를 Q-Score 가중합으로 그룹별 HI 생성 후 최종 합산
3. MinMax 스케일링 + EMA 스무딩

---

## 수정 이력

### [2026-05-14] rul_train.py — 비대칭 손실함수 교체

**변경 내용:**
`nn.MSELoss()` → `AsymmetricRULLoss` (대회 채점 방식과 동일한 비대칭 손실)

```
Er = 100 * (actual - predicted) / actual  (%)
Er ≤ 0 (과대예측): score = exp(-ln(0.5) * Er / 20)  → 20% 과대 시 점수 0.5
Er > 0 (과소예측): score = exp( ln(0.5) * Er / 50)  → 50% 과소 시 점수 0.5
loss = -mean(score)  (점수 최대화)
```

과대예측이 과소예측보다 2.5배 빠르게 페널티 증가.
`y_true.abs().clamp(min=1.0)` 로 RUL≈0 구간 분모 0 방지.

---

### [2026-05-14] hi_test.py — Bug Fix (Critical × 2)

#### Bug 1: 정규화를 두 번 하는 문제

**기존 코드 흐름:**
```
Test 피처값
  → Signal Transformation: (x - μ_train) / σ_train  ← z-score, 값이 0 근처로 됨
  → FDR 비율: (값 - 기준값) / 기준값                ← 기준값이 0이라 분모 폭발
  → HI
```

FDR의 기준값으로 z-score 결과(≈ 0)가 들어가면서 `0 / 0` 문제가 발생했다.
코드에서 `1e-8`로 0 나눔을 막았지만, 결과적으로 모든 값이 수천만 배로 뻥튀기됐다.
그 뒤 `robust_clip(1%, 99%)`이 이 폭발값을 잘라냈기 때문에 코드가 돌아가긴 했지만,
**Signal Transformation이 사실상 아무 효과가 없었다.**

**수정:**
Signal Transformation(z-score) 전체 제거. FDR 하나만 사용.

#### Bug 2: 피처 소스가 달랐던 문제

**기존 코드:**
- baseline μ/σ 계산: `04140103_initial_pca_result` (구형 PCA 피처)
- 실제 정규화 적용: `05072245_signal_transform_v5_test` (v5 피처)

두 디렉토리는 서로 다른 처리과정을 거친 다른 값들이라, 계산한 μ를 다른 피처에 적용하는 셈이었다.

**수정:**
baseline 계산 소스를 `04142304_signal_transform_v2`로 변경.
v2 train 피처와 v5_test 피처는 **동일한 raw feature 컬럼** (`ch3_rms`, `ch3_p2p` 등)을 공유한다.

#### Bug 1+2 통합 수정 내용

| 항목 | 이전 | 수정 후 |
|------|------|---------|
| 정규화 방식 | z-score → FDR (이중) | FDR 단일 |
| Train baseline 소스 | `04140103_initial_pca_result` | `04142304_signal_transform_v2` |
| baseline 타입 | μ, σ (z-score용) | μ만 (FDR 기준점) |
| Test baseline 계산 | test 자체 첫 10% | Train 4개 베어링 건강 구간 평균 |
| 함수 | `signal_transform_transfer()` + `compute_train_baseline_params()` | `compute_train_fdr_baseline()` + `build_feature_ratios_from_train()` |

#### 수정 후 HI 흐름

```
Test 피처값
  → FDR 비율: (x - μ_train_건강) / |μ_train_건강|  ← Train 레짐별 건강 평균 기준
  → 그룹별 가중합 → MinMax → EMA → HI
```

이제 Test2, Test6처럼 관측 시작부터 이미 열화된 베어링은,
HI가 처음부터 높은 값으로 시작한다 (Train 건강 기준 대비 이미 크게 벗어남).

---

### [2026-05-14] rul_test.py — AsymmetricRULLoss 추가 및 교체

`nn.MSELoss()` → `AsymmetricRULLoss` (rul_train.py와 동일한 비대칭 손실 적용).
클래스 정의가 없었으므로 rul_train.py와 동일한 구현을 rul_test.py에 추가.

---

## 현재 결과 (2026-05-14 최신)

### HI Q-Score (hi_test.py 버그 픽스 후, v3: FFT+FDR Only)

v1 = 기존 K-Means+SignalTransform, v3 = 버그픽스 후 FFT+FDR

| Test | v1_Q | v3_Q | Delta | HI_end |
|------|------|------|-------|--------|
| Test1 | 0.819 | 0.517 | -0.302 | 0.769 |
| Test2 | 0.133 | 0.379 | **+0.246** | 0.366 |
| Test3 | 0.598 | 0.658 | +0.059 | 0.931 |
| Test4 | 0.594 | 0.676 | +0.082 | 0.994 |
| Test5 | 0.754 | 0.786 | +0.031 | 0.986 |
| Test6 | 0.592 | 0.654 | **+0.062** | 0.985 |
| **전체** | **0.582** | **0.612** | **+0.030** | |

- Test2, Test6 목표했던 개선 달성 ✓
- Test1 역방향 악화 (-0.30): FFT 레짐 분류가 19/31→25/25으로 바뀌면서 HI 패턴 변화

### HI Train 그리드 탐색 결과 (hi_train.py)

- V3 Best: br=0.10, alpha=0.1 → **Q-Score 0.8098** (v5 SC 최고 성능과 동일)
- V4 Best: br=0.05, alpha=0.1, latent_dim=1 → Q-Score 0.7498

### RUL 예측 (2026-05-14 전체 재실행)

| Test | 최종 예측 RUL | 시간 | HI_end | 비고 |
|------|-------------|------|--------|------|
| Test1 | 4.8 cycles | 0.81 hr | 0.77 | HI slope 음수, 신뢰도 낮음 |
| Test2 | 76.8 cycles | 12.80 hr | 0.37 | 선형 외삽 96.6과 근접, 합리적 |
| Test3 | 3.0 cycles | 0.50 hr | 0.93 | 선형 외삽 3.5과 일치 |
| Test4 | 2.6 cycles | 0.43 hr | 0.99 | 수명 末期 |
| Test5 | 2.6 cycles | 0.44 hr | 0.99 | 수명 末期 |
| Test6 | 2.6 cycles | 0.43 hr | 0.98 | 이전 70.5→2.6 (버그 픽스 효과) |

LSTM LOOCV Score: **0.5749** (이전 0.4579 대비 +0.127 개선)

---

## TODO

### HI

- [x] **hi_test.py 재실행** — 버그 픽스 후 재실행 완료 (2026-05-14). Test2/6 개선, Test1 악화.
- [x] **hi_train.py 입력 경로 수정** — 확인 결과 v5가 v2 피처 동일 사용. 경로 변경 불필요. train_v4 폴더 생성 후 그리드 탐색 완료 (2026-05-14).

### RUL

- [x] **비대칭 Loss 함수 적용** — rul_train.py, rul_test.py 모두 `AsymmetricRULLoss` 적용 완료 (2026-05-14)
- [x] **Test6 RUL 재확인** — HI 수정 후 70.5 → 2.6 cycles로 정상화 (2026-05-14)
- [x] **RUL 스케일 불일치 검토** — 분석 완료 (2026-05-14). HI 0→1 정규화로 실질 영향 제한적. 다만 Bearing4 HI 종점(0.63)이 낮아 LOOCV 0.43으로 낮음. Test1 HI slope 음수 → RUL 예측 불안정.

### 전반

- [x] hi_test.py 수정 후 rul_test.py 전체 재실행 완료 (2026-05-14). LOOCV 0.4579 → 0.5749.

---

## [2026-05-14] 구조적 개선 — Train/Test HI 파이프라인 대칭화

### 문제
- hi_train.py: 자기 자신의 첫 10% 데이터를 baseline으로 사용 (내부 baseline)
- hi_test.py: Train 4개 베어링 건강구간 평균을 baseline으로 사용 (외부 baseline)
- → LSTM이 학습한 HI 패턴 ≠ Test에서 보는 HI 패턴 (분포 이동)

### 수정 내용

**hi_train.py**
- `build_feature_ratios()` → `build_feature_ratios_external()` (레짐별 LOO 외부 baseline)
- `compute_bearing_baseline(dfs, br, exclude_bid)` 추가: 현재 베어링 제외한 3개 기준
- `v3_pipeline(df, br, alpha, baseline)`: `baseline` 파라미터 받아서 레짐별 FDR 적용
- 그리드 탐색: 각 베어링마다 LOO baseline 계산 후 평가

**hi_test.py**
- `BASELINE_RATIO = 0.10` (LOO 그리드 탐색 결과 반영)

**rul_train.py / rul_test.py — RUL 정규화 추가**
- `create_sequences()`: RUL 타겟을 `MEAN_TRAIN_LIFE=116.5`로 정규화 → [0, ~1.2] 범위 균일화
- `AsymmetricRULLoss`: 손실 계산은 절대 cycles로 역변환 후 수행 (clamp(min=1.0) 유지)
- LOOCV 역정규화: `rul_pred_abs = rul_pred_norm * N_test`
- Test 역정규화: `rul_pred_abs = rul_pred_norm * MEAN_TRAIN_LIFE`

### 결과

**Train 그리드 탐색 (LOO 방식)**
- V3 Best: br=0.10, alpha=0.1 → LOO Q-Score 0.7311

**Test HI Q-Score (br=0.10, LOO)**

| Test | v1_Q | v3_Q | Delta |
|------|------|------|-------|
| Test1 | 0.819 | 0.517 | -0.302 |
| Test2 | 0.133 | 0.379 | +0.246 |
| Test3 | 0.598 | 0.658 | +0.059 |
| Test4 | 0.594 | 0.676 | +0.082 |
| Test5 | 0.754 | 0.786 | +0.031 |
| Test6 | 0.592 | 0.654 | +0.062 |
| **전체** | **0.582** | **0.612** | **+0.030** |

**LOOCV RUL Score: 0.5956** (이전 0.5749 대비 +0.021)

| Bearing | 이전 | 현재 |
|---------|------|------|
| 1 | 0.714 | 0.695 |
| 2 | 0.578 | 0.521 |
| 3 | 0.599 | 0.619 |
| 4 | 0.430 | **0.551** |

**Test RUL 최종 예측 (2026-05-14 최신)**

| Test | 예측 RUL | 시간 | HI_end | 비고 |
|------|----------|------|--------|------|
| Test1 | 4.8 cycles | 0.80 hr | 0.796 | HI slope 불안정, 신뢰도 낮음 |
| Test2 | 45.6 cycles | 7.61 hr | 0.343 | 중간 열화, 수명 남음 |
| Test3 | 3.3 cycles | 0.55 hr | 0.933 | 수명 末期 |
| Test4 | 3.0 cycles | 0.49 hr | 0.982 | 수명 末期 |
| Test5 | 3.0 cycles | 0.49 hr | 0.995 | 수명 末期 |
| Test6 | 2.9 cycles | 0.49 hr | 0.885 | 수명 末期 |

---

## [2026-05-14] 앙상블 파이프라인 (옵션 B)

단일 최고 성능을 낸 LSTM-A 외에, 시간 도메인(HI-B) 기반 LSTM-B와 비선형 Tree 계열인 LightGBM(flat 피처)을 결합하는 Multi-Model 앙상블 파이프라인을 구축했습니다.

### 문제 및 해결 (LightGBM 비대칭 손실함수)
- 초기에는 LightGBM 학습 시 공식 대회 스코어를 그대로 사용하려 했으나, 오차가 클 경우 지수 함수의 꼬리 부분에서 **Vanishing Gradient**가 발생하여 점수가 0.25에 머무는 현상 확인.
- 이를 해결하기 위해 과대 예측 시 2.5배 페널티를 주는 **Proxy Asymmetric MSE**를 커스텀 손실함수(`lgbm_asymmetric_obj`)로 도입하여 학습 정상화.

### LOOCV 결과 (앙상블 재도전)

| Test Bearing | LSTM-A (HI-A, 5 seeds) | LSTM-B (HI-B, 5 seeds) | LightGBM (HI-A flat) | **Ensemble** |
|--------------|------------------------|------------------------|----------------------|--------------|
| Bearing 1    | 0.6225                 | 0.5592                 | 0.5473               | **0.6691**   |
| Bearing 2    | 0.6038                 | 0.4283                 | 0.3472               | 0.4757       |
| Bearing 3    | 0.6800                 | 0.3859                 | 0.4424               | 0.6451       |
| Bearing 4    | 0.5209                 | 0.5736                 | 0.5830               | **0.5850**   |
| **평균**     | **0.6068**             | 0.4867                 | 0.4800               | **0.5937**   |

- **결과:** Bearing 1과 Bearing 4에서는 앙상블 점수가 단일 모델 최고점(LSTM-A)보다 높게 나타나며 상호 보완성을 입증.
- 다만, Bearing 2에서 LSTM-B와 LGBM의 저조한 성적이 앙상블 점수를 깎아 평균적으로는 0.5937을 기록 (LSTM-A 단독 평균 0.6068에 살짝 미치지 못함).

### Test 예측 결과 (최종 앙상블 가중치: LSTM-A 39%, LSTM-B 31%, LGBM 30%)

| Test | 예측 RUL (cycles) | 예측 시간 (hr) |
|------|-------------------|----------------|
| Test1| 7.9 cycles        | 1.32 hr        |
| Test2| 22.5 cycles       | 3.76 hr        |
| Test3| 22.7 cycles       | 3.79 hr        |
| Test4| 12.0 cycles       | 1.99 hr        |
| Test5| 10.6 cycles       | 1.76 hr        |
| Test6| 11.1 cycles       | 1.85 hr        |

*비고: 다양한 모델의 의견이 결합되어 기존에 너무 짧았던 수명 예측이 완화되고 현실적인 값으로 예측됨.*

---

## 다음 개선 후보

- [ ] **Test1 HI 개선** — slope 음수 (Q=0.517). FFT 레짐 분류 결과 확인 및 postprocess_score 부호 반전 동작 디버깅.
- [ ] **Test2 RUL 신뢰도 검증** — 현재 앙상블 기준 22.5 cycles 예측 중. 선형 외삽 등 독립적 추정치와 비교.
- [ ] **LSTM 아키텍처 추가 탐색** — hidden_size, seq_length 그리드 탐색으로 LOOCV 0.6+ 안정화.
- [ ] **앙상블 가중치 튜닝** — 성능이 낮은 모델의 가중치(최소 0.1) 하한선을 낮추거나 동적으로 클램핑하여 평균 점수 최적화.
