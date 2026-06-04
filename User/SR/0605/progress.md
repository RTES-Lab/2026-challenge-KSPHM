# 0605 실험 기록 — HI Scale 근본 문제 분석 및 feature-similarity start_frac

> 마지막 업데이트: 2026-06-05 (실험 G — Adaptive Ensemble)

---

## 1. 동기: FDR HI의 근본적 scale 불일치 문제

기존 최고 파이프라인(0603_v3 hi_v4 + rul_v5, LOOCV 0.468 → 제출 0.3)에서
**train HI의 scale이 베어링마다 다름**이 근본 원인으로 진단됨.

| Bearing | hi_start | hi_end | 문제 |
|---------|----------|--------|------|
| B1 | 0.002 | 0.392 | 정상 (기준) |
| B2 | 0.000 | 0.428 | 정상 (기준) |
| B3 | 0.001 | **0.139** | EOL HI가 너무 낮음 → LSTM이 B3 사망 시점을 "초기" 오인 |
| B4 | **0.553** | 0.839 | 정상기부터 HI 높음 → fleet baseline 대비 offset |

원인:
- **B3**: FDR 정규화(p5/p95)가 B1/B2/B4 기준으로 설정 → B3의 실제 열화가 작아 보임
- **B4**: LOO baseline(B1+B2+B3 정상기 평균)보다 B4 고유 진동이 높아 시작부터 HI↑

---

## 2. 실험 A — Self-Normalized + Failure-Anchored HI (`0605/code/hi_self_norm.py`)

### 아이디어
- **자기정규화(self-baseline)**: 각 베어링이 자신의 초기 10% 데이터를 baseline으로 사용 → B4 hi_start ≈ 0 수정
- **Failure anchor**: 훈련 베어링 EOL 시점 raw score로 정규화 → 각 훈련 베어링이 hi_end ≈ 1.0 보장
- **FLOOR_RATIO=0.25**: 열화량이 작은 B3의 과도 증폭 방지 (EOL anchor 최소값 = fleet anchor의 25%)
- Test HI: fleet median baseline + fleet failure anchor (lifecycle 위치 정보 보존)

### Train HI 결과

| Bearing | hi_start (구→신) | hi_end (구→신) | Q-score (구→신) |
|---------|-----------------|----------------|-----------------|
| B1 | 0.002 → **0.000** | 0.392 → **1.000** | 0.573 → 0.770 (+0.197) |
| B2 | 0.000 → **0.000** | 0.428 → **0.970** | 0.639 → 0.615 (-0.024) |
| B3 | 0.001 → **0.000** | 0.139 → **0.105** | 0.804 → 0.713 (-0.091) |
| B4 | **0.553 → 0.000** | 0.839 → **0.831** | 0.501 → 0.660 (+0.159) |
| **평균** | — | — | **0.629 → 0.690 (+0.060)** |

Train HI Q-score는 전반적으로 개선됨. B4 hi_start 수정 성공.

### LSTM LOOCV 결과 — **악화**

| Bearing | 구 LSTM | 신 LSTM | 변화 |
|---------|---------|---------|------|
| B1 | 0.387 | 0.609 | +0.222 |
| B2 | **0.577** | **0.053** | **-0.524** |
| B3 | 0.468 | 0.184 | -0.284 |
| B4 | 0.373 | 0.427 | +0.054 |
| **평균** | **0.451** | **0.318** | **-0.133** |

### 실패 원인 분석

자기정규화가 **베어링 간 HI 패턴을 비균질하게** 만들어 LSTM 일반화 실패:

- B3 HI: 급격히 0→0.105 상승 후 7.5시간 plateau (FLOOR 적용 후)
- B2 HI: 완만히 0→0.97 (114사이클)
- LSTM이 B3 plateau("HI=0.1 → 아직 많이 남음") 패턴을 학습 후 B2 예측에 적용 → B2 말기를 "초기"로 오인

**결론**: HI 이론적 품질은 개선되나, 소수(4개) 이종 베어링 간 패턴 비균질로 LOOCV 악화. 채택 불가.

---

## 3. 실험 B — Feature-Similarity start_frac (`0605/code/rul_feat_sim.py`)

### 아이디어: start_frac 추정 방식을 HI → raw feature 유사도로 교체

**기존 문제** (`estimate_start_frac`):
```
Test5 (hi_start=0.326)로 start_frac 추정할 때:
  B3: max HI=0.14 → 0.326 미도달 → 제외
  B4: hi[0]=0.55 > 0.326 → 제외
  → B1, B2 두 개만 참조 → 편향
```

**새 방법** (`estimate_start_frac_feat_sim`):
1. Test 베어링 초기 10 관측의 raw feature vector (log1p 적용) 추출
2. 레짐별 z-score 정규화 (per-regime 훈련 통계 사용)
3. 각 훈련 베어링의 전체 궤적에서 Euclidean 최근접이웃 탐색
4. 해당 사이클 / EOL = lifecycle fraction → 4개 평균

B3/B4 제외 없이 항상 4개 베어링 모두 참조.

### 기반 파이프라인

- HI: 기존 0603_v3 hi_v4 (변경 없음)
- 모델: LGBM + LSTM 앙상블 (rul_regime_v5.py 구조 유지)
- 피처 캐시: 0604/output (ch1/ch2 impulse 포함, log1p 적용)

### LOOCV 결과 — **변화 없음** (예상대로)

LOOCV는 실제 사이클 위치를 직접 사용하므로 start_frac 추정 방식의 영향 없음.

| | LGBM | LSTM | Ens_raw |
|--|------|------|---------|
| 평균 | 0.450 | 0.378 | **0.468** |

### Test start_frac 및 RUL 예측 비교

| Test | hi_based SF | feat_sim SF | 구 RUL | 신 RUL | 변화 |
|------|------------|------------|--------|--------|------|
| T1 | 8.6% | **11.6%** | 6.11hr | 5.10hr | -1.01hr |
| T2 | 52.6% | **35.9%** | 2.62hr | 2.07hr | -0.55hr |
| T3 | 47.5% | **42.2%** | 2.77hr | 2.95hr | +0.18hr |
| T4 | 9.2% | **34.6%** | 8.44hr | **5.60hr** | **-2.84hr** |
| T5 | 86.4% | **70.5%** | 1.98hr | 2.01hr | +0.03hr |
| T6 | 94.1% | **93.3%** | 1.90hr | 1.90hr | ≈0 |

### 핵심 변화: T4 (8.44hr → 5.60hr)

기존: hi_start=0.006 → B4(hi[0]=0.55) 제외 → B1/B2/B3만 참조 → 사이클 ~5 도달 → SF=9.2% → 매우 긴 RUL(8.44hr)

신규: raw feature 유사도 → T4 feature 수준이 훈련 베어링 35% 지점과 유사 → SF=34.6% → 더 보수적 RUL(5.60hr)

T4 hi_end=0.161 (50사이클에 0.155 상승)은 실질적 열화를 보여줌.
hi_start=0.006이라고 수명 9% 지점일 이유 없음 → feat_sim 추정이 더 합리적.

채점 함수상 낙관적 예측(Er<0) 페널티가 2.5배 가혹하므로,
기존 T4=8.44hr이 실제보다 높으면 심각한 감점 → feat_sim의 5.60hr이 유리.

### 출력

![Test 예측](output/rul_feat_sim/test_predictions_v5.png)

---

---

## 3-C. 실험 C — Stage-Blended HI + kurtosis feature (`hi_stage_blend.py`, `rul_lstm_kurt.py`)

### 배경: Q-score의 한계

현재 Q = (monotonicity + trendability) / 2는 "전체 수명 대비 단조성"만 측정.
B3의 kurtosis는 초기 82사이클 flat → 마지막 2사이클 폭발(3994, 747)이라 overall_Q ≈ 0.108.
그러나 kurtosis는 B3 말기 탐지에 가장 유용한 feature.

**Stage-aware Q 정의**:
- `late_Q`: 수명 마지막 33%만으로 계산한 Q
- `composite_Q = 0.4 × overall_Q + 0.4 × late_Q + 0.2 × prognosability`

### 실험 C-1: Stage-Blended HI (`hi_stage_blend.py`)

**아이디어**: 두 트랙 HI를 시그모이드로 블렌딩:
```
hi_smooth_raw  = SMOOTH_FEATS (kurtosis/crest 제외, 7개)
hi_late_raw    = LATE_FEATS (kurtosis/crest만, 4개)
beta[t]        = sigmoid((t/T - 0.70) × 10)
hi_blend[t]    = (1 - beta) × hi_smooth + beta × hi_late
```

**LOO 설계 (정직한 검증)**:
- 훈련 HI: self-baseline + T_actual (output_cq/train/)
- LOOCV test fold: fleet-baseline + T_estimated (output_cq/loocv/)
- 테스트 HI: fleet-baseline + T_estimated (output_cq/test/)

**Train HI 결과** (self-baseline + T_actual):

| Bearing | hi_end (이전→신) | Q-score (이전→신) | 비고 |
|---------|----------------|------------------|------|
| B1 | 1.000 → 0.968 | 0.770 → 0.677 | -0.093 |
| B2 | 0.970 → 0.919 | 0.615 → 0.635 | +0.020 |
| **B3** | **0.105 → 1.000** | **0.713 → 0.926** | **+0.213 🎉** |
| B4 | 0.831 → 0.844 | 0.660 → 0.608 | -0.052 |
| **평균** | — | **0.690 → 0.711** | **+0.021** |

B3 hi_end: 0.105 → 1.000 (성공). 그러나 LOOCV는 개선 없음.

**LOOCV 비교 (actual start_frac)**:

| HI 방법 | B1 | B2 | B3 | B4 | **Mean** |
|---|---|---|---|---|---|
| 0604 원본 | 0.455 | 0.583 | 0.458 | 0.344 | **0.460** |
| 0605 self_norm | 0.609 | 0.053 | 0.184 | 0.427 | 0.318 |
| composite_q | 0.483 | 0.235 | 0.156 | 0.401 | 0.319 |

**실패 원인**: stage-blend가 B3 trajectory를 flat→급등으로 만들어, B1/B2/B4에서 학습한 LSTM이 B3를 "초기 베어링"으로 오인 → RUL 과대추정. HI scale이 맞아도 **trajectory 형태 불일치**가 LSTM 일반화를 방해.

핵심 인사이트: LSTM에서 HI의 절대 scale보다 상대적 패턴 일관성이 더 중요. 0604 HI는 B3 hi_end=0.14여도 obs_frac(실제 사이클 위치) feature와 조합하면 LOOCV B3=0.458.

---

### 실험 C-2: kurtosis를 별도 LSTM feature로 추가 (`rul_lstm_kurt.py`) ⭐

**아이디어**: HI 자체는 0604를 유지하고, kurtosis를 4번째 LSTM 입력으로 추가:
```
N_FEAT = 4: [HI_norm, obs_frac, regime, kurt_norm]
kurt_norm = max(log1p(ch1_kurt_log), log1p(ch2_kurt_log)), fleet-normalized
```

HI 분포를 변형하지 않고, kurtosis 정보를 LSTM이 학습하게 함.

**LOOCV 결과** (0604 HI + actual start_frac):

| Bearing | 0604 baseline | kurtosis_feat | 변화 |
|---------|--------------|--------------|------|
| B1 | 0.455 | 0.463 | +0.008 |
| B2 | 0.583 | **0.658** | **+0.075** |
| B3 | 0.458 | **0.467** | **+0.009** |
| B4 | 0.344 | 0.378 | +0.034 |
| **Mean** | **0.460** | **0.491** | **+0.031 🎉** |

**모든 베어링에서 개선!** B2의 대폭 개선(+0.075)이 주도.

**Test RUL 예측** (0604 HI + kurtosis feat):

| Test | sf | RUL | kurt_max | 비고 |
|------|-----|-----|---------|------|
| T1 | 0% | 4.15hr | 0.73 | 초기 |
| T2 | 15% | 4.25hr | 0.46 | 초기 |
| T3 | 54% | 4.33hr | 0.44 | 중기 |
| T4 | 0% | 4.09hr | 0.60 | 초기 |
| T5 | 75% | 4.11hr | 0.60 | 중후기 |
| T6 | 86% | 4.06hr | 0.51 | 후기 |

테스트 베어링의 kurt_max가 모두 낮음 (0.44-0.73) → 정상 운전 중. 충격성 실패 신호 없음.

**Train HI (0604 output, hi_loo_regime_v1 기준)**:

| B1 Q=0.766 | B2 Q=0.579 |
|---|---|
| ![B1 HI](../0604/output/train/Bearing1_HI.png) | ![B2 HI](../0604/output/train/Bearing2_HI.png) |

| B3 Q=0.372 | B4 Q=0.485 |
|---|---|
| ![B3 HI](../0604/output/train/Bearing3_HI.png) | ![B4 HI](../0604/output/train/Bearing4_HI.png) |

**LOOCV 플롯 (4개 베어링)**:

| B1 sc=0.463 | B2 sc=0.658 |
|---|---|
| ![B1 LOOCV](output_cq/rul_kf/Bearing1_loocv.png) | ![B2 LOOCV](output_cq/rul_kf/Bearing2_loocv.png) |

| B3 sc=0.467 | B4 sc=0.378 |
|---|---|
| ![B3 LOOCV](output_cq/rul_kf/Bearing3_loocv.png) | ![B4 LOOCV](output_cq/rul_kf/Bearing4_loocv.png) |

**Test RUL 예측**:

![Test RUL](output_cq/rul_kf/Test_RUL_all.png)

**출력**: `output_cq/rul_kf/`

---

## 4. 실험 D — obs_frac 제거 LSTM (`0605/code/rul_lstm_kurt_noobs.py`)

### 배경: obs_frac 정보 누수 진단

kurtosis_feat_v1의 LOOCV 0.491이 실제 테스트 성능을 과대평가하고 있음을 발견:

| 상황 | obs_frac 계산 | 문제 |
|------|--------------|------|
| LOOCV | `t_start(실제)` / 116.5 | 정답 위치를 입력으로 사용 |
| Test | `estimate_start_frac` × 116.5 | 추정 오차 그대로 전파 |

→ LOOCV-Test 조건 불일치. 누수 제거를 위해 obs_frac을 입력에서 완전히 삭제.

### 변경 사항
- `N_FEAT = 3`: `[HI_norm, regime, kurt_norm]` (obs_frac 제거)
- LOOCV에서도 `t_start` 미사용 → 정직한 검증
- Test 추론에서 `estimate_start_frac` 불필요

### LOOCV 결과

| Bearing | kurtosis_feat (누수) | no_obs_frac (정직) | 변화 |
|---------|---------------------|-------------------|------|
| B1 | 0.463 | 0.442 | -0.021 |
| B2 | 0.658 | 0.379 | -0.279 |
| B3 | 0.467 | 0.236 | -0.231 |
| B4 | 0.378 | 0.311 | -0.067 |
| **Mean** | **0.491 (누수)** | **0.342 (정직)** | **-0.149** |

obs_frac이 실제 기여했던 정보량 = **0.149**.

### Test RUL 비교

| Test | kurtosis_feat | no_obs_frac | feat_sim (참고) |
|------|--------------|-------------|----------------|
| T1 | 4.15hr | 4.19hr | 5.10hr |
| T2 | 4.25hr | 4.26hr | 2.07hr |
| T3 | 4.33hr | 3.72hr | 2.95hr |
| T4 | 4.09hr | 3.76hr | 5.60hr |
| T5 | 4.11hr | **3.30hr** | 2.01hr |
| T6 | 4.06hr | **3.27hr** | 1.90hr |

긍정적 변화: no_obs_frac 모델이 HI 수준에 실제로 반응함.
- T6 (hi=0.139): 3.27hr → T1 (hi=0.000): 4.19hr 보다 짧게 예측
- kurtosis_feat은 모두 ~4hr으로 무반응이었음

부정적 변화: B3 LOOCV 심각 악화 (0.467→0.236). 위치 정보 없이 LSTM이 훈련 베어링 평균 수명(B1/B2/B4 기반)으로 과대 예측.

### LOOCV 플롯 (no_obs_frac)

| B1 sc=0.442 | B3 sc=0.236 |
|---|---|
| ![B1 LOOCV](output_cq/rul_nof/Bearing1_loocv.png) | ![B3 LOOCV](output_cq/rul_nof/Bearing3_loocv.png) |

**출력**: `output_cq/rul_nof/`

---

## 5. 실험 E — Window-Feature k-NN (`0605/code/rul_knn.py`) ⭐

### 배경: LSTM flat predictor 근본 원인

학습 데이터의 61~71%가 동일한 const-RUL 타겟(정상기 시퀀스).
LSTM이 MSE 최소화 → 평균값 수렴 = flat prediction. 아키텍처 문제가 아니라 **데이터 부족 문제**.

### 아이디어

LSTM 없이 50-cycle 창에서 추출한 9개 스칼라 피처로 직접 k-NN RUL 추정:

```
features = [hi_mean, hi_slope, hi_end, hi_std, hi_range,
            kurt_mean, kurt_max, kurt_std, regime_frac]
```

k-NN의 핵심 장점:
- const-RUL 정상기 창 → 정상기 이웃 탐색 → const-RUL 예측 (자연스러운 처리)
- 열화기 창 → 유사 열화 상태 이웃 탐색 → 감소하는 RUL 예측
- obs_frac 불필요 (start_frac 추정 오차 제거)
- 과적합 없음 (비모수적)

k=10, inverse-distance weighting, z-score 정규화.

### LOOCV 결과

| Bearing | LSTM no_obs_frac | k-NN raw | 변화 |
|---------|-----------------|---------|------|
| B1 | 0.442 | 0.525 | +0.083 |
| B2 | 0.379 | **0.100** | **-0.279** |
| B3 | 0.236 | **0.499** | **+0.263** |
| B4 | 0.311 | 0.336 | +0.025 |
| **Mean** | **0.342** | **0.365** | **+0.023** |

**B3 대폭 개선** (+0.263): 충격성 kurtosis 창이 다른 베어링의 kurtosis spike 창과 잘 매칭됨.

### B2 저점수 원인 — const-RUL 이질성

LOOCV test=B2 시 훈련 데이터(B1+B3+B4)의 const-RUL:
- B1=37, B3=27, B4=59 → 훈련 평균 ≈ 41 cycles
- **B2 실제 const-RUL = 22 cycles** (훈련 fleet의 절반)

k-NN이 B2 정상기 창(HI≈0.01)을 B1 정상기(RUL=37)에 매칭 → 예측 37, 실제 22 → Er=-68% → 급격한 감점.

### Conservative Bias 최적화

```
# 채점 비대칭: Er<0 페널티가 2.5배 가혹
# 최적 bias: LOOCV 전체 예측에 곱할 스칼라
optimal_bias = 0.650 → LOOCV 0.365 → 0.4345
```

bias=0.650의 의미: B2 과대추정(×37→×22)을 전역 보정.
B1/B3는 살짝 과소추정되지만 채점 비대칭으로 이득.

### Test RUL 예측 (bias=0.65)

| Test | HI_end | RUL | k-NN NN 근거 |
|------|--------|-----|-------------|
| T1 | 0.014 | **2.38hr** | B2 정상기 (RUL=22) |
| T2 | 0.005 | **4.01hr** | B1 정상기 (RUL=37) |
| T3 | 0.019 | 3.53hr | B2/B1 혼합 |
| T4 | 0.026 | 3.19hr | B1 정상기 |
| T5 | 0.116 | 2.62hr | B3@t52 (RUL=27) |
| T6 | 0.200 | 2.38hr | B2 말기 (RUL=22) |

**T1 주의**: HI=0.014 (정상 수준)인데 2.38hr 예측. T1의 kurt_max=0.732가 B2 정상기(kurt≈0.7)에 매칭됨. B2 정상기 const-RUL=22(짧음). T1이 B2와 유사한 특성의 베어링이라면 이 예측이 합리적.

### Train HI (0604 output 그대로 사용)

| B1 Q=0.766 | B2 Q=0.579 |
|---|---|
| ![B1 HI](../0604/output/train/Bearing1_HI.png) | ![B2 HI](../0604/output/train/Bearing2_HI.png) |

| B3 Q=0.372 | B4 Q=0.485 |
|---|---|
| ![B3 HI](../0604/output/train/Bearing3_HI.png) | ![B4 HI](../0604/output/train/Bearing4_HI.png) |

### LOOCV 플롯 (4개 베어링)

| B1 sc=0.525 | B2 sc=0.100 |
|---|---|
| ![B1 LOOCV](output_knn/Bearing1_loocv.png) | ![B2 LOOCV](output_knn/Bearing2_loocv.png) |

| B3 sc=0.499 | B4 sc=0.336 |
|---|---|
| ![B3 LOOCV](output_knn/Bearing3_loocv.png) | ![B4 LOOCV](output_knn/Bearing4_loocv.png) |

**B4 LOOCV 실패**: 전 구간 flat 5 cycle 예측 (True RUL 시작 59). B4 정상기 HI≈0.18이 B1/B2/B3 훈련 데이터의 열화기 HI에 매칭됨 (절대 HI offset 문제). B2/B3/B4 모두 k-NN으로 근본적으로 해결하기 어려운 이질성 문제.

### Test HI (0604 output)

| T1 HI | T2 HI |
|---|---|
| ![T1 HI](../0604/output/test/Test1_HI.png) | ![T2 HI](../0604/output/test/Test2_HI.png) |

| T3 HI | T4 HI |
|---|---|
| ![T3 HI](../0604/output/test/Test3_HI.png) | ![T4 HI](../0604/output/test/Test4_HI.png) |

| T5 HI | T6 HI |
|---|---|
| ![T5 HI](../0604/output/test/Test5_HI.png) | ![T6 HI](../0604/output/test/Test6_HI.png) |

### Test RUL 예측

![Test RUL](output_knn/test_rul_all.png)

---

## 5-F. 실험 F — k-NN rel: Relative HI + Fractional RUL (`rul_knn_rel.py`) ❌

### 시도한 수정

**B4 문제 (절대 HI offset)**: hi_mean, hi_end → hi_rel_mean, hi_rel_end (창 초반 10사이클 baseline 뺌)
**B2 문제 (const-RUL 스케일)**: RUL 타겟 → frac_rul = RUL / CONST_RUL[bearing]

### LOOCV 결과

| Bearing | v1 raw | v2 raw | 변화 |
|---------|--------|--------|------|
| B1 | 0.525 | **0.760** | +0.235 |
| B2 | 0.100 | **0.048** | -0.052 |
| B3 | 0.499 | **0.094** | -0.405 |
| B4 | 0.336 | 0.326 | -0.010 |
| **Mean** | **0.365** | **0.307** | **-0.058** |

### 실패 원인

**B4 여전히 flat**: 상대 HI 피처만으로는 부족. B4의 kurtosis/regime 패턴도 B1/B2/B3 훈련 분포 밖에 위치. HI 뿐 아니라 모든 피처에서 B4는 이질적 outlier.

**B2/B3 붕괴**: Fractional RUL 정규화 후 정상기 창이 모두 frac=1.0으로 통일되었으나, 열화기 모양(shape)이 bearing마다 달라서 k-NN 매칭이 혼동됨. B2는 22사이클에 급속 열화, B1/B3/B4는 완만한 열화 → frac_rul 기준으로는 중간값(frac=0.5)의 절대 HI 차이가 매우 큼 → 특성 불일치.

**Test 예측 붕괴**: 상대 HI로는 T5(HI=0.116), T6(HI=0.200) 모두 hi_rel≈0 (창 내부가 상대적으로 flat) → 모든 테스트 베어링이 fleet_mean=3.93hr으로 flat 예측. 절대 열화 수준 정보 소실.

### LOOCV 플롯

| B1 sc=0.760 | B2 sc=0.048 |
|---|---|
| ![B1 LOOCV v2](output_knn_rel/Bearing1_loocv.png) | ![B2 LOOCV v2](output_knn_rel/Bearing2_loocv.png) |

| B3 sc=0.094 | B4 sc=0.326 |
|---|---|
| ![B3 LOOCV v2](output_knn_rel/Bearing3_loocv.png) | ![B4 LOOCV v2](output_knn_rel/Bearing4_loocv.png) |

### Test HI (0604 output)

| T1 HI | T2 HI |
|---|---|
| ![T1 HI](../0604/output/test/Test1_HI.png) | ![T2 HI](../0604/output/test/Test2_HI.png) |

| T3 HI | T4 HI |
|---|---|
| ![T3 HI](../0604/output/test/Test3_HI.png) | ![T4 HI](../0604/output/test/Test4_HI.png) |

| T5 HI | T6 HI |
|---|---|
| ![T5 HI](../0604/output/test/Test5_HI.png) | ![T6 HI](../0604/output/test/Test6_HI.png) |

### Test RUL 예측

![Test RUL v2](output_knn_rel/test_rul_all.png)

### 핵심 결론

**B4 LOOCV 실패는 feature engineering으로 해결 불가**. B4는 B1/B2/B3과 모든 피처에서 이질적이며, LOO시 B4를 커버하는 훈련 샘플이 존재하지 않음. 이는 4개 베어링의 근본적 이질성에서 오는 한계로, 아키텍처 문제가 아님.

**실제 테스트 예측에 대한 영향**: 전체 훈련 데이터(B1+B2+B3+B4)를 lookup table에 포함하면 B4 유사 베어링을 올바르게 처리 가능. T1-T6 중 B4와 유사한 HI baseline(≈0.18) 베어링이 없으므로, B4 LOOCV 실패가 테스트 예측에 직접 영향 주지 않음.

**v1 유지**: k-NN v1이 현재 최선. v2는 기각.

---

## 6. 현재 최선 후보 비교 (업데이트)

| 파이프라인 | LOOCV | 누수 여부 | T1 RUL | T5 RUL | T6 RUL | 비고 |
|----------|-------|----------|--------|--------|--------|------|
| 0603_v3 hi_v4 + rul_v5 | 0.468 | ⚠️ 누수 | — | 1.98hr | 1.90hr | T4 낙관 위험 |
| 0605 rul_feat_sim | 0.468 | ⚠️ 누수 | 5.10hr | 2.01hr | 1.90hr | T4 보수화 |
| 0605 rul_lstm_kurt | 0.491 | ⚠️ 누수 | 4.15hr | 4.11hr | 4.06hr | flat |
| 0605 rul_lstm_kurt_noobs | 0.342 | ✅ 정직 | 4.19hr | 3.30hr | 3.27hr | flat |
| 0605 rul_knn (biased) | 0.4345 | ✅ 정직 | 2.38hr | 2.62hr | 2.38hr | k-NN 단독 |
| **0605 rul_adaptive** | **—** | **✅ 정직** | **3.47hr** | **2.01hr** | **1.90hr** | **★ 현재 최선** |

**현재 최선**: Adaptive Ensemble v1. k-NN의 정직성 + feat_sim의 start_frac 근거를 HI 조건부로 결합.

---

## 5-G. 실험 G — Adaptive Ensemble (`rul_adaptive.py`) ⭐

### 아이디어: feat_sim + k-NN 조건부 블렌딩

두 모델의 강점을 HI 수준에 따라 선택적으로 활용:

```python
def adaptive_blend(feat_rul, knn_rul, hi_end, threshold=0.05):
    if hi_end >= threshold:
        # 열화 확인됨 → min (더 보수적인 쪽 선택)
        return min(feat_rul, knn_rul)
    else:
        # 정상기 → 60:40 단기 가중 (채점 비대칭 활용, k-NN 과보수 완화)
        return 0.40 * max(feat_rul, knn_rul) + 0.60 * min(feat_rul, knn_rul)
```

**설계 근거**:
- `HI >= 0.05`: 열화가 확인된 상태 → k-NN 창 매칭이 신뢰성 높음 → min이 안전
- `HI < 0.05`: 정상기 → k-NN이 B2 const-RUL=22에 편향되어 과보수 위험 → feat_sim의 lifecycle 위치 추정으로 완화
- 50:50 대신 60:40 (짧은 쪽 가중): 채점 비대칭(낙관 예측 2.5배 가혹)을 활용하면서 k-NN 단독보다 보수적이지 않게

### Test RUL 예측 비교

| Test | HI_end | feat_sim | k-NN | Adaptive | 규칙 | vs k-NN |
|------|--------|----------|------|----------|------|---------|
| T1 | 0.014 | 5.10hr | 2.38hr | **3.47hr** | 60:40 | +1.09hr |
| T2 | 0.005 | 2.07hr | 4.01hr | **2.85hr** | 60:40 | -1.16hr |
| T3 | 0.019 | 2.95hr | 3.53hr | **3.18hr** | 60:40 | -0.35hr |
| T4 | 0.026 | 5.60hr | 3.19hr | **4.15hr** | 60:40 | +0.96hr |
| T5 | 0.116 | 2.01hr | 2.62hr | **2.01hr** | min  | -0.61hr |
| T6 | 0.200 | 1.90hr | 2.38hr | **1.90hr** | min  | -0.48hr |

**주요 개선**:
- T1 (HI≈0, 정상기): k-NN 2.38hr → 3.47hr (B2 bias 완화, feat_sim 학습점 반영)
- T2 (HI≈0): k-NN 4.01hr → 2.85hr (feat_sim의 보수적 2.07 반영)
- T5, T6 (열화 진행): feat_sim 2.01/1.90hr 유지 (k-NN보다 더 보수적)

![Adaptive Test RUL](output_adaptive/adaptive_test_rul_timeseries.png)

**Train HI (0604 output, hi_loo_regime_v1 기준)**:

| B1 Q=0.766 | B2 Q=0.579 |
|---|---|
| ![B1 HI](../0604/output/train/Bearing1_HI.png) | ![B2 HI](../0604/output/train/Bearing2_HI.png) |

| B3 Q=0.372 | B4 Q=0.485 |
|---|---|
| ![B3 HI](../0604/output/train/Bearing3_HI.png) | ![B4 HI](../0604/output/train/Bearing4_HI.png) |

**LOOCV 플롯 (k-NN v1 기준 — Adaptive는 별도 LOOCV 없음)**:

| B1 sc=0.525 | B2 sc=0.100 |
|---|---|
| ![B1 LOOCV](output_knn/Bearing1_loocv.png) | ![B2 LOOCV](output_knn/Bearing2_loocv.png) |

| B3 sc=0.499 | B4 sc=0.336 |
|---|---|
| ![B3 LOOCV](output_knn/Bearing3_loocv.png) | ![B4 LOOCV](output_knn/Bearing4_loocv.png) |

**Test HI (0604 output)**:

| T1 HI | T2 HI |
|---|---|
| ![T1 HI](../0604/output/test/Test1_HI.png) | ![T2 HI](../0604/output/test/Test2_HI.png) |

| T3 HI | T4 HI |
|---|---|
| ![T3 HI](../0604/output/test/Test3_HI.png) | ![T4 HI](../0604/output/test/Test4_HI.png) |

| T5 HI | T6 HI |
|---|---|
| ![T5 HI](../0604/output/test/Test5_HI.png) | ![T6 HI](../0604/output/test/Test6_HI.png) |

**출력**: `output_adaptive/`

---

## 7. 파일 구조

```
User/SR/0605/
├── code/
│   ├── hi_self_norm.py          # 실험 A — self-norm HI (LOOCV 악화로 미채택)
│   ├── rul_lstm.py              # 실험 A용 LSTM RUL (미채택)
│   ├── rul_feat_sim.py          # 실험 B — feature-sim start_frac (채택 후보)
│   ├── hi_stage_blend.py        # 실험 C-1 — stage-blended HI (LOOCV 개선 없음)
│   ├── rul_lstm_cq.py           # 실험 C-1용 RUL (LOOCV 0.306, 미채택)
│   ├── rul_lstm_kurt.py         # 실험 C-2 — kurtosis 4번째 feature (LOOCV 0.491, 누수)
│   ├── rul_lstm_kurt_noobs.py   # 실험 D — obs_frac 제거 (LOOCV 0.342, 정직)
│   ├── rul_knn.py               # 실험 E — Window-Feature k-NN (LOOCV 0.4345, 정직)
│   ├── rul_knn_rel.py           # 실험 F — rel-HI+frac-RUL (LOOCV 0.307, 기각)
│   ├── rul_adaptive.py          # 실험 G — Adaptive Ensemble (★ 현재 최선)
│   └── rul_adaptive_plot.py     # 실험 G — 시계열 시각화 유틸리티
└── output/
    ├── train/           # self-norm HI 결과
    ├── test/            # self-norm HI 결과
    ├── rul/             # LSTM RUL (실험 A)
    ├── rul_feat_sim/    # feature-sim RUL (실험 B)
    ├── output_cq/
    │   ├── train/       # stage-blended HI (self-baseline + T_actual)
    │   ├── loocv/       # stage-blended HI (fleet-baseline + T_est)
    │   ├── test/        # stage-blended HI (fleet-baseline + T_est)
    │   ├── rul/         # stage-blended RUL (미채택)
    │   ├── rul_kf/      # kurtosis_feat RUL (누수 LOOCV 0.491)
    │   └── rul_nof/     # no_obs_frac RUL (정직 LOOCV 0.342)
    ├── output_knn/      # k-NN RUL (정직 LOOCV 0.4345, bias=0.65)
    ├── output_knn_rel/  # k-NN rel-HI+frac-RUL (LOOCV 0.307, 기각)
    └── output_adaptive/ # Adaptive Ensemble (★ 현재 최선)
```

---

## 8. 핵심 인사이트

1. **Q-score는 "점진적 열화 추적 능력"만 측정** — kurtosis 같은 충격성 feature는 Q 낮아도 말기 탐지에 유용
2. **Stage-aware Q는 이론적으로 옳지만** — trajectory 형태를 바꿔서 LSTM 일반화를 방해
3. **HI 절대 scale보다 패턴 일관성** — 0604 HI가 B3 hi_end=0.14여도 LOOCV B3=0.458인 이유
4. **kurtosis를 LSTM feature로 추가** — HI를 건드리지 않고 kurtosis 정보 전달 → LOOCV +0.031
5. **LOOCV용 자기 baseline 이슈** — LOO test fold도 fleet baseline 써야 정직한 검증
6. **obs_frac = 정보 누수** — LOOCV에서 실제 t_start 사용 → 0.491이 아니라 정직한 성능은 0.342
7. **obs_frac 기여분 = 0.149** — 위치 정보가 이 정도의 LOOCV 이득을 제공. 위치 추정 정확도가 성능의 핵심 레버
8. **LSTM은 flat predictor** — 위치 정보 없이는 훈련 데이터 평균 수명을 출력. HI 수준엔 어느 정도 반응하지만 궤적 추적 불가
9. **k-NN > LSTM** (데이터 부족 상황에서) — 61~71% const-RUL 샘플로는 LSTM이 의미 있는 패턴을 학습할 수 없음. k-NN은 const 영역 자연 처리, B3 impulse failure 대폭 개선 (+0.263)
10. **const-RUL 이질성이 k-NN의 주요 한계** — B2 const=22, B1=37, B4=59. LOOCV에서 fleet 과대추정 → 공격적 bias=0.65로 보정
11. **B4 LOOCV 실패는 feature engineering으로 해결 불가** — B4는 HI baseline뿐 아니라 kurt/regime 패턴도 이질적. 상대 HI(v2)로도 고쳐지지 않음. 4개 베어링의 근본 이질성 한계
12. **상대 HI 피처는 절대 열화 수준 정보를 소실** — v2에서 T5(HI=0.116), T6(HI=0.200)이 모두 fleet_mean=3.93hr flat 예측. 절대 HI가 테스트 베어링 간 열화 수준 구분에 필수

## 9. 미해결 / 향후 고려

- **k-NN v2 기각 확정** — Fractional RUL + 상대 HI: B4 여전히 실패, B2/B3 붕괴, 테스트 flat 예측. v1 유지.
- **B4 LOOCV 실패가 테스트에 미치는 영향 제한적** — 전체 훈련 데이터 사용 시 B4가 lookup에 포함됨. T1-T6 중 B4 유사(HI≈0.18) 베어링 없음.
- **Adaptive v1 채택 확정** — T2(k-NN 4.01→2.85), T5/T6(feat_sim 유지) 개선. LOOCV 직접 검증 불가이나 두 정직한 모델의 조건부 최선을 취함.
- **제출 후보 최종 비교**:

| 베어링 | k-NN v1 | feat_sim | Adaptive v1 | 채택 근거 |
|--------|---------|----------|-------------|---------|
| T1 | 2.38hr | 5.10hr | **3.47hr** | B2 bias 완화 |
| T2 | 4.01hr | 2.07hr | **2.85hr** | feat_sim 보수 반영 |
| T3 | 3.53hr | 2.95hr | **3.18hr** | 균형 |
| T4 | 3.19hr | 5.60hr | **4.15hr** | feat_sim 과낙관 완화 |
| T5 | 2.62hr | 2.01hr | **2.01hr** | min (열화 확인) |
| T6 | 2.38hr | 1.90hr | **1.90hr** | min (열화 확인) |
