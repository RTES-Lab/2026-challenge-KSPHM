# KSPHM 2026 Challenge - 베어링 RUL 예측 실험 정리

**작성 기준**: User/SR/test.ipynb 노트북 실험 결과 요약
**목표**: 베어링 진동 데이터로부터 잔여수명(RUL)을 예측하고, 대회 평가 Score를 최대화

# 1. 데이터 개요

- Train1: TDMS 126개, EOL 75,251초 (~20.9h)
- Train2: TDMS 114개, EOL 67,979초 (~18.9h)
- Train3: TDMS 89개, EOL 53,225초 (~14.8h)
- Train4: TDMS 137개, EOL 82,613초 (~22.9h)
- **수집 주기**: 10분(600초) 간격
- **채널**: CH1~CH4 (4채널 진동 데이터)
- **EOL 판정 기준**: 토크 ≤ -20 또는 온도 ≥ 200

**추출 피처 (28개 → feature_df)**

- 운전 조건: torque, rpm, temp_front, temp_rear
- 진동 통계 (CH1~CH4 각각): RMS, Kurtosis, P2P, Crest Factor, Std

# 2. 평가 방식

**Challenge Score 공식**

Er = 100 × (실제값 - 예측값) / 실제값

- 과소 예측(Er ≥ 0): s = exp(-ln(0.5) × Er / 20)
- 과대 예측(Er < 0): s = exp(+ln(0.5) × Er / 10)
- **최종 Score = mean(s)** → 1.0에 가까울수록 좋음
- ⚠️ **과대 예측(실제보다 높게 예측)에 2배 더 큰 페널티** 부여

**검증 전략: Leave-One-Out Cross Validation (LOO-CV)**

- Train 데이터 4개(Train1~4) 중 1개를 테스트용으로, 나머지 3개를 학습에 사용
- 4번 반복 후 Score 평균을 산출

# 3. 실험 결과 요약 🔑

- **V1 (Baseline)**: LightGBM + LOO-CV + RPM 정규화 → **Score 0.4220** ← 기본 모델
- **V2**: log(RUL) + 비대칭 커스텀 Loss → **Score 0.4016** ❌ 오히려 하락
- **V3**: Optuna 하이퍼파라미터 튜닝 + Scale 조정 → **Score 0.4224** △ 미미한 개선
- **V4**: HI Dynamics 피처 추가 (16개) → **Score 0.4212** △ 미미한 변화
- **HI 외삽**: Similarity-based HI 외삽 모델 → **Score 0.4080** ❌ Baseline 미만
- **Time-split**: 시간순 70/30 분할 → **Score 0.0496** ❌ 과대예측 심각

# 4. 각 실험 상세

## 4.1. V1 — Baseline (Score: 0.4220)

**핵심 구성:**

1. **HI(Health Index) 구성**: PCA 기반
    - 진동 피처를 RPM으로 정규화 → 전체 스케일링 → PCA(1차원)
    - Baseline 구간: 각 run의 처음 10% 데이터
    - Smoothing: Median filter (w=5)
2. **학습 모델**: LightGBM
    - 학습 피처: 진동 원본 + 정규화 피처 + HI + slope 등 **116개**
    - 타겟: rul_sec (잔여수명, 초 단위)
    - 검증: LOO-CV (4-fold)
3. **Startup 제거**: 운전 초기 불안정 구간 제외
    - rpm==0 이전의 첫 번째 구간을 startup으로 판별

**개별 결과:**

- Train1: Score=0.4318 (과대 48, 과소 77, Er 평균 +2.0%, Er 중앙 +15.7%)
- Train2: Score=0.4407 (과대 58, 과소 55, Er 평균 -7.6%, Er 중앙 +2.3%)
- Train3: Score=0.4527 (과대 41, 과소 47, Er 평균 -3.1%, Er 중앙 +3.5%)
- Train4: Score=0.3627 (과대 89, 과소 45, Er 평균 -80.9%, Er 중앙 -42.0%)

> 💡 **관찰**: Train4의 Score가 특히 낮아 전체 평균을 끌어내림 (과대예측 경향 심각)

## 4.2. V2 — log(RUL) + 비대칭 Loss (Score: 0.4016)

**변경점:**

- **RUL을 log 스케일로 변환**: log1p(rul_sec) → 학습 후 expm1으로 복원
- **비대칭 커스텀 Loss**: 과대예측(Er<0)에 더 큰 페널티를 부여하는 gradient/hessian 직접 구현

**결과:**

- Train1: Score=0.4339 (과대 58, 과소 67, Er 평균 -20.5%, Er 중앙 +11.2%)
- Train2: Score=0.4473 (과대 63, 과소 50, Er 평균 -180.6%, Er 중앙 -12.0%)
- Train3: Score=0.4244 (과대 47, 과소 41, Er 평균 -49.3%, Er 중앙 +3.5%)
- Train4: Score=0.3010 (과대 97, 과소 37, Er 평균 -199.1%, Er 중앙 -76.3%)

> ❌ **결론**: Baseline 대비 **0.02점 하락** — log 변환과 비대칭 Loss가 오히려 성능 저하를 유발

## 4.3. V3 — Optuna 하이퍼파라미터 튜닝 (Score: 0.4224)

**변경점:**

- Optuna로 LightGBM 하이퍼파라미터 자동 탐색 (40 trials)
- 최적 파라미터를 다수 seed로 앙상블 → 최적 scale factor 탐색

**튜닝된 파라미터:**

- learning_rate: 0.0547
- num_leaves: 125
- max_depth: 9
- min_child_samples: 23
- feature_fraction: 0.778
- bagging_fraction: 0.620

**결과:**

- Train1: Score=0.4385
- Train2: Score=0.4484
- Train3: Score=0.4254
- Train4: Score=0.3742

> △ **결론**: Baseline 대비 **+0.0004점** — 거의 차이 없음. 스케일 조정(×0.95)을 해봐도 비슷

## 4.4. V4 — HI Dynamics 피처 추가 (Score: 0.4212)

**추가 피처 (16개):**

- HI_diff1, HI_global_diff1: HI의 1차 미분 (변화율)
- HI_accel, HI_global_accel: HI의 2차 미분 (가속도)
- HI_cumsum, HI_global_cumsum: HI 누적합
- HI_recent_max, HI_global_recent_max: 최근 10구간 최대값
- HI_mono10, HI_global_mono10: 최근 10구간 단조증가 비율
- CH1_HI ~ CH4_HI: 채널별 개별 HI
- HI_max_ch: 최대 HI를 가진 채널 번호

**결과:**

- Train1: Score=0.4385
- Train2: Score=0.4513
- Train3: Score=0.4142
- Train4: Score=0.3808

> △ **결론**: Baseline 대비 **-0.0008점** — 피처 추가가 유의미한 개선을 가져오지 못함

## 4.5. HI 외삽 모델 (Similarity-based) (Score: 0.4080)

**방법론:**

1. 각 training run의 전체 HI 시퀀스를 DB로 저장
2. Test 시점에서 최근 k개 HI 값을 query로 사용
3. DB의 모든 window와 유사도(거리) 계산
4. Top-N 유사 window의 "EOL까지 남은 시간" 평균을 예측값으로 반환

> ❌ **결론**: LightGBM 모델 대비 성능 떨어짐 (0.4080 vs 0.4220)

## 4.6. Time-split 평가 (Score: 0.0496)

**방법론:**

- 각 run을 시간순 정렬 → 앞 70% train, 뒤 30% test
- 4개 run의 앞 70%를 모두 모아서 학습, 뒤 30%를 각각 예측

**결과:**

- Train1: Score=0.0779 (과대 38, 과소 0, Er 평균 -756.3%, Er 중앙 -140.8%)
- Train2: Score=0.0175 (과대 34, 과소 0, Er 평균 -1168.6%, Er 중앙 -290.4%)
- Train3: Score=0.0207 (과대 27, 과소 0, Er 평균 -864.3%, Er 중앙 -308.8%)
- Train4: Score=0.0823 (과대 39, 과소 0, Er 평균 -437.3%, Er 중앙 -68.9%)

> ❌ **결론**: **심각한 과대예측**으로 거의 0에 가까운 Score. Time-split은 RUL 예측 문제에 적합하지 않은 검증 전략 (학습 데이터가 열화 초기만 포함하기 때문)

# 5. 주요 인사이트

1. **과대예측이 성능의 핵심 병목**: 대회 Score 공식에서 과대예측의 패널티가 과소예측의 2배이므로, 과대예측을 줄이는 것이 가장 중요
2. **Train4가 가장 어려운 데이터**: 모든 실험에서 Train4의 Score가 가장 낮았음 (과대예측 경향)
3. **LOO-CV가 적절한 검증 전략**: Time-split은 RUL 예측 문제의 특성상 부적절
4. **단순 피처 추가/Loss 변경만으로는 큰 개선 어려움**: Baseline 모델이 이미 LightGBM의 한계에 근접한 것으로 보임
5. **HI(Health Index)의 품질이 핵심**: RPM 정규화 → PCA 기반 HI가 가장 중요한 피처로 작용

# 6. Top Feature Importance (V4 기준)

1. CH4_RMS_norm — 평균 중요도 101.0
2. HI_cumsum — 평균 중요도 75.0
3. CH1_RMS — 평균 중요도 32.0
4. CH4_RMS_norm_rmean10 — 평균 중요도 21.0
5. CH3_RMS — 평균 중요도 ~20

> 💡 RPM으로 정규화된 CH4_RMS가 압도적으로 높은 Feature Importance를 보여, **RPM 정규화의 효과**가 확인됨

# 7. 향후 방향 제안

- [ ] 딥러닝 기반 시계열 모델 (LSTM, Transformer) 시도
- [ ] 주파수 도메인 피처 (FFT, Envelope 분석) 추가
- [ ] 열화 단계별 예측 모델 분리 (초기/중기/말기)
- [ ] Train4 특성 분석 및 별도 처리 전략 수립
- [ ] 앙상블: HI 외삽 모델 + LightGBM 결합 최적화
