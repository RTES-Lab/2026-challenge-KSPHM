# 0605_v2 — F2S2 Bug-Fixed Implementation

**날짜:** 2026-06-05  
**기반 논문:** Li et al., "Remaining useful life prediction of machinery under time-varying operating conditions based on a two-factor state-space model", RESS 186 (2019)  
**참조 원본:** `User/SC/HI/06031410_f2s2_rul_raw` (수정 대상, 원본 불변)

---

## 1. 작업 개요

원 논문 F2S2 구현체(`06031410_f2s2_rul_raw/code/f2s2_core.py`)의 버그를 엄밀히 분석하고,  
`User/SR/0605_v2/code/` 에 완전히 새로 작성하여 수정한 실험.

---

## 2. 발견된 버그 및 수정 내용

### 버그 1 — `tau_path` 인덱스 오류 (중간급, c 추정 왜곡)

```python
# 원본 (오류)
inc = np.array([R[profile[max(j - 1, 0)]] for j in range(len(profile))])

# j=0: max(-1,0)=0 → profile[0]
# j=1: max( 0,0)=0 → profile[0]  ← 두 번 참조 (버그)
# j=2: max( 1,0)=1 → profile[1]  ← 한 칸씩 밀림
```

교번 프로파일 `[0,1,0,1,...]`에서 짝수 인덱스의 τ 값이 틀림.  
`fit_measurement`의 c 추정에만 영향 (MLE, PF 직접 무관).

```python
# 수정
inc = np.array([R[p] for p in profile])
return np.concatenate([[0.0], np.cumsum(inc)[:-1]])  # τ[0]=0
```

### 버그 2 — `b_B` 평균화 방식 (소소한 수치 오차)

```python
# 원본: mean(sm[0]) / mean(a_{B,n})  — 비율의 평균이 아닌 평균의 비율
b_B = np.mean([sm[0] for sm in sm_list]) / a_B

# 수정: 논문 eq 35 — 유닛별 b_{B,n} 먼저 계산 후 평균
b_B = np.mean([sm[0] / a_Bn for sm, a_Bn in zip(sm_list, a_Bns)])
```

### 버그 3 — `future_regime` last-regime padding (심각, RUL 2배 이상 과대추정 원인)

```python
# 원본: 인위적 고정 스케줄 (저속 3사이클 → 6사이클씩 교번)
# → 실제 운전 조건 프로파일과 무관

# 1차 수정 시도: last-seen regime으로 padding
# → B2 마지막 사이클이 HIGH(r=0.2)이면 padding 337사이클 전부 HIGH
#    미래 평균 dtau: 0.59 → 0.26으로 절반 이하 → RUL 2배 과대추정
```

**진단:** B2 LOO at k=50:
- 실제 미래 63사이클의 dtau 평균 = 0.594 (50/50 split)
- last-regime padding 적용 시 dtau 평균 = 0.262 (337사이클 전부 HIGH with r=0.2)

```python
# 최종 수정: cyclic tiling — 실제 이력 프로파일을 반복
n_pad = n_future - len(avail)
tile  = np.tile(full_profile, (n_pad // len(full_profile)) + 1)[:n_pad]
return np.concatenate([avail, tile])
```

실제 운전 조건 프로파일(Operation.csv)을 사용하여 알려진 구간은 정확히,  
그 이후는 동일 패턴으로 반복. 경험적 low/high 비율 보존.

### 수정 4 — `sigma2` 내부 파라미터 표현 정리 (결과 동일, 가독성)

```python
# 원본: sigma2 = mean_residual / a_B²  저장 → PF에서 a_B² 곱하여 복원
# 수정: sigma2 = mean_residual 직접 저장 → PF에서 그대로 사용
```

### 추가 변경 — Test 베어링 레짐 감지

원본에는 test 베어링 지원 없음. TDMS에서 직접 추출 구현:
- ch3_rms: CH3 전체 샘플 RMS
- 레짐: CH2 FFT 피크 (8~20 Hz 구간) → 축 주파수 → RPM × 60 → 850 기준 이진 분류
- FFT 길이: 131072 샘플(2^17, ≈5.1초, 주파수 분해능 0.2 Hz)

---

## 3. 코드 구조

```
code/
  f2s2_fixed.py   — 수정된 F2S2 알고리즘 core (논문 섹션별 구현)
  run_f2s2.py     — Train LOO + Test RUL 실행 스크립트
output/
  train_hi.png         — Train 베어링 HI (raw → baseline 변환 → PF state)
  train_rul_loo.png    — Train LOO RUL 곡선 + 점수
  test_hi.png          — Test 베어링 HI
  test_rul.png         — Test 베어링 RUL 예측 곡선
  train_rul_results.csv
  test_rul_results.csv
```

---

## 4. Train LOO 결과

| Bearing | score(fpt) | score(full) | mean_er% | FPT | r_high | eta | c |
|---------|-----------|------------|----------|-----|--------|-----|---|
| B1 | 0.463 | 0.520 | -188.2 | 17 | 0.383 | 0.01260 | 1.58 |
| B2 | 0.537 | 0.582 | -25.7 | 15 | 0.200 | 0.01394 | 1.69 |
| B3 | 0.070 | 0.090 | -265.8 | 10 | 0.461 | 0.01084 | 2.08 |
| B4 | 0.468 | 0.505 | +59.4 | 17 | 0.367 | 0.01313 | 2.70 |
| **avg** | **0.385** | **0.424** | | | | | |

원본 avg score(fpt) = 0.414. B2, B4 개선 / B1 저하.

**B1 저하 원인:** B1의 ch3_rms 진폭 변화(~0.35)가 B2/B4(~1.5~1.9)와 크게 달라,  
LOO시 B2,B3,B4로 추정한 a_B가 B1 신호에 맞지 않음. 버그와 무관한 데이터 이질성.

**Full-train 파라미터:** r_high=0.254, eta=0.01348, c=1.99

**r_high < 1 해석:** 4개 베어링 모두 low/high 비율이 유사(~50/50)하여  
MLE에서 r_high가 잘 식별되지 않음. 고속이 저속보다 단위 사이클당 열화 기여가 작다고 추정.

---

## 5. Test 베어링 RUL 예측 (50사이클 관측 후)

| Test | K | state[-1] | 예측 RUL (cycles) |
|------|---|-----------|------------------|
| 1 | 50 | 0.347 | 77.5 |
| 2 | 50 | 0.357 | 74.7 |
| 3 | 50 | 0.354 | 74.8 |
| 4 | 50 | 0.363 | 75.2 |
| 5 | 50 | 0.406 | 71.7 |
| 6 | 50 | 0.483 | 60.4 |

모든 테스트 베어링의 state[-1]이 0.35~0.48 (열화 진행 중, 아직 실패 전).

---

## 6. 알려진 한계

- r_high < 1: 4개 베어링만으로는 두 레짐의 열화율 비를 안정적으로 추정하기 어려움
- B3(89 cycle, 최단수명) LOO 점수 낮음: 장수명 베어링(B1/B2/B4) 파라미터로 과대추정
- B1 LOO 점수 낮음: 신호 진폭이 다른 베어링에 비해 현저히 작아 측정함수 파라미터 불일치
- Test 레짐 감지: TDMS CH2 FFT 피크 기반으로 강건성 제한적
