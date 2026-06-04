# 0604 작업 진행 기록

> 마지막 업데이트: 2026-06-04
> 출발점: `SR/0603_v3` (FDR HI + LGBM+LSTM, 전체수명 LOOCV Ens_raw 0.466)

---

## 1. 출발 문제 — "HI 스케일이 너무 다르다"

B3 EOL HI = 0.14, B4 EOL HI = 0.84. LGBM이 B3 말기를 "B1 초기"로 오인 → RUL 과대예측(LGBM B3 score 0.132).

---

## 2. Feature 분석 — 어떤 피처가 4개 베어링 모두에서 잘 나타내는가

### 2-1. 56개 피처 Q-score 분석 (`User/SR/0604` 세션 내 임시 실행)

| Feature | B1 | B2 | B3 | B4 | min |
|---------|----|----|----|----|-----|
| ch3_rms | 0.370 | 0.497 | 0.414 | 0.363 | **0.363** |
| ch3_total_power | 0.362 | 0.490 | 0.427 | 0.370 | **0.362** |
| ch1_kurt_log | 0.158 | 0.296 | 0.108 | 0.206 | 0.108 |

**결론**: 현재 쓰는 7개 피처(ch3/ch4 에너지·주파수)가 min Q-score 기준 최선. kurtosis 등 새 피처는 모두 하위.

### 2-2. B3 열화 패턴 분석

- B3 ch1_kurtosis: 파일 084~085에서 3994/747 스파이크 (마지막 2파일만)
- B3 ch3_rms: EOL에서 1.47x 증가 (B2의 3.5x, B1의 1.93x 대비 매우 작음)
- **B3는 점진적 열화가 아닌 충격성(impulsive) 말기 실패 패턴**

### 2-3. 핵심 발견

B3 HI(FDR) Q-score = **0.804** — HI 자체는 잘 계산되고 있음.
문제는 EOL HI 절댓값(0.14)이 B1의 초기 30% HI와 겹침 → RUL 모델 오인.

---

## 3. HI 방법 실험 — 모두 실패 또는 현상 유지

### 3-1. kurtosis/crest 피처 추가 (`hi_loo_regime_v1.py`)

- B3 Q-score: 0.804 → **0.372** (악화)
- B3 kurtosis 스파이크가 마지막 2파일뿐 → moving average(w=7)에 희석
- crest factor가 초기에도 비零 값 → HI 시작점 부풀림
- **폐기**

### 3-2. Z-score HI (`hi_zscore_v1.py`)

FDR 대신 (feature - baseline_mean) / baseline_std로 정규화.
- B2의 ch3_p2p가 23x 증가 → z-score 수천 → eol_p95 폭발 → 모든 HI 0에 수렴
- 전 베어링 Q-score 악화 (mean 0.629 → 0.469)
- **폐기. FDR이 B3에 오히려 더 적합했음.**

**결론: HI 방법 변경으로 B3 문제 해결 불가. B3는 데이터 구조적 한계.**

---

## 4. 핵심 발견 — Sliding Window LOOCV

현재 LOOCV 문제: 전체 수명(89~137사이클) 평가 ≠ 실제 test (50사이클 창, 임의 시작점).

### 4-1. v1: 50사이클 슬라이딩 창 LOOCV (`rul_sliding_v1.py`)

| 방법 | B1 | B2 | B3 | B4 | **평균** |
|------|----|----|----|----|---------|
| LGBM | 0.329 | 0.076 | 0.010 | 0.283 | 0.175 |
| LSTM | **0.387** | **0.577** | **0.468** | **0.373** | **0.451** |
| 50/50 ens | 0.368 | 0.285 | 0.150 | 0.339 | 0.285 |

**LGBM이 sliding 시나리오에서 완전히 붕괴.** 이유:
- `elapsed_frac` 피처가 `estimate_start_frac` 추정에 의존
- start_frac 추정 오류 → elapsed_frac 오염 → LGBM 예측 전체 망가짐
- LSTM은 obs_frac이 3개 피처 중 하나라 상대적으로 robust

→ **LSTM 단독 사용 결정**

### 4-2. v2: LSTM only (`rul_sliding_v2.py`)

sliding LOOCV LSTM avg: **0.451**

**두 가지 문제 발견:**
1. T2~T6 예측이 2.8~2.9hr로 수렴 — LSTM flat 예측 (lifecycle 위치 무관)
2. 예측 진동 — 레짐 교번으로 HI가 흔들리고 LSTM이 따라 흔들림 → 마지막 10개 중앙값 사용

---

## 5. start_frac 추정 방법 실험

### 5-1. 문제 진단

```
start_frac (level 기반):
  "train 베어링에서 HI가 hi_start에 처음 도달하는 시점 / MEAN_TRAIN_LIFE"

문제:
  B3 EOL HI=0.14 = B1의 30% 시점 HI
  → B3가 EOL이어도 start_frac ≈ 30% (실제는 100%)
  → obs_frac 피처 오염 → LSTM도 잘못된 위치 인식
```

### 5-2. v3: Slope 기반 start_frac (`rul_sliding_v3.py`)

50사이클 창의 선형 기울기 → train 베어링의 모든 창 기울기와 비교 → inverse-distance weighted average.

| 방법 | B1 | B2 | B3 | B4 | 평균 |
|------|----|----|----|----|------|
| level (v2) | 0.387 | **0.577** | 0.468 | 0.373 | 0.451 |
| slope (v3) | **0.445** | 0.539 | 0.468 | **0.374** | **0.457** |

- B1 +0.058 개선
- **T5/T6 치명적 실패**: 평탄한 HI(slope≈0) → 초기 단계(18~21%)로 오해 → 4.5hr 예측 (실제는 말기)

### 5-3. v4: Hybrid start_frac (`rul_sliding_v4.py`)

```python
w = clip(mean(hi_window) / 0.2, 0, 1)
start_frac = w * level_sf + (1-w) * slope_sf
# 고HI → level 신뢰, 저HI → slope 신뢰
```

| 방법 | B1 | B2 | B3 | B4 | 평균 |
|------|----|----|----|----|------|
| level | 0.387 | **0.577** | 0.468 | 0.373 | 0.451 |
| slope | **0.445** | 0.539 | 0.468 | 0.374 | **0.457** |
| hybrid | 0.399 | 0.533 | 0.468 | 0.373 | 0.443 |

**hybrid가 가장 낮음** — 두 방법을 평균하면 각자의 장점이 상쇄됨.
그러나 T5/T6는 w=1.0 (level 완전 신뢰)으로 올바르게 처리됨.

---

## 6. 방법별 Test RUL 예측 비교

| Test | 시작 위치 | 0603_v4 | level (v2) | slope (v3) | hybrid (v4) | 비고 |
|------|---------|---------|-----------|-----------|------------|------|
| T1 | 8% | **5.97hr** | 3.78 | 3.66 | 3.75 | 초기, slope 무관 |
| T2 | 52% | 1.94 | 2.85 | 3.77 | **2.94** | 중기 |
| T3 | 47% | 2.68 | 2.87 | 3.79 | **3.40** | 중기 |
| T4 | 8% | **8.29hr** | 3.85 | 3.18 | 3.98 | 초기, slope 무관 |
| T5 | 86% | **1.72** | **2.83** | 4.23 | **2.86** | 말기, slope 실패 |
| T6 | 94% | **1.64** | **2.80** | 4.52 | **2.87** | 말기, slope 실패 |

---

## 7. 현재 상황 정리

### 남아있는 구조적 한계

1. **B3 문제**: EOL HI(0.14)가 다른 베어링 초기 HI 범위와 겹침 → 4개 데이터로 해결 불가
2. **LSTM flat 예측**: 3개 train 베어링으로 lifecycle-dependent RUL 학습 불충분 → ~17사이클 상수 예측
3. **T1/T4 vs T5/T6 트레이드오프**:
   - 0603_v4: T1/T4 길게(5.97/8.29hr), T5/T6 짧게(1.64/1.72hr) → 물리적으로 합리적
   - 0604 sliding: T1/T4 짧게(3.75/3.98hr), T5/T6 적당(2.83/2.87hr) → 보수적

### 방법별 장단점

| 방법 | LOOCV | T1/T4 | T5/T6 | 특징 |
|------|-------|-------|-------|------|
| 0603_v4 (LGBM+LSTM) | 0.466* | 좋음 | 더 보수적 | *전체수명, 비교 불가 |
| 0604 level (v2) | 0.451 | 짧음 | 올바름 | 안전한 기준 |
| 0604 slope (v3) | 0.457 | 짧음 | **틀림** | T5/T6 사용 불가 |
| 0604 hybrid (v4) | 0.443 | 짧음 | 올바름 | T5/T6 안전 |

---

## 8. 파일 구조

```
User/SR/0604/
├── code/
│   ├── hi_loo_regime_v1.py   # kurtosis 추가 실험 (폐기)
│   ├── hi_zscore_v1.py       # z-score HI 실험 (폐기)
│   ├── rul_sliding_v1.py     # LGBM+LSTM sliding, LGBM 붕괴 확인
│   ├── rul_sliding_v2.py     # LSTM only, level start_frac
│   ├── rul_sliding_v3.py     # LSTM only, slope start_frac
│   └── rul_sliding_v4.py     # LSTM only, hybrid start_frac ← 현재 최선
├── output/
│   ├── train/                # hi_loo_regime_v1 출력 (kurtosis 실험)
│   ├── test/                 # hi_loo_regime_v1 출력
│   ├── rul_sliding/          # v1, v2 출력
│   ├── rul_sliding_v3/       # slope 출력
│   └── rul_sliding_v4/       # hybrid 출력 ← 현재 최신
└── progress.md               # 이 파일
```

---

## 9. 다음 작업 후보

- [ ] 제출 결정: 0603_v4 vs 0604 hybrid 중 선택
  - 0603_v4: T1/T4 더 길게 예측 (낙관적 리스크)
  - 0604 hybrid: T1/T4 더 짧게 예측 (보수적, 채점 비대칭상 유리)
- [ ] Conservative bias 추가: 전체 ×0.85 정도 적용 시 test 점수 개선 가능성
- [ ] 제출 파일 생성

---

## 10. 핵심 인사이트 요약

1. **HI는 FDR이 최선** — z-score, kurtosis 추가 모두 오히려 악화
2. **B3 문제는 HI가 아닌 RUL 모델** — B3 HI Q=0.804으로 우수, EOL 절댓값이 낮은 게 문제
3. **실제 test 시나리오 = 50사이클 창** — 전체수명 LOOCV(0.466)는 부풀려진 수치
4. **LGBM은 sliding에서 무용** — start_frac 추정 오류 → elapsed_frac 오염 → 붕괴
5. **LSTM sliding LOOCV** = 0.451~0.457 (실제 test 성능 추정치)
6. **T5/T6는 level 기반 start_frac 필수** — slope 기반 쓰면 말기를 초기로 오해
