# 0603_regime_f2s2 Progress

## 목표
F2S2 기반 per-regime LOO HI + LGBM/LSTM 앙상블 RUL 예측 파이프라인 구축

---

## 작업 흐름

### Phase 1: 원래 코드 분석 (0603_v2, 구 디렉토리명)

원래 `hi_loo_regime_v1.py`는 외부 디렉토리에서 사전 계산된 CSV를 읽어오는 구조였음:
- Train 특징: `SC/HI/04142304_signal_transform_v2/output/BearingN_features_transformed.csv` (F2S2 변환 완료)
- Test 특징: `SC/HI/05072245_signal_transform_v5_test/output/TestN_features.csv` (raw)
- 레짐(cond): `BearingN_SSM_result.csv`

**발견된 문제점:**

| 문제 | 심각도 | 설명 |
|------|--------|------|
| Train/Test 파이프라인 불일치 | 🔴 심각 | Train은 F2S2 변환된 특징, Test는 raw 특징 |
| F2S2 변환 temporal leakage | 🔴 심각 | 변환 파라미터 추정 시 전체 시계열(미래 포함) 사용 |
| `FEATURE_Q_GLOBAL` hardcode | 🟡 중간 | 전체 4개 베어링 기반으로 계산된 값 — LOO fold에서 test_bid 정보 누출 |
| Test 파이프라인 temporal leakage | 🟡 중간 | minmax_scale/robust_clip이 전체 시계열 min/max 사용 |

---

### Phase 2: Self-contained 재작성

**변경 내용:**
- TDMS에서 직접 특징 추출 (`extract_features_from_file`) — 외부 SC/ 디렉토리 의존 완전 제거
- Train 레짐 분류: `TrainN_Operation.csv` → RPM 850 기준 이진 분류
- Test 레짐 분류: CH2 FFT 피크 → RPM 추정
- F2S2 변환 제거 (per-regime FDR ratio로 대체)
- `FEATURE_Q_GLOBAL` hardcode 제거 → exclude_bid 기반 LOO fallback
- 특징 캐시: `output/cache/` (재실행 속도 개선)
- `rul_regime_v1.py` SR_BASE 경로 수정 (`0603_v2` → `0603_regime_f2s2`)

**1차 실행 결과:**

| 지표 | 값 |
|------|----|
| Train avg Q | 0.629 |
| Test avg Q | 0.339 |
| LOOCV RUL | 0.4281 |

**발견된 문제:** Bearing4 HI가 0.55에서 시작 (LOO 베어링 B1~3 기반 baseline이 B4의 절대값과 크게 다름)

```
ch3_p2p:    B4 baseline = B1~3 평균의 8.1배
ch3_energy: B4 baseline = B1~3 평균의 2.5배
ch3_rms:    B4 baseline = B1~3 평균의 1.6배
```

---

### Phase 3: Own-baseline 적용

**변경 내용:**
- `compute_own_baseline()` 추가: 각 레짐별 자기 자신의 첫 10% 평균
- `compute_regime_stats()`: feat_q/direction/p5/p95 계산 시에도 각 베어링의 own baseline 기준 FDR 사용
- `apply_regime_hi()`: regime_stats의 baseline 대신 own baseline으로 FDR ratio 계산

**설계 근거:**
- Validation data는 정상~열화 어느 시점에서나 시작할 수 있음
- Own baseline = "관측 시작 시점 대비 얼마나 더 열화됐나"
- RUL 모델은 HI 절대값이 아닌 **열화 속도(slope/rate)**로 예측

**2차 실행 결과:**

| 지표 | Phase 2 | Phase 3 | 변화 |
|------|---------|---------|------|
| Train avg Q | 0.629 | **0.712** | ↑ +0.083 |
| Test avg Q | 0.339 | **0.582** | ↑ +0.243 |
| B4 hi_start | 0.553 (버그) | **0.002** | ✅ 수정 |
| LOOCV RUL | **0.4281** | 0.3521 | ↓ -0.076 |

**LOOCV 상세:**

| Bearing | LGBM | LSTM | Ens+CF |
|---------|------|------|--------|
| B1 | 0.4846 | 0.3747 | 0.3863 |
| B2 | 0.4363 | 0.3840 | 0.4692 |
| B3 | **0.0528** | 0.4785 | **0.1869** |
| B4 | 0.4853 | 0.3644 | 0.3661 |
| 평균 | 0.3647 | 0.4004 | **0.3521** |

**LOOCV 하락 원인:**
- B3 hi_end=0.185로 낮고, B1~4 중 열화 패턴이 가장 완만 (89 cycles, 최단 수명)
- Own baseline으로 전환 후 전체 HI scale이 낮아지면서 LGBM이 B3 패턴을 제대로 학습 못함
- LSTM은 B3에서도 0.48로 준수 → LGBM이 낮은 HI end 케이스에 취약

---

## 현재 상태

- HI 품질(Q score): 개선됨, Test HI도 의미있는 트렌드 포착
- LOOCV RUL: 이전 대비 하락 (0.4281 → 0.3521)
- **미결 과제**: LGBM이 낮은 hi_end 베어링에 취약한 문제 해결 필요

## 파일 구조

```
code/
  hi_loo_regime_v1.py    # TDMS → 특징 추출 → LOO HI (own baseline)
  rul_regime_v1.py       # LOO LGBM+LSTM 앙상블 RUL
output/
  cache/                 # 특징 추출 캐시 (TDMS 재파싱 방지)
  train/                 # BearingN_HI.csv, summary.csv, 시각화
  test/                  # TestN_HI.csv, summary.csv, 시각화
  rul/                   # TestN_RUL.csv, loocv_log.txt, 시각화
```
