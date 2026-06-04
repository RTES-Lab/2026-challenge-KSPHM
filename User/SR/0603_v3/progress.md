# KSPHM 2026 Challenge — 작업 기록

> 마지막 업데이트: 2026-06-04 (v4 HI + v5 RUL 실험 추가)

---

## 1. 프로젝트 개요

- **목표**: 베어링 진동 데이터로 Health Index(HI) 구성 및 잔여 수명(RUL) 예측
- **데이터**: Train 베어링 4개 (전체 수명), Test 베어링 6개 (관측 창 ~50 사이클)
- **채점 함수**: Er = 100×(RUL_true − RUL_pred)/RUL_true → 낙관적 예측(Er<0) 페널티 2.5배 가혹
- **레짐**: 저속(0, <850 RPM) / 고속(1, ≥850 RPM) 교번, 1시간 단위
- **실제 제출 결과**: LOOCV 0.466 → 대회 채점 **0.3** (갭 원인 불명확, 개별 결과 미공개)

| Bearing | Total cycles | Normal until | EOL HI | 비고 |
|---------|-------------|-------------|--------|------|
| B1 | 126 | 89 | 불명 | 정상 |
| B2 | 114 | 92 | 불명 | 정상 |
| B3 | 89 | 62 | **~0.14** | EOL HI가 타 베어링 정상기 수준과 겹침 |
| B4 | 137 | 78 | 불명 | **시작부터 HI≈0.55 (이미 열화)** |

---

## 2. 현재 최고 파이프라인 — `hi_loo_regime_v4.py` + `rul_regime_v5.py`

### HI (`hi_loo_regime_v4.py`)

- v2 대비: Energy/Total Power/RMS → **log1p 적용** (step function → gradual trend)
- 피처 구성 동일 (7개), LOO baseline, per-regime FDR 가중합

| Bearing | HI Q-score (v4) | v2 대비 |
|---------|----------------|--------|
| B1 | 0.571 | -0.002 |
| B2 | 0.641 | +0.002 |
| B3 | **0.815** | +0.011 |
| B4 | 0.501 | ±0.000 |
| **평균** | **0.632** | **+0.003** |

| Test | HI start | HI end | Q-score | 비고 |
|------|----------|--------|---------|------|
| T1 | 0.005 | 0.083 | 0.627 | 정상 초기 |
| T2 | 0.104 | 0.133 | 0.144 | 중기, 평탄 |
| T3 | 0.085 | 0.143 | 0.383 | 중기 |
| T4 | 0.006 | 0.158 | 0.679 | 정상 초기 |
| T5 | 0.326 | 0.343 | 0.124 | 이미 열화 상태 |
| T6 | 0.513 | 0.517 | 0.054 | 심각 열화 상태 |

**Train HI (LOO)**

![Train LOO HI](output/train_v4/Bearing_LOO_Regime_HI.png)

| B1 | B2 |
|----|----|
| ![](output/train_v4/Bearing1_HI.png) | ![](output/train_v4/Bearing2_HI.png) |

| B3 | B4 |
|----|----|
| ![](output/train_v4/Bearing3_HI.png) | ![](output/train_v4/Bearing4_HI.png) |

**Test HI**

![Test HI](output/test_v4/Test_Regime_HI.png)

| T1 | T2 | T3 |
|----|----|----|
| ![](output/test_v4/Test1_HI.png) | ![](output/test_v4/Test2_HI.png) | ![](output/test_v4/Test3_HI.png) |

| T4 | T5 | T6 |
|----|----|----|
| ![](output/test_v4/Test4_HI.png) | ![](output/test_v4/Test5_HI.png) | ![](output/test_v4/Test6_HI.png) |

### RUL LOOCV (`rul_regime_v5.py`)

모델: LGBM + LSTM 앙상블 (LOO score 비례 가중치)
피처: HI window(10) + slope/mean/std/max/last/delta + regime + elapsed_frac + hi_slope_5/10/hi_delta

| | B1 | B2 | B3 | B4 | 평균 |
|--|---|---|---|---|---|
| LGBM | 0.499 | 0.548 | 0.135 | 0.619 | 0.450 |
| LSTM | 0.381 | 0.380 | 0.403 | 0.347 | 0.378 |
| **Ens_raw** | **0.437** | **0.447** | **0.532** | **0.458** | **0.468** |

**LOOCV 예측**

![LOOCV](output/rul_v5/loocv_predictions_v5.png)

| B1 | B2 |
|----|----|
| ![](output/rul_v5/Bearing1_RUL_v5.png) | ![](output/rul_v5/Bearing2_RUL_v5.png) |

| B3 | B4 |
|----|----|
| ![](output/rul_v5/Bearing3_RUL_v5.png) | ![](output/rul_v5/Bearing4_RUL_v5.png) |

### Test RUL 예측 (최종)

| Test | start_frac | RUL (hr) |
|------|-----------|---------|
| T1 | 8.6% | 6.11 |
| T2 | 52.6% | 2.62 |
| T3 | 47.5% | 2.77 |
| T4 | 9.2% | 8.44 |
| T5 | 86.4% | 1.98 |
| T6 | 94.1% | 1.90 |

**Test 예측**

![Test](output/rul_v5/test_predictions_v5.png)

| T1 | T2 | T3 |
|----|----|----|
| ![](output/rul_v5/Test1_RUL_v5.png) | ![](output/rul_v5/Test2_RUL_v5.png) | ![](output/rul_v5/Test3_RUL_v5.png) |

| T4 | T5 | T6 |
|----|----|----|
| ![](output/rul_v5/Test4_RUL_v5.png) | ![](output/rul_v5/Test5_RUL_v5.png) | ![](output/rul_v5/Test6_RUL_v5.png) |

---

## 3. 구조적 한계 (해결 불가)

- **B3**: EOL HI≈0.14가 타 베어링 정상기 HI와 겹침 → LGBM이 B3 말기를 "초기 상태"로 오인 → cross-bearing 학습으로는 어떤 tabular 모델도 해결 불가
- **B4**: 시작부터 HI≈0.55 → LOO baseline 오염, estimate_start_frac에서 자동 skip됨
- **LSTM**: 300개 시퀀스로 수렴 한계, flat 예측 경향. 그러나 B3에서 LSTM(0.403)이 유일하게 작동하므로 제거 불가

---

## 4. 실험 이력 (요약)

| 실험 | 결과 | 결론 |
|------|------|------|
| F2S2 변환 Train+Test 적용 | HI Q 동일 | per-regime baseline이 이미 스케일 통일 → 중복 |
| 레짐별 피처 Q-score 필터링 | Ens_raw 0.447 (-0.018) | B4 오염으로 멀쩡한 피처까지 제거 |
| HI slope/delta 피처 추가 | Ens_raw 0.466 (+0.001) | B3 구조 문제라 피처로 해결 불가 |
| LSTM → Ridge 교체 (rul v5) | Ens_raw 0.462 | B3 Ridge=0.109, LSTM 제거 불가 |
| 3-way 앙상블 LGBM+Ridge+LSTM (rul v6) | Ens_raw 0.453 | B3 fold에서 LSTM 가중치 24%만 → B3 0.526→0.333 |
| HI 우회, raw z-score 직접 예측 | LOOCV 동일, Test 추론 불가 | start_frac 추정이 HI 스케일에 의존 |
| Conservative bias T2/T5/T6 ×0.875 | — | 수동 구분 근거 불충분 → 제거 |
| **HI: Energy/Power/RMS log1p 변환** (hi v4 + rul v5) | **Ens_raw 0.468 (+0.002)** | B3 HI Q 0.804→0.815, RUL 소폭 개선. Kurtosis/CF 추가는 B3 0.804→0.358 붕괴로 폐기 |

---

## 5. 파일 구조

```
User/SR/0603_v3/
├── code/
│   ├── hi_loo_regime_v4.py    # HI (현용, ★) — log1p on energy/power/rms
│   ├── rul_regime_v5.py       # RUL (현용, ★ 최종) — hi_v4 기반
│   ├── hi_loo_regime_v2.py    # HI 이전 버전 (raw 피처)
│   ├── rul_regime_v4.py       # RUL 이전 버전 (hi_v2 기반)
│   ├── rul_regime_v3.py       # 기준선
│   ├── rul_regime_v6.py       # 3-way 앙상블 실험 (폐기)
│   ├── rul_direct_v1.py       # raw z-score 실험 (폐기)
│   ├── hi_loo_regime_v3.py    # F2S2 both 실험 (폐기)
│   └── plot_features.py       # 피처 시각화 유틸
└── output/
    ├── train_v4/  # Bearing{1-4}_HI.csv/.png, features_raw.csv (log1p)
    ├── test_v4/   # Test{1-6}_HI.csv/.png, features_raw.csv (log1p)
    ├── train/     # v2 HI 결과
    ├── test/      # v2 HI 결과
    ├── rul_v5/    # loocv/test 예측 결과 (현용)
    ├── rul/       # v4 RUL 결과
    └── rul_direct/
```
