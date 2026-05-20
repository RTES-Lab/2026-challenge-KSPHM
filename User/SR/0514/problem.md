# 0514 파이프라인 알려진 문제점

## [P-LEAK] LOOCV 0.5937 — Data Leakage (삭제 예정)

### 점수 출처

`rul_ensemble.py` 구버전 (커밋 `2b070b1`) + **사전계산 HI (v3 방식)**으로 얻은 결과.

| 모델 | LOOCV |
|------|-------|
| LSTM-A (HI-A v3) | 0.6068 |
| LSTM-B (HI-B v3) | 0.4867 |
| LGBM (HI-A v3 flat) | 0.4800 |
| **앙상블** | **0.5937** |

문서 위치: `README.md` (옵션 B 섹션), `0514_공유용.md` (섹션 4)

---

### 원인: 사전계산 HI의 data leakage

**사전계산(v3) 방식**: LOOCV 실행 전에 Train 4개 베어링 전부를 사용해 HI를 한 번 계산하고 파일로 저장한 뒤, 모든 fold가 그 파일을 읽어서 씀.

```
hi_train.py (B1+B2+B3+B4 전체 기준으로 HI 계산)
→ hi/output/train/Bearing1_best.csv  ← B1이 자기 baseline 계산에 참여
→ hi/output/train/Bearing2_best.csv
   ...

LOOCV fold (B1 = test):
  학습: B2, B3, B4 HI (사전계산 파일)
  평가: B1 HI (사전계산 파일) ← leakage: B1이 자신의 HI 생성에 이미 기여함
```

**결론**: test bearing의 피처 분포가 baseline 추정에 개입하므로 LOOCV가 낙관적으로 뻥튀기됨. 0.5937은 실제 일반화 성능이 아님.

---

### Inline HI (v4 방식)로 수정

LOOCV 각 fold 안에서 test bearing을 제외한 나머지로만 baseline을 계산:

```
fold (B1 = test):
  1. {B2, B3, B4} 기준으로만 baseline/p5/p95 계산
  2. 이 기준으로 B1~B4 HI 생성
  3. {B2, B3, B4}로 모델 학습
  4. B1 HI로 평가
```

실제 배포 상황(test bearing HI를 train 기준으로 생성)과 동일한 구조 → leakage 없음.

**Inline HI 적용 후 점수**:

| 구성 | LOOCV |
|------|-------|
| LGBM + LSTM-C (`rul_ensemble.py` 현재) | 0.4257 |
| LGBM + LSTM-A + LSTM-C (`rul_ensemble_v2.py`) | 0.4267 |
| LGBM + LSTM-A, cf=0.76 (`rul_ensemble_v3.py`) | **0.4326** |

0.5937 → 0.4326: leakage 제거 시 실제 gap이 약 **-0.161** 수준임.

---

### 부수 문제: v4 Inline HI의 절대 스케일 역전

Inline HI로 전환하자 새로운 문제가 노출됨. v4 HI는 4개 베어링의 FDR 분포 전체 기준으로 절대 스케일을 유지하는데, 베어링 간 실제 FDR 크기가 너무 달라서 HI 범위가 역전됨:

| 베어링 | 수명 | v4 HI 범위 |
|--------|------|------------|
| Bearing3 | 89 cycles (빠른 열화) | 0.016 ~ **0.152** |
| Bearing4 | 137 cycles (느린 열화) | **0.244** ~ 0.814 |

B3 HI 최댓값(0.152) < B4 HI 최솟값(0.244) → 연속된 HI 값이 정반대의 RUL을 가짐 → LSTM 수렴 실패 (전 구간 0 예측, score=0.25).

**해결 (Approach A)**: window 내부 minmax 정규화 + `obs_fraction` 추가 → LSTM-A 부활, 현재 `rul_ensemble_v3.py`에 적용.
