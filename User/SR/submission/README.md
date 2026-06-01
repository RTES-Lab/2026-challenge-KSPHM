# RULulalla — Validation Submission

예비 제출 파일(`RULulalla_validation.xlsx`)을 재현하기 위한 안내입니다.

---

## 결과 요약

| File | RUL_Score (seconds) |
|:----:|:-------------------:|
| Validation1 | 15408 |
| Validation2 | 48780 |
| Validation3 | 32616 |
| Validation4 | 12420 |
| Validation5 | 15192 |
| Validation6 | 11772 |

**사용 모델**: Exp-L_Asym (`User/SR/Ensemble_6/experiments/ExpL_model_diversity/run_expL_v2.py`)  
**학습 데이터**: Train1~4 전체 (4개 베어링 모두 사용)  
**추론 대상**: Validation1~6 데이터셋 각각의 마지막 측정 시점 기준 RUL(초)

---

## 사전 준비 조건

### 환경

```bash
conda activate ksphm_env
```

주요 패키지: `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `torch`, `nptdms`, `openpyxl`

### 필요 입력 파일

| 데이터 | 경로 |
|--------|------|
| Train HI (SP 파이프라인 출력) | `User/SP/05-26/V1b/output/HI_Bearing{1-4}.csv` |
| Validation HI (SP 파이프라인 출력) | `User/SP/05-26/V1b/output/test/HI_Test{1-6}.csv` |
| Validation TDMS (원본) | `dataset/Test/Test{1-6}/` |

> **주의**: Validation set에는 진동 데이터(TDMS)만 제공됩니다. Operation 데이터(토크, RPM, 온도)는 제공되지 않습니다.

---

## 실행 방법

### Step 1: HI 생성 (아직 생성되지 않은 경우만)

Train 및 Validation 베어링의 Health Index를 생성합니다.

```bash
cd /data/home/ksphm/2026-challenge-KSPHM
/data/home/ksphm/anaconda3/envs/ksphm_env/bin/python User/SP/05-26/V1b/code/hi_v1b.py
```

출력:
- `User/SP/05-26/V1b/output/HI_Bearing{1-4}.csv`
- `User/SP/05-26/V1b/output/test/HI_Test{1-6}.csv`

> HI 파일이 이미 존재하면 이 단계는 생략합니다.

### Step 2: RUL 추론 실행

Train1~4 전체로 모델을 학습하고, Validation1~6에 대해 RUL을 추론합니다.

```bash
cd /data/home/ksphm/2026-challenge-KSPHM
/data/home/ksphm/anaconda3/envs/ksphm_env/bin/python \
    User/SR/Ensemble_6/experiments/ExpL_model_diversity/run_expL_v2.py
```

출력:
- `User/SR/Ensemble_6/experiments/ExpL_model_diversity/results/test_rul_results_v2.csv`  
  → `test`, `rul_hr` 컬럼 확인 (마지막 행이 해당 Validation 베어링의 RUL 예측값)

### Step 3: 제출 xlsx 생성

Step 2 결과를 읽어 제출 형식으로 저장합니다.

```bash
/data/home/ksphm/anaconda3/envs/ksphm_env/bin/python - << 'EOF'
import pandas as pd

src = 'User/SR/Ensemble_6/experiments/ExpL_model_diversity/results/test_rul_results_v2.csv'
df_rul = pd.read_csv(src)

rows = []
for _, row in df_rul.iterrows():
    rows.append({
        'File': f'Validation{int(row["test"])}',
        'RUL_Score': round(row['rul_hr'] * 3600),
    })

df_out = pd.DataFrame(rows)
print(df_out)
df_out.to_excel('User/SR/submission/RULulalla_validation.xlsx', index=False)
EOF
```

---

## 출력 파일 설명

| 파일 | 설명 |
|------|------|
| `RULulalla_validation.xlsx` | 제출용 파일 |

컬럼:

| 컬럼 | 설명 |
|------|------|
| `File` | Validation1 ~ Validation6 |
| `RUL_Score` | 예측 잔여수명 (초 단위) |

---

## 모델 구조 요약 (Exp-L_Asym)

**하이퍼파라미터**는 LOOCV(Leave-One-Out Cross Validation, Train1~4 대상)로 선정.  
**실제 추론**은 Train1~4 전체로 모델을 재학습한 뒤 Validation1~6에 적용.

```
obs_frac = HI / 0.75          # HI 기반 수명 위치 추정

base      = DTW 예측 × 0.68   # DTW를 안정 기저로 고정
bilstm_up = 0.6 × clip(BiLSTM 예측 - base, 0, base)
tcnres_up = 0.6 × clip(TCN-Res 예측 - base, 0, base)
transf_up = 0.6 × clip(Transformer 예측 - base, 0, base)
final     = (base + bilstm_up + tcnres_up + transf_up) × 0.765
```

- BiLSTM / TCN-Res / Transformer는 `AsymmetricHuberLoss(over_penalty=2.8)` 로 훈련 (과대예측 억제)
- `safety_margin=0.90` (cf를 10% 하향): T2처럼 HI가 역진하는 케이스의 고예측 방지

---

## 주의사항

1. **Validation set에는 operation 데이터 없음**: 진동(TDMS)만 제공. HI 생성 시 진동 데이터만 사용됨.
2. **재현성**: BiLSTM/TCN-Res/Transformer는 3 seeds(`SEEDS=[42, 43, 44]`) 평균. GPU/라이브러리 버전에 따라 소수점 미만 차이 가능.
3. **구 코드 혼동 주의**: `run_expL.py`(Phase 1+2)는 CF 탐색 시 검증 베어링 RUL을 참조하는 결함이 있으므로 제출에 사용하지 마십시오. **반드시 `run_expL_v2.py`를 사용**하십시오.
4. **실행 시간**: 전체 파이프라인(HI + RUL) 실행에 수 시간 소요될 수 있습니다.

---

## 관련 경로

| 항목 | 경로 |
|------|------|
| HI 생성 코드 | `User/SP/05-26/V1b/code/hi_v1b.py` |
| RUL 추론 코드 (최종) | `User/SR/Ensemble_6/experiments/ExpL_model_diversity/run_expL_v2.py` |
| 신규 모델 정의 | `User/SR/Ensemble_6/experiments/ExpL_model_diversity/new_models.py` |
| Test RUL 예측 결과 | `User/SR/Ensemble_6/experiments/ExpL_model_diversity/results/test_rul_results_v2.csv` |
| 실험 상세 기록 | `User/SR/Ensemble_6/progress.md` |
