# models_4_sub — RUL Prediction Benchmark

다양한 RUL 모델을 **LOO(Leave-One-Out) Cross-Validation**으로 leakage 없이 평가하고, 여러 HI 소스를 바꿔가며 비교할 수 있는 벤치마크 툴킷.

---

## 파일 구조

```
models_4_sub/
  run.py       ← 실행 진입점 (CLI)
  models.py    ← 모델 구현 (9종)
  utils.py     ← 공통 유틸: 데이터 로딩, 스코어링, 피처 추출
  view.py      ← result.csv 뷰어 (mean 기준 내림차순 정렬)
  result.csv   ← 누적 결과 로그
  output/      ← 실행 결과 자동 저장
```

---

## 입력 형식 — 단일 CSV, 2열 규칙

누가 HI를 뽑든 이 포맷으로 만들어서 경로만 넘기면 됩니다.

```
train_hi.csv          test_hi.csv
────────────          ───────────
id, HI [,regime]      id, HI [,regime]
1, 0.001 [,0]         1, 0.012 [,0]
1, 0.014 [,0]         1, 0.025 [,1]
...                   ...
2, 0.003 [,1]         2, 0.008 [,0]
...                   ...
```

- `id` : bearing 번호 (train 1~4, test 1~6)
- `HI` : 건강 지수 (스케일 무관)
- `regime` : 선택 컬럼, 없으면 자동으로 0.5로 채움
- 같은 `id` 내 행은 **시간순 정렬** 필수

기본 파일 위치: `hi_data/train_hi.csv`, `hi_data/test_hi.csv`

---

## 빠른 시작

```bash
# 기본 실행 (hi_data/ 폴더 사용)
python run.py

# 다른 사람 HI 넣기 — 경로만 바꾸면 끝
python run.py \
    --hi_train /path/to/their_train_hi.csv \
    --hi_test  /path/to/their_test_hi.csv

# Bias search 포함 (권장)
python run.py \
    --hi_train /path/to/train_hi.csv --bias_search

# 원하는 모델만
python run.py --models "KNN,SVR,RF"

# LSTM 제외하고 빠르게
python run.py \
    --models "KNN,SVR,RF,LGBM,GPR,Linear,LevelLookup,SlopeExtrapolate"
```

### 결과 조회 (view.py)

```bash
# 전체 결과 — mean 기준 내림차순
python view.py

# 상위 3개만
python view.py --top 3

# 특정 HI 소스 필터
python view.py --hi hi_data
```

---

## 모델 목록

| 모델 | 분류 | 학습 방식 | 입력 | 특징 |
|------|------|-----------|------|------|
| `SlopeExtrapolate` | Signal-based | 없음 | HI | HI slope → 고장임계치까지 외삽 |
| `LevelLookup` | 통계 | 히스토그램 | HI | HI 구간별 평균 RUL 조회 |
| `Linear` | ML / 선형 | Ridge Regression | hi_mean, hi_slope, hi_end, hi_std, hi_range, hi_max, hi_recent, regime_frac | 선형 회귀 |
| `KNN` | ML / 유사도 | k-NN (k=10) | hi_mean, hi_slope, hi_end, hi_std, hi_range, hi_max, hi_recent, regime_frac | 최근접 이웃 |
| `RF` | ML / 트리 앙상블 | Random Forest | hi_mean, hi_slope, hi_end, hi_std, hi_range, hi_max, hi_recent, regime_frac | Random Forest |
| `LGBM` | ML / 부스팅 | LightGBM | hi_mean, hi_slope, hi_end, hi_std, hi_range, hi_max, hi_recent, regime_frac | LightGBM |
| `SVR` | ML / 커널 | SVR (RBF) | hi_mean, hi_slope, hi_end, hi_std, hi_range, hi_max, hi_recent, regime_frac | SVM 회귀 |
| `GPR` | 확률 모델 | Gaussian Process | hi_mean, hi_slope, hi_end, hi_std, hi_range, hi_max, hi_recent, regime_frac | RBF+White noise 커널 |
| `LSTM` | 딥러닝 | LSTM 앙상블 (5 seeds) | HI | 순환 신경망 |

---

## Leakage 방지 설계

LOO 각 fold에서 다음을 철저히 보장:

1. **`model.fit()`** 은 반드시 `train_bids` 데이터만 접근
2. **정규화 통계** (mean, std, scaler) 는 train fold에서만 계산
3. **test bearing 의 HI/RUL** 는 fit() 에 전달되지 않음
4. `EOL`, `NORMAL_UNTIL` 는 train bearing 에만 사용 (label 생성 용)

---

## 출력 구조

```
models_4_sub/
  result.csv                 ← ★ 누적 결과 로그 (모든 실행 축적, 실시간 업데이트)

output/{MMDD_HHMMSS}/
  loocv_summary.csv          ← 해당 실행의 모델별 bearing 스코어 표
  loocv_comparison.png       ← 모델 순위 비교 바 차트
  {ModelName}/
    loocv_all.png            ← 2×2 LOO 예측 패널
    B{i}_loo.png             ← 베어링별 LOO 개별 그림
    loocv_detail.png         ← 산점도 + 잔차 히스토그램
  test/
    test_predictions.csv     ← long-form (model, test_id, rul_hours, rul_raw_hr, bias)
    test_pivot.csv           ← wide-form (model × test bearing)
    test_comparison.png      ← 모델별 test RUL 비교 바 차트
```

### result.csv 열 구성

| 열 | 설명 |
|----|------|
| `timestamp` | 결과 기록 시각 |
| `run_id` | 실행 식별자 (타임스탬프, output 폴더명과 동일) |
| `input_hi` | 사용한 HI 소스 (`--hi_name` 또는 `--hi_train` 경로의 부모 디렉토리명) |
| `model` | 모델명 |
| `B1`–`B4` | bearing별 LOO 스코어 |
| `mean` | 평균 LOO 스코어 |
| `bias` | 최적 bias (`--bias_search` 미사용 시 1.0) |
| `elapsed_s` | 소요 시간 (초) |

각 모델의 LOO 평가가 끝나는 즉시 한 행씩 append 되므로, 실행 도중에도 중간 결과를 확인할 수 있습니다.

---

## 전체 CLI 옵션

```
usage: python run.py
         [--hi_train PATH] [--hi_test PATH] [--out DIR]
         [--win_size N] [--stride N]
         [--models MODEL1,MODEL2,...]
         [--bias_search]
         [--hi_name LABEL]
         [--eol 1:126,2:114,3:89,4:137]
         [--nu  1:89,2:92,3:62,4:78]
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--hi_train` | `hi_data/train_hi.csv` | 학습 HI CSV |
| `--hi_test` | `hi_data/test_hi.csv` | 테스트 HI CSV |
| `--out` | `output/<타임스탬프>` | 결과 저장 경로 |
| `--win_size` | 50 | 슬라이딩 윈도우 크기 (cycles) |
| `--stride` | 1 | LOO 평가 stride |
| `--models` | 전체 모델 (LSTM 포함) | 쉼표 구분 모델명 |
| `--bias_search` | off | LOO pool에서 최적 bias 탐색 |
| `--hi_name` | `--hi_train`의 부모 디렉토리명 | `result.csv`에 기록할 HI 소스 레이블 |
| `--eol` | 경쟁 기본값 | bearing별 수명 사이클 override |
| `--nu` | 경쟁 기본값 | bearing별 normal phase 종료 시점 override |

---

## 다른 사람 HI 연결 워크플로우

1. 팀원이 HI 뽑고 → `train_hi.csv` / `test_hi.csv` 로 저장 (포맷: `id,HI[,regime]`)
2. 경로 + 레이블 지정해서 실행:

```bash
python run.py \
    --hi_train /path/to/their_train_hi.csv \
    --hi_test  /path/to/their_test_hi.csv \
    --models "KNN,SVR,RF,LGBM" \
    --bias_search \
    --hi_name "0603_v3" \
    --out output/their_hi_eval
```

3. `result.csv` 에서 여러 HI 소스 간 스코어 비교:

```
timestamp,           run_id,      input_hi,   model, B1,     B2,     B3,     B4,     mean,   bias, elapsed_s
2026-06-05 14:32:01, 0605_143201, 0603_v3,    SVR,   0.5012, 0.6282, 0.4634, 0.3981, 0.4977, 0.90, 12.3
2026-06-05 14:32:14, 0605_143201, 0603_v3,    KNN,   0.4421, 0.6587, 0.4798, 0.3282, 0.4772, 0.95, 8.1
2026-06-05 15:01:44, 0605_150144, teammate_A, SVR,   0.5230, 0.6011, 0.4901, 0.4120, 0.5066, 0.90, 11.8
...
```

4. 결과 조회:

```bash
python view.py --hi 0603_v3
python view.py --hi teammate_A
```

---

## 스코어 공식 (대회 기준)

```
Er = 100 × (RUL_true − RUL_pred) / RUL_true

score = exp(−ln(0.5) × Er / 20)   if Er ≤ 0  (과예측, 벌점 강함)
        exp( ln(0.5) × Er / 50)   if Er > 0  (과소예측, 벌점 약함)
```

- score = 1.0 : 완벽 예측
- Er = −20 (20% 과예측) → score = 0.5
- Er = +50 (50% 과소예측) → score = 0.5
- 과예측이 2.5배 더 가혹하므로 최적 bias는 대부분 < 1.0
