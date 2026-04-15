"""RPM 실제 계단 패턴 분석"""
import numpy as np
import pandas as pd

DATA_DIR = "/data/home/ksphm/2026-challenge-KSPHM/dataset"

for i in [1,2,3,4]:
    df = pd.read_csv(f'{DATA_DIR}/Train{i}_Operation.csv', encoding='cp949')
    df.columns = [c.strip() for c in df.columns]
    rpm = df['Motor speed[rpm]'].values
    t   = df['Time[sec]'].values

    # 진동 파일 단위 (10분) 평균 RPM
    file_rpms = []
    for k in range(0, int(t.max()), 600):
        mask = (t >= k) & (t < k+60)
        if mask.any():
            file_rpms.append(rpm[mask].mean())
    file_rpms = np.array(file_rpms)

    # RPM 클러스터링 (10단위 반올림)
    rounded = np.round(file_rpms / 10) * 10
    unique_rpms = sorted(np.unique(rounded))
    counts = [int(np.sum(rounded==v)) for v in unique_rpms]
    print(f'Train{i}: RPM levels = {unique_rpms}')
    print(f'         Counts    = {counts}')

    # 1시간 단위 패턴
    hourly = []
    for h in range(0, len(file_rpms), 6):
        chunk = file_rpms[h:h+6]
        hourly.append(int(round(np.mean(chunk))))
    print(f'         Hourly    = {hourly}')
    print()
