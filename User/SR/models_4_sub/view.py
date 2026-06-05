#!/usr/bin/env python3
"""
models_4_sub/view.py  —  RUL 모델 결과 조회

사용법:
  python3 view.py                      # 전체 결과 (mean 기준 내림차순)
  python3 view.py --run 0605_155119    # 특정 run_id만
  python3 view.py --hi hi_data         # 특정 input_hi만
  python3 view.py --top 5              # 상위 N개만
"""

import argparse
import pandas as pd
from pathlib import Path

RESULT_CSV = Path(__file__).parent / "result.csv"


def _print_table(df: pd.DataFrame):
    if df.empty:
        print("결과 없음")
        return

    b_cols = [c for c in ["B1", "B2", "B3", "B4"] if c in df.columns]

    # column widths
    RK = 5
    HI = 12
    MDL = 20
    BC = 8
    MN = 8
    BIAS = 7
    EL = 9

    W = RK + 1 + HI + 1 + MDL + len(b_cols) * (BC + 1) + 1 + MN + 1 + BIAS + 1 + EL

    b_hdr = "".join(f" {c:>{BC}}" for c in b_cols)
    header = (
        f"{'Rank':<{RK}} {'input_hi':<{HI}} "
        f"{'Model':<{MDL}}{b_hdr} {'Mean':>{MN}} {'Bias':>{BIAS}} {'Time(s)':>{EL}}"
    )

    print("\n" + "=" * W)
    print(f"  models_4_sub  결과  (mean 기준 내림차순, 총 {len(df)}건)")
    print("=" * W)
    print(header)
    print("-" * W)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        star = "★" if i == 1 else " "
        b_vals = "".join(
            f" {row[c]:>{BC}.4f}" if pd.notnull(row.get(c)) else f" {'-':>{BC}}"
            for c in b_cols
        )
        hi  = str(row.get("input_hi", ""))[:HI]
        mdl = str(row.get("model", ""))[:MDL]
        mn  = row.get("mean")
        bs  = row.get("bias")
        el  = row.get("elapsed_s")

        mn_s  = f"{mn:>{MN}.4f}"  if pd.notnull(mn)  else f"{'–':>{MN}}"
        bs_s  = f"{bs:>{BIAS}.4f}" if pd.notnull(bs)  else f"{'–':>{BIAS}}"
        el_s  = f"{el:>{EL}.1f}"  if pd.notnull(el)  else f"{'–':>{EL}}"

        print(
            f"{str(i)+star:<{RK}} {hi:<{HI}} "
            f"{mdl:<{MDL}}{b_vals} {mn_s} {bs_s} {el_s}"
        )

    print("=" * W)


def main():
    pa = argparse.ArgumentParser(
        description="models_4_sub result.csv 뷰어",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    pa.add_argument("--run",  default=None, help="특정 run_id 필터")
    pa.add_argument("--hi",   default=None, help="특정 input_hi 필터")
    pa.add_argument("--top",  type=int, default=None, help="상위 N개만 표시")
    args = pa.parse_args()

    if not RESULT_CSV.exists():
        print(f"result.csv 없음: {RESULT_CSV}")
        return

    df = pd.read_csv(RESULT_CSV)
    if args.run:
        df = df[df["run_id"] == args.run]
    if args.hi:
        df = df[df["input_hi"] == args.hi]

    df = df.sort_values("mean", ascending=False).reset_index(drop=True)

    if args.top:
        df = df.head(args.top)

    _print_table(df)


if __name__ == "__main__":
    main()
