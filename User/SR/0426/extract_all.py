"""
Run preprocess.features_one_run on every train_dir under dataset/, save CSVs.

Output: User/SR/0426/features/Train{i}_features.csv (one row per .tdms file)
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import pandas as pd

from preprocess import features_one_run

ROOT = Path("/data/home/ksphm/2026-challenge-KSPHM")
DATASET = ROOT / "dataset"
OUT_DIR = Path(__file__).parent / "features"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="e.g. Train1 Train2  (default: all train sets found)")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    if args.runs:
        run_names = args.runs
    else:
        run_names = sorted({p.name.split("_")[0]
                            for p in DATASET.glob("Train*_Vibration")})

    for name in run_names:
        run_dir = DATASET / f"{name}_Vibration"
        if not run_dir.exists():
            print(f"[skip] {run_dir} not found"); continue
        print(f"\n=== {name} ({len(list(run_dir.glob('*.tdms')))} files) ===")
        t0 = time.time()
        df = features_one_run(run_dir, samples_per_rev=256, verbose=True)
        df.to_csv(out_dir / f"{name}_features.csv", index=False)
        print(f"[save] {out_dir / f'{name}_features.csv'} "
              f"({len(df)} rows, {df.shape[1]} cols, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
