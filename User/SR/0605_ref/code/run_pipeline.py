"""
Master Pipeline Runner
======================
Stage 0 → Stage 1 → Stage 2+3

Usage:
  python run_pipeline.py                          # full pipeline
  python run_pipeline.py --skip_s0               # skip RPM validation
  python run_pipeline.py --from_stage 2           # resume from stage 2
  python run_pipeline.py --epochs 200 --percentile 0.42
"""

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM")
CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def run_stage0(step: int = 5) -> tuple:
    from s0_rpm_estimation import run_validation, estimate_test_rpm
    t0 = time.time()
    best_method, best_rmse, proceed = run_validation(step=step)
    if proceed:
        estimate_test_rpm(best_method)
    else:
        print(f"  [Stage 0] RMSE={best_rmse:.1f} ≥ 30 rpm → fallback to avg RPM for test")
    print(f"  Stage 0 elapsed: {fmt_time(time.time()-t0)}")
    return best_method, best_rmse, proceed


def run_stage1(epochs: int = 300, lr: float = 1e-5):
    from s1_dei_cnn import main as s1_main
    t0 = time.time()
    model = s1_main(epochs=epochs, lr=lr)
    print(f"  Stage 1 elapsed: {fmt_time(time.time()-t0)}")
    return model


def run_stage2(percentile: float = 0.40, n_particles: int = 2000):
    from s2_f2s2_rul import main as s2_main
    t0 = time.time()
    results = s2_main(percentile=percentile, n_particles=n_particles)
    print(f"  Stage 2+3 elapsed: {fmt_time(time.time()-t0)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="DEI-CNN + F2S2 RUL Pipeline")
    parser.add_argument("--skip_s0",     action="store_true",
                        help="Skip Stage 0 (tacholess RPM validation)")
    parser.add_argument("--from_stage",  type=int, default=0,
                        help="Start from stage N (0, 1, 2)")
    parser.add_argument("--s0_step",     type=int,   default=5,
                        help="Stage 0: evaluate every Nth file")
    parser.add_argument("--epochs",      type=int,   default=300)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--percentile",  type=float, default=0.40,
                        help="RUL point estimate percentile (0.40 → under-predict bias)")
    parser.add_argument("--n_particles", type=int,   default=2000)
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("  DEI-CNN + F2S2 RUL Prediction Pipeline")
    print("=" * 60)

    # ── Stage 0 ───────────────────────────────────────────────────────────────
    if not args.skip_s0 and args.from_stage <= 0:
        print("\n── Stage 0: Tacholess RPM Estimation ──")
        run_stage0(step=args.s0_step)
    else:
        print("\n── Stage 0: SKIPPED ──")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if args.from_stage <= 1:
        print("\n── Stage 1: DEI Extraction + CNN Training ──")
        run_stage1(epochs=args.epochs, lr=args.lr)
    else:
        print("\n── Stage 1: SKIPPED (using existing model) ──")

    # ── Stage 2+3 ─────────────────────────────────────────────────────────────
    if args.from_stage <= 2:
        print("\n── Stage 2+3: F2S2 State Estimation + RUL Prediction ──")
        results = run_stage2(percentile=args.percentile, n_particles=args.n_particles)
    else:
        print("\n── Stage 2+3: SKIPPED ──")

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete.  Total: {fmt_time(total)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    pred_csv = BASE_DIR / "User/SR/0605_ref/output/predictions/test_rul_predictions.csv"
    if pred_csv.exists():
        import pandas as pd
        df = pd.read_csv(pred_csv)
        print("\n  Final RUL predictions:")
        print(f"  {'Test':>5s}  {'RUL [s]':>10s}  {'RUL [h]':>10s}")
        for _, row in df.iterrows():
            print(f"  Test{int(row['test_id']):>1d}  "
                  f"{row['rul_seconds']:>10.0f}  "
                  f"{row['rul_seconds']/3600:>10.2f}")


if __name__ == "__main__":
    main()
