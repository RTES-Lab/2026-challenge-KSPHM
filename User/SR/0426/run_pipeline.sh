#!/bin/bash
# Full pipeline: feature extraction (GLCT-GWO) → LOOCV training → sanity checks
set -e
cd /data/home/ksphm/2026-challenge-KSPHM/User/SR/0426

LOG=results/pipeline_$(date +%Y%m%d_%H%M%S).log
mkdir -p results

echo "[$(date)] === Step 1: Feature extraction (GLCT-GWO) ===" | tee -a "$LOG"
python extract_all.py 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[$(date)] === Step 2: LOOCV training ===" | tee -a "$LOG"
python train_loocv.py --epochs 120 --bs 32 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[$(date)] === Step 3: Sanity checks (LOOCV fold checkpoints) ===" | tee -a "$LOG"
for run in Train1 Train2 Train3 Train4; do
    CKPT="checkpoints/loocv/fold_${run}/best.pt"
    if [ -f "$CKPT" ]; then
        echo "[$(date)] -- $run" | tee -a "$LOG"
        python sanity_check.py --ckpt "$CKPT" --run "$run" 2>&1 | tee -a "$LOG"
    else
        echo "[$(date)] SKIP $run (no checkpoint at $CKPT)" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "[$(date)] === Pipeline complete ===" | tee -a "$LOG"
echo "Log saved to: $LOG"
