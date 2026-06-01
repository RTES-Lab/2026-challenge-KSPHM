import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/data/home/ksphm/2026-challenge-KSPHM/User/SR/Ensemble_6/experiments/ExpL_model_diversity")
import run_expL_v2 as exp2

def main():
    print("=== Generating In-Sample (NO_LOO) Predictions ===")
    hi_tr = exp2.load_hi("train")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Train global models on all 4 bearings
    MODELS = {"tcn_res": exp2.TCNRes, "transformer": exp2.MiniTransformer, "bilstm": exp2.BiLSTM}
    use = ["bilstm", "tcn_res", "transformer"]
    alphas = {"bilstm": 0.6, "tcn_res": 0.6, "transformer": 0.6}
    cap = 2.0
    margin = 0.90
    
    print("Training global models...")
    nn_models = {}
    for name, Cls in MODELS.items():
        print(f"  {name} (3 seeds)...")
        nn_models[name] = exp2.train_ens(hi_tr, exp2.BEARINGS, Cls, dev, exp2.SEEDS_FINAL)
        
    # 2. Get global predictions of each base model on the 4 training bearings
    print("Predicting in-sample...")
    raw_global = {m: {} for m in ["dtw"] + use}
    meta = {}
    for b in exp2.BEARINGS:
        n = len(hi_tr[b])
        obs, p_dtw = exp2.pred_dtw(hi_tr, exp2.BEARINGS, hi_tr[b])
        raw_global["dtw"][b] = p_dtw
        meta[b] = {"N": n, "obs": obs}
        
        for mk in use:
            ms, sc = nn_models[mk]
            _, pp = exp2.pred_nn(ms, sc, hi_tr[b], dev)
            raw_global[mk][b] = pp

    # 3. Calculate global cfs
    global_cfs = {}
    for m in raw_global:
        cf_glob, _ = exp2.calibrate(raw_global[m], meta, exp2.BEARINGS)
        global_cfs[m] = cf_glob
        
    # 4. Compute global ensemble blend
    global_blend = {}
    for b in exp2.BEARINGS:
        base = raw_global["dtw"][b] * global_cfs["dtw"]
        ref = np.maximum(base, exp2.MIN_RUL)
        val = base.copy()
        for mk in use:
            pp_cal = raw_global[mk][b] * global_cfs[mk]
            val = val + alphas[mk] * np.clip(pp_cal - base, 0, (cap-1)*ref)
        global_blend[b] = np.maximum(val, exp2.MIN_RUL)
        
    global_cf_ens, _ = exp2.calibrate(global_blend, meta, exp2.BEARINGS, safety_margin=margin)
    
    # 5. Calculate scores and errors for NO_LOO registration
    rows = []
    for b in exp2.BEARINGS:
        final_pred = global_blend[b] * global_cf_ens
        sc = exp2.sc_curve(meta[b]["N"], meta[b]["obs"], final_pred)
        er = exp2.err_summ(meta[b]["N"], meta[b]["obs"], final_pred)["mean_er"]
        rows.append({
            "test_bearing": b,
            "score": sc,
            "mean_er": er,
            "dataset": "Train"
        })
        print(f"  Bearing {b}: score={sc:.4f} mean_er={er:.1f}%")
        
    output_path = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/Ensemble_6/experiments/ExpL_model_diversity/output/rul_results_noloo.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"In-sample predictions written to {output_path}!")

if __name__ == "__main__":
    main()
