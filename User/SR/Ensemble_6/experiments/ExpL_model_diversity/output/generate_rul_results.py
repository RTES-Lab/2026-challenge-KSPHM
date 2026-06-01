from pathlib import Path
import pandas as pd

results_dir = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/Ensemble_6/experiments/ExpL_model_diversity/results")
output_dir = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SR/Ensemble_6/experiments/ExpL_model_diversity/output")
output_dir.mkdir(parents=True, exist_ok=True)

# Load best_config_v2.csv
df = pd.read_csv(results_dir / "best_config_v2.csv")

# We need columns: test_bearing, score, mean_er, dataset
rows = []
for b in [1, 2, 3, 4]:
    rows.append({
        "test_bearing": b,
        "score": df[f"B{b}_score"].iloc[0],
        "mean_er": df[f"B{b}_er"].iloc[0],
        "dataset": "Train"
    })

pd.DataFrame(rows).to_csv(output_dir / "rul_results.csv", index=False)
print("output/rul_results.csv generated successfully!")
