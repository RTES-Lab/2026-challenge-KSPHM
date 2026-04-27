"""
Plot HI trajectories per train run for sanity (rms, kurt, BPFI/BPFO/BSF amps).
Saves PNGs to ./plots/.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
FEAT = ROOT / "features"
OUT = ROOT / "plots"; OUT.mkdir(exist_ok=True)

# pick representative HI columns (CH2 used in older pipeline; pool-mean across channels)
def pool_channels(df: pd.DataFrame, key: str) -> pd.Series:
    cols = [c for c in df.columns if c.endswith(f"_{key}") and c.split("_")[0] in ("CH1","CH2","CH3","CH4")]
    return df[cols].mean(axis=1) if cols else pd.Series(np.zeros(len(df)))

HIS = ["rms", "kurt", "crest", "BPFI_h1", "BPFO_h1", "BSF_h1",
       "BPFI_h2", "E_low", "E_mid", "E_high"]

for fp in sorted(FEAT.glob("Train*_features.csv")):
    name = fp.stem.split("_")[0]
    df = pd.read_csv(fp)
    fig, axes = plt.subplots(len(HIS) + 1, 1, figsize=(10, 1.4 * (len(HIS)+1)),
                             sharex=True)
    t = df["t_min"] / 60.0
    axes[0].plot(t, df["rpm_est"], ".", ms=2)
    axes[0].axhline(700, color="gray", ls="--", alpha=0.4)
    axes[0].axhline(950, color="gray", ls="--", alpha=0.4)
    axes[0].set_ylabel("RPM"); axes[0].grid(alpha=0.3)
    for ax, key in zip(axes[1:], HIS):
        y = pool_channels(df, key).values
        ax.plot(t, y, ".", ms=2, alpha=0.4)
        ax.plot(t, pd.Series(y).rolling(11, center=True, min_periods=1).median(),
                "-", lw=1.2, color="crimson")
        ax.set_ylabel(key); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [hours]")
    fig.suptitle(f"{name}: HI trajectories ({len(df)} files)", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_hi.png", dpi=110); plt.close(fig)
    print(f"[save] {OUT / f'{name}_hi.png'}")
