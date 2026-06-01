"""
HI Loader — Exchange Interface
================================
Allows any experiment to load HI from:
  1. hi_input/{BearingN}.csv and hi_input/{TestN}.csv  (external override)
  2. V1b default output (fallback)

Usage:
    from hi_loader import HILoader

    loader = HILoader(hi_input_dir="path/to/hi_input")  # None = use default
    hi_train = loader.load_train()  # {1: array, 2: array, 3: array, 4: array}
    hi_test  = loader.load_test()   # {1: array, ..., 6: array}
"""

from pathlib import Path
import numpy as np
import pandas as pd

BEARINGS = [1, 2, 3, 4]
TEST_IDS = [1, 2, 3, 4, 5, 6]

DEFAULT_HI_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM/User/SP/05-26/V1b/output")


def _load_hi_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "HI" in df.columns:
        return df["HI"].values.astype(float)
    # Try first numeric column as fallback
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return df[numeric_cols[0]].values.astype(float)
    raise ValueError(f"No numeric column found in {path}")


class HILoader:
    def __init__(self, hi_input_dir=None):
        self.hi_input_dir = Path(hi_input_dir) if hi_input_dir else None
        self._validate()

    def _validate(self):
        if self.hi_input_dir is not None and not self.hi_input_dir.exists():
            raise FileNotFoundError(f"hi_input_dir not found: {self.hi_input_dir}")

    def _get_train_path(self, bearing_id: int) -> Path:
        if self.hi_input_dir:
            p = self.hi_input_dir / f"Bearing{bearing_id}.csv"
            if p.exists():
                return p
        return DEFAULT_HI_DIR / f"HI_Bearing{bearing_id}.csv"

    def _get_test_path(self, test_id: int) -> Path:
        if self.hi_input_dir:
            p = self.hi_input_dir / f"Test{test_id}.csv"
            if p.exists():
                return p
        return DEFAULT_HI_DIR / "test" / f"HI_Test{test_id}.csv"

    def load_train(self) -> dict:
        result = {}
        for b in BEARINGS:
            path = self._get_train_path(b)
            result[b] = _load_hi_csv(path)
            src = "external" if (self.hi_input_dir and (self.hi_input_dir / f"Bearing{b}.csv").exists()) else "default"
            print(f"  [HI] Train B{b}: {src}  len={len(result[b])}  {result[b][0]:.3f}→{result[b][-1]:.3f}")
        return result

    def load_test(self) -> dict:
        result = {}
        for t in TEST_IDS:
            path = self._get_test_path(t)
            result[t] = _load_hi_csv(path)
            src = "external" if (self.hi_input_dir and (self.hi_input_dir / f"Test{t}.csv").exists()) else "default"
            print(f"  [HI] Test  T{t}: {src}  len={len(result[t])}  {result[t][0]:.3f}→{result[t][-1]:.3f}")
        return result

    def describe(self) -> str:
        if self.hi_input_dir is None:
            return "Default V1b HI"
        return f"External HI from {self.hi_input_dir}"
