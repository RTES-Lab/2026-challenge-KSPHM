"""
Stage 1: DEI Extraction and 1D-CNN Training
=============================================
Paper: Cheng et al. (2020) "A Deep Learning-Based RUL Prediction Approach for Bearings"

DEI label (per measurement file):
  For each fault frequency (BPFI, BPFO, BSF) scaled by actual RPM:
    1. Bandpass filter full 60-s signal around fault frequency
    2. Hilbert transform → instantaneous amplitude envelope
    3. Marginal Hilbert Spectrum (MHS) at f ≈ mean of instantaneous amplitude
       (bandpass pre-selects frequency, so amplitude ≈ MHS contribution)
  DEI = max(MHS_BPFI, MHS_BPFO, MHS_BSF)
  DEI_norm = (DEI - DEI_min + eps) / (DEI_max - DEI_min + 2*eps)

1D-CNN architecture (Table III in paper):
  Input:  (B, 1, 2560)   – center 2560 samples of CH1 (0.1 s at 25.6 kHz)
  Conv1:  64 filters, k=100, stride=50  → (B, 64, 50)
  Pool1:  maxpool k=2, stride=2         → (B, 64, 25)
  Conv2:  64 filters, k=2,  stride=1   → (B, 64, 24)
  Pool2:  maxpool k=2, stride=2         → (B, 64, 12)
  Flatten → (B, 768)
  FC1:    768→100, ReLU
  Output: 100→1,  Sigmoid   (value in (0,1))
  Loss:   MSE vs normalized DEI label

Usage:
  python s1_dei_cnn.py
  python s1_dei_cnn.py --epochs 300 --lr 1e-4
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nptdms import TdmsFile
from scipy.signal import butter, sosfiltfilt, hilbert, savgol_filter
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM")
DATA_DIR = BASE_DIR / "dataset"
OUT_DEI  = BASE_DIR / "User/SR/0605_ref/output/dei"
OUT_MDL  = BASE_DIR / "User/SR/0605_ref/output/models"
for d in (OUT_DEI, OUT_MDL):
    d.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
FS           = 25600
INTERVAL_SEC = 600
WIN_SIZE     = 2560        # CNN input window (0.1 s)
WIN_K        = 20          # average K windows per file at inference
EPS          = 1e-4        # DEI normalization guard

BPFI_1000    = 140.0
BPFO_1000    =  93.0
BSF_1000     =  78.0
BW_FRAC      =  0.20       # bandpass half-width = BW_FRAC * fault_freq
RPM_BASELINE = 825.0       # (700+950)/2
RPM_ALPHA    = 2.0         # DEI ∝ rpm^alpha for healthy bearing; divide to normalize
SG_WINDOW    = 13          # Savitzky-Golay smoothing window (must be odd)
SG_POLYORDER = 2

TRAIN_EOL    = {1: 126, 2: 114, 3: 89, 4: 137}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_ch1(path: Path) -> np.ndarray:
    tf = TdmsFile.read(str(path))
    return tf["Vibration"]["CH1"][:].astype(np.float32)


def get_actual_rpm(op_csv: Path, file_idx: int) -> float:
    df = pd.read_csv(op_csv, encoding="cp949")
    df.columns = df.columns.str.strip()
    t_center = (file_idx - 1) * INTERVAL_SEC + 30
    idx = (df["Time[sec]"] - t_center).abs().idxmin()
    return float(df.loc[idx, "Motor speed[rpm]"])


def center_window(signal: np.ndarray, size: int = WIN_SIZE) -> np.ndarray:
    """Extract center `size` samples."""
    mid   = len(signal) // 2
    start = mid - size // 2
    return signal[start: start + size]


def random_windows(signal: np.ndarray, size: int = WIN_SIZE, k: int = WIN_K
                   ) -> np.ndarray:
    """Sample k non-overlapping random windows; shape (k, size)."""
    max_start = len(signal) - size
    starts    = np.linspace(0, max_start, k, dtype=int)
    return np.stack([signal[s: s + size] for s in starts])


# ── DEI label extraction ──────────────────────────────────────────────────────

def mhs_at_freq(signal: np.ndarray, fault_freq: float, bw_frac: float = BW_FRAC
                ) -> float:
    """
    Approximate Marginal Hilbert Spectrum (MHS) at fault_freq.
    Bandpass the full signal around fault_freq, compute mean instantaneous
    amplitude (= energy contribution at that frequency).
    """
    nyq   = FS / 2.0
    f_lo  = max(fault_freq * (1 - bw_frac), 1.0)
    f_hi  = min(fault_freq * (1 + bw_frac), nyq - 1)
    if f_lo >= f_hi:
        return 0.0
    sos   = butter(4, [f_lo / nyq, f_hi / nyq], "bandpass", output="sos")
    filt  = sosfiltfilt(sos, signal)
    env   = np.abs(hilbert(filt))
    return float(env.mean())


def compute_dei(signal: np.ndarray, rpm: float) -> float:
    """
    DEI for one measurement file.
    DEI_raw = max(MHS at BPFI, BPFO, BSF) at actual-RPM fault frequencies.
    """
    f_bpfi = BPFI_1000 * rpm / 1000.0
    f_bpfo = BPFO_1000 * rpm / 1000.0
    f_bsf  = BSF_1000  * rpm / 1000.0
    return max(mhs_at_freq(signal, f_bpfi),
               mhs_at_freq(signal, f_bpfo),
               mhs_at_freq(signal, f_bsf))


def rpm_correct_and_smooth(dei_arr: np.ndarray, rpm_arr: np.ndarray) -> np.ndarray:
    """
    Remove RPM-induced amplitude variation from DEI sequence.

    Physical model: DEI_healthy ∝ (rpm / rpm_baseline)^alpha
    Correction: DEI_corr = DEI_raw / (rpm/baseline)^alpha  →  ≈ const for healthy bearing

    Then apply Savitzky-Golay smoothing to suppress remaining noise.
    The smoothed signal preserves the long-term degradation trend.
    """
    rpm_factor   = (rpm_arr / RPM_BASELINE) ** RPM_ALPHA
    dei_corrected = dei_arr / (rpm_factor + 1e-12)

    # Adaptive SG window: at least 3, at most half the sequence, must be odd
    n   = len(dei_corrected)
    win = min(SG_WINDOW, n - (1 if n % 2 == 0 else 0))
    win = win if win % 2 == 1 else win - 1
    win = max(win, 3)
    polyord = min(SG_POLYORDER, win - 1)

    if n >= win:
        dei_smooth = savgol_filter(dei_corrected, window_length=win, polyorder=polyord)
    else:
        dei_smooth = dei_corrected.copy()

    # Clip to non-negative (smoothing can introduce small negatives)
    dei_smooth = np.clip(dei_smooth, 0.0, None)
    return dei_smooth


def normalize_dei(dei_raw: np.ndarray) -> np.ndarray:
    lo = dei_raw.min()
    hi = dei_raw.max()
    return (dei_raw - lo + EPS) / (hi - lo + 2 * EPS)


# ── dataset construction ──────────────────────────────────────────────────────

HI_DIR = BASE_DIR / "User/SR/0605_ref/output/hi"


def load_hi_label(b: int) -> np.ndarray:
    """Load FDR HI from c2_hi_pipeline output (leakage-fixed version)."""
    csv = HI_DIR / f"Bearing{b}_HI.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} — run c2_hi_pipeline.py first.")
    return pd.read_csv(csv)["HI"].values.astype(np.float32)


def build_training_dataset() -> tuple:
    """
    Returns
    -------
    X_norm : ndarray (N, 1, WIN_SIZE)  – z-score normalized vibration windows
    y_norm : ndarray (N,)              – FDR HI labels from c2_hi_pipeline
    meta   : list of dicts             – bearing id, file index, hi value
    """
    X_list, y_list, meta = [], [], []

    for b in range(1, 5):
        vib_dir    = DATA_DIR / f"Train{b}_Vibration"
        tdms_files = sorted(vib_dir.glob("*.tdms"))
        hi_labels  = load_hi_label(b)

        if len(hi_labels) != len(tdms_files):
            # Train4 has one missing TDMS — truncate label to available files
            n = min(len(hi_labels), len(tdms_files))
            hi_labels  = hi_labels[:n]
            tdms_files = tdms_files[:n]

        print(f"\nBearing {b}: {len(tdms_files)} files, "
              f"HI range [{hi_labels.min():.3f}, {hi_labels.max():.3f}]")

        for i, tf_path in enumerate(tdms_files):
            file_idx = int(tf_path.stem)
            sig  = load_ch1(tf_path)
            win  = center_window(sig, WIN_SIZE).astype(np.float32)
            win  = (win - win.mean()) / (win.std() + 1e-8)
            X_list.append(win)
            y_list.append(float(hi_labels[i]))
            meta.append({"bearing": b, "file": file_idx, "hi": float(hi_labels[i])})

    X = np.stack(X_list)[:, np.newaxis, :]   # (N, 1, WIN_SIZE)
    y = np.array(y_list, dtype=np.float32)
    return X, y, meta


def save_dei_csv(meta: list):
    """Save per-bearing DEI CSVs (raw + normalized)."""
    df = pd.DataFrame(meta)
    for b in range(1, 5):
        sub = df[df["bearing"] == b].reset_index(drop=True)
        sub.to_csv(OUT_DEI / f"Train{b}_DEI.csv", index=False)
        print(f"  Saved: Train{b}_DEI.csv  ({len(sub)} rows)")


def plot_dei(meta: list):
    df = pd.DataFrame(meta)

    # Raw DEI vs RPM
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, b in zip(axes.flat, range(1, 5)):
        sub = df[df["bearing"] == b]
        ax2 = ax.twinx()
        ax.plot(sub["file"].values, sub["dei_norm"].values, "b-", lw=1.5, label="DEI corr+norm")
        ax2.plot(sub["file"].values, sub["rpm"].values, "r--", lw=0.8, alpha=0.5, label="RPM")
        ax.set_xlabel("File index")
        ax.set_ylabel("DEI (RPM-corrected, normalized)", color="blue")
        ax2.set_ylabel("RPM", color="red")
        ax.set_title(f"Bearing {b}")
    plt.suptitle(f"Training DEI after RPM^{RPM_ALPHA} correction + SG smoothing", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DEI / "train_dei.png", dpi=120)
    plt.close()

    # 4-panel comparison: raw vs corrected
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, b in zip(axes.flat, range(1, 5)):
        sub = df[df["bearing"] == b]
        raw_norm = (sub["dei_raw"].values - sub["dei_raw"].min()) / \
                   (sub["dei_raw"].max() - sub["dei_raw"].min() + 1e-12)
        ax.plot(sub["file"].values, raw_norm, "b-", lw=1, alpha=0.6, label="DEI raw (norm)")
        ax.plot(sub["file"].values, sub["dei_norm"].values, "r-", lw=1.5, label="DEI corrected")
        ax.set_xlabel("File index"); ax.set_ylabel("DEI")
        ax.set_title(f"Bearing {b}"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle("DEI: raw (norm) vs RPM-corrected+smoothed", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DEI / "train_dei_comparison.png", dpi=120)
    plt.close()
    print(f"Plots saved → {OUT_DEI / 'train_dei.png'}, train_dei_comparison.png")


# ── CNN model ─────────────────────────────────────────────────────────────────

class BearingCNN(nn.Module):
    """1D-CNN from Table III of Cheng et al. (2020)."""

    def __init__(self, win_size: int = WIN_SIZE):
        super().__init__()
        L = win_size
        # Conv1: k=100, stride=50
        L = (L - 100) // 50 + 1                    # 50
        # MaxPool1: k=2, stride=2
        L = (L - 2) // 2 + 1                        # 25
        # Conv2: k=2, stride=1
        L = (L - 2) // 1 + 1                        # 24
        # MaxPool2: k=2, stride=2
        L = (L - 2) // 2 + 1                        # 12
        self._flat = 64 * L

        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=100, stride=50),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(x)).squeeze(1)


# ── training ──────────────────────────────────────────────────────────────────

def train_cnn(X: np.ndarray, y: np.ndarray,
              epochs: int = 300, lr: float = 1e-5,
              batch_size: int = 32, val_frac: float = 0.15) -> BearingCNN:
    """
    Train 1D-CNN using Adam optimizer with MSE loss.
    Parameters mirror the original paper (lr=1e-5, optimizer=Adam).
    """
    N = len(y)
    idx     = np.random.permutation(N)
    n_val   = max(1, int(N * val_frac))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    Xtr = torch.tensor(X[tr_idx], dtype=torch.float32)
    ytr = torch.tensor(y[tr_idx], dtype=torch.float32)
    Xvl = torch.tensor(X[val_idx], dtype=torch.float32).to(DEVICE)
    yvl = torch.tensor(y[val_idx], dtype=torch.float32).to(DEVICE)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size,
                        shuffle=True, drop_last=False)

    model = BearingCNN(WIN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    tr_losses, vl_losses = [], []
    best_vl, best_state  = np.inf, None

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(yb)
        ep_loss /= len(tr_idx)

        model.eval()
        with torch.no_grad():
            vl_loss = criterion(model(Xvl), yvl).item()

        tr_losses.append(ep_loss)
        vl_losses.append(vl_loss)

        if vl_loss < best_vl:
            best_vl    = vl_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 50 == 0 or ep == 1:
            print(f"  Epoch {ep:4d}/{epochs}  train_mse={ep_loss:.6f}  val_mse={vl_loss:.6f}")

    model.load_state_dict(best_state)

    # save loss plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(tr_losses, label="Train")
    ax.semilogy(vl_losses, label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (log)")
    ax.set_title("CNN Training Loss"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_MDL / "cnn_loss.png", dpi=100)
    plt.close()
    print(f"  Best val MSE = {best_vl:.6f}")
    return model


# ── inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def predict_dei_file(model: BearingCNN, signal: np.ndarray, k: int = WIN_K
                     ) -> float:
    """
    Estimate DEI for one TDMS signal by averaging K random-window predictions.
    """
    model.eval()
    wins  = random_windows(signal, WIN_SIZE, k).astype(np.float32)
    # z-score per window
    mu    = wins.mean(axis=1, keepdims=True)
    sd    = wins.std(axis=1, keepdims=True) + 1e-8
    wins  = (wins - mu) / sd
    X     = torch.tensor(wins[:, np.newaxis, :], dtype=torch.float32).to(DEVICE)
    preds = model(X).cpu().numpy()
    return float(preds.mean())


def predict_dei_sequence(model: BearingCNN, vib_dir: Path) -> np.ndarray:
    """Return estimated DEI sequence for all TDMS files in a directory."""
    tdms_files = sorted(vib_dir.glob("*.tdms"))
    dei_seq    = []
    for tf_path in tdms_files:
        sig = load_ch1(tf_path)
        dei_seq.append(predict_dei_file(model, sig))
    return np.array(dei_seq, dtype=np.float32)


# ── main ──────────────────────────────────────────────────────────────────────

def main(epochs: int = 300, lr: float = 1e-5):
    print("=" * 60)
    print("Stage 1 – DEI Extraction + CNN Training")
    print("=" * 60)

    # ── Step 1: build training dataset ────────────────────────────────────────
    print("\n[1/4] Loading FDR HI labels (from c2_hi_pipeline) …")
    X, y, meta = build_training_dataset()
    print(f"      Dataset: X={X.shape}, y={y.shape}  (device: {DEVICE})")

    # ── Step 2: train CNN ─────────────────────────────────────────────────────
    print(f"\n[2/4] Training 1D-CNN ({epochs} epochs, lr={lr}) …")
    model = train_cnn(X, y, epochs=epochs, lr=lr)
    torch.save(model.state_dict(), OUT_MDL / "bearing_cnn.pt")
    print(f"      Model saved → {OUT_MDL / 'bearing_cnn.pt'}")

    # ── Step 3: re-estimate DEI on training bearings (CNN output) ─────────────
    print("\n[3/4] Re-estimating DEI on training bearings via CNN …")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    df_meta   = pd.DataFrame(meta)
    for ax, b in zip(axes.flat, range(1, 5)):
        vib_dir = DATA_DIR / f"Train{b}_Vibration"
        dei_cnn = predict_dei_sequence(model, vib_dir)
        sub     = df_meta[df_meta["bearing"] == b]

        ax.plot(sub["file"].values, sub["hi"].values,
                "b-", lw=1.5, label="HI label (FDR)")
        ax.plot(np.arange(1, len(dei_cnn) + 1), dei_cnn,
                "r--", lw=1.2, label="HI CNN estimate")
        ax.set_xlabel("File index"); ax.set_ylabel("HI")
        ax.set_title(f"Bearing {b}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # save CSV
        pd.DataFrame({"file": np.arange(1, len(dei_cnn) + 1),
                      "dei_cnn": dei_cnn}).to_csv(
            OUT_DEI / f"Train{b}_DEI_CNN.csv", index=False)

    plt.suptitle("Training: FDR HI label vs CNN estimate", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DEI / "train_dei_vs_cnn.png", dpi=120)
    plt.close()
    print(f"      Plot saved → {OUT_DEI / 'train_dei_vs_cnn.png'}")

    # ── Step 4: estimate DEI on test bearings ─────────────────────────────────
    print("\n[4/4] Estimating DEI on test bearings …")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, t_id in zip(axes.flat, range(1, 7)):
        test_dir = DATA_DIR / "Test" / f"Test{t_id}"
        dei_cnn  = predict_dei_sequence(model, test_dir)
        ax.plot(np.arange(1, len(dei_cnn) + 1), dei_cnn, "b-", lw=1.5)
        ax.set_xlabel("File index"); ax.set_ylabel("DEI")
        ax.set_title(f"Test {t_id}"); ax.grid(alpha=0.3)

        pd.DataFrame({"file": np.arange(1, len(dei_cnn) + 1),
                      "dei_cnn": dei_cnn}).to_csv(
            OUT_DEI / f"Test{t_id}_DEI_CNN.csv", index=False)

    plt.suptitle("Test Bearings — CNN-Estimated DEI", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DEI / "test_dei_cnn.png", dpi=120)
    plt.close()
    print(f"      Plot saved → {OUT_DEI / 'test_dei_cnn.png'}")

    print("\nStage 1 complete.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=300)
    parser.add_argument("--lr",         type=float, default=1e-5)
    args = parser.parse_args()
    main(epochs=args.epochs, lr=args.lr)
