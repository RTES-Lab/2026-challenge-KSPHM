"""
s1a_cnn_improved.py  —  Approach A: Improved 1D-CNN
=====================================================
Key improvements over the original s1_dei_cnn.py:
  1. Window size: 2560 (0.1 s) → 25600 (1 s)   more fault cycles per window
  2. Data augmentation: 1 window/file → K=20 random windows/file (466→9320 samples)
  3. Regularisation: Dropout(0.3) added after every FC layer
  4. Learning rate: 1e-5 → 1e-4 (faster convergence with bigger dataset)
  5. Larger FC head: 768 → 256 → 1  (matches increased flat dimension)
  6. Architecture adjusted for 25600-sample input

Output:
  output/hi/Train{b}_HI_CNN.csv   — CNN re-estimate on training (for diagnostic)
  output/hi/Test{t}_HI_CNN.csv    — CNN estimate on test  (Approach A HI)
"""

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
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/data/home/ksphm/2026-challenge-KSPHM")
DATA_DIR = BASE_DIR / "dataset"
HI_DIR   = BASE_DIR / "User/SR/0605_ref/output/hi"
MDL_DIR  = BASE_DIR / "User/SR/0605_ref/output/models"
MDL_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
FS          = 25600
WIN_SIZE    = 25600      # 1 s
K_AUG       = 20         # augmentation windows per file
BATCH_SIZE  = 64
EPOCHS      = 400
LR          = 1e-4
DROPOUT     = 0.3
TRAIN_EOL   = {1: 126, 2: 114, 3: 89, 4: 137}
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_channels(path: Path) -> dict:
    tf = TdmsFile.read(str(path))
    return {ch: tf["Vibration"][ch][:].astype(np.float32)
            for ch in ["CH1", "CH2", "CH3", "CH4"]}


def random_windows(signal: np.ndarray, size: int = WIN_SIZE,
                   k: int = K_AUG) -> np.ndarray:
    """k evenly-spaced random windows; shape (k, size)."""
    max_start = len(signal) - size
    starts    = np.linspace(0, max_start, k, dtype=int)
    return np.stack([signal[s: s + size] for s in starts])


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


# ── CNN architecture ──────────────────────────────────────────────────────────

class ImprovedCNN(nn.Module):
    """1D-CNN for 25600-sample input with dropout regularisation."""

    def __init__(self):
        super().__init__()
        # feature extraction
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=100, stride=50),  # → 510
            nn.ReLU(),
            nn.MaxPool1d(2, 2),                             # → 255
            nn.Conv1d(64, 64, kernel_size=2, stride=1),    # → 254
            nn.ReLU(),
            nn.MaxPool1d(2, 2),                             # → 127
        )
        flat = 64 * 127   # 8128
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.regressor(self.features(x)).squeeze(1)


# ── Dataset construction ──────────────────────────────────────────────────────

def build_dataset():
    """
    Build (X, y) with K_AUG augmented windows per file.
    X shape: (N*K_AUG, 1, WIN_SIZE),  y shape: (N*K_AUG,)
    """
    X_list, y_list = [], []

    for b in range(1, 5):
        vib_dir    = DATA_DIR / f"Train{b}_Vibration"
        tdms_files = sorted(vib_dir.glob("*.tdms"))
        hi_labels  = pd.read_csv(HI_DIR / f"Bearing{b}_HI.csv")["HI"].values

        n = min(len(hi_labels), len(tdms_files))
        hi_labels  = hi_labels[:n]
        tdms_files = tdms_files[:n]

        print(f"  Bearing {b}: {n} files, HI [{hi_labels.min():.3f}, {hi_labels.max():.3f}]")

        for i, tf_path in enumerate(tdms_files):
            sig   = load_channels(tf_path)["CH1"]
            wins  = random_windows(sig, WIN_SIZE, K_AUG)
            label = float(hi_labels[i])
            for win in wins:
                X_list.append(zscore(win))
                y_list.append(label)

    X = np.stack(X_list)[:, np.newaxis, :]
    y = np.array(y_list, dtype=np.float32)
    print(f"  Total: X={X.shape}, y={y.shape}")
    return X, y


# ── Training ──────────────────────────────────────────────────────────────────

def train(X, y):
    N       = len(y)
    idx     = np.random.permutation(N)
    n_val   = max(1, int(N * 0.12))
    tr_idx  = idx[n_val:]
    vl_idx  = idx[:n_val]

    Xtr = torch.tensor(X[tr_idx], dtype=torch.float32)
    ytr = torch.tensor(y[tr_idx], dtype=torch.float32)
    Xvl = torch.tensor(X[vl_idx], dtype=torch.float32).to(DEVICE)
    yvl = torch.tensor(y[vl_idx], dtype=torch.float32).to(DEVICE)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=False)

    model     = ImprovedCNN().to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR,
                                  weight_decay=1e-5)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_vl, best_state = np.inf, None
    tr_losses, vl_losses = [], []

    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimiser.step()
            ep_loss += loss.item() * len(yb)
        ep_loss /= len(tr_idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            vl_loss = criterion(model(Xvl), yvl).item()

        tr_losses.append(ep_loss)
        vl_losses.append(vl_loss)

        if vl_loss < best_vl:
            best_vl    = vl_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 50 == 0 or ep == 1:
            print(f"  Epoch {ep:4d}/{EPOCHS}  tr={ep_loss:.5f}  val={vl_loss:.5f}")

    model.load_state_dict(best_state)

    # loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(tr_losses, label="Train"); ax.semilogy(vl_losses, label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Improved CNN — Loss")
    plt.tight_layout()
    plt.savefig(MDL_DIR / "cnn_improved_loss.png", dpi=100)
    plt.close()
    print(f"  Best val MSE = {best_vl:.5f}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_sequence(model: ImprovedCNN, vib_dir: Path,
                   k_avg: int = 10) -> np.ndarray:
    """Average k_avg window predictions per file."""
    model.eval()
    tdms_files = sorted(vib_dir.glob("*.tdms"))
    preds = []
    for tf_path in tdms_files:
        sig  = load_channels(tf_path)["CH1"]
        wins = random_windows(sig, WIN_SIZE, k_avg)
        X    = torch.tensor(np.stack([zscore(w) for w in wins])[:, np.newaxis, :],
                             dtype=torch.float32).to(DEVICE)
        preds.append(float(model(X).cpu().numpy().mean()))
    return np.array(preds, dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Approach A — Improved 1D-CNN")
    print(f"  WIN={WIN_SIZE} ({WIN_SIZE/FS:.2f}s)  K_AUG={K_AUG}  "
          f"DROPOUT={DROPOUT}  LR={LR}")
    print("=" * 60)

    # ── Build & train ─────────────────────────────────────────────────────────
    print("\n[1/3] Building augmented dataset …")
    X, y = build_dataset()

    print(f"\n[2/3] Training CNN ({EPOCHS} epochs, device={DEVICE}) …")
    model = train(X, y)
    torch.save(model.state_dict(), MDL_DIR / "cnn_improved.pt")
    print(f"  Saved → {MDL_DIR / 'cnn_improved.pt'}")

    # ── Apply to training bearings ────────────────────────────────────────────
    print("\n[3/3] Generating HI sequences …")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, b in zip(axes.flat, range(1, 5)):
        gt      = pd.read_csv(HI_DIR / f"Bearing{b}_HI.csv")["HI"].values
        cnn_hi  = infer_sequence(model, DATA_DIR / f"Train{b}_Vibration")
        n = min(len(gt), len(cnn_hi))
        ax.plot(gt[:n],     "b-",  lw=1.5, label="HI label (FDR)")
        ax.plot(cnn_hi[:n], "r--", lw=1.2, label="HI CNN (A)")
        ax.set_title(f"Bearing {b}"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        pd.DataFrame({"file": np.arange(1, n+1), "HI_CNN": cnn_hi[:n]
                      }).to_csv(HI_DIR / f"Train{b}_HI_CNN.csv", index=False)
    plt.suptitle("Approach A: FDR label vs Improved CNN estimate", fontsize=11)
    plt.tight_layout()
    plt.savefig(HI_DIR / "train_hi_cnn_improved.png", dpi=120)
    plt.close()

    # ── Apply to test bearings ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, t in zip(axes.flat, range(1, 7)):
        test_dir = DATA_DIR / "Test" / f"Test{t}"
        hi       = infer_sequence(model, test_dir)
        ax.plot(hi, "b-", lw=1.5)
        ax.set_title(f"Test {t}  (Approach A)"); ax.grid(alpha=0.3)
        pd.DataFrame({"file": np.arange(1, len(hi)+1), "HI_CNN": hi}
                     ).to_csv(HI_DIR / f"Test{t}_HI_CNN.csv", index=False)
    plt.suptitle("Approach A: Test Bearings — Improved CNN HI", fontsize=11)
    plt.tight_layout()
    plt.savefig(HI_DIR / "test_hi_cnn_improved.png", dpi=120)
    plt.close()
    print(f"  Plots saved → {HI_DIR}")
    print("\nApproach A complete.")
    return model


if __name__ == "__main__":
    main()
