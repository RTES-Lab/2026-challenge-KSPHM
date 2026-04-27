"""
Leave-One-Out Cross Validation (LOOCV) training for Adaptive MAGNN-TCN.

4개 Train run 중 매번 1개를 hold-out으로 빼고 나머지 3개로 학습.
총 4 fold를 실행하여 모든 run에 대한 out-of-sample 성능을 측정한 후,
최종 제출용으로 전체 데이터로 재학습한다.

출력:
  checkpoints/loocv/fold_{val_run}/best.pt     — fold별 best 체크포인트
  checkpoints/loocv/final/best.pt              — 전체 데이터 학습 (제출용)
  results/loocv_results.txt                    — 전체 LOOCV 결과 요약
  plots/loocv_scores.png                       — fold별 score 비교 그래프

Usage:
  python train_loocv.py [--epochs 120] [--bs 32] [--lr 5e-4]
"""
from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import AdaptiveMAGNN_TCN

ROOT = Path(__file__).parent
FEAT_DIR = ROOT / "features"
PLOT_DIR = ROOT / "plots"; PLOT_DIR.mkdir(exist_ok=True)
RESULT_DIR = ROOT / "results"; RESULT_DIR.mkdir(exist_ok=True)


# ── Dataset (from train.py) ──────────────────────────────────────────────
class RULDataset(Dataset):
    def __init__(self, runs, feat_cols, window=16, scaler=None):
        self.window = window
        self.feat_cols = feat_cols
        Xs, Ys, run_ids = [], [], []
        for name, df, lifetime_min in runs:
            df = df.sort_values("idx").reset_index(drop=True)
            F = df[feat_cols].to_numpy(dtype=np.float32)
            t = df["t_min"].to_numpy(dtype=np.float32)
            rul = np.clip((lifetime_min - t) / max(lifetime_min, 1e-6),
                          0.0, 1.0).astype(np.float32)
            Xs.append(F); Ys.append(rul); run_ids.append(name)
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(np.concatenate(Xs, axis=0))
        self.scaler = scaler
        self.windows = []
        for name, X, y in zip(run_ids, Xs, Ys):
            X_s = scaler.transform(X).astype(np.float32)
            n = len(X_s)
            for i in range(n):
                start = max(0, i - window + 1)
                seg = X_s[start:i + 1]
                if seg.shape[0] < window:
                    pad = np.zeros((window - seg.shape[0], seg.shape[1]),
                                   dtype=np.float32)
                    seg = np.concatenate([pad, seg], axis=0)
                self.windows.append((seg.T.copy(), float(y[i]), name))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        seg, y, _ = self.windows[i]
        return torch.from_numpy(seg), torch.tensor(y, dtype=torch.float32)


# ── Loss (from train.py) ─────────────────────────────────────────────────
class CompScoreLoss(nn.Module):
    def __init__(self, eps=1e-2, mse_weight=1.0):
        super().__init__()
        self.eps, self.mse_w = eps, mse_weight
        self.ln_half = math.log(0.5)

    def forward(self, y_hat, y):
        err = 100.0 * (y - y_hat) / torch.clamp(y, min=self.eps)
        loss_score = -self.ln_half * torch.where(
            err <= 0, -err / 20.0, err / 50.0)
        mse = ((y_hat - y) ** 2).mean()
        return loss_score.mean() + self.mse_w * mse


# ── Competition Score ─────────────────────────────────────────────────────
def comp_score_batch(y_hat, y, eps=1e-2):
    """Per-sample competition score ∈ (0, 1]."""
    err = 100.0 * (y - y_hat) / torch.clamp(y, min=eps)
    ln_half = math.log(0.5)
    s_over  = torch.exp(-ln_half * err / 20.0)
    s_under = torch.exp( ln_half * err / 50.0)
    return torch.where(err <= 0, s_over, s_under)


# ── Single Fold Training ──────────────────────────────────────────────────
def train_one_fold(
    train_runs, val_runs, feat_cols, args, ckpt_dir, fold_name, device
):
    """Train one fold, return best val_score and history."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = RULDataset(train_runs, feat_cols, window=args.window)
    val_ds = RULDataset(val_runs, feat_cols, window=args.window,
                        scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                            num_workers=0)

    model = AdaptiveMAGNN_TCN(
        n_vars=len(feat_cols), t_in=args.window,
        c_hidden=args.c_hidden, n_scales=args.n_scales,
        top_k=args.top_k, tcn_hidden=args.tcn_hidden,
        tcn_blocks=args.tcn_blocks).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = CompScoreLoss()

    best_score = -math.inf
    history = []

    for ep in range(args.epochs):
        # ── train ─────
        model.train()
        tr_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            yhat = model(xb)
            loss = crit(yhat, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0); n += xb.size(0)
        tr_loss /= max(n, 1)
        sched.step()

        # ── val ───────
        model.eval()
        vl, m = 0.0, 0
        scores = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                yhat = model(xb)
                vl += crit(yhat, yb).item() * xb.size(0); m += xb.size(0)
                scores.append(comp_score_batch(yhat, yb).cpu())
        val_loss = vl / max(m, 1)
        val_score = torch.cat(scores).mean().item()

        history.append({"epoch": ep, "train_loss": tr_loss,
                        "val_loss": val_loss, "val_score": val_score})

        # save best
        if val_score > best_score:
            best_score = val_score
            torch.save({
                "model_state": model.state_dict(),
                "feat_cols": feat_cols,
                "scaler_mean": train_ds.scaler.mean_.astype(np.float32),
                "scaler_std":  train_ds.scaler.scale_.astype(np.float32),
                "args": vars(args),
                "lifetimes": {n: lt for n, (_, lt) in
                              {r[0]: (r[1], r[2]) for r in train_runs + val_runs}.items()},
                "fold": fold_name,
                "val_run": val_runs[0][0],
            }, ckpt_dir / "best.pt")

        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"    ep {ep:3d}  train={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"vscore={val_score:.3f}  (best={best_score:.3f})")

    pd.DataFrame(history).to_csv(ckpt_dir / "history.csv", index=False)
    return best_score, history


# ── Full-data retraining (for submission) ──────────────────────────────────
def train_final(all_runs, feat_cols, args, ckpt_dir, device):
    """Train on all data for final submission model."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print("FINAL: Training on ALL data for submission model")
    print("=" * 60)

    train_ds = RULDataset(all_runs, feat_cols, window=args.window)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=0, drop_last=True)

    model = AdaptiveMAGNN_TCN(
        n_vars=len(feat_cols), t_in=args.window,
        c_hidden=args.c_hidden, n_scales=args.n_scales,
        top_k=args.top_k, tcn_hidden=args.tcn_hidden,
        tcn_blocks=args.tcn_blocks).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = CompScoreLoss()

    # late-life score for checkpoint selection (same as original train.py)
    late_xs, late_ys = [], []
    for name, df, lifetime_min in all_runs:
        df = df.sort_values("idx").reset_index(drop=True)
        F = df[feat_cols].to_numpy(dtype=np.float32)
        F = (F - train_ds.scaler.mean_) / np.maximum(train_ds.scaler.scale_, 1e-6)
        n_late = max(1, int(len(F) * 0.25))
        for i in range(len(F) - n_late, len(F)):
            start = max(0, i - args.window + 1)
            seg = F[start:i + 1]
            if seg.shape[0] < args.window:
                pad = np.zeros((args.window - seg.shape[0], seg.shape[1]),
                               dtype=np.float32)
                seg = np.concatenate([pad, seg], axis=0)
            late_xs.append(seg.T)
            t_i = i * 10.0
            rul_norm = max((lifetime_min - t_i) / max(lifetime_min, 1e-6), 0.0)
            late_ys.append(float(rul_norm))
    late_xs = np.stack(late_xs).astype(np.float32)
    late_ys = np.array(late_ys, dtype=np.float32)

    best_score = -math.inf
    for ep in range(args.epochs):
        model.train()
        tr_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            yhat = model(xb)
            loss = crit(yhat, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0); n += xb.size(0)
        tr_loss /= max(n, 1)
        sched.step()

        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(late_xs).to(device)
            yb = torch.from_numpy(late_ys).to(device)
            yhat = model(xb)
            s = comp_score_batch(yhat, yb)
            late_score = s.mean().item()

        if late_score > best_score:
            best_score = late_score
            torch.save({
                "model_state": model.state_dict(),
                "feat_cols": feat_cols,
                "scaler_mean": train_ds.scaler.mean_.astype(np.float32),
                "scaler_std":  train_ds.scaler.scale_.astype(np.float32),
                "args": vars(args),
                "lifetimes": {r[0]: r[2] for r in all_runs},
            }, ckpt_dir / "best.pt")

        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"    ep {ep:3d}  train={tr_loss:.4f}  late_score={late_score:.3f}  "
                  f"(best={best_score:.3f})")

    print(f"  [final] best late_score = {best_score:.4f}")
    return best_score


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default=str(FEAT_DIR))
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--n-scales", type=int, default=3)
    ap.add_argument("--c-hidden", type=int, default=16)
    ap.add_argument("--tcn-hidden", type=int, default=32)
    ap.add_argument("--tcn-blocks", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-final", action="store_true",
                    help="skip final full-data retraining")
    args = ap.parse_args()

    feat_dir = Path(args.feat_dir)
    loocv_dir = ROOT / "checkpoints" / "loocv"
    loocv_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load all runs ────────────────────────────────────────────────
    csvs = sorted(feat_dir.glob("Train*_features.csv"))
    if not csvs:
        raise RuntimeError(f"No features under {feat_dir}; run extract_all.py first")

    runs_all = {}
    for fp in csvs:
        df = pd.read_csv(fp)
        name = fp.stem.split("_")[0]
        N = len(df)
        lifetime = (N if name == "Train4" else N - 1) * 10.0
        runs_all[name] = (df, lifetime)
        print(f"[load] {name}: {N} files, lifetime ≈ {lifetime:.0f} min")

    meta_cols = {"file", "idx", "t_min"}
    sample_df = next(iter(runs_all.values()))[0]
    feat_cols = [c for c in sample_df.columns
                 if c not in meta_cols and pd.api.types.is_numeric_dtype(sample_df[c])]
    print(f"[features] {len(feat_cols)} feature columns")

    run_names = sorted(runs_all.keys())

    # ── LOOCV: 4 folds ──────────────────────────────────────────────
    fold_results = []
    t_start = time.time()

    for val_name in run_names:
        print(f"\n{'=' * 60}")
        print(f"FOLD: val={val_name}  train={[r for r in run_names if r != val_name]}")
        print(f"{'=' * 60}")

        torch.manual_seed(args.seed); np.random.seed(args.seed)

        train_runs = [(n, df, lt) for n, (df, lt) in runs_all.items() if n != val_name]
        val_runs   = [(val_name, runs_all[val_name][0], runs_all[val_name][1])]

        fold_dir = loocv_dir / f"fold_{val_name}"
        best_score, history = train_one_fold(
            train_runs, val_runs, feat_cols, args, fold_dir, val_name, device)

        fold_results.append({
            "val_run": val_name,
            "best_val_score": best_score,
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
        })
        print(f"  [fold {val_name}] best val_score = {best_score:.4f}")

    # ── LOOCV summary ────────────────────────────────────────────────
    avg_score = np.mean([r["best_val_score"] for r in fold_results])
    print(f"\n{'=' * 60}")
    print(f"LOOCV AVERAGE val_score = {avg_score:.4f}")
    print(f"{'=' * 60}")

    # ── Final full-data training ──────────────────────────────────────
    final_score = None
    if not args.skip_final:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        all_runs = [(n, df, lt) for n, (df, lt) in runs_all.items()]
        final_dir = loocv_dir / "final"
        final_score = train_final(all_runs, feat_cols, args, final_dir, device)

    elapsed = time.time() - t_start

    # ── Save results ──────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("LOOCV (Leave-One-Out Cross Validation) 결과")
    lines.append("=" * 70)
    lines.append("")
    lines.append("학습 설정:")
    for k, v in vars(args).items():
        lines.append(f"  {k}: {v}")
    lines.append(f"  device: {device}")
    lines.append(f"  features: {len(feat_cols)}")
    lines.append("")
    lines.append("-" * 70)
    lines.append(f"{'Fold':<10} {'Val Run':<10} {'Best Val Score':>15} {'Final Train Loss':>18}")
    lines.append("-" * 70)
    for r in fold_results:
        lines.append(f"{'fold_'+r['val_run']:<10} {r['val_run']:<10} "
                     f"{r['best_val_score']:>15.4f} {r['final_train_loss']:>18.4f}")
    lines.append("-" * 70)
    lines.append(f"{'AVERAGE':<10} {'':10} {avg_score:>15.4f}")
    lines.append("")
    if final_score is not None:
        lines.append(f"최종 모델 (전체 데이터 학습) late_score: {final_score:.4f}")
        lines.append(f"  체크포인트: checkpoints/loocv/final/best.pt")
    lines.append("")
    lines.append(f"총 소요 시간: {elapsed:.0f}초 ({elapsed/60:.1f}분)")
    lines.append("")
    lines.append("해석:")
    lines.append(f"  LOOCV 평균 val_score = {avg_score:.4f}")
    lines.append("  이 값은 새로운 베어링에 대한 예상 성능의 비편향 추정치입니다.")
    lines.append("  각 fold의 val_score 편차가 크면, 모델이 특정 run에 민감한 것을 의미합니다.")
    lines.append("=" * 70)

    result_text = "\n".join(lines) + "\n"
    result_path = RESULT_DIR / "loocv_results.txt"
    result_path.write_text(result_text, encoding="utf-8")
    print(f"\n[save] {result_path}")

    # ── Fold comparison plot ──────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart: per-fold val score
    names = [r["val_run"] for r in fold_results]
    scores = [r["best_val_score"] for r in fold_results]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax1.bar(names, scores, color=colors[:len(names)], alpha=0.8, edgecolor="black")
    ax1.axhline(avg_score, color="crimson", ls="--", lw=2,
                label=f"LOOCV avg = {avg_score:.3f}")
    ax1.set_ylabel("Best Val Score", fontsize=11)
    ax1.set_title("LOOCV: Per-Fold Validation Score", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    for bar, s in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{s:.3f}", ha="center", fontsize=10, fontweight="bold")

    # Learning curves: per-fold val_score over epochs
    for i, val_name in enumerate(names):
        h = pd.read_csv(loocv_dir / f"fold_{val_name}" / "history.csv")
        ax2.plot(h["epoch"], h["val_score"], "-", lw=1.5, color=colors[i],
                 label=f"fold_{val_name}", alpha=0.8)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Val Score", fontsize=11)
    ax2.set_title("LOOCV: Val Score Learning Curves", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "loocv_scores.png", dpi=130)
    plt.close(fig)
    print(f"[save] {PLOT_DIR / 'loocv_scores.png'}")

    print(f"\n✓ LOOCV 완료. 평균 val_score = {avg_score:.4f}")
    if final_score is not None:
        print(f"✓ 최종 모델 (전체 학습) late_score = {final_score:.4f}")
        print(f"  제출용 체크포인트: checkpoints/loocv/final/best.pt")


if __name__ == "__main__":
    main()
