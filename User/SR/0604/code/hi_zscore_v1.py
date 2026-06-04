"""
HI Z-score v1 (0604)
====================
FDR 대신 z-score 기반 HI 계산.

핵심 차이:
- FDR: (feature - baseline_mean) / |baseline_mean|  → baseline 크기로 나눔
         → B3처럼 신호가 작은 베어링은 B2의 21x 변화 범위에 묻혀 압축됨
- Z-score: (feature - baseline_mean) / baseline_std → baseline 변동성으로 나눔
            → 각 피처의 신호 대비 잡음 비율로 측정 → 베어링 간 스케일 독립

추가:
- Baseline 오염 감지: 특정 베어링의 baseline 기간 중앙값이 다른 베어링 대비
  2σ 이상 이탈하면 baseline 계산에서 제외 (B4 오염 문제 대응)
- EOL p95로 정규화: train 베어링 말기 z-score의 95th percentile → HI [0,1]
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.signal import welch
import nptdms
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
BASE        = "/data/home/ksphm/2026-challenge-KSPHM"
SR_BASE     = f"{BASE}/User/SR/0604"
DATASET_DIR = os.path.join(BASE, "dataset")
TEST_DIR    = os.path.join(DATASET_DIR, "Test")
OUT_TRAIN   = os.path.join(SR_BASE, "output/train")
OUT_TEST    = os.path.join(SR_BASE, "output/test")
os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_TEST,  exist_ok=True)

BEARINGS       = [1, 2, 3, 4]
TEST_IDS       = [1, 2, 3, 4, 5, 6]
FS             = 25600
INTERVAL_SEC   = 600
RPM_BOUNDARY   = 850
BASELINE_RATIO = 0.10
EOL_RATIO      = 0.20       # 말기 정의: 마지막 20%
CONTAM_THRESH  = 2.0        # baseline 오염 감지 z-score 임계값
MIN_REGIME_WIN = 5

ALL_FEATS = [
    "ch3_high_band", "ch4_high_band",
    "ch3_total_power", "ch3_energy", "ch3_rms",
    "ch3_std", "ch3_p2p",
]


# ══════════════════════════════════════════════════════════════════
# Feature extraction (캐시: 0603_v3 것 재사용 가능하면 재사용)
# ══════════════════════════════════════════════════════════════════
def _extract_one(fpath):
    f   = nptdms.TdmsFile(fpath)
    grp = f["Vibration"]
    feat = {}
    for ch_name in ["CH3", "CH4"]:
        x  = grp[ch_name][:]
        ch = ch_name.lower()
        feat[f"{ch}_rms"]         = float(np.sqrt(np.mean(x ** 2)))
        feat[f"{ch}_p2p"]         = float(np.max(x) - np.min(x))
        feat[f"{ch}_std"]         = float(np.std(x))
        feat[f"{ch}_energy"]      = float(np.sum(x ** 2))
        nperseg = min(4096, len(x) // 8)
        fv, psd = welch(x, fs=FS, nperseg=nperseg)
        tp = float(np.sum(psd))
        feat[f"{ch}_total_power"] = tp
        idx_hb = (fv >= 3000) & (fv < 8000)
        feat[f"{ch}_high_band"]   = float(np.sum(psd[idx_hb]) / (tp + 1e-12))
    return feat


def _load_features(cache_path, tdms_dir):
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    files = sorted(glob.glob(os.path.join(tdms_dir, "*.tdms")))
    print(f"      TDMS 추출 ({len(files)}개)...")
    rows = [dict(_extract_one(fp), file_idx=i+1) for i, fp in enumerate(files)]
    df = pd.DataFrame(rows)[["file_idx"] + ALL_FEATS]
    df.to_csv(cache_path, index=False)
    return df


def get_train_cond(bid, n):
    op = pd.read_csv(
        os.path.join(DATASET_DIR, f"Train{bid}_Operation.csv"), encoding="cp949")
    op.columns = [c.strip() for c in op.columns]
    op = op.rename(columns={"Time[sec]": "time_sec", "Motor speed[rpm]": "rpm"})
    rpm = np.zeros(n)
    for k in range(n):
        t0, t1 = k * INTERVAL_SEC, k * INTERVAL_SEC + 60
        mask = (op["time_sec"] >= t0) & (op["time_sec"] < t1)
        vals = op.loc[mask, "rpm"].values
        rpm[k] = vals.mean() if len(vals) else np.nan
    nans = np.isnan(rpm)
    if nans.any():
        idx = np.arange(n)
        rpm[nans] = np.interp(idx[nans], idx[~nans], rpm[~nans])
    return (rpm >= RPM_BOUNDARY).astype(int)


def load_bearing(bid):
    # 0603_v3 캐시 우선 사용, 없으면 새로 추출
    cache = os.path.join(BASE, f"User/SR/0603_v3/output/train/Bearing{bid}_features_raw.csv")
    if not os.path.exists(cache):
        cache = os.path.join(OUT_TRAIN, f"Bearing{bid}_features_raw.csv")
    feat = _load_features(cache, os.path.join(DATASET_DIR, f"Train{bid}_Vibration"))
    cond = get_train_cond(bid, len(feat))
    df = feat[ALL_FEATS].copy().reset_index(drop=True)
    df["cond"] = cond
    return df


def load_test(tid):
    cache = os.path.join(BASE, f"User/SR/0603_v3/output/test/Test{tid}_features_raw.csv")
    if not os.path.exists(cache):
        cache = os.path.join(OUT_TEST, f"Test{tid}_features_raw.csv")
    return _load_features(cache, os.path.join(TEST_DIR, f"Test{tid}"))[ALL_FEATS].reset_index(drop=True)


def classify_regime_fft(test_id):
    files = sorted(glob.glob(os.path.join(TEST_DIR, f"Test{test_id}", "*.tdms")))
    cond = np.zeros(len(files), dtype=int)
    for k, fp in enumerate(files):
        x = nptdms.TdmsFile(fp)["Vibration"]["CH2"][:]
        freqs, psd = welch(x, fs=FS, nperseg=min(65536, len(x)//4))
        mask = (freqs >= 8) & (freqs <= 20)
        rpm = float(freqs[mask][np.argmax(psd[mask])] * 60)
        cond[k] = 1 if rpm >= RPM_BOUNDARY else 0
    return cond


# ══════════════════════════════════════════════════════════════════
# Z-score Regime Stats
# ══════════════════════════════════════════════════════════════════
def compute_zscore_stats(dfs, exclude_bid=None):
    """
    Returns per-regime: baseline_mean, baseline_std, feat_direction,
                        feat_q, eol_p95 (for normalization).
    """
    train_bids = [b for b in BEARINGS if b != exclude_bid]
    result = {}

    for regime in [0, 1]:
        # ── 1. Baseline samples per bearing ──────────────────────
        bear_base = {}
        for bid in train_bids:
            df    = dfs[bid]
            idx_r = np.where(df["cond"].values == regime)[0]
            n_b   = max(3, int(len(idx_r) * BASELINE_RATIO))
            bear_base[bid] = df[ALL_FEATS].values[idx_r[:n_b]]

        # ── 2. Contamination detection ───────────────────────────
        medians = {bid: np.median(v, axis=0) for bid, v in bear_base.items()}
        all_meds = np.array(list(medians.values()))     # (n_bear, n_feat)
        ref_med  = np.median(all_meds, axis=0)
        ref_std  = np.std(all_meds, axis=0) + 1e-8
        clean_bids = []
        for bid in train_bids:
            max_dev = float(np.max((medians[bid] - ref_med) / ref_std))
            if max_dev < CONTAM_THRESH:
                clean_bids.append(bid)
            else:
                lbl = "LOW" if regime == 0 else "HIGH"
                print(f"    [WARN][{lbl}] B{bid} baseline 오염 감지 "
                      f"(max_dev={max_dev:.1f}σ) → baseline 제외")
        if not clean_bids:
            clean_bids = train_bids   # fallback

        # ── 3. Pooled baseline mean / std (clean bearings only) ──
        pooled = np.concatenate([bear_base[bid] for bid in clean_bids], axis=0)
        base_mean = np.mean(pooled, axis=0)
        base_std  = np.std(pooled,  axis=0) + 1e-8   # 분모 0 방지

        # ── 4. Feature direction (majority vote over train bearings)
        dir_votes = {f: [] for f in ALL_FEATS}
        feat_q    = {f: [] for f in ALL_FEATS}
        for bid in train_bids:
            df    = dfs[bid]
            idx_r = np.where(df["cond"].values == regime)[0]
            if len(idx_r) < MIN_REGIME_WIN:
                continue
            vals_t = np.arange(len(idx_r))
            for fi, feat in enumerate(ALL_FEATS):
                rho, _ = spearmanr(vals_t, df[feat].values[idx_r])
                if not np.isnan(rho):
                    dir_votes[feat].append(1 if rho >= 0 else -1)
                    feat_q[feat].append(abs(rho))

        feat_dir = {f: (+1 if sum(dir_votes[f]) >= 0 else -1) for f in ALL_FEATS}
        feat_q   = {f: float(np.mean(feat_q[f])) if feat_q[f] else 0.3
                    for f in ALL_FEATS}

        # ── 5. EOL p95 for [0,1] normalization ───────────────────
        eol_zscores = []
        for bid in train_bids:
            df    = dfs[bid]
            idx_r = np.where(df["cond"].values == regime)[0]
            if len(idx_r) < MIN_REGIME_WIN:
                continue
            n_eol   = max(3, int(len(idx_r) * EOL_RATIO))
            eol_mat = df[ALL_FEATS].values[idx_r[-n_eol:]]
            z       = (eol_mat - base_mean) / base_std
            # direction-corrected, Q-weighted aggregate
            dir_vec = np.array([feat_dir[f] for f in ALL_FEATS])
            w_vec   = np.array([feat_q[f]   for f in ALL_FEATS])
            w_vec  /= w_vec.sum() + 1e-12
            z_agg   = (z * dir_vec * w_vec).sum(axis=1)
            eol_zscores.extend(z_agg.tolist())

        eol_p95 = float(np.percentile(eol_zscores, 95)) if eol_zscores else 1.0
        eol_p95 = max(eol_p95, 1e-3)

        result[regime] = {
            "base_mean":  base_mean,
            "base_std":   base_std,
            "feat_dir":   feat_dir,
            "feat_q":     feat_q,
            "eol_p95":    eol_p95,
            "clean_bids": clean_bids,
        }

    return result


# ══════════════════════════════════════════════════════════════════
# Apply HI
# ══════════════════════════════════════════════════════════════════
def moving_average(x, w=7):
    if w <= 1: return x.copy()
    pad   = w // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, np.ones(w)/w, mode="valid")[:len(x)]

def monotonicity(s):
    d = np.diff(s)
    return abs(np.sum(d>0) - np.sum(d<0)) / max(len(d), 1)

def trendability(s):
    rho, _ = spearmanr(np.arange(len(s)), s)
    return abs(rho) if not np.isnan(rho) else 0.0


def apply_zscore_hi(feat_mat, cond, zstats):
    """
    feat_mat: (N, n_feats) — raw feature values
    Returns: hi [0,1] smoothed, hi_raw (unclipped z-score aggregate)
    """
    hi_raw = np.zeros(len(feat_mat))
    hi     = np.zeros(len(feat_mat))

    for regime in [0, 1]:
        idx_r = np.where(cond == regime)[0]
        if len(idx_r) == 0:
            continue
        rs = zstats[regime]
        bm, bs = rs["base_mean"], rs["base_std"]
        dir_vec = np.array([rs["feat_dir"][f] for f in ALL_FEATS])
        w_vec   = np.array([rs["feat_q"][f]   for f in ALL_FEATS])
        w_vec  /= w_vec.sum() + 1e-12
        eol_p95 = rs["eol_p95"]

        z     = (feat_mat[idx_r] - bm) / bs           # z-scores
        z_agg = (z * dir_vec * w_vec).sum(axis=1)     # directed aggregate
        hi_raw[idx_r] = z_agg
        hi[idx_r]     = np.clip(z_agg / eol_p95, 0.0, 1.0)

    return np.clip(moving_average(hi, 7), 0.0, 1.0), moving_average(hi_raw, 7)


# ══════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════
def _plot_hi(ax, t, hi, cond, title, q):
    for lbl, col, name in [(0,"#4C72B0","Low"),(1,"#DD8452","High")]:
        idx = cond == lbl
        ax.scatter(t[idx], hi[idx], s=12, color=col, alpha=0.6, label=name, zorder=3)
    ax.plot(t, hi, color="gray", lw=0.8, alpha=0.4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{title}  Q={q:.3f}", fontsize=10)
    ax.set_xlabel("Time [hr]"); ax.set_ylabel("HI")
    ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.3)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  HI Z-score v1 (0604)")
    print("  FDR → z-score  |  B4 오염 감지  |  EOL p95 정규화")
    print("=" * 70)

    print("\n[Train 데이터 로드]")
    dfs = {}
    for bid in BEARINGS:
        print(f"  Bearing{bid}...")
        dfs[bid] = load_bearing(bid)

    # ── Train LOO HI ──────────────────────────────────────────────
    print("\n[Train LOO HI 계산]")
    train_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Train LOO HI — Z-score (0604)", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for i, bid in enumerate(BEARINGS):
        print(f"\n  Bearing{bid} (LOO)")
        zstats = compute_zscore_stats(dfs, exclude_bid=bid)

        # EOL p95 출력
        for regime in [0, 1]:
            lbl = "LOW" if regime == 0 else "HIGH"
            rs = zstats[regime]
            print(f"    [{lbl}] eol_p95={rs['eol_p95']:.3f}  "
                  f"clean_bids={rs['clean_bids']}")

        df   = dfs[bid]
        cond = df["cond"].values
        hi, hi_raw = apply_zscore_hi(df[ALL_FEATS].values, cond, zstats)

        pd.DataFrame({"HI": hi, "HI_raw": hi_raw, "regime": cond}).to_csv(
            os.path.join(OUT_TRAIN, f"Bearing{bid}_HI.csv"), index=False)

        mon  = monotonicity(hi)
        tred = trendability(hi)
        q    = (mon + tred) / 2
        print(f"    hi_start={hi[0]:.4f}  hi_end={hi[-1]:.4f}"
              f"  Mon={mon:.4f}  Tred={tred:.4f}  Q={q:.4f}")

        train_rows.append(dict(bearing=bid, hi_start=round(float(hi[0]),4),
                               hi_end=round(float(hi[-1]),4),
                               mon=round(mon,4), tred=round(tred,4),
                               q_score=round(q,4)))

        t = np.arange(len(hi)) * INTERVAL_SEC / 3600
        _plot_hi(axes[i], t, hi, cond, f"Bearing{bid}", q)

    df_tr = pd.DataFrame(train_rows)
    df_tr.to_csv(os.path.join(OUT_TRAIN, "summary.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_TRAIN, "Bearing_LOO_Regime_HI.png"), dpi=150)
    plt.close()

    print(f"\n  Train avg Q: {df_tr['q_score'].mean():.4f}")
    print(df_tr[["bearing","hi_start","hi_end","q_score"]].to_string(index=False))

    # ── Test HI ───────────────────────────────────────────────────
    print("\n[Test HI 계산]")
    all_zstats = compute_zscore_stats(dfs, exclude_bid=None)

    test_rows = []
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle("Test HI — Z-score (0604)", fontsize=13, fontweight="bold")
    axes2 = axes2.flatten()

    for i, tid in enumerate(TEST_IDS):
        print(f"\n  Test{tid}")
        feat_df  = load_test(tid)
        cond     = classify_regime_fft(tid)
        n        = min(len(feat_df), len(cond))
        hi, hi_raw = apply_zscore_hi(feat_df[ALL_FEATS].values[:n], cond[:n], all_zstats)

        pd.DataFrame({"HI": hi, "HI_raw": hi_raw, "regime": cond[:n]}).to_csv(
            os.path.join(OUT_TEST, f"Test{tid}_HI.csv"), index=False)

        mon  = monotonicity(hi)
        tred = trendability(hi)
        q    = (mon + tred) / 2
        print(f"    n={n}  start={hi[0]:.4f}  end={hi[-1]:.4f}  Q={q:.4f}")

        test_rows.append(dict(test_id=tid, hi_start=round(float(hi[0]),4),
                              hi_end=round(float(hi[-1]),4),
                              mon=round(mon,4), tred=round(tred,4),
                              q_score=round(q,4)))

        t = np.arange(len(hi)) * INTERVAL_SEC / 3600
        _plot_hi(axes2[i], t, hi, cond[:n], f"Test{tid}", q)

    df_te = pd.DataFrame(test_rows)
    df_te.to_csv(os.path.join(OUT_TEST, "summary.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_TEST, "Test_Regime_HI.png"), dpi=150)
    plt.close()

    print(f"\n  Test avg Q: {df_te['q_score'].mean():.4f}")
    print(df_te[["test_id","hi_start","hi_end","q_score"]].to_string(index=False))

    # ── 비교 요약 ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  v3(FDR) vs 0604(z-score) Train LOO Q-score 비교")
    print("=" * 70)
    v3_q = {1: 0.573, 2: 0.639, 3: 0.804, 4: 0.501}
    for _, row in df_tr.iterrows():
        b = int(row["bearing"])
        print(f"  B{b}: v3={v3_q[b]:.3f}  →  0604={row['q_score']:.3f}  "
              f"({'↑' if row['q_score'] > v3_q[b] else '↓'})")
    print(f"  Mean: v3=0.629  →  0604={df_tr['q_score'].mean():.3f}")


if __name__ == "__main__":
    main()
