"""
diagnose_outliers.py — raw feature 극단 노이즈가 HI 스케일 왜곡을 유발하는지 진단
=================================================================================
목표:
1) 각 베어링별 raw feature의 분포를 확인 (min, p1, p5, median, p95, p99, max)
2) IQR 기반으로 극단 outlier 개수 / 비율 파악
3) fleet-wide p5/p95 계산 시 outlier 포함/제외에 따른 변화 확인
4) FDR ratio → group score → p5/p95 스케일링에서 어떤 지점에서 왜곡이 발생하는지 추적
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────
BASE      = "/data/home/ksphm/2026-challenge-KSPHM"
TRAIN_OUT = os.path.join(BASE, "User/SR/0604/output/train")
DIAG_OUT  = os.path.join(BASE, "User/SR/0605/output_diag")
os.makedirs(DIAG_OUT, exist_ok=True)

BEARINGS = [1, 2, 3, 4]

FEATURE_GROUPS = {
    "highfreq":  ["ch3_high_band", "ch4_high_band"],
    "energy":    ["ch3_total_power", "ch3_energy", "ch3_rms"],
    "variation": ["ch3_std", "ch3_p2p"],
    "impulse":   ["ch1_kurt_log", "ch2_kurt_log", "ch1_crest", "ch2_crest"],
}
ALL_FEATS = [f for feats in FEATURE_GROUPS.values() for f in feats]


def load_raw(bid):
    return pd.read_csv(os.path.join(TRAIN_OUT, f"Bearing{bid}_features_raw.csv"))


def load_hi(bid):
    return pd.read_csv(os.path.join(TRAIN_OUT, f"Bearing{bid}_HI.csv"))


def iqr_bounds(series, k=3.0):
    """IQR 기반 outlier 경계"""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def main():
    print("=" * 70)
    print("  Raw Feature Outlier Diagnosis")
    print("=" * 70)

    # ── 1. 각 베어링별 raw feature 분포 ──
    all_dfs = {}
    for bid in BEARINGS:
        df = load_raw(bid)
        all_dfs[bid] = df

    print("\n[1] Per-bearing feature distribution (key percentiles)")
    print("-" * 70)
    for feat in ALL_FEATS:
        print(f"\n  Feature: {feat}")
        for bid in BEARINGS:
            s = all_dfs[bid][feat]
            pcts = s.describe(percentiles=[.01, .05, .25, .50, .75, .95, .99])
            print(f"    B{bid}: min={s.min():.6g}  p1={s.quantile(.01):.6g}  p5={s.quantile(.05):.6g}  "
                  f"med={s.median():.6g}  p95={s.quantile(.95):.6g}  p99={s.quantile(.99):.6g}  "
                  f"max={s.max():.6g}  (n={len(s)})")

    # ── 2. IQR 기반 outlier 탐지 ──
    print("\n\n[2] IQR(k=3) Outlier Detection")
    print("-" * 70)
    outlier_summary = []
    for feat in ALL_FEATS:
        print(f"\n  Feature: {feat}")
        for bid in BEARINGS:
            s = all_dfs[bid][feat]
            lo, hi = iqr_bounds(s, k=3.0)
            n_out = int(((s < lo) | (s > hi)).sum())
            pct = n_out / len(s) * 100
            # outlier 값 확인
            outliers = s[(s < lo) | (s > hi)]
            if n_out > 0:
                print(f"    B{bid}: {n_out} outliers ({pct:.1f}%)  range=[{lo:.6g}, {hi:.6g}]")
                print(f"           outlier values: {outliers.values[:10]}")  # 최대 10개만
                outlier_summary.append({
                    "feature": feat, "bearing": bid, "n_outlier": n_out,
                    "pct": round(pct, 2), "lo": lo, "hi": hi,
                    "outlier_min": outliers.min(), "outlier_max": outliers.max()
                })
            else:
                print(f"    B{bid}: No outliers (IQR range: [{lo:.6g}, {hi:.6g}])")

    # ── 3. Fleet-wide FDR ratio 분포 분석 ──
    print("\n\n[3] Fleet-wide FDR → Group Score → p5/p95 Analysis")
    print("-" * 70)

    # Baseline: 전체 베어링 초기 10% 평균
    baseline = {}
    for feat in ALL_FEATS:
        vals = []
        for bid in BEARINGS:
            df = all_dfs[bid]
            n_base = max(3, int(len(df) * 0.10))
            vals.extend(df[feat].values[:n_base].tolist())
        baseline[feat] = np.mean(vals)
    
    print("\n  Baseline (fleet-wide initial 10% mean):")
    for feat in ALL_FEATS:
        print(f"    {feat}: {baseline[feat]:.6g}")

    # 각 그룹별 score 분포 확인
    for gname, feats in FEATURE_GROUPS.items():
        print(f"\n  ── Group: {gname} ──")
        all_scores = []
        for bid in BEARINGS:
            df = all_dfs[bid]
            feat_mat = df[feats].values
            # FDR ratio
            bvec = np.array([baseline[f] for f in feats])
            bvec = np.where(np.abs(bvec) < 1e-8, 1e-8, bvec)
            ratios = (feat_mat - bvec) / (np.abs(bvec) + 1e-8)
            
            # Uniform weight for simplicity
            weights = np.ones(len(feats)) / len(feats)
            score = (ratios * weights).sum(axis=1)
            all_scores.extend(score.tolist())
            
            print(f"    B{bid}: score min={score.min():.4f}  p5={np.percentile(score,5):.4f}  "
                  f"med={np.median(score):.4f}  p95={np.percentile(score,95):.4f}  "
                  f"max={score.max():.4f}  range_span={score.max()-score.min():.4f}")
        
        all_scores = np.array(all_scores)
        fleet_p5 = np.percentile(all_scores, 5)
        fleet_p95 = np.percentile(all_scores, 95)
        fleet_max = all_scores.max()
        fleet_min = all_scores.min()
        
        print(f"    ── Fleet-wide: p5={fleet_p5:.4f}  p95={fleet_p95:.4f}  "
              f"min={fleet_min:.4f}  max={fleet_max:.4f}")
        print(f"    ── p95/p5 span: {fleet_p95-fleet_p5:.4f}")
        print(f"    ── max/p95 ratio: {fleet_max / (fleet_p95+1e-12):.2f}x")

    # ── 4. HI 스케일 확인 ──
    print("\n\n[4] HI Scale Summary")
    print("-" * 70)
    for bid in BEARINGS:
        hi = load_hi(bid)["HI"].values
        print(f"  B{bid}: start={hi[0]:.4f}  end={hi[-1]:.4f}  min={hi.min():.4f}  max={hi.max():.4f}  "
              f"range={hi.max()-hi.min():.4f}")

    # ── 5. 극단값 제거 전후 p5/p95 변화 시뮬레이션 ──
    print("\n\n[5] Simulated p5/p95 Change with Outlier Clipping")
    print("-" * 70)
    for gname, feats in FEATURE_GROUPS.items():
        print(f"\n  ── Group: {gname} ──")
        
        # Original scores
        all_scores_orig = []
        per_bearing_scores = {}
        for bid in BEARINGS:
            df = all_dfs[bid]
            feat_mat = df[feats].values
            bvec = np.array([baseline[f] for f in feats])
            bvec = np.where(np.abs(bvec) < 1e-8, 1e-8, bvec)
            ratios = (feat_mat - bvec) / (np.abs(bvec) + 1e-8)
            weights = np.ones(len(feats)) / len(feats)
            score = (ratios * weights).sum(axis=1)
            all_scores_orig.extend(score.tolist())
            per_bearing_scores[bid] = score
        
        all_scores_orig = np.array(all_scores_orig)
        p5_orig = np.percentile(all_scores_orig, 5)
        p95_orig = np.percentile(all_scores_orig, 95)
        
        # Winsorized scores (clip at p1/p99 per feature before FDR)
        all_scores_clip = []
        for bid in BEARINGS:
            df = all_dfs[bid]
            feat_mat = df[feats].values.copy()
            for fi, f in enumerate(feats):
                # Fleet-wide p1/p99 per feature
                all_vals = np.concatenate([all_dfs[b][f].values for b in BEARINGS])
                lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
                feat_mat[:, fi] = np.clip(feat_mat[:, fi], lo, hi)
            bvec = np.array([baseline[f] for f in feats])
            bvec = np.where(np.abs(bvec) < 1e-8, 1e-8, bvec)
            ratios = (feat_mat - bvec) / (np.abs(bvec) + 1e-8)
            weights = np.ones(len(feats)) / len(feats)
            score = (ratios * weights).sum(axis=1)
            all_scores_clip.extend(score.tolist())
        
        all_scores_clip = np.array(all_scores_clip)
        p5_clip = np.percentile(all_scores_clip, 5)
        p95_clip = np.percentile(all_scores_clip, 95)
        
        print(f"    Original: p5={p5_orig:.4f}  p95={p95_orig:.4f}  span={p95_orig-p5_orig:.4f}")
        print(f"    Clipped:  p5={p5_clip:.4f}  p95={p95_clip:.4f}  span={p95_clip-p5_clip:.4f}")
        print(f"    Change:   p5={p5_clip-p5_orig:+.4f}  p95={p95_clip-p95_orig:+.4f}  "
              f"span_change={((p95_clip-p5_clip)-(p95_orig-p5_orig)):+.4f}")

    # ── 6. Raw feature time-series plot with outliers highlighted ──
    print("\n\n[6] Generating outlier highlight plots...")
    # 핵심 feature만 시각화 (ch3_high_band, ch3_rms, ch1_kurt_log, ch2_kurt_log)
    key_feats = ["ch3_high_band", "ch3_rms", "ch1_kurt_log", "ch2_kurt_log", "ch1_crest", "ch2_crest"]
    fig, axes = plt.subplots(len(key_feats), 4, figsize=(20, 3 * len(key_feats)))
    
    for fi, feat in enumerate(key_feats):
        for bi, bid in enumerate(BEARINGS):
            ax = axes[fi][bi]
            s = all_dfs[bid][feat].values
            t = np.arange(len(s))
            lo, hi_bound = iqr_bounds(pd.Series(s), k=3.0)
            mask_out = (s < lo) | (s > hi_bound)
            
            ax.plot(t, s, 'b-', lw=0.5, alpha=0.5)
            ax.scatter(t[~mask_out], s[~mask_out], s=3, c='blue', alpha=0.3)
            ax.scatter(t[mask_out], s[mask_out], s=15, c='red', marker='x', zorder=5, label=f'{mask_out.sum()} outliers')
            
            if mask_out.any():
                ax.axhline(lo, color='orange', ls='--', lw=0.8, alpha=0.5)
                ax.axhline(hi_bound, color='orange', ls='--', lw=0.8, alpha=0.5)
            
            ax.set_title(f"B{bid} {feat}", fontsize=8)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIAG_OUT, "outlier_highlight.png"), dpi=150)
    plt.close()
    print(f"  Saved: {DIAG_OUT}/outlier_highlight.png")

    # ── 7. 각 베어링의 EOL 구간 raw feature 급등 분석 ──
    print("\n\n[7] EOL region (last 10%) feature spike analysis")
    print("-" * 70)
    for bid in BEARINGS:
        df = all_dfs[bid]
        n = len(df)
        eol_start = int(n * 0.9)
        print(f"\n  B{bid} (n={n}, EOL starts at idx={eol_start}):")
        for feat in ALL_FEATS:
            normal = df[feat].values[:eol_start]
            eol = df[feat].values[eol_start:]
            ratio = eol.max() / (normal.mean() + 1e-12)
            if ratio > 3.0:  # 정상기 대비 3배 이상 급등
                print(f"    {feat}: normal_mean={normal.mean():.6g}  eol_max={eol.max():.6g}  "
                      f"ratio={ratio:.1f}x  ← SPIKE")

    print("\n\n[DONE]")


if __name__ == "__main__":
    main()
