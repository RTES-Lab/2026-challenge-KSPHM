import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/data/home/ksphm/2026-challenge-KSPHM")
TRAIN_FEAT_DIR = BASE / "User/TH/common_source"
TEST_FEAT_DIR  = BASE / "User/TH/FI/06_v6/output/validation_features"
OUT_DIR = BASE / "User/SR/0526/rul/output/th742_ensemble3_svdd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
BEARINGS        = [1, 2, 3, 4]
TEST_IDS        = [1, 2, 3, 4, 5, 6]
SEQ_LENGTH      = 10
MEAN_TRAIN_LIFE = 116.5
INTERVAL_SEC    = 600

NORMAL_UNTIL = {1: 89, 2: 92, 3: 62, 4: 78}
EOL          = {1: 126, 2: 114, 3: 89, 4: 137}

MAIN_ALL_FEATS = ["ch3_total_power", "ch3_energy", "ch3_rms", "ch3_std", "ch3_p2p"]

# ── DEGRADATION PROGNOSIS MODELS ─────────────────────────────────────────────
def _exp_fn(x, a, b, c):
    return a * np.exp(b * x) + c

def predict_wiener_degraded(hi_prefix, hi_threshold=1.0):
    n = len(hi_prefix)
    t = np.arange(n, dtype=float)
    if n < 2 or hi_prefix.std() < 1e-6:
        return float(50.0)
    r, b = np.polyfit(t, hi_prefix, 1)
    if r <= 1e-6:
        return float(50.0)
    t_fail = (hi_threshold - b) / r
    t_fail = min(t_fail, 180.0)
    return float(max(0.0, t_fail - (n - 1)))

def predict_exp_fit_degraded(hi_prefix, hi_threshold=1.0):
    n = len(hi_prefix)
    if n < 5:
        return predict_wiener_degraded(hi_prefix, hi_threshold)
    t = np.arange(n, dtype=float)
    try:
        popt, _ = curve_fit(_exp_fn, t, hi_prefix,
                            p0=[0.01, 0.05, hi_prefix[0]],
                            maxfev=2000)
        a, b, c = popt
        if a <= 1e-8 or b <= 1e-8 or (hi_threshold - c) / a <= 0:
            return predict_wiener_degraded(hi_prefix, hi_threshold)
        t_fail = np.log((hi_threshold - c) / a) / b
        t_fail = min(t_fail, 180.0)
        return float(max(0.0, t_fail - (n - 1)))
    except Exception:
        return predict_wiener_degraded(hi_prefix, hi_threshold)

def predict_rul_physical(hi_full, obs, t_d, mean_life=MEAN_TRAIN_LIFE):
    if obs < t_d:
        return float(max(0.0, mean_life - obs))
    else:
        hi_degraded = hi_full[t_d:obs]
        return predict_exp_fit_degraded(hi_degraded, hi_threshold=1.0)

# ── SVDD HI Smoothing Helper Functions ─────────────────────────────────────
def ema_smooth(x, alpha=0.2):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if len(x) == 0:
        return y
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def moving_average(x, window=7):
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    return pd.Series(x).rolling(window=window, center=True, min_periods=1).mean().values

def compute_smoothed_svdd_scores(df, ocsvm, scaler):
    X = df[MAIN_ALL_FEATS].values
    X_scaled = scaler.transform(X)
    scores = -ocsvm.decision_function(X_scaled)
    s_scores = ema_smooth(scores)
    s_scores = moving_average(s_scores)
    return scores, s_scores

# ── Global SVDD Training ──────────────────────────────────────────────────
def train_global_svdd(train_dfs, train_bids=BEARINGS):
    healthy_features = []
    for b in train_bids:
        df = train_dfs[b]
        healthy_features.append(df[MAIN_ALL_FEATS].values[:50])
    X_healthy = np.vstack(healthy_features)
    
    scaler = StandardScaler()
    X_healthy_scaled = scaler.fit_transform(X_healthy)
    
    ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
    ocsvm.fit(X_healthy_scaled)
    return ocsvm, scaler

# ── Feature Loading ───────────────────────────────────────────────────────
def load_train_features():
    return {b: pd.read_csv(TRAIN_FEAT_DIR / f"Bearing{b}_features.csv") for b in BEARINGS}

def load_test_features():
    return {t: pd.read_csv(TEST_FEAT_DIR / f"Test{t}_features.csv") for t in TEST_IDS}

# ── Competition Scoring ────────────────────────────────────────────────────
def competition_score(rul_true, rul_pred):
    if rul_true <= 0:
        return np.nan
    Er = 100.0 * (rul_true - rul_pred) / rul_true
    return (np.exp(-np.log(0.5) * Er / 20.0) if Er <= 0
            else np.exp(np.log(0.5) * Er / 50.0))

# ── LOOCV (Physical Pure SVDD version) ─────────────────────────────────────
def run_loocv(train_dfs):
    results = {}
    
    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        print(f"{'='*60}")
        print(f"[LOOCV] Bearing {test_bid} held out (LOO-SVDD HI recalculation)")

        # 1. LOO-SVDD 학습
        ocsvm_loo, scaler_loo = train_global_svdd(train_dfs, train_bids)

        # 2. 4개 베어링에 대해 SVDD 스코어 및 스무딩 계산
        raw_scores_loo = {}
        smoothed_scores_loo = {}
        for b in BEARINGS:
            raw, smooth = compute_smoothed_svdd_scores(train_dfs[b], ocsvm_loo, scaler_loo)
            raw_scores_loo[b] = raw
            smoothed_scores_loo[b] = smooth

        # 3. LOO train 베어링들의 bounds로 글로벌 정규화
        all_train_loo_smoothed = np.concatenate([smoothed_scores_loo[b] for b in train_bids])
        global_min_loo = all_train_loo_smoothed.min()
        global_max_loo = all_train_loo_smoothed.max()

        hi_train_loo = {}
        for b in BEARINGS:
            hi = (smoothed_scores_loo[b] - global_min_loo) / (global_max_loo - global_min_loo + 1e-8)
            hi_train_loo[b] = np.clip(hi, 0.0, 1.0)

        # 4. Onset detection (t_d) using raw LOO scores
        raw_val = raw_scores_loo[test_bid]
        t_d_loo = None
        for i_c in range(len(raw_val) - 2):
            if raw_val[i_c] > 0.2 and raw_val[i_c+1] > 0.2 and raw_val[i_c+2] > 0.2:
                t_d_loo = i_c
                break
        if t_d_loo is None:
            t_d_loo = len(raw_val)

        hi_val = hi_train_loo[test_bid]
        N_test = len(hi_val)
        obs_pts = np.arange(SEQ_LENGTH, N_test)

        # Generate RUL predictions
        preds = []
        for obs in obs_pts:
            p = predict_rul_physical(hi_val, obs, t_d_loo, mean_life=MEAN_TRAIN_LIFE)
            preds.append(p)

        results[test_bid] = {
            "preds": np.array(preds),
            "obs_pts": list(obs_pts), 
            "N": N_test,
            "t_d": t_d_loo,
            "svdd_hi": list(hi_val)
        }

    # Search for the best calibration factor cf
    best_cf, best_score = 1.0, -np.inf
    for cf in np.arange(0.20, 2.51, 0.02):
        scores = []
        for b in BEARINGS:
            data = results[b]
            N = data["N"]
            obs_pts = data["obs_pts"]
            sc = np.nanmean([competition_score(N - obs, p * cf) for obs, p in zip(obs_pts, data["preds"])])
            scores.append(sc)
        mean_sc = np.mean(scores)
        if mean_sc > best_score:
            best_score = mean_sc
            best_cf = cf

    print(f"\n[LOOCV Results (best_cf = {best_cf:.2f})]")
    print(f"  LOOCV Mean Score: {best_score:.4f}")
    for b in BEARINGS:
        data = results[b]
        sc = np.nanmean([competition_score(data["N"] - obs, p * best_cf) for obs, p in zip(data["obs_pts"], data["preds"])])
        print(f"    Bearing {b}: score = {sc:.4f}")

    # Plot LOOCV predictions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, b in enumerate(BEARINGS):
        ax = axes[idx]
        data = results[b]
        obs_pts = data["obs_pts"]
        N = data["N"]
        preds_calibrated = data["preds"] * best_cf
        
        true_rul = [N - obs for obs in obs_pts]
        ax.plot(obs_pts, true_rul, "k-", lw=2, label="True RUL")
        ax.plot(obs_pts, preds_calibrated, "b-", lw=2, label=f"SVDD Exp Fit (score={np.nanmean([competition_score(N-obs, p) for obs, p in zip(obs_pts, preds_calibrated)]):.4f})")
        
        ax.set_title(f"Bearing {b} RUL Extrapolation (t_d={data['t_d']})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Observed Cycle")
        ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.4)
        
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loocv_predictions.png", dpi=150)
    plt.close()

    return results, best_cf, best_score

# ── Test inference with Pure SVDD-HI Pipeline ──────────────────────────────
def run_test_inference(best_cf, test_dfs, ocsvm_global, scaler_global, global_min, global_max):
    print(f"\n{'='*60}")
    print(f"  Test inference — Pure Physical SVDD-HI + Exponential Curve Fitting, cf={best_cf:.2f}\n")

    REF = {
        "0514": {1: 5.05, 2: 5.08, 3: 4.69, 4: 3.43, 5: 7.46, 6: 5.40},
        "v4":   {1: 7.40, 2: 8.58, 3: 5.21, 4: 9.78, 5: 8.31, 6: 6.79},
        "afull": {1: 7.08, 2: 0.44, 3: 2.38, 4: 5.31, 5: 1.90, 6: 0.56},
        "ens3": {1: 6.80, 2: 0.27, 3: 2.23, 4: 5.56, 5: 1.86, 6: 0.53},
    }

    summary_rows = []
    fig, axes = plt.subplots(6, 2, figsize=(15, 24))
    fig.suptitle(f"SR/0526 Test Inference — SVDD Diagnostics & Physical Extrapolation RUL (cf={best_cf:.2f})", fontsize=14, fontweight="bold")

    for i, tid in enumerate(TEST_IDS):
        feat_df = test_dfs[tid]

        # 1. SVDD HI 계산
        raw_scores_t, smoothed_scores_t = compute_smoothed_svdd_scores(feat_df, ocsvm_global, scaler_global)
        hi_t = (smoothed_scores_t - global_min) / (global_max - global_min + 1e-8)
        hi_t = np.clip(hi_t, 0.0, 1.0)

        # 2. Onset Detection using raw scores
        t_d = None
        for i_c in range(len(raw_scores_t) - 2):
            if raw_scores_t[i_c] > 0.2 and raw_scores_t[i_c+1] > 0.2 and raw_scores_t[i_c+2] > 0.2:
                t_d = i_c
                break

        obs_pts = np.arange(SEQ_LENGTH, len(hi_t) + 1)
        
        # Generate RUL predictions over time
        preds_cyc = []
        for o in obs_pts:
            p = predict_rul_physical(hi_t, o, t_d if t_d is not None else len(hi_t), mean_life=MEAN_TRAIN_LIFE)
            preds_cyc.append(p)
            
        preds_cyc = np.array(preds_cyc)
        preds_final = preds_cyc * best_cf
        final_cyc = preds_final[-1]
        final_hr  = final_cyc * INTERVAL_SEC / 3600.0
        
        print(f"  [Test{tid}] SVDD HI: [{hi_t[0]:.4f} → {hi_t[-1]:.4f}]")
        print(f"    SVDD Onset t_d: {t_d if t_d is not None else 'None'}")
        print(f"    → 최종 RUL = {final_hr:.2f}hr  (참고용: Ens3={REF['ens3'][tid]:.2f}, afull={REF['afull'][tid]:.2f})\n")

        # CSV 출력 저장
        pd.DataFrame({
            "obs_cycle": obs_pts,
            "svdd_hi":   hi_t[SEQ_LENGTH - 1:],
            "rul_cycles": preds_final,
            "rul_hours": preds_final * INTERVAL_SEC / 3600.0,
        }).to_csv(OUT_DIR / f"Test{tid}_RUL.csv", index=False)

        summary_rows.append({
            "test_id":             tid,
            "hi_svdd_start":       round(float(hi_t[0]),  4),
            "hi_svdd_end":         round(float(hi_t[-1]), 4),
            "t_d":                 t_d if t_d is not None else -1,
            "rul_hours":           round(final_hr, 2),
            "afull_hours":         REF["afull"][tid],
            "ens3_hours":          REF["ens3"][tid],
            "base0514":            REF["0514"][tid],
        })

        # Left plot: SVDD HI & Onset Shading
        ax_svdd = axes[i, 0]
        ax_svdd.plot(np.arange(len(hi_t)), hi_t, "b-", lw=2, label="SVDD HI")
        ax_svdd.axhline(y=0.2, color="r", linestyle="--", lw=1.5, label="Threshold (0.2)")
        if t_d is not None and t_d < len(hi_t):
            ax_svdd.axvline(x=t_d, color="g", linestyle=":", lw=2, label=f"Detected Onset (t_d={t_d})")
            ax_svdd.axvspan(0, t_d, color="lightgreen", alpha=0.15, label="Healthy Phase")
            ax_svdd.axvspan(t_d, len(hi_t), color="salmon", alpha=0.15, label="Degraded Phase")
        else:
            ax_svdd.axvspan(0, len(hi_t), color="lightgreen", alpha=0.15, label="Healthy Phase")
            
        ax_svdd.set_title(f"Test {tid} SVDD Health Indicator", fontsize=10, fontweight="bold")
        ax_svdd.set_xlabel("Observed Cycle"); ax_svdd.set_ylabel("Health / Anomaly Score")
        ax_svdd.set_ylim(-0.05, 1.05)
        ax_svdd.legend(fontsize=8, loc="upper left")
        ax_svdd.grid(True, alpha=0.3)

        # Right plot: Predicted RUL
        ax_rul = axes[i, 1]
        ax_rul.plot(obs_pts, preds_final, "m-",  lw=2.5,             label="Physical SVDD Exp Fit")
        
        ax_rul.set_title(f"Test {tid} RUL Prediction (Final RUL = {final_hr:.2f} hr)", fontsize=10, fontweight="bold")
        ax_rul.set_xlabel("Observed Cycle"); ax_rul.set_ylabel("RUL (cycles)")
        ax_rul.legend(fontsize=8, loc="upper right")
        ax_rul.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUT_DIR / "test_predictions.png", dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "test_summary.csv", index=False)

    print(f"\n  {'Test':>5} | {'0514':>6} | {'afull':>6} | {'ens3':>6} | {'svdd_hours':>10} | t_d | hi_start | hi_end")
    print(f"  {'-'*105}")
    for row in summary_rows:
        print(f"  {row['test_id']:>5} | {row['base0514']:>6.2f} | "
              f"{row['afull_hours']:>6.2f} | {row['ens3_hours']:>6.2f} | "
              f"{row['rul_hours']:>10.2f} | "
              f"{row['t_d']:>9} | {row['hi_svdd_start']:>8.4f} | "
              f"{row['hi_svdd_end']:.4f}")

    with open(OUT_DIR / "loocv_log.txt", "a", encoding="utf-8") as f:
        f.write("\nTest inference (Pure SVDD-HI & Physical Prognostics):\n")
        f.write(df_sum.to_string(index=False) + "\n")

    print(f"\n[완료] 결과 저장 위치: {OUT_DIR}")

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("SR/0526 — Pure Physical SVDD-HI + Exponential Extrapolation RUL Pipeline")
    print("=" * 60)

    # 1. Features 로드
    print("\nLoading features...")
    train_dfs = load_train_features()
    test_dfs  = load_test_features()

    for b in BEARINGS:
        print(f"  Bearing{b}: feat={len(train_dfs[b])}")
    for t in TEST_IDS:
        print(f"  Test{t}:    feat={len(test_dfs[t])}")

    # 2. Train 전체 기준의 global SVDD 피팅 및 HI 사전 계산
    print("\nTraining global SVDD on training healthy data...")
    ocsvm_global, scaler_global = train_global_svdd(train_dfs)

    # 3. 4개 Train 베어링의 SVDD HI 계산 및 global normalization bounds 설정
    raw_scores_all = {}
    smoothed_scores_all = {}
    for b in BEARINGS:
        raw, smooth = compute_smoothed_svdd_scores(train_dfs[b], ocsvm_global, scaler_global)
        raw_scores_all[b] = raw
        smoothed_scores_all[b] = smooth

    all_smoothed_all = np.concatenate(list(smoothed_scores_all.values()))
    global_min = all_smoothed_all.min()
    global_max = all_smoothed_all.max()

    hi_train_all = {}
    for b in BEARINGS:
        hi = (smoothed_scores_all[b] - global_min) / (global_max - global_min + 1e-8)
        hi_train_all[b] = np.clip(hi, 0.0, 1.0)
        print(f"  Bearing{b}: n={len(hi)}, range=[{hi.min():.3f}, {hi.max():.3f}]")

    # 4. LOOCV 실행
    print("\nStarting LOOCV...")
    results, best_cf, sc_ens_cf = run_loocv(train_dfs)

    # 5. Test Inference 실행
    print("\nStarting Test Inference...")
    run_test_inference(best_cf, test_dfs, ocsvm_global, scaler_global, global_min, global_max)
