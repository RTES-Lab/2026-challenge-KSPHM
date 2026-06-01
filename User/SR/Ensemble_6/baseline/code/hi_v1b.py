"""
05-26/V1b — HI pipeline (V1 문제 수정)
========================================
V1 대비 변경:
  1. apply_personal_offset 제거 (train/test 모두)
  2. smooth()  center=True → center=False
  3. calibrate() center=True → center=False

Config: r=0.3, N=7, method=by_mean
Features: top-7 by Q_mean (SP only)
"""
import os, warnings, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nptdms, numpy as np, pandas as pd
from scipy import stats as sp_stats
from scipy.signal import butter, sosfiltfilt, hilbert
warnings.filterwarnings("ignore")

BASE       = "/data/home/ksphm/2026-challenge-KSPHM"
SP         = os.path.join(BASE, "User", "SP")
TRAIN_FEAT = os.path.join(SP, "compare", "output", "01_features")
TH_TEST    = os.path.join(BASE, "User", "TH", "FI", "06_v6", "output", "validation_features")
TEST_DIR   = os.path.join(BASE, "dataset", "Test")
KMEANS_PKL = os.path.join(SP, "outputs", "05_kmeans", "ch4", "kmeans_ch4_model.pkl")
QSCORE_CSV = os.path.join(SP, "05-25", "output", "qscore_all.csv")
OUT_DIR    = os.path.join(SP, "05-26", "V1b", "output")
os.makedirs(OUT_DIR, exist_ok=True)

FAULT_POINTS = {1: 65, 2: 85, 3: 65, 4: 75}
Q_BIDS       = [1, 2, 3, 4]
NORMAL_RATIO = 0.15
EMA_ALPHA    = 0.2
MA_WINDOW    = 7
Z_CAP        = 10.0
FS           = 25_600
LPF_CUT      = 50.0
PEAK_LOW, PEAK_HIGH = 10.0, 20.0
BASELINE_R   = 0.3
N_FEATS      = 7

EXCLUDE_FEATS = {
    "ch3_env_kurtosis", "ch3_rvf", "ch3_spectral_flatness", "ch3_gini_index",
    "ch4_env_kurtosis", "ch4_rvf", "ch4_spectral_flatness", "ch4_gini_index",
}

km_bundle = joblib.load(KMEANS_PKL)
km_scaler, km_model, km_lmap = km_bundle["scaler"], km_bundle["km"], km_bundle["label_map"]
lpf_sos = butter(8, LPF_CUT / (FS / 2), btype="low", output="sos")


def extract_sp_extra(x):
    x = x.astype(np.float64); N = len(x)
    mad     = float(sp_stats.median_abs_deviation(x))
    env_rms = float(np.sqrt(np.mean(np.abs(hilbert(x)) ** 2)))
    fft_amp = np.abs(np.fft.rfft(x)); freqs = np.fft.rfftfreq(N, d=1.0 / FS)
    power   = fft_amp ** 2; total = power.sum()
    rmsf    = float(np.sqrt((freqs**2 * power).sum() / total)) if total > 0 else 0.0
    return {"mad": mad, "env_rms": env_rms, "rmsf": rmsf}

def extract_peak_ch4(path):
    ch4 = nptdms.TdmsFile.read(path)["Vibration"]["CH4"][:].astype(np.float64)
    flt = sosfiltfilt(lpf_sos, ch4)
    amp = np.abs(np.fft.rfft(flt)); frq = np.fft.rfftfreq(len(flt), d=1.0 / FS)
    msk = (frq >= PEAK_LOW) & (frq <= PEAK_HIGH)
    return float(frq[msk][np.argmax(amp[msk])]) if msk.any() else 0.0

def load_regime_train(bid):
    vib_dir = os.path.join(BASE, "dataset", f"Train{bid}_Vibration")
    records = []
    for fname in sorted(f for f in os.listdir(vib_dir) if f.endswith(".tdms")):
        fidx = int(os.path.splitext(fname)[0])
        try:    label = km_lmap[int(km_model.predict(km_scaler.transform([[extract_peak_ch4(os.path.join(vib_dir, fname))]]))[0])]
        except: label = None
        records.append({"file_idx": fidx, "rpm_label": label})
    df = pd.DataFrame(records)
    df["rpm_label"] = df["rpm_label"].ffill().bfill()
    return df.set_index("file_idx")["rpm_label"].to_dict()

def load_regime_test(test_id):
    df = pd.read_csv(os.path.join(SP, "outputs", "05_kmeans", "ch4", "test", f"test{test_id}_file_labels.csv"))
    return df.set_index("file_idx")["rpm_label"].to_dict()

def add_sp_features(df_th, tdms_dir):
    extra_rows = []
    idx_set = set(df_th["file_idx"].astype(int).tolist())
    for fname in sorted(f for f in os.listdir(tdms_dir) if f.endswith(".tdms")):
        fidx = int(os.path.splitext(fname)[0])
        if fidx not in idx_set: continue
        try:
            tdms = nptdms.TdmsFile(os.path.join(tdms_dir, fname))
            row = {"file_idx": fidx}
            for ch in ["CH3", "CH4"]:
                for k, v in extract_sp_extra(tdms["Vibration"][ch][:]).items():
                    row[f"{ch.lower()}_{k}"] = v
            extra_rows.append(row)
        except: pass
    return df_th.merge(pd.DataFrame(extra_rows), on="file_idx", how="inner") \
                .sort_values("file_idx").reset_index(drop=True)

def load_features_train(bid, feat_names):
    df = pd.read_csv(os.path.join(TRAIN_FEAT, f"Bearing{bid}_features_new.csv"))
    return df[["file_idx"] + feat_names].sort_values("file_idx").reset_index(drop=True)

def load_features_test(test_id, feat_names):
    df_th = pd.read_csv(os.path.join(TH_TEST, f"Test{test_id}_features.csv"))
    df = add_sp_features(df_th, os.path.join(TEST_DIR, f"Test{test_id}"))
    return df[["file_idx"] + feat_names].sort_values("file_idx").reset_index(drop=True)


def select_features():
    qdf = pd.read_csv(QSCORE_CSV)
    qdf = qdf[qdf["source"] == "SP"]
    qdf = qdf[~qdf["feature"].isin(EXCLUDE_FEATS)]
    return qdf.sort_values("Q_mean", ascending=False)["feature"].iloc[:N_FEATS].tolist()

def compute_directions(train_dfs, feat_names):
    directions = {}
    for feat in feat_names:
        votes = []
        for bid in Q_BIDS:
            df = train_dfs[bid]; fp = FAULT_POINTS[bid]
            early = df[df["file_idx"] <= fp * 0.3][feat].dropna()
            late  = df[(df["file_idx"] > fp * 0.7) & (df["file_idx"] <= fp)][feat].dropna()
            if len(early) == 0 or len(late) == 0: continue
            votes.append(1 if late.mean() > early.mean() else -1)
        d = int(np.sign(sum(votes))) if votes else 1
        directions[feat] = d if d != 0 else 1
    return directions

def monotonicity(x):
    d = np.diff(x)
    return abs(np.sum(d > 0) - np.sum(d < 0)) / len(d) if len(d) > 0 else 0.0

def trendability(x):
    t = np.arange(len(x))
    return abs(np.corrcoef(t, x)[0, 1]) if np.std(x) > 1e-12 else 0.0

def compute_weights(dfs, feat_names):
    def qs(s):
        x = s.values.astype(float); mn, mx = x.min(), x.max()
        if mx - mn < 1e-12: return 0.0
        xn = (x - mn) / (mx - mn)
        return (monotonicity(xn) + trendability(xn)) / 2
    scores = {f: float(np.mean([qs(dfs[b][f]) for b in Q_BIDS])) for f in feat_names}
    total  = sum(scores.values()) or 1.0
    return scores, {f: v / total for f, v in scores.items()}

def compute_baseline_sigma(dfs, regime_maps, feat_names):
    baseline, sigma = {}, {}
    for regime in ("low", "high"):
        for feat in feat_names:
            vals = []
            for bid in Q_BIDS:
                fp   = FAULT_POINTS[bid]
                rmap = regime_maps[bid]
                df   = dfs[bid]
                cutoff = max(int(fp * BASELINE_R), 1)
                mask = (df["file_idx"] <= cutoff) & \
                       (df["file_idx"].map(rmap).fillna("low") == regime)
                vals.extend(df.loc[mask, feat].dropna().tolist())
            baseline[(regime, feat)] = float(np.mean(vals)) if vals else 0.0
            s = float(np.std(vals)) if len(vals) > 1 else 1.0
            sigma[(regime, feat)] = max(s, 1e-12)
    return baseline, sigma

def compute_raw(df, regime_map, feat_names, directions, baseline, sigma, weights):
    rows = []
    for _, row in df.iterrows():
        fidx   = int(row["file_idx"])
        regime = regime_map.get(fidx, "low")
        z_sum  = 0.0
        for feat in feat_names:
            d   = directions[feat]
            bl  = baseline.get((regime, feat), 0.0)
            sig = sigma.get((regime, feat), 1.0)
            z   = d * (row[feat] - bl) / sig
            z   = float(np.clip(z, -Z_CAP, Z_CAP))
            z   = float(np.sign(z) * np.log1p(abs(z)))
            z_sum += max(z, 0.0) * weights[feat]
        rows.append({"file_idx": fidx, "raw_score": z_sum})
    return pd.DataFrame(rows).sort_values("file_idx").reset_index(drop=True)

def smooth(series):
    ema = series.ewm(alpha=EMA_ALPHA, adjust=False).mean()
    return ema.rolling(MA_WINDOW, center=False, min_periods=1).mean()

def compute_calibration_params(raw_dfs):
    normal_raws, all_raws = [], []
    for bid in Q_BIDS:
        df = raw_dfs[bid]
        n0 = max(3, int(len(df) * NORMAL_RATIO))
        normal_raws.extend(df["raw_score"].iloc[:n0].tolist())
        all_raws.extend(df["raw_score"].tolist())
    offset = float(np.percentile(normal_raws, 25))
    tau    = float(np.percentile(all_raws, 70)) - offset
    return offset, max(tau, 1e-6)

def calibrate(raw_series, offset, tau):
    r  = (raw_series - offset).clip(lower=0)
    hi = 1.0 - np.exp(-r / tau)
    hi = pd.Series(hi).ewm(alpha=EMA_ALPHA, adjust=False).mean()
    hi = hi.rolling(MA_WINDOW, center=False, min_periods=1).mean()
    return hi.clip(lower=0.0, upper=1.0)

def hi_metric(hi):
    x = np.array(hi, dtype=float)
    return (monotonicity(x) + trendability(x)) / 2.0


def main():
    print("=== 05-26/V1b: V1 문제 수정 (no p_off, no center=True) ===\n")

    feat_names = select_features()
    print(f"Features ({len(feat_names)}): {feat_names}\n")

    print("[1] Train features & regime maps...")
    train_dfs   = {bid: load_features_train(bid, feat_names) for bid in Q_BIDS}
    regime_maps = {bid: load_regime_train(bid) for bid in Q_BIDS}

    print("[2] Directions & weights...")
    directions = compute_directions(
        {bid: pd.read_csv(os.path.join(TRAIN_FEAT, f"Bearing{bid}_features_new.csv"))
         for bid in Q_BIDS},
        feat_names)
    scores, weights = compute_weights(train_dfs, feat_names)
    for f in feat_names:
        print(f"  {f}: dir={directions[f]:+d}  Q={scores[f]:.3f}  w={weights[f]:.3f}")

    print("\n[3] Baseline/sigma (r=0.3)...")
    baseline, sigma = compute_baseline_sigma(train_dfs, regime_maps, feat_names)

    print("\n[4] Train HI (no p_offset, no center=True)...")
    raw_dfs = {}
    for bid in Q_BIDS:
        raw_df = compute_raw(train_dfs[bid], regime_maps[bid], feat_names,
                             directions, baseline, sigma, weights)
        raw_df["raw_score"] = smooth(raw_df["raw_score"])
        raw_dfs[bid] = raw_df

    offset, tau = compute_calibration_params(raw_dfs)
    print(f"  calibration: offset={offset:.4f}  tau={tau:.4f}")

    summary_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for i, bid in enumerate(Q_BIDS):
        hi = calibrate(raw_dfs[bid]["raw_score"], offset, tau)
        raw_dfs[bid]["HI"] = hi.values
        q = hi_metric(raw_dfs[bid]["HI"])
        summary_rows.append({"bearing": bid, "q_score": round(q, 4),
                              "HI_max": round(float(hi.max()), 4), "dataset": "Train"})
        print(f"  B{bid}: Q={q:.3f}  HI_start={hi.iloc[0]:.3f}  HI_max={hi.max():.3f}")
        raw_dfs[bid][["file_idx", "raw_score", "HI"]].to_csv(
            os.path.join(OUT_DIR, f"HI_Bearing{bid}.csv"), index=False)
        axes.flatten()[i].plot(raw_dfs[bid]["file_idx"], hi.values, lw=1.5)
        axes.flatten()[i].axvline(FAULT_POINTS[bid], color="red", ls="--", lw=1)
        axes.flatten()[i].set_title(f"B{bid}  Q={q:.3f}  start={hi.iloc[0]:.3f}")
        axes.flatten()[i].set_ylim(-0.05, 1.05)
        axes.flatten()[i].grid(alpha=0.3)
    fig.suptitle("05-26/V1b Train HI  (no p_off, no center=True)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "HI_train.png"), dpi=150)
    plt.close()

    print("\n[5] Test HI...")
    test_out = os.path.join(OUT_DIR, "test")
    os.makedirs(test_out, exist_ok=True)
    test_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for i, tid in enumerate(range(1, 7)):
        print(f"  Test{tid}...")
        df = load_features_test(tid, feat_names)
        rm = load_regime_test(tid)
        raw_df = compute_raw(df, rm, feat_names, directions, baseline, sigma, weights)
        raw_df["raw_score"] = smooth(raw_df["raw_score"])
        hi = calibrate(raw_df["raw_score"], offset, tau)
        raw_df["HI"] = hi.values
        q = hi_metric(raw_df["HI"])
        test_rows.append({"bearing": tid, "q_score": round(q, 4),
                           "HI_max": round(float(hi.max()), 4), "dataset": "Test"})
        print(f"    T{tid}: Q={q:.3f}  HI_start={hi.iloc[0]:.3f}  HI_max={hi.max():.3f}")
        raw_df[["file_idx", "raw_score", "HI"]].to_csv(
            os.path.join(test_out, f"HI_Test{tid}.csv"), index=False)
        axes.flatten()[i].plot(raw_df["file_idx"], hi.values, lw=1.4)
        axes.flatten()[i].set_title(f"T{tid}  Q={q:.3f}  start={hi.iloc[0]:.3f}")
        axes.flatten()[i].set_ylim(-0.05, 1.05)
        axes.flatten()[i].grid(alpha=0.3)
    fig.suptitle("05-26/V1b Test HI  (no p_off, no center=True)")
    plt.tight_layout()
    plt.savefig(os.path.join(test_out, "HI_test.png"), dpi=150)
    plt.close()

    pd.DataFrame(test_rows).to_csv(os.path.join(test_out, "hi_summary_test.csv"), index=False)
    pd.DataFrame(summary_rows + test_rows).to_csv(
        os.path.join(OUT_DIR, "hi_summary.csv"), index=False)
    print(f"\n완료 → {OUT_DIR}")


if __name__ == "__main__":
    main()
