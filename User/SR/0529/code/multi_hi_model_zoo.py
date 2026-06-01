"""
TH v8_3 Model Zoo — Multi-HI Fusion Version in SR/0529
======================================================
Fuses HI inputs from three teams:
  - TH v8 HI (v8_train_anchored_baseline)
  - SP V10c HI (no personal offset, absolute HI)
  - SC HI (post-align RPM-scale)

Model Architecture:
  - Multi-feature tabular LightGBM
  - LSTM sequence model (10 input channels: 3 HIs * 3 channels + obs_frac)
  - GRU sequence model (10 input channels)
  - TCN sequence model (10 input channels)
  - DTW/kNN curve matching using 3D segment distance
  - LOOCV score-weighted ensemble
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb

BASE = Path("/data/home/ksphm/2026-challenge-KSPHM")
TH_HI_DIR = BASE / "User" / "TH" / "FI" / "08_v8" / "output" / "v8_train_anchored_baseline"
SP_HI_DIR = BASE / "User" / "SP" / "compare" / "V10c" / "output"
SC_HI_DIR = BASE / "User" / "SC" / "HI" / "05170000_postalign_rpmscale_hi" / "output"
OUT_DIR = BASE / "User" / "SR" / "0529" / "output" / "multi_hi_zoo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS = [1, 2, 3, 4]
TEST_IDS = [1, 2, 3, 4, 5, 6]

SEQ_LENGTH = 25
MATCH_LEN = 18
INTERVAL_SEC = 600
MEAN_TRAIN_LIFE = 116.5
SEEDS = [42]

MIN_RUL_CYCLES = 1.0
EPS = 1e-8


def load_train_hi():
    out = {}
    for b in BEARINGS:
        th = pd.read_csv(TH_HI_DIR / f"v8_Bearing{b}_HI.csv")["HI_v8"].values.astype(float)
        sp = pd.read_csv(SP_HI_DIR / f"HI_Bearing{b}.csv")["HI"].values.astype(float)
        sc = pd.read_csv(SC_HI_DIR / f"Bearing{b}_HI.csv")["HI_final"].values.astype(float)
        
        min_len = min(len(th), len(sp), len(sc))
        out[b] = np.stack([th[:min_len], sp[:min_len], sc[:min_len]], axis=1)
    return out


def load_test_hi():
    out = {}
    for t in TEST_IDS:
        th = pd.read_csv(TH_HI_DIR / f"v8_Test{t}_HI.csv")["HI_v8"].values.astype(float)
        sp = pd.read_csv(SP_HI_DIR / "test" / f"HI_Test{t}.csv")["HI"].values.astype(float)
        sc = pd.read_csv(SC_HI_DIR / f"Test{t}_HI.csv")["HI_final"].values.astype(float)
        
        min_len = min(len(th), len(sp), len(sc))
        out[t] = np.stack([th[:min_len], sp[:min_len], sc[:min_len]], axis=1)
    return out


def minmax_norm(x):
    x = np.asarray(x, dtype=float)
    mn, mx = np.min(x), np.max(x)
    if mx - mn < EPS:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def slope_of(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(x)), x, 1)[0])


def competition_score(rul_true, rul_pred):
    if rul_true <= 0:
        return np.nan
    er = 100.0 * (rul_true - rul_pred) / rul_true
    if er <= 0:
        return np.exp(-np.log(0.5) * er / 20.0)
    return np.exp(np.log(0.5) * er / 50.0)


def true_rul(n, obs_pts):
    obs_pts = np.asarray(obs_pts, dtype=float)
    return np.maximum(n - obs_pts, 1.0)


def score_curve(n, obs_pts, preds):
    y = true_rul(n, obs_pts)
    return float(np.nanmean([competition_score(t, p) for t, p in zip(y, preds)]))


def error_summary(n, obs_pts, preds):
    y = true_rul(n, obs_pts)
    p = np.asarray(preds, dtype=float)
    err = p - y
    er = np.asarray([100.0 * (t - pp) / t for t, pp in zip(y, p) if t > 0], dtype=float)
    return {
        "score": score_curve(n, obs_pts, p),
        "mean_er": float(np.nanmean(er)),
        "mae_cycles": float(np.mean(np.abs(err))),
        "bias_cycles": float(np.mean(err)),
        "over_pred_rate": float(np.mean(err > 0)),
        "last_true": float(y[-1]),
        "last_pred": float(p[-1]),
    }


def make_tabular(hi_matrix, start_obs=0):
    # hi_matrix: (N, 3) [TH, SP, SC]
    n = len(hi_matrix)
    x, obs_pts = [], []
    for i in range(SEQ_LENGTH, n):
        row_feats = []
        obs_frac = float(np.clip((start_obs + i) / MEAN_TRAIN_LIFE, 0.0, 2.0))
        for col_idx in range(3):
            w = hi_matrix[i - SEQ_LENGTH:i, col_idx]
            wn = minmax_norm(w)
            w0 = float(hi_matrix[0, col_idx])
            feats = (
                list(wn)
                + [
                    slope_of(wn),
                    float(w[-1]),
                    float(w.mean()),
                    float(w.max()),
                    float(w.min()),
                    float(w.std()),
                    slope_of(w),
                    float(w[-1] - w[0]),
                    float(w[-1] - w0),
                    float(w[-1] * obs_frac)
                ]
            )
            row_feats.extend(feats)
        row_feats.append(obs_frac)
        x.append(row_feats)
        obs_pts.append(i)
    return np.asarray(x), np.asarray(obs_pts)


def train_lgbm(hi_train, train_bids):
    xs, ys = [], []
    for b in train_bids:
        x, obs = make_tabular(hi_train[b], start_obs=0)
        xs.append(x)
        ys.append(true_rul(len(hi_train[b]), obs))
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    def asym_obj(y_pred, dataset):
        y_true = dataset.get_label()
        diff = y_true - y_pred
        weight = np.where(diff < 0, 2.8, 1.0)
        return -diff * weight, np.ones_like(diff) * weight

    params = {
        "num_leaves": 15,
        "learning_rate": 0.04,
        "min_child_samples": 5,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "verbose": -1,
        "objective": asym_obj,
    }
    return lgb.train(params, lgb.Dataset(x_train, label=y_train), num_boost_round=260)


def predict_lgbm(model, hi_matrix, start_obs=0):
    x, obs = make_tabular(hi_matrix, start_obs=start_obs)
    return obs, np.maximum(model.predict(x), MIN_RUL_CYCLES)


def make_seq(hi_matrix, start_obs=0):
    # hi_matrix: (N, 3) [TH, SP, SC]
    n = len(hi_matrix)
    x, obs_pts = [], []
    for i in range(n - SEQ_LENGTH):
        channels = []
        for col_idx in range(3):
            w = hi_matrix[i:i + SEQ_LENGTH, col_idx]
            wn = minmax_norm(w)
            raw = w.copy()
            w0 = float(hi_matrix[0, col_idx])
            delta = w - w0
            channels.extend([wn, raw, delta])
        
        obs_frac = np.clip((start_obs + i + np.arange(SEQ_LENGTH)) / MEAN_TRAIN_LIFE, 0.0, 2.0)
        channels.append(obs_frac)
        
        x.append(np.stack(channels, axis=1))
        obs_pts.append(i + SEQ_LENGTH)
    return np.asarray(x), np.asarray(obs_pts)


class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(10, 48, num_layers=2, batch_first=True, dropout=0.15)
        self.fc = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.fc(y[:, -1, :]).squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(10, 48, num_layers=2, batch_first=True, dropout=0.15)
        self.fc = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.fc(y[:, -1, :]).squeeze(-1)


class TCNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(10, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Conv1d(32, 32, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Conv1d(32, 32, kernel_size=3, dilation=4, padding=4),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(32, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x):
        z = x.transpose(1, 2)
        z = self.net(z)
        return self.fc(z[:, :, -1]).squeeze(-1)


def build_model(kind):
    if kind == "lstm":
        return LSTMRegressor()
    if kind == "gru":
        return GRURegressor()
    if kind == "tcn":
        return TCNRegressor()
    raise ValueError(kind)


def train_torch_model(x_train, y_train, scale, kind, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(kind).to(device)
    ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train / scale, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=min(32, len(ds)), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    best_state, best_loss, patience = None, np.inf, 0
    for _ in range(160):
        model.train()
        losses = []
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        cur = float(np.mean(losses))
        if cur < best_loss - 1e-5:
            best_loss = cur
            patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 18:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_torch_ensemble(hi_train, train_bids, kind, device):
    xs, ys = [], []
    for b in train_bids:
        x, obs = make_seq(hi_train[b], start_obs=0)
        xs.append(x)
        ys.append(true_rul(len(hi_train[b]), obs))
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    scale = float(max(y_train.max(), 1.0))
    models = [train_torch_model(x_train, y_train, scale, kind, seed, device) for seed in SEEDS]
    return models, scale


def predict_torch(models, scale, hi_matrix, start_obs, device):
    x, obs = make_seq(hi_matrix, start_obs=start_obs)
    xt = torch.tensor(x, dtype=torch.float32).to(device)
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            preds.append(np.maximum(model(xt).cpu().numpy() * scale, MIN_RUL_CYCLES))
    return obs, np.median(preds, axis=0)


def seg_dist(a, b):
    # a, b: 2D arrays of shape (L, 3)
    dists = []
    for col in range(3):
        ac = a[:, col]
        bc = b[:, col]
        d = float(
            0.25 * abs(ac[-1] - bc[-1]) / 0.25
            + 0.20 * abs(ac.mean() - bc.mean()) / 0.25
            + 0.20 * abs((ac[-1] - ac[0]) - (bc[-1] - bc[0])) / 0.25
            + 0.15 * abs(slope_of(ac) - slope_of(bc)) / 0.03
            + 0.20 * np.mean(np.abs(minmax_norm(ac) - minmax_norm(bc)))
        )
        dists.append(d)
    return float(np.mean(dists))


def estimate_start_obs(hi_train, hi_target, train_bids):
    target = np.asarray(hi_target, dtype=float)
    l = min(MATCH_LEN, len(target))
    target_seg = target[:l]
    candidates = []
    for b in train_bids:
        hi = np.asarray(hi_train[b], dtype=float)
        for s in range(0, max(1, len(hi) - l - 3)):
            d = seg_dist(target_seg, hi[s:s + l])
            candidates.append((d, s, b))
    if not candidates:
        return 0
    top = sorted(candidates, key=lambda x: x[0])[:8]
    pos = np.asarray([s for d, s, b in top], dtype=float)
    wt = np.asarray([1.0 / (d + 1e-6) for d, s, b in top], dtype=float)
    est = int(round(np.average(pos, weights=wt)))
    
    # Gain heuristic across the 3 HIs
    gains = [float(target[-1, col] - target[0, col]) for col in range(3)]
    avg_gain = float(np.mean(gains))
    est = max(est, int(round(np.clip((avg_gain - 0.25) / 0.35, 0, 1) * 25)))
    return int(np.clip(est, 0, int(MEAN_TRAIN_LIFE)))


def predict_dtw_knn(hi_train, train_bids, hi_target):
    hi_target = np.asarray(hi_target, dtype=float)
    obs_pts = np.arange(SEQ_LENGTH, len(hi_target))
    preds = []
    for obs in obs_pts:
        l = min(MATCH_LEN, obs)
        seg = hi_target[obs - l:obs]
        candidates = []
        for b in train_bids:
            hi = np.asarray(hi_train[b], dtype=float)
            for end in range(l, len(hi)):
                d = seg_dist(seg, hi[end - l:end])
                pred = max(len(hi) - end, MIN_RUL_CYCLES)
                candidates.append((d, pred))
        top = sorted(candidates, key=lambda x: x[0])[:6]
        wt = np.asarray([1.0 / (d + 1e-6) for d, p in top], dtype=float)
        pv = np.asarray([p for d, p in top], dtype=float)
        preds.append(float(np.average(pv, weights=wt)))
    return obs_pts, np.asarray(preds)


def calibrate_model(pred_by_bearing, results):
    best_cf, best_score = 1.0, -np.inf
    for cf in np.arange(0.60, 1.61, 0.02):
        scores = []
        for b, pred in pred_by_bearing.items():
            r = results[b]
            scores.append(score_curve(r["N"], r["obs_pts"], np.asarray(pred) * cf))
        sc = float(np.mean(scores))
        if sc > best_score:
            best_cf, best_score = float(cf), sc
    return best_cf, best_score


def ensemble_predictions(preds_by_model, weights):
    keys = list(weights)
    total = sum(weights.values()) + EPS
    out = None
    for k in keys:
        p = np.asarray(preds_by_model[k], dtype=float)
        out = p * (weights[k] / total) if out is None else out + p * (weights[k] / total)
    return np.maximum(out, MIN_RUL_CYCLES)


def score_weights(model_scores):
    vals = {k: max(v, 1e-4) ** 4 for k, v in model_scores.items()}
    total = sum(vals.values())
    return {k: v / total for k, v in vals.items()}


def run_loocv(hi_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    model_names = ["lgbm", "lstm", "gru", "tcn", "dtw"]
    results = {}

    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        print("=" * 72)
        print(f"[LOOCV] Bearing {test_bid} held out")
        target_hi = hi_train[test_bid]
        n_test = len(target_hi)

        fold = {"N": n_test}

        lgbm_model = train_lgbm(hi_train, train_bids)
        obs, pred = predict_lgbm(lgbm_model, target_hi)
        fold["obs_pts"] = obs
        fold["lgbm"] = pred

        for kind in ["lstm", "gru", "tcn"]:
            print(f"  {kind.upper()} training ({len(SEEDS)} seeds)...")
            models, scale = train_torch_ensemble(hi_train, train_bids, kind, device)
            obs2, pred2 = predict_torch(models, scale, target_hi, start_obs=0, device=device)
            assert np.array_equal(obs, obs2)
            fold[kind] = pred2

        _, pred_dtw = predict_dtw_knn(hi_train, train_bids, target_hi)
        fold["dtw"] = pred_dtw

        raw_scores = {m: score_curve(n_test, obs, fold[m]) for m in model_names}
        weights = score_weights(raw_scores)
        fold["zoo_raw"] = ensemble_predictions({m: fold[m] for m in model_names}, weights)
        fold["raw_scores"] = raw_scores
        fold["weights"] = weights
        results[test_bid] = fold

        msg = "  " + "  ".join([f"{m}={raw_scores[m]:.4f}" for m in model_names])
        print(msg)
        print("  zoo_raw={:.4f} weights={}".format(
            score_curve(n_test, obs, fold["zoo_raw"]),
            {k: round(v, 3) for k, v in weights.items()},
        ))

    print("\nCalibration search")
    cf_rows = []
    for m in model_names + ["zoo_raw"]:
        cf, sc = calibrate_model({b: results[b][m] for b in BEARINGS}, results)
        cf_rows.append({"model": m, "cf": cf, "score": sc})
    cf_df = pd.DataFrame(cf_rows).sort_values("score", ascending=False)
    print(cf_df.to_string(index=False))

    cal_scores = {r["model"]: r["score"] for _, r in cf_df.iterrows() if r["model"] in model_names}
    cal_cfs = {r["model"]: r["cf"] for _, r in cf_df.iterrows() if r["model"] in model_names}
    final_weights = score_weights(cal_scores)
    for b in BEARINGS:
        preds = {m: np.asarray(results[b][m]) * cal_cfs[m] for m in model_names}
        results[b]["zoo"] = ensemble_predictions(preds, final_weights)

    zoo_cf, zoo_sc = calibrate_model({b: results[b]["zoo"] for b in BEARINGS}, results)

    best_row = cf_df.iloc[0]
    best_model = str(best_row["model"])
    best_cf = float(best_row["cf"])
    best_score = float(best_row["score"])
    if zoo_sc > best_score:
        final_kind = "zoo"
        final_cf = zoo_cf
        final_score = zoo_sc
        for b in BEARINGS:
            results[b]["final"] = np.asarray(results[b]["zoo"]) * zoo_cf
    else:
        final_kind = best_model
        final_cf = best_cf
        final_score = best_score
        for b in BEARINGS:
            results[b]["final"] = np.asarray(results[b][best_model]) * best_cf

    print(f"\nZoo candidate: cf={zoo_cf:.2f}, score={zoo_sc:.4f}, weights={{{', '.join(f'{k}: {v:.3f}' for k, v in final_weights.items())}}}")
    print(f"Final selection: {final_kind}, cf={final_cf:.2f}, score={final_score:.4f}")

    rows = []
    for b in BEARINGS:
        for m in model_names + ["zoo_raw", "final"]:
            s = error_summary(results[b]["N"], results[b]["obs_pts"], results[b][m])
            rows.append({"bearing": b, "model": m, **s})
    err_df = pd.DataFrame(rows)
    err_df.to_csv(OUT_DIR / "loocv_error_summary.csv", index=False)
    cf_df.to_csv(OUT_DIR / "calibration_summary.csv", index=False)
    pd.DataFrame([{"model": k, "weight": v, "cf": cal_cfs[k]} for k, v in final_weights.items()]).to_csv(
        OUT_DIR / "ensemble_weights.csv", index=False
    )

    with open(OUT_DIR / "loocv_log.txt", "w", encoding="utf-8") as f:
        f.write("SR Multi-HI Model Zoo LOOCV\n")
        f.write(cf_df.to_string(index=False) + "\n\n")
        f.write(f"Zoo candidate cf={zoo_cf:.2f}, score={zoo_sc:.4f}\n")
        f.write(f"Final selection: {final_kind}, cf={final_cf:.2f}, score={final_score:.4f}\n")
        f.write("Weights:\n")
        for k, v in final_weights.items():
            f.write(f"  {k}: weight={v:.4f}, cf={cal_cfs[k]:.2f}, calibrated_score={cal_scores[k]:.4f}\n")
        f.write("\nFinal errors:\n")
        f.write(err_df[err_df["model"] == "final"].to_string(index=False) + "\n")

    plot_loocv(results, model_names)
    return results, cal_cfs, final_weights, zoo_cf, final_kind, final_cf


def plot_loocv(results, model_names):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SR Multi-HI Model Zoo LOOCV", fontsize=12)
    for ax, b in zip(axes.flatten(), BEARINGS):
        r = results[b]
        obs = r["obs_pts"]
        ax.plot(obs, true_rul(r["N"], obs), "k-", lw=2, label="true")
        for m in model_names:
            ax.plot(obs, r[m], lw=0.9, alpha=0.55, label=m)
        ax.plot(obs, r["final"], "b-", lw=2, label="final")
        ax.set_title(f"B{b} final score={score_curve(r['N'], obs, r['final']):.3f}")
        ax.set_xlabel("obs cycle")
        ax.set_ylabel("RUL cycles")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loocv_predictions.png", dpi=150)
    plt.close()


def run_test(hi_train, hi_test, cal_cfs, final_weights, zoo_cf, final_kind, final_cf):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nTraining final models on all train bearings...")
    lgbm_model = train_lgbm(hi_train, BEARINGS)
    torch_models = {}
    for kind in ["lstm", "gru", "tcn"]:
        print(f"  {kind.upper()} final training...")
        torch_models[kind] = train_torch_ensemble(hi_train, BEARINGS, kind, device)

    summary = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("SR Multi-HI Model Zoo Test RUL", fontsize=12)
    for ax, tid in zip(axes.flatten(), TEST_IDS):
        hi = hi_test[tid]
        start_obs = estimate_start_obs(hi_train, hi, BEARINGS)

        obs, p_lgbm = predict_lgbm(lgbm_model, hi, start_obs=start_obs)
        preds = {"lgbm": p_lgbm * cal_cfs["lgbm"]}
        for kind in ["lstm", "gru", "tcn"]:
            models, scale = torch_models[kind]
            obs2, pp = predict_torch(models, scale, hi, start_obs=start_obs, device=device)
            assert np.array_equal(obs, obs2)
            preds[kind] = pp * cal_cfs[kind]
        _, p_dtw = predict_dtw_knn(hi_train, BEARINGS, hi)
        preds["dtw"] = p_dtw * cal_cfs["dtw"]

        if final_kind == "zoo":
            final = ensemble_predictions(preds, final_weights) * zoo_cf
        else:
            final = preds[final_kind] / max(cal_cfs[final_kind], EPS) * final_cf
        final = np.maximum(final, MIN_RUL_CYCLES)
        hours = final * INTERVAL_SEC / 3600.0

        out = pd.DataFrame({"obs_cycle": obs})
        for k, v in preds.items():
            out[f"rul_{k}"] = v
        out["rul_final"] = final
        out["rul_hours"] = hours
        out.to_csv(OUT_DIR / f"Test{tid}_RUL.csv", index=False)

        summary.append({
            "test_id": tid,
            "hi_start": round(float(hi[0, 0]), 3),  # Use TH v8 for representation in summary
            "hi_end": round(float(hi[-1, 0]), 3),
            "start_obs": start_obs,
            "obs_frac0": round(start_obs / MEAN_TRAIN_LIFE, 3),
            "rul_hours": round(float(hours[-1]), 2),
            "rul_cycles": round(float(final[-1]), 2),
        })

        ax.plot(obs, final, "b-", lw=2, label="final")
        for k, v in preds.items():
            ax.plot(obs, v, lw=0.9, alpha=0.45, label=k)
        ax.set_title(f"T{tid} start={start_obs} RUL={hours[-1]:.1f}h")
        ax.set_xlabel("obs cycle")
        ax.set_ylabel("RUL cycles")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_predictions.png", dpi=150)
    plt.close()

    df = pd.DataFrame(summary)
    df.to_csv(OUT_DIR / "test_summary.csv", index=False)
    print("\nTest summary")
    print(df.to_string(index=False))


def make_rul_results():
    df = pd.read_csv(OUT_DIR / "loocv_error_summary.csv")
    final = df[df["model"] == "final"].copy().sort_values("bearing")
    out = final[["bearing", "score", "mean_er"]].rename(columns={"bearing": "test_bearing"})
    out.insert(0, "dataset", "Train")
    out["score"] = out["score"].round(4)
    out["mean_er"] = out["mean_er"].round(4)
    out.to_csv(OUT_DIR / "rul_results.csv", index=False)
    print("\nRUL registration summary")
    print(out.to_string(index=False))
    print("avg_score =", round(float(out["score"].mean()), 4))


if __name__ == "__main__":
    print("Loading Train/Test Multi-HI fusion matrices...")
    hi_train = load_train_hi()
    hi_test = load_test_hi()
    for b in BEARINGS:
        hi = hi_train[b]
        print(f"  B{b}: shape={hi.shape}, final_TH={hi[-1, 0]:.3f}, final_SP={hi[-1, 1]:.3f}, final_SC={hi[-1, 2]:.3f}")
    results, cal_cfs, final_weights, zoo_cf, final_kind, final_cf = run_loocv(hi_train)
    run_test(hi_train, hi_test, cal_cfs, final_weights, zoo_cf, final_kind, final_cf)
    make_rul_results()
    print("\nSaved to:", OUT_DIR)
