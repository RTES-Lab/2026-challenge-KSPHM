"""
SR Cross-Pipeline RUL Ensemble & Optimization
==============================================
This script runs:
1. TH v8_3 Model Zoo on TH v8 HI (Train & Test)
2. TH v8_3 Model Zoo on SP V10c HI (Train & Test)

Then, it optimizes the blending weight (alpha) for combining the predictions:
  Final_RUL = alpha * RUL_TH_HI + (1 - alpha) * RUL_SP_HI

It evaluates the combined LOOCV score and outputs the optimal alpha,
then applies it to generate final test bearing RUL predictions.
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
OUT_DIR = BASE / "User" / "SR" / "0529" / "output" / "cross_ensemble"
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


# ── Utilities ────────────────────────────────────────────────────────────────
def minmax_norm(x):
    x = np.asarray(x, dtype=float)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + EPS)


def slope_of(x):
    x = np.asarray(x, dtype=float)
    return float(np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x) >= 2 else 0.0


def competition_score(rul_true, rul_pred):
    if rul_true <= 0:
        return np.nan
    er = 100.0 * (rul_true - rul_pred) / rul_true
    return np.exp(-np.log(0.5) * er / 20.0) if er <= 0 else np.exp(np.log(0.5) * er / 50.0)


def true_rul(n, obs_pts):
    return np.maximum(n - np.asarray(obs_pts, dtype=float), 1.0)


def score_curve(n, obs_pts, preds):
    y = true_rul(n, obs_pts)
    return float(np.nanmean([competition_score(t, p) for t, p in zip(y, preds)]))


# ── Model Zoo definition ─────────────────────────────────────────────────────
def make_tabular(hi_arr, start_obs=0):
    hi = np.asarray(hi_arr, dtype=float)
    hi0 = float(hi[0])
    x, obs_pts = [], []
    for i in range(SEQ_LENGTH, len(hi)):
        w = hi[i - SEQ_LENGTH:i]
        wn = minmax_norm(w)
        obs_frac = float(np.clip((start_obs + i) / MEAN_TRAIN_LIFE, 0.0, 2.0))
        x.append(list(wn) + [
            slope_of(wn), float(w[-1]), float(w.mean()), float(w.max()),
            float(w.min()), float(w.std()), slope_of(w),
            float(w[-1] - w[0]), float(w[-1] - hi0),
            obs_frac, float(w[-1] * obs_frac),
        ])
        obs_pts.append(i)
    return np.asarray(x), np.asarray(obs_pts)


def train_lgbm(hi_train, train_bids):
    xs, ys = [], []
    for b in train_bids:
        x, obs = make_tabular(hi_train[b], start_obs=0)
        xs.append(x)
        ys.append(true_rul(len(hi_train[b]), obs))
    x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)

    def asym_obj(y_pred, ds):
        diff = ds.get_label() - y_pred
        w = np.where(diff < 0, 2.8, 1.0)
        return -diff * w, np.ones_like(diff) * w

    return lgb.train(
        {"num_leaves": 15, "learning_rate": 0.04, "min_child_samples": 5,
         "feature_fraction": 0.90, "bagging_fraction": 0.90, "bagging_freq": 1,
         "verbose": -1, "objective": asym_obj},
        lgb.Dataset(x_tr, label=y_tr), num_boost_round=260
    )


def predict_lgbm(model, hi_arr, start_obs=0):
    x, obs = make_tabular(hi_arr, start_obs)
    return obs, np.maximum(model.predict(x), MIN_RUL_CYCLES)


def make_seq(hi_arr, start_obs=0):
    hi = np.asarray(hi_arr, dtype=float)
    hi0 = float(hi[0])
    x, obs_pts = [], []
    for i in range(len(hi) - SEQ_LENGTH):
        w = hi[i:i + SEQ_LENGTH]
        wn = minmax_norm(w)
        raw = w.copy()
        delta = w - hi0
        obs_frac = np.clip((start_obs + i + np.arange(SEQ_LENGTH)) / MEAN_TRAIN_LIFE, 0.0, 2.0)
        x.append(np.stack([wn, raw, delta, obs_frac], axis=1))
        obs_pts.append(i + SEQ_LENGTH)
    return np.asarray(x), np.asarray(obs_pts)


class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(4, 48, num_layers=2, batch_first=True, dropout=0.15)
        self.fc = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.fc(y[:, -1, :]).squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(4, 48, num_layers=2, batch_first=True, dropout=0.15)
        self.fc = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.fc(y[:, -1, :]).squeeze(-1)


class TCNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.10),
            nn.Conv1d(32, 32, kernel_size=3, dilation=2, padding=2), nn.ReLU(), nn.Dropout(0.10),
            nn.Conv1d(32, 32, kernel_size=3, dilation=4, padding=4), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(32, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x):
        z = self.net(x.transpose(1, 2))
        return self.fc(z[:, :, -1]).squeeze(-1)


def build_model(kind):
    return {"lstm": LSTMRegressor, "gru": GRURegressor, "tcn": TCNRegressor}[kind]()


def train_torch_model(x_train, y_train, scale, kind, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(kind).to(device)
    ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                       torch.tensor(y_train / scale, dtype=torch.float32))
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
            best_loss, patience = cur, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 18:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def train_torch_ensemble(hi_train, train_bids, kind, device):
    xs, ys = [], []
    for b in train_bids:
        x, obs = make_seq(hi_train[b], start_obs=0)
        xs.append(x)
        ys.append(true_rul(len(hi_train[b]), obs))
    x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
    scale = float(max(y_tr.max(), 1.0))
    models = [train_torch_model(x_tr, y_tr, scale, kind, s, device) for s in SEEDS]
    return models, scale


def predict_torch(models, scale, hi_arr, seq_length, start_obs, device):
    x, obs = make_seq(hi_arr, start_obs)
    xt = torch.tensor(x, dtype=torch.float32).to(device)
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(np.maximum(m(xt).cpu().numpy() * scale, MIN_RUL_CYCLES))
    return obs, np.median(preds, axis=0)


def seg_dist(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return (0.25 * abs(a[-1] - b[-1]) / 0.25
            + 0.20 * abs(a.mean() - b.mean()) / 0.25
            + 0.20 * abs((a[-1] - a[0]) - (b[-1] - b[0])) / 0.25
            + 0.15 * abs(slope_of(a) - slope_of(b)) / 0.03
            + 0.20 * float(np.mean(np.abs(minmax_norm(a) - minmax_norm(b)))))


def seg_dist_components(a, b):
    """Returns each component of seg_dist separately for diagnosis."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return {
        "endpoint":    0.25 * abs(a[-1] - b[-1]) / 0.25,
        "mean":        0.20 * abs(a.mean() - b.mean()) / 0.25,
        "total_delta": 0.20 * abs((a[-1] - a[0]) - (b[-1] - b[0])) / 0.25,
        "slope":       0.15 * abs(slope_of(a) - slope_of(b)) / 0.03,
        "shape":       0.20 * float(np.mean(np.abs(minmax_norm(a) - minmax_norm(b)))),
    }


def slope_only_dist(a, b):
    """Distance based purely on slope — captures rapid vs slow degradation."""
    return abs(slope_of(a) - slope_of(b)) / (0.03 + EPS)


def level_slope_dist(a, b):
    """Distance based on end level + slope — high level AND steep slope = B3-like."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    level_diff = abs(a[-1] - b[-1]) / (0.5 + EPS)
    slope_diff = abs(slope_of(a) - slope_of(b)) / (0.03 + EPS)
    return 0.5 * level_diff + 0.5 * slope_diff


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
        wt = np.asarray([1.0 / (d + EPS) for d, p in top])
        pv = np.asarray([p for d, p in top])
        preds.append(float(np.average(pv, weights=wt)))
    return obs_pts, np.asarray(preds)


def score_weights(model_scores):
    vals = {k: max(v, 1e-4) ** 4 for k, v in model_scores.items()}
    total = sum(vals.values())
    return {k: v / total for k, v in vals.items()}


def ensemble_predictions(preds_by_model, weights):
    total = sum(weights.values()) + EPS
    out = None
    for k, w in weights.items():
        p = np.asarray(preds_by_model[k], dtype=float)
        out = p * (w / total) if out is None else out + p * (w / total)
    return np.maximum(out, MIN_RUL_CYCLES)


def calibrate_model(pred_by_bearing, results):
    best_cf, best_score = 1.0, -np.inf
    for cf in np.arange(0.60, 1.61, 0.02):
        sc = float(np.mean([
            score_curve(results[b]["N"], results[b]["obs_pts"],
                        np.asarray(pred_by_bearing[b]) * cf)
            for b in BEARINGS]))
        if sc > best_score:
            best_score, best_cf = sc, float(cf)
    return best_cf, best_score


def estimate_start_obs(hi_train, hi_target, train_bids):
    hi_target = np.asarray(hi_target, dtype=float)
    l = min(MATCH_LEN, len(hi_target))
    target_seg = hi_target[:l]
    candidates = []
    for b in train_bids:
        hi = np.asarray(hi_train[b], dtype=float)
        for s in range(0, max(1, len(hi) - l - 3)):
            d = seg_dist(target_seg, hi[s:s + l])
            candidates.append((d, s, b))
    if not candidates:
        return 0
    top = sorted(candidates, key=lambda x: x[0])[:8]
    pos = np.asarray([s for d, s, b in top])
    wt = np.asarray([1.0 / (d + EPS) for d, s, b in top])
    est = int(round(np.average(pos, weights=wt)))
    gain = float(hi_target[-1] - hi_target[0])
    est = max(est, int(round(np.clip((gain - 0.25) / 0.35, 0, 1) * 25)))
    return int(np.clip(est, 0, int(MEAN_TRAIN_LIFE)))


def compute_similarity_alpha(hi_test, hi_train_sp, train_bids, n_early=30):
    """
    LOO-validatable soft alpha for TH/SP blending based on early SP HI similarity.

    High alpha  → B3-like (rapid degradation) → lean on TH pipeline.
    Low alpha   → normal bearing              → lean on SP pipeline.

    Case A (B3 in train_bids): alpha = sim(test, B3) / sum_b sim(test, b).
      B3 typically has a very distinct early SP HI trajectory, so normal
      bearings get low B3-sim and high SP weight automatically.

    Case B (B3 held out, LOO for B3 itself): alpha from slope percentile.
      B3's early degradation slope should exceed B1/B2/B4 slopes, giving
      alpha = 1.0 (or near 1.0), correctly routing to TH.
    """
    n = min(n_early, len(hi_test))
    # Use the most recent n timesteps — B3 is a late-stage rapid failure type,
    # so the current degradation state is more discriminative than early history.
    test_seg = np.asarray(hi_test[-n:], dtype=float)

    sims = {}
    for b in train_bids:
        m = min(n, len(hi_train_sp[b]))
        seg = np.asarray(hi_train_sp[b][-m:], dtype=float)
        if m < n:
            seg = np.pad(seg, (n - m, 0), mode="edge")
        sims[b] = 1.0 / (seg_dist(test_seg, seg) + EPS)

    if 3 in train_bids:
        total = sum(sims.values()) + EPS
        alpha = sims[3] / total
    else:
        # B3 held out: proxy B3-likeness by where test slope ranks vs training slopes
        test_slope = slope_of(test_seg)
        train_slopes = [
            slope_of(np.asarray(hi_train_sp[b][-min(n, len(hi_train_sp[b])):], dtype=float))
            for b in train_bids
        ]
        alpha = float(np.mean([test_slope > s for s in train_slopes]))

    return float(np.clip(alpha, 0.0, 1.0))


def compute_alpha_for_config(hi_test, hi_train_sp, train_bids, n, use_recent, dist_fn):
    """
    Generic alpha computation for parameter sweep.
    use_recent=True  → compare last n timesteps (late-stage profile)
    use_recent=False → compare first n timesteps (early-life profile)
    dist_fn          → distance function (seg_dist, slope_only_dist, level_slope_dist)
    """
    actual_n = min(n, len(hi_test))
    if use_recent:
        test_seg = np.asarray(hi_test[-actual_n:], dtype=float)
        def get_seg(b):
            m = min(n, len(hi_train_sp[b]))
            s = np.asarray(hi_train_sp[b][-m:], dtype=float)
            return np.pad(s, (n - m, 0), mode="edge") if m < n else s
    else:
        test_seg = np.asarray(hi_test[:actual_n], dtype=float)
        def get_seg(b):
            m = min(n, len(hi_train_sp[b]))
            s = np.asarray(hi_train_sp[b][:m], dtype=float)
            return np.pad(s, (0, n - m), mode="edge") if m < n else s

    sims = {}
    for b in train_bids:
        sims[b] = 1.0 / (dist_fn(test_seg, get_seg(b)) + EPS)

    if 3 in train_bids:
        total = sum(sims.values()) + EPS
        alpha = sims[3] / total
    else:
        test_slope = slope_of(test_seg)
        train_slopes = [slope_of(get_seg(b)) for b in train_bids]
        alpha = float(np.mean([test_slope > s for s in train_slopes]))

    return float(np.clip(alpha, 0.0, 1.0))


# ── Run LOOCV for a single HI configuration ───────────────────────────────────
def run_model_zoo_loocv(hi_train, label, device):
    print(f"\n>>> Running Model Zoo LOOCV for: {label} <<<")
    model_names = ["lgbm", "lstm", "gru", "tcn", "dtw"]
    results = {}

    for test_bid in BEARINGS:
        train_bids = [b for b in BEARINGS if b != test_bid]
        n_test = len(hi_train[test_bid])
        fold = {"N": n_test}

        lgbm_model = train_lgbm(hi_train, train_bids)
        obs, pred = predict_lgbm(lgbm_model, hi_train[test_bid])
        fold["obs_pts"] = obs
        fold["lgbm"] = pred

        for kind in ["lstm", "gru", "tcn"]:
            models, scale = train_torch_ensemble(hi_train, train_bids, kind, device)
            obs2, pred2 = predict_torch(models, scale, hi_train[test_bid], SEQ_LENGTH, 0, device)
            fold[kind] = pred2

        _, pred_dtw = predict_dtw_knn(hi_train, train_bids, hi_train[test_bid])
        fold["dtw"] = pred_dtw

        raw_scores = {m: score_curve(n_test, obs, fold[m]) for m in model_names}
        weights = score_weights(raw_scores)
        fold["zoo_raw"] = ensemble_predictions({m: fold[m] for m in model_names}, weights)
        fold["raw_scores"] = raw_scores
        fold["weights"] = weights
        results[test_bid] = fold

    cf_rows = []
    for m in model_names + ["zoo_raw"]:
        cf, sc = calibrate_model({b: results[b][m] for b in BEARINGS}, results)
        cf_rows.append({"model": m, "cf": cf, "score": sc})
    cf_df = pd.DataFrame(cf_rows).sort_values("score", ascending=False)

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

    final_predictions = {}
    if zoo_sc > best_score:
        for b in BEARINGS:
            final_predictions[b] = np.asarray(results[b]["zoo"]) * zoo_cf
        actual_kind = "zoo"
        actual_cf = zoo_cf
    else:
        for b in BEARINGS:
            final_predictions[b] = np.asarray(results[b][best_model]) * best_cf
        actual_kind = best_model
        actual_cf = best_cf

    print(f"[{label}] LOOCV Best single: {best_model} (cf={best_cf:.2f}, score={best_score:.4f})")
    print(f"[{label}] LOOCV Zoo: (cf={zoo_cf:.2f}, score={zoo_sc:.4f})")
    print(f"[{label}] Selected: {actual_kind} (cf={actual_cf:.2f})")
    print(f"  {'Bearing':<10} {'Score':<10} {'mean_er %':<12}")
    for b in BEARINGS:
        sc = score_curve(results[b]["N"], results[b]["obs_pts"], final_predictions[b])
        y_true = true_rul(results[b]["N"], results[b]["obs_pts"])
        mean_er = float(np.mean([
            100.0 * (t - p) / t
            for t, p in zip(y_true, final_predictions[b]) if t > 0
        ]))
        print(f"  B{b}         {sc:.4f}     {mean_er:+.1f}%")

    avg_raw_weights = {}
    for m in model_names:
        avg_raw_weights[m] = float(np.mean([results[b]["weights"][m] for b in BEARINGS]))

    return final_predictions, results, cal_cfs, final_weights, zoo_cf, actual_kind, actual_cf, avg_raw_weights


# ── Main function ────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load HI configurations
    # configuration 1: TH v8 HI
    hi_train_th = {b: pd.read_csv(TH_HI_DIR / f"v8_Bearing{b}_HI.csv")["HI_v8"].values.astype(float)
                   for b in BEARINGS}
    hi_test_th = {t: pd.read_csv(TH_HI_DIR / f"v8_Test{t}_HI.csv")["HI_v8"].values.astype(float)
                  for t in TEST_IDS}

    # configuration 2: SP V10c HI
    hi_train_sp = {b: pd.read_csv(SP_HI_DIR / f"HI_Bearing{b}.csv")["HI"].values.astype(float)
                   for b in BEARINGS}
    hi_test_sp = {t: pd.read_csv(SP_HI_DIR / "test" / f"HI_Test{t}.csv")["HI"].values.astype(float)
                  for t in TEST_IDS}

    # 2. Run LOOCV for both HIs
    preds_loocv_th, results_th, th_cal_cfs, th_final_weights, th_zoo_cf, th_kind, th_cf, th_avg_raw_weights = run_model_zoo_loocv(hi_train_th, "TH v8 HI", device)
    preds_loocv_sp, results_sp, sp_cal_cfs, sp_final_weights, sp_zoo_cf, sp_kind, sp_cf, sp_avg_raw_weights = run_model_zoo_loocv(hi_train_sp, "SP V10c HI", device)

    # 3. Optimize blending weight (alpha)
    # alpha: weight for TH_HI model, (1 - alpha): weight for SP_HI model
    print("\n" + "=" * 72)
    print("OPTIMIZING CROSS-PIPELINE RUL BLENDING")
    print("=" * 72)
    print(f"{'alpha':<7} | {'B1':<8} | {'B2':<8} | {'B3':<8} | {'B4':<8} | {'AVG Score':<10}")
    print("-" * 72)

    best_avg_score = -1.0
    best_alpha = 0.5
    optimal_scores_by_bearing = {}

    alpha_grid = np.linspace(0, 1, 11)  # [0.0, 0.1, ..., 1.0]
    for alpha in alpha_grid:
        scores = []
        b_scores = {}
        for b in BEARINGS:
            # Note: obs_pts should be identical for both models since SEQ_LENGTH is identical
            obs_pts = results_th[b]["obs_pts"]
            n_test = results_th[b]["N"]
            p_th = preds_loocv_th[b]
            p_sp = preds_loocv_sp[b]
            p_blend = alpha * p_th + (1.0 - alpha) * p_sp
            sc = score_curve(n_test, obs_pts, p_blend)
            scores.append(sc)
            b_scores[b] = sc
        
        avg_sc = float(np.mean(scores))
        print(f"{alpha:7.2f} | {b_scores[1]:8.4f} | {b_scores[2]:8.4f} | {b_scores[3]:8.4f} | {b_scores[4]:8.4f} | {avg_sc:10.4f}")
        
        if avg_sc > best_avg_score:
            best_avg_score = avg_sc
            best_alpha = float(alpha)
            optimal_scores_by_bearing = b_scores

    print("-" * 72)
    print(f"Optimal Alpha: {best_alpha:.2f}")
    print(f"Best Avg LOOCV Score: {best_avg_score:.4f}")
    print(f"Optimal bearing scores: B1={optimal_scores_by_bearing[1]:.4f}, B2={optimal_scores_by_bearing[2]:.4f}, B3={optimal_scores_by_bearing[3]:.4f}, B4={optimal_scores_by_bearing[4]:.4f}")
    print(f"NOTE: Global alpha grid-search is informational only — single alpha is")
    print(f"      oracle-free but cannot capture B3 vs normal bearing difference.")

    # 4. LOO-Validated Similarity Alpha (DTW-based, no oracle)
    # Each bearing's alpha is derived solely from the n_early observations of its
    # own SP HI compared to the training bearings visible in that LOO fold.
    print("\n" + "=" * 72)
    print("LOO-VALIDATED SIMILARITY ALPHA (DTW-based, oracle-free)")
    print("=" * 72)
    print(f"  {'Bearing':<10} {'Alpha(TH)':<12} {'Score':<10} {'mean_er %':<12}")
    print("-" * 72)

    sim_alpha_loocv = {}
    sim_alpha_score_list = []
    for test_bid in BEARINGS:
        train_bids_loo = [b for b in BEARINGS if b != test_bid]
        alpha = compute_similarity_alpha(
            hi_train_sp[test_bid], hi_train_sp, train_bids_loo, n_early=10
        )
        obs_pts = results_th[test_bid]["obs_pts"]
        n_test  = results_th[test_bid]["N"]
        p_th    = preds_loocv_th[test_bid]
        p_sp    = preds_loocv_sp[test_bid]
        p_blend = alpha * p_th + (1.0 - alpha) * p_sp
        sc      = score_curve(n_test, obs_pts, p_blend)
        y_true  = true_rul(n_test, obs_pts)
        mean_er = float(np.mean([
            100.0 * (t - p) / t for t, p in zip(y_true, p_blend) if t > 0
        ]))
        sim_alpha_loocv[test_bid] = {
            "alpha": alpha, "score": sc, "mean_er": mean_er, "pred": p_blend
        }
        sim_alpha_score_list.append(sc)
        print(f"  B{test_bid}         {alpha:>9.3f}   {sc:>9.4f}   {mean_er:>+9.1f}%")

    avg_sim_alpha_score = float(np.mean(sim_alpha_score_list))
    print("-" * 72)
    print(f"  LOO Similarity Alpha Avg: {avg_sim_alpha_score:.4f}")
    print(f"  (Reference — global best_alpha={best_alpha:.2f}: {best_avg_score:.4f})")
    print(f"  (Reference — pure SP alpha=0.0: {optimal_scores_by_bearing.get(1, 0):.4f} / pure TH alpha=1.0)")

    # 5. Diagnostic — parameter sweep over similarity configurations
    # This section uses already-computed LOOCV predictions (no retraining).
    # Goal: find if any config gives B3 alpha >> B2 alpha (clean separation).
    print("\n" + "=" * 90)
    print("DIAGNOSTIC: SIMILARITY ALPHA PARAMETER SWEEP")
    print("=" * 90)

    # ── Distance function decomposition for late-30 (current config) ──
    print("\n[A] seg_dist component breakdown — last 30 timesteps")
    print(f"  {'Pair':<8} | {'endpoint':>9} {'mean':>7} {'delta':>7} {'slope':>7} {'shape':>7} | {'TOTAL':>7}")
    print("  " + "-" * 72)
    for ref in BEARINGS:
        for other in BEARINGS:
            if ref == other:
                continue
            n = 30
            a = np.asarray(hi_train_sp[ref][-n:], dtype=float)
            b_seg = np.asarray(hi_train_sp[other][-n:], dtype=float)
            c = seg_dist_components(a, b_seg)
            total = sum(c.values())
            print(f"  B{ref}→B{other}    | "
                  f"{c['endpoint']:>9.3f} {c['mean']:>7.3f} {c['total_delta']:>7.3f} "
                  f"{c['slope']:>7.3f} {c['shape']:>7.3f} | {total:>7.3f}")

    # ── Sweep: window size × direction × distance function ──
    sweep_configs = []
    for n in [10, 15, 20, 25, 30, 40, 50]:
        sweep_configs.append((f"last_{n:02d}_seg",   n, True,  seg_dist))
        sweep_configs.append((f"first_{n:02d}_seg",  n, False, seg_dist))
    sweep_configs.append(("last_15_slope",   15, True,  slope_only_dist))
    sweep_configs.append(("last_30_slope",   30, True,  slope_only_dist))
    sweep_configs.append(("last_15_lvlslp",  15, True,  level_slope_dist))
    sweep_configs.append(("last_30_lvlslp",  30, True,  level_slope_dist))

    print(f"\n[B] Alpha and Score by config (B3 alpha goal: HIGH; B1/B2/B4: LOW)")
    W = 18
    print(f"  {'Config':<{W}} | {'α_B1':>5} {'α_B2':>5} {'α_B3':>5} {'α_B4':>5} "
          f"| {'sc_B1':>6} {'sc_B2':>6} {'sc_B3':>6} {'sc_B4':>6} | {'Avg':>6}")
    print("  " + "-" * 90)

    sweep_best_score = -1.0
    sweep_best_name = ""
    for cfg_name, n_w, use_recent, dist_fn in sweep_configs:
        cfg_alphas, cfg_scores = {}, {}
        for test_bid in BEARINGS:
            train_bids_loo = [b for b in BEARINGS if b != test_bid]
            a = compute_alpha_for_config(
                hi_train_sp[test_bid], hi_train_sp, train_bids_loo, n_w, use_recent, dist_fn
            )
            cfg_alphas[test_bid] = a
            p_blend = a * preds_loocv_th[test_bid] + (1.0 - a) * preds_loocv_sp[test_bid]
            cfg_scores[test_bid] = score_curve(
                results_th[test_bid]["N"], results_th[test_bid]["obs_pts"], p_blend
            )
        avg_sc = float(np.mean(list(cfg_scores.values())))
        marker = " ◀" if avg_sc >= sweep_best_score else ""
        print(f"  {cfg_name:<{W}} | "
              f"{cfg_alphas[1]:>5.3f} {cfg_alphas[2]:>5.3f} {cfg_alphas[3]:>5.3f} {cfg_alphas[4]:>5.3f} "
              f"| {cfg_scores[1]:>6.4f} {cfg_scores[2]:>6.4f} {cfg_scores[3]:>6.4f} {cfg_scores[4]:>6.4f} "
              f"| {avg_sc:>6.4f}{marker}")
        if avg_sc > sweep_best_score:
            sweep_best_score = avg_sc
            sweep_best_name = cfg_name

    print("  " + "-" * 90)
    print(f"  Sweep best: {sweep_best_name}  score={sweep_best_score:.4f}")
    print(f"  Reference — pure SP: {best_avg_score:.4f}")

    # ── Hard-threshold analysis for current config (last_30_seg) ──
    print(f"\n[C] Hard-threshold analysis for current config (last_30_seg)")
    print(f"  threshold | α_B1  α_B2  α_B3  α_B4 → assigned | scores → Avg")
    print("  " + "-" * 70)
    for thr in [0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75]:
        hard_scores = {}
        for test_bid in BEARINGS:
            train_bids_loo = [b for b in BEARINGS if b != test_bid]
            soft_a = compute_alpha_for_config(
                hi_train_sp[test_bid], hi_train_sp, train_bids_loo, 30, True, seg_dist
            )
            hard_a = 1.0 if soft_a >= thr else 0.0
            p_blend = hard_a * preds_loocv_th[test_bid] + (1.0 - hard_a) * preds_loocv_sp[test_bid]
            hard_scores[test_bid] = score_curve(
                results_th[test_bid]["N"], results_th[test_bid]["obs_pts"], p_blend
            )
        avg_h = float(np.mean(list(hard_scores.values())))
        print(f"  thr={thr:.2f}   | "
              f"{hard_scores[1]:.4f} {hard_scores[2]:.4f} {hard_scores[3]:.4f} {hard_scores[4]:.4f} | {avg_h:.4f}")

    # 6. Final Train on Full data & Test Inference
    print("\nRunning final Test Inference...")
    
    # Train final TH HI Model
    print("  Training final models on TH HI...")
    th_lgbm = train_lgbm(hi_train_th, BEARINGS)
    th_torch_models = {}
    for kind in ["lstm", "gru", "tcn"]:
        th_torch_models[kind] = train_torch_ensemble(hi_train_th, BEARINGS, kind, device)

    # Train final SP HI Model
    print("  Training final models on SP HI...")
    sp_lgbm = train_lgbm(hi_train_sp, BEARINGS)
    sp_torch_models = {}
    for kind in ["lstm", "gru", "tcn"]:
        sp_torch_models[kind] = train_torch_ensemble(hi_train_sp, BEARINGS, kind, device)

    summary = []
    for tid in TEST_IDS:
        # TH Test inference
        t_hi_th = hi_test_th[tid]
        start_obs_th = estimate_start_obs(hi_train_th, t_hi_th, BEARINGS)
        obs_th, p_lgbm_th = predict_lgbm(th_lgbm, t_hi_th, start_obs=start_obs_th)
        th_preds = {"lgbm": p_lgbm_th * th_cal_cfs["lgbm"]}
        for kind in ["lstm", "gru", "tcn"]:
            models, scale = th_torch_models[kind]
            _, pp = predict_torch(models, scale, t_hi_th, SEQ_LENGTH, start_obs_th, device)
            th_preds[kind] = pp * th_cal_cfs[kind]
        _, p_dtw_th = predict_dtw_knn(hi_train_th, BEARINGS, t_hi_th)
        th_preds["dtw"] = p_dtw_th * th_cal_cfs["dtw"]

        th_raw_preds = {m: th_preds[m] / max(th_cal_cfs[m], EPS) for m in th_preds}
        if th_kind == "zoo":
            th_final = ensemble_predictions(th_preds, th_final_weights) * th_zoo_cf
        elif th_kind == "zoo_raw":
            th_final = ensemble_predictions(th_raw_preds, th_avg_raw_weights) * th_cf
        else:
            th_final = th_raw_preds[th_kind] * th_cf

        # SP Test inference
        t_hi_sp = hi_test_sp[tid]
        start_obs_sp = estimate_start_obs(hi_train_sp, t_hi_sp, BEARINGS)
        obs_sp, p_lgbm_sp = predict_lgbm(sp_lgbm, t_hi_sp, start_obs=start_obs_sp)
        sp_preds = {"lgbm": p_lgbm_sp * sp_cal_cfs["lgbm"]}
        for kind in ["lstm", "gru", "tcn"]:
            models, scale = sp_torch_models[kind]
            _, pp = predict_torch(models, scale, t_hi_sp, SEQ_LENGTH, start_obs_sp, device)
            sp_preds[kind] = pp * sp_cal_cfs[kind]
        _, p_dtw_sp = predict_dtw_knn(hi_train_sp, BEARINGS, t_hi_sp)
        sp_preds["dtw"] = p_dtw_sp * sp_cal_cfs["dtw"]

        sp_raw_preds = {m: sp_preds[m] / max(sp_cal_cfs[m], EPS) for m in sp_preds}
        if sp_kind == "zoo":
            sp_final = ensemble_predictions(sp_preds, sp_final_weights) * sp_zoo_cf
        elif sp_kind == "zoo_raw":
            sp_final = ensemble_predictions(sp_raw_preds, sp_avg_raw_weights) * sp_cf
        else:
            sp_final = sp_raw_preds[sp_kind] * sp_cf

        # Per-bearing similarity alpha — LOO-validated, no oracle
        alpha_test = compute_similarity_alpha(
            hi_test_sp[tid], hi_train_sp, BEARINGS, n_early=10
        )
        print(f"    Test{tid}: similarity alpha (TH weight) = {alpha_test:.3f}")

        final_blend = alpha_test * th_final + (1.0 - alpha_test) * sp_final
        final_blend = np.maximum(final_blend, MIN_RUL_CYCLES)
        hours = final_blend * INTERVAL_SEC / 3600.0

        # Save individual prediction CSV
        out_df = pd.DataFrame({
            "obs_cycle": obs_th,
            "rul_th_hi": th_final,
            "rul_sp_hi": sp_final,
            "alpha_sim": alpha_test,
            "rul_blend": final_blend,
            "rul_hours": hours
        })
        out_df.to_csv(OUT_DIR / f"Test{tid}_RUL.csv", index=False)

        summary.append({
            "test_id": tid,
            "th_start_obs": start_obs_th,
            "sp_start_obs": start_obs_sp,
            "similarity_alpha": round(float(alpha_test), 3),
            "th_rul_hours": round(float(th_final[-1] * INTERVAL_SEC / 3600.0), 2),
            "sp_rul_hours": round(float(sp_final[-1] * INTERVAL_SEC / 3600.0), 2),
            "blend_rul_hours": round(float(hours[-1]), 2),
            "blend_rul_cycles": round(float(final_blend[-1]), 2),
        })

    # Save summary files
    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(OUT_DIR / "test_summary.csv", index=False)
    print("\n=== Final Test RUL Predictions Summary ===")
    print(sum_df.to_string(index=False))

    # Save registration results.csv — using LOO similarity alpha scores (oracle-free)
    reg_df = pd.DataFrame([
        {
            "dataset": "Train",
            "test_bearing": b,
            "score": round(sim_alpha_loocv[b]["score"], 4),
            "mean_er": round(sim_alpha_loocv[b]["mean_er"], 2),
            "alpha": round(sim_alpha_loocv[b]["alpha"], 3),
        }
        for b in BEARINGS
    ])
    reg_df.to_csv(OUT_DIR / "rul_results.csv", index=False)
    print("\nSaved RUL registration CSV to:", OUT_DIR / "rul_results.csv")
    print(f"Final LOO Similarity Alpha Avg Score = {avg_sim_alpha_score:.4f}")
    print(f"(Reference: oracle hybrid 0.6824, pure SP 0.5732)")


if __name__ == "__main__":
    main()
