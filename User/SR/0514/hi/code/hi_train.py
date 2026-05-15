import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# =========================================================
# 경로 및 상수
# =========================================================
SR_BASE  = "/data/home/ksphm/2026-challenge-KSPHM/User/SR/0514"
SC_BASE  = "/data/home/ksphm/2026-challenge-KSPHM/User/SC"

V2_DIR = Path(f"{SC_BASE}/HI/04142304_signal_transform_v2/output")

import os
os.makedirs(f"{SR_BASE}/hi/output/train", exist_ok=True)
V3_OUT = Path(f"{SR_BASE}/hi/output/train")
V4_OUT = Path(f"{SR_BASE}/hi/output/train_v4")

BEARING_IDS = [1, 2, 3, 4]

# ── HI-A: 주파수도메인 위주 (열화 초기 민감) ──────────────────────
FEATURE_Q = {
    "ch3_high_band": 0.4315732105779938,
    "ch4_high_band": 0.41934581236265145,
    "ch3_std": 0.4143663846438889,
    "ch3_total_power": 0.41207524516168137,
    "ch3_energy": 0.41108403369865365,
    "ch3_rms": 0.41108403369865365,
    "ch3_p2p": 0.3665441916808586,
}

FEATURE_GROUPS = {
    "highfreq": ["ch3_high_band", "ch4_high_band"],
    "energy": ["ch3_total_power", "ch3_energy", "ch3_rms"],
    "variation": ["ch3_std", "ch3_p2p"],
}

ALL_FEATS = list(FEATURE_Q.keys())

# ── HI-B: 시간도메인 위주 (열화 후기 민감) ────────────────────────
FEATURE_Q_B = {
    "ch3_kurtosis": 0.40,    # 충격성 — 결함 후기 급등
    "ch3_crest_f":  0.38,    # Crest Factor — 임펄스성
    "ch3_rms":      0.36,    # 전반적 에너지
    "ch3_p2p":      0.35,    # Peak-to-Peak
    "ch4_kurtosis": 0.37,    # ch4 충격성
    "ch4_rms":      0.34,    # ch4 에너지
}

FEATURE_GROUPS_B = {
    "impulse":   ["ch3_kurtosis", "ch3_crest_f", "ch4_kurtosis"],
    "amplitude": ["ch3_rms", "ch3_p2p", "ch4_rms"],
}

ALL_FEATS_B = list(FEATURE_Q_B.keys())

# =========================================================
# 평가 및 전처리 함수
# =========================================================
def monotonicity(series: np.ndarray) -> float:
    if len(series) <= 1: return 0.0
    diff = np.diff(series)
    return abs(np.sum(diff > 0) - np.sum(diff < 0)) / len(diff)

def trendability(series: np.ndarray) -> float:
    if len(series) <= 1: return 0.0
    rho, _ = spearmanr(np.arange(len(series)), series)
    return abs(rho) if not np.isnan(rho) else 0.0

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1: return x.copy()
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    y = np.convolve(x_pad, kernel, mode="valid")
    return y[:len(x)]

def minmax_scale(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

def robust_clip(x: np.ndarray, low_q: float = 0.01, high_q: float = 0.99) -> np.ndarray:
    lo = np.quantile(x, low_q)
    hi = np.quantile(x, high_q)
    return np.clip(x, lo, hi)

def ema_smooth(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

# =========================================================
# V3 FDR 함수 — 외부 baseline (hi_test.py와 동일 방식)
# =========================================================
def compute_bearing_baseline(dfs: dict, baseline_ratio: float, exclude_bid: int) -> dict:
    """
    LOOCV-style external baseline.
    exclude_bid를 제외한 나머지 베어링들의 레짐별 건강구간(첫 baseline_ratio%) 평균 μ.
    hi_test.py의 compute_train_fdr_baseline()과 동일한 방식 — Train/Test HI 분포 일치.
    """
    feat_vals = {(r, f): [] for r in [0, 1] for f in ALL_FEATS}

    for bid, df in dfs.items():
        if bid == exclude_bid:
            continue
        cond = df["cond"].values
        n = len(df)
        n_base = max(3, int(n * baseline_ratio))

        for regime in [0, 1]:
            idx_regime = np.where(cond == regime)[0]
            base_idx = idx_regime[:n_base]
            if len(base_idx) == 0:
                continue
            for f in ALL_FEATS:
                feat_vals[(regime, f)].extend(df[f].values[base_idx].tolist())

    baseline = {}
    for regime in [0, 1]:
        for f in ALL_FEATS:
            vals = feat_vals[(regime, f)]
            baseline[(regime, f)] = float(np.mean(vals)) if vals else 1.0

    return baseline


def build_feature_ratios_external(feat_matrix: np.ndarray,
                                   feature_names: list,
                                   cond: np.ndarray,
                                   baseline: dict,
                                   eps: float = 1e-8) -> np.ndarray:
    """레짐별 외부 baseline μ로 FDR 비율 계산."""
    ratios = np.zeros_like(feat_matrix, dtype=float)
    for regime in [0, 1]:
        idx = np.where(cond == regime)[0]
        if len(idx) == 0:
            continue
        baseline_vec = np.array([baseline[(regime, f)] for f in feature_names])
        baseline_vec = np.where(np.abs(baseline_vec) < eps, eps, baseline_vec)
        ratios[idx] = (feat_matrix[idx] - baseline_vec) / (np.abs(baseline_vec) + eps)
    return ratios


def postprocess_score(score: np.ndarray) -> np.ndarray:
    score = robust_clip(score, 0.01, 0.99)
    corr = np.corrcoef(np.arange(len(score)), score)[0, 1]
    if not np.isnan(corr) and corr < 0:
        score = -score
    return score


def make_group_hi_fdr(feat_matrix: np.ndarray,
                       feature_names: list,
                       cond: np.ndarray,
                       baseline: dict,
                       ema_alpha: float,
                       feat_q_map: dict = None) -> np.ndarray:
    """범용 FDR HI 생성. feat_q_map이 None이면 FEATURE_Q 사용."""
    if feat_q_map is None:
        feat_q_map = FEATURE_Q
    ratios  = build_feature_ratios_external(feat_matrix, feature_names, cond, baseline)
    weights = np.array([feat_q_map[f] for f in feature_names], dtype=float)
    weights = weights / (weights.sum() + 1e-12)
    score   = (ratios * weights.reshape(1, -1)).sum(axis=1)
    score   = postprocess_score(score)
    score   = minmax_scale(score)
    score   = ema_smooth(score, alpha=ema_alpha)
    return minmax_scale(score)


def v3_pipeline(df: pd.DataFrame, br: float, alpha: float, baseline: dict) -> np.ndarray:
    """
    HI-A: FDR Group Weight HI 파이프라인 (주파수도메인 위주).
    baseline: compute_bearing_baseline()로 계산한 외부 LOO baseline.
    """
    cond = df["cond"].values
    sub_his       = {}
    group_weights = {}
    for gname, feats in FEATURE_GROUPS.items():
        mat    = df[feats].values
        sub_hi = make_group_hi_fdr(mat, feats, cond, baseline, alpha, FEATURE_Q)
        sub_his[gname]       = sub_hi
        group_weights[gname] = np.mean([FEATURE_Q[f] for f in feats])

    sub_mat = np.column_stack([sub_his[g] for g in FEATURE_GROUPS.keys()])
    w = np.array([group_weights[g] for g in FEATURE_GROUPS.keys()], dtype=float)
    w = w / (w.sum() + 1e-12)
    final_hi = (sub_mat * w.reshape(1, -1)).sum(axis=1)
    return moving_average(minmax_scale(final_hi), 7)


def compute_bearing_baseline_b(dfs: dict, baseline_ratio: float, exclude_bid: int) -> dict:
    """HI-B용 외부 LOO baseline (ALL_FEATS_B 피처 기준)."""
    feat_vals = {(r, f): [] for r in [0, 1] for f in ALL_FEATS_B}

    for bid, df in dfs.items():
        if bid == exclude_bid:
            continue
        cond = df["cond"].values
        n = len(df)
        n_base = max(3, int(n * baseline_ratio))

        for regime in [0, 1]:
            idx_regime = np.where(cond == regime)[0]
            base_idx = idx_regime[:n_base]
            if len(base_idx) == 0:
                continue
            for f in ALL_FEATS_B:
                if f in df.columns:
                    feat_vals[(regime, f)].extend(df[f].values[base_idx].tolist())

    baseline = {}
    for regime in [0, 1]:
        for f in ALL_FEATS_B:
            vals = feat_vals[(regime, f)]
            baseline[(regime, f)] = float(np.mean(vals)) if vals else 1.0
    return baseline


def build_feature_ratios_hib(feat_matrix: np.ndarray,
                              feature_names: list,
                              cond: np.ndarray,
                              baseline: dict,
                              eps: float = 1e-8) -> np.ndarray:
    """HI-B용 FDR 비율 계산."""
    ratios = np.zeros_like(feat_matrix, dtype=float)
    for regime in [0, 1]:
        idx = np.where(cond == regime)[0]
        if len(idx) == 0:
            continue
        baseline_vec = np.array([baseline[(regime, f)] for f in feature_names])
        baseline_vec = np.where(np.abs(baseline_vec) < eps, eps, baseline_vec)
        ratios[idx] = (feat_matrix[idx] - baseline_vec) / (np.abs(baseline_vec) + eps)
    return ratios


def v5_pipeline_hib(df: pd.DataFrame, br: float, alpha: float, baseline_b: dict) -> np.ndarray:
    """
    HI-B: 시간도메인 FDR HI 파이프라인 (Kurtosis/Crest/RMS/P2P 위주).
    baseline_b: compute_bearing_baseline_b()로 계산한 외부 LOO baseline.
    """
    cond = df["cond"].values
    sub_his       = {}
    group_weights = {}
    for gname, feats in FEATURE_GROUPS_B.items():
        available = [f for f in feats if f in df.columns]
        if not available:
            continue
        mat = df[available].values
        ratios = build_feature_ratios_hib(mat, available, cond, baseline_b)
        weights = np.array([FEATURE_Q_B[f] for f in available], dtype=float)
        weights = weights / (weights.sum() + 1e-12)
        score = (ratios * weights.reshape(1, -1)).sum(axis=1)
        score = postprocess_score(score)
        score = minmax_scale(score)
        score = ema_smooth(score, alpha=alpha)
        sub_his[gname]       = minmax_scale(score)
        group_weights[gname] = np.mean([FEATURE_Q_B[f] for f in available])

    if not sub_his:
        return np.zeros(len(df))

    sub_mat = np.column_stack([sub_his[g] for g in sub_his])
    w = np.array([group_weights[g] for g in sub_his], dtype=float)
    w = w / (w.sum() + 1e-12)
    final_hi = (sub_mat * w.reshape(1, -1)).sum(axis=1)
    return moving_average(minmax_scale(final_hi), 7)


# =========================================================
# V4 VAE 함수 (변경 없음)
# =========================================================
class SimpleVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=1):
        super(SimpleVAE, self).__init__()
        hidden = max(2, input_dim // 2)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, input_dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def train_vae(feat_matrix, baseline_ratio, latent_dim):
    n = feat_matrix.shape[0]
    n_base = max(3, int(n * baseline_ratio))

    scaler = StandardScaler()
    scaler.fit(feat_matrix[:n_base])
    X_scaled = scaler.transform(feat_matrix)

    X_train = torch.tensor(X_scaled[:n_base], dtype=torch.float32)
    dataset = TensorDataset(X_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    input_dim = feat_matrix.shape[1]
    model = SimpleVAE(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    def loss_function(recon_x, x, mu, logvar):
        BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    model.train()
    for _ in range(50):
        for batch in loader:
            batch_x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            loss = loss_function(recon_x, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_all = torch.tensor(X_scaled, dtype=torch.float32)
        recon_all, _, _ = model(X_all)
        mse = torch.mean((X_all - recon_all)**2, dim=1).numpy()

    return mse

def make_group_hi_vae(feat_matrix: np.ndarray, br: float, alpha: float, latent_dim: int) -> np.ndarray:
    score = train_vae(feat_matrix, br, latent_dim)
    score = robust_clip(score, 0.01, 0.99)
    corr = np.corrcoef(np.arange(len(score)), score)[0, 1]
    if not np.isnan(corr) and corr < 0:
        score = -score
    score = minmax_scale(score)
    score = ema_smooth(score, alpha=alpha)
    return minmax_scale(score)

def v4_pipeline(df: pd.DataFrame, br: float, alpha: float, latent_dim: int) -> np.ndarray:
    sub_his = {}
    group_weights = {}
    for gname, feats in FEATURE_GROUPS.items():
        mat = df[feats].values
        sub_hi = make_group_hi_vae(mat, br, alpha, latent_dim)
        sub_his[gname] = sub_hi
        group_weights[gname] = np.mean([FEATURE_Q[f] for f in feats])

    sub_mat = np.column_stack([sub_his[g] for g in FEATURE_GROUPS.keys()])
    w = np.array([group_weights[g] for g in FEATURE_GROUPS.keys()], dtype=float)
    w = w / (w.sum() + 1e-12)
    final_hi = (sub_mat * w.reshape(1, -1)).sum(axis=1)
    return moving_average(minmax_scale(final_hi), 7)


# =========================================================
# 그리드 탐색 및 평가
# =========================================================
def evaluate_hi(hi_series, cond_series):
    scores = []
    for lbl in [0, 1]:
        idx = cond_series == lbl
        if sum(idx) > 1:
            sub_hi = hi_series[idx]
            scores.append((monotonicity(sub_hi) + trendability(sub_hi)) / 2)
    return np.mean(scores) if scores else 0.0

def load_data():
    dfs = {}
    for bid in BEARING_IDS:
        df = pd.read_csv(V2_DIR / f"Bearing{bid}_features_transformed.csv")
        cond_df = pd.read_csv(V2_DIR / f"Bearing{bid}_SSM_result.csv")[["cond"]]
        df["cond"] = cond_df["cond"]
        dfs[bid] = df
    return dfs

def run_grid_search():
    dfs = load_data()

    br_cands    = [0.05, 0.10, 0.15, 0.20, 0.25]
    alpha_cands = [0.1, 0.2, 0.3]

    best_v3_score  = 0
    best_v3_params = None

    print("--- Starting V3/HI-A (FDR, LOO External Baseline) Grid Search ---")
    for br, alpha in itertools.product(br_cands, alpha_cands):
        scores = []
        for bid, df in dfs.items():
            # LOOCV baseline: 현재 베어링 제외한 나머지 3개 기준
            baseline = compute_bearing_baseline(dfs, br, exclude_bid=bid)
            hi = v3_pipeline(df, br, alpha, baseline)
            score = evaluate_hi(hi, df["cond"])
            scores.append(score)

        mean_score = np.mean(scores)
        print(f"V3/HI-A Params (br={br}, alpha={alpha}) -> Q-Score: {mean_score:.4f}")
        if mean_score > best_v3_score:
            best_v3_score  = mean_score
            best_v3_params = (br, alpha)

    print(f"\n[V3/HI-A Best] br={best_v3_params[0]}, alpha={best_v3_params[1]} -> Score: {best_v3_score:.4f}\n")

    # ── HI-B 그리드 탐색 ──────────────────────────────────────────
    best_hib_score  = 0
    best_hib_params = None

    print("--- Starting HI-B (시간도메인: Kurtosis/Crest/RMS/P2P) Grid Search ---")
    for br, alpha in itertools.product(br_cands, alpha_cands):
        scores = []
        for bid, df in dfs.items():
            baseline_b = compute_bearing_baseline_b(dfs, br, exclude_bid=bid)
            hi = v5_pipeline_hib(df, br, alpha, baseline_b)
            score = evaluate_hi(hi, df["cond"])
            scores.append(score)

        mean_score = np.mean(scores)
        print(f"HI-B Params (br={br}, alpha={alpha}) -> Q-Score: {mean_score:.4f}")
        if mean_score > best_hib_score:
            best_hib_score  = mean_score
            best_hib_params = (br, alpha)

    print(f"\n[HI-B Best] br={best_hib_params[0]}, alpha={best_hib_params[1]} -> Score: {best_hib_score:.4f}\n")

    # V4 Grid Search (변경 없음)
    latent_cands = [1, 2]
    best_v4_score  = 0
    best_v4_params = None

    print("--- Starting V4 (VAE Group Weight) Grid Search ---")
    for br, alpha, ldim in itertools.product(br_cands, alpha_cands, latent_cands):
        scores = []
        torch.manual_seed(42)
        np.random.seed(42)

        for bid, df in dfs.items():
            hi = v4_pipeline(df, br, alpha, ldim)
            score = evaluate_hi(hi, df["cond"])
            scores.append(score)

        mean_score = np.mean(scores)
        print(f"V4 Params (br={br}, alpha={alpha}, latent_dim={ldim}) -> Q-Score: {mean_score:.4f}")
        if mean_score > best_v4_score:
            best_v4_score  = mean_score
            best_v4_params = (br, alpha, ldim)

    print(f"\n[V4 Best] br={best_v4_params[0]}, alpha={best_v4_params[1]}, latent_dim={best_v4_params[2]} -> Score: {best_v4_score:.4f}\n")

    # Best 결과 저장
    def save_best(method, params, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(15, 10))
        summary_rows = []
        for bid, df in dfs.items():
            if method == "v3":
                baseline = compute_bearing_baseline(dfs, params[0], exclude_bid=bid)
                hi = v3_pipeline(df, params[0], params[1], baseline)
            elif method == "hib":
                baseline_b = compute_bearing_baseline_b(dfs, params[0], exclude_bid=bid)
                hi = v5_pipeline_hib(df, params[0], params[1], baseline_b)
            else:
                torch.manual_seed(42)
                np.random.seed(42)
                hi = v4_pipeline(df, params[0], params[1], params[2])

            pd.DataFrame({"HI": hi}).to_csv(out_dir / f"Bearing{bid}_best.csv", index=False)

            plt.subplot(2, 2, bid)
            plt.plot(hi, "o-", label="HI", markersize=3, lw=1)
            plt.title(f"Bearing {bid} Best HI ({method})")
            plt.grid(True, alpha=0.3)

            for lbl, rname in [(0, "레짐1(저속)"), (1, "레짐2(고속)")]:
                idx = df["cond"] == lbl
                if sum(idx) > 1:
                    sub_hi = hi[idx]
                    summary_rows.append({
                        "bearing": bid,
                        "regime":  rname,
                        "mon":     monotonicity(sub_hi),
                        "tred":    trendability(sub_hi),
                    })

        plt.tight_layout()
        plt.savefig(out_dir / f"best_{method}_plot.png", dpi=150)
        plt.close()
        pd.DataFrame(summary_rows).to_csv(out_dir / "summary_best.csv", index=False)

    # HI-B 출력 디렉토리
    HIB_OUT = Path(f"{SR_BASE}/hi/output/train_hib")

    save_best("v3",  best_v3_params,  V3_OUT)
    save_best("v4",  best_v4_params,  V4_OUT)
    save_best("hib", best_hib_params, HIB_OUT)
    print("Done! Saved best results (HI-A v3, HI-A v4, HI-B).")

# =========================================================
# V4 FDR (Train-Anchored Absolute Scaling, LOO 방식)
# =========================================================
def train_anchored_scale(x: np.ndarray, p5: float, p95: float) -> np.ndarray:
    denom = p95 - p5
    if abs(denom) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return np.clip((x - p5) / denom, 0.0, 1.0)


def compute_group_score_stats_loo(dfs: dict, br: float, exclude_bid: int) -> dict:
    """
    LOO fold의 학습 베어링들(exclude_bid 제외)에서 그룹별 direction/p5/p95 계산.
    baseline도 동일한 LOO fold 기준으로 사용.
    """
    baseline = compute_bearing_baseline(dfs, br, exclude_bid=exclude_bid)
    all_scores = {g: [] for g in FEATURE_GROUPS}
    dir_votes  = {g: [] for g in FEATURE_GROUPS}

    for bid, df in dfs.items():
        if bid == exclude_bid:
            continue
        cond = df["cond"].values
        for gname, feats in FEATURE_GROUPS.items():
            mat     = df[feats].values
            ratios  = build_feature_ratios_external(mat, feats, cond, baseline)
            weights = np.array([FEATURE_Q[f] for f in feats], dtype=float)
            weights /= weights.sum() + 1e-12
            score   = (ratios * weights.reshape(1, -1)).sum(axis=1)
            rho, _  = spearmanr(np.arange(len(score)), score)
            dir_votes[gname].append(+1 if (not np.isnan(rho) and rho >= 0) else -1)
            all_scores[gname].extend(score.tolist())

    stats = {}
    for gname in FEATURE_GROUPS:
        direction = +1 if sum(dir_votes[gname]) >= 0 else -1
        arr = np.array(all_scores[gname]) * direction
        stats[gname] = {
            "direction": direction,
            "p5":  float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }
    return stats


def compute_group_score_stats_loo_b(dfs: dict, br: float, exclude_bid: int) -> dict:
    """HI-B용 LOO group score stats."""
    baseline_b = compute_bearing_baseline_b(dfs, br, exclude_bid=exclude_bid)
    all_scores = {g: [] for g in FEATURE_GROUPS_B}
    dir_votes  = {g: [] for g in FEATURE_GROUPS_B}

    for bid, df in dfs.items():
        if bid == exclude_bid:
            continue
        cond = df["cond"].values
        for gname, feats in FEATURE_GROUPS_B.items():
            available = [f for f in feats if f in df.columns]
            if not available:
                continue
            mat     = df[available].values
            ratios  = build_feature_ratios_hib(mat, available, cond, baseline_b)
            weights = np.array([FEATURE_Q_B[f] for f in available], dtype=float)
            weights /= weights.sum() + 1e-12
            score   = (ratios * weights.reshape(1, -1)).sum(axis=1)
            rho, _  = spearmanr(np.arange(len(score)), score)
            dir_votes[gname].append(+1 if (not np.isnan(rho) and rho >= 0) else -1)
            all_scores[gname].extend(score.tolist())

    stats = {}
    for gname in FEATURE_GROUPS_B:
        direction = +1 if sum(dir_votes[gname]) >= 0 else -1
        arr = np.array(all_scores[gname]) * direction
        stats[gname] = {
            "direction": direction,
            "p5":  float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }
    return stats


def make_group_hi_fdr_v4(feat_matrix, feature_names, cond, baseline, group_stat, ema_alpha):
    ratios  = build_feature_ratios_external(feat_matrix, feature_names, cond, baseline)
    weights = np.array([FEATURE_Q[f] for f in feature_names], dtype=float)
    weights /= weights.sum() + 1e-12
    score   = (ratios * weights.reshape(1, -1)).sum(axis=1)
    score   = score * group_stat["direction"]
    score   = train_anchored_scale(score, group_stat["p5"], group_stat["p95"])
    score   = ema_smooth(score, alpha=ema_alpha)
    return np.clip(score, 0.0, 1.0)


def make_group_hi_fdr_v4_b(feat_matrix, feature_names, cond, baseline_b, group_stat, ema_alpha):
    ratios  = build_feature_ratios_hib(feat_matrix, feature_names, cond, baseline_b)
    weights = np.array([FEATURE_Q_B[f] for f in feature_names], dtype=float)
    weights /= weights.sum() + 1e-12
    score   = (ratios * weights.reshape(1, -1)).sum(axis=1)
    score   = score * group_stat["direction"]
    score   = train_anchored_scale(score, group_stat["p5"], group_stat["p95"])
    score   = ema_smooth(score, alpha=ema_alpha)
    return np.clip(score, 0.0, 1.0)


def v4fdr_pipeline(df: pd.DataFrame, baseline: dict, group_stats: dict, ema_alpha: float) -> np.ndarray:
    cond = df["cond"].values
    sub_his, group_weights = {}, {}
    for gname, feats in FEATURE_GROUPS.items():
        sub_his[gname]       = make_group_hi_fdr_v4(df[feats].values, feats, cond,
                                                      baseline, group_stats[gname], ema_alpha)
        group_weights[gname] = np.mean([FEATURE_Q[f] for f in feats])
    sub_mat  = np.column_stack([sub_his[g] for g in FEATURE_GROUPS])
    w        = np.array([group_weights[g] for g in FEATURE_GROUPS], dtype=float)
    w       /= w.sum() + 1e-12
    final_hi = (sub_mat * w.reshape(1, -1)).sum(axis=1)
    return np.clip(moving_average(final_hi, 7), 0.0, 1.0)


def v4fdr_pipeline_b(df: pd.DataFrame, baseline_b: dict, group_stats_b: dict, ema_alpha: float) -> np.ndarray:
    cond = df["cond"].values
    sub_his, group_weights = {}, {}
    for gname, feats in FEATURE_GROUPS_B.items():
        available = [f for f in feats if f in df.columns]
        if not available:
            continue
        sub_his[gname]       = make_group_hi_fdr_v4_b(df[available].values, available, cond,
                                                        baseline_b, group_stats_b[gname], ema_alpha)
        group_weights[gname] = np.mean([FEATURE_Q_B[f] for f in available])
    if not sub_his:
        return np.zeros(len(df))
    sub_mat  = np.column_stack([sub_his[g] for g in sub_his])
    w        = np.array([group_weights[g] for g in sub_his], dtype=float)
    w       /= w.sum() + 1e-12
    final_hi = (sub_mat * w.reshape(1, -1)).sum(axis=1)
    return np.clip(moving_average(final_hi, 7), 0.0, 1.0)


def run_v4fdr_grid_search():
    """HI-A/B v4 FDR (Train-Anchored Absolute Scaling) LOO 그리드서치 및 저장."""
    dfs = load_data()
    br_cands    = [0.05, 0.10, 0.15, 0.20, 0.25]
    alpha_cands = [0.1, 0.2, 0.3]

    # ── HI-A v4 ──────────────────────────────────────────────────
    best_a_score, best_a_params = 0.0, None
    print("\n--- V4 FDR HI-A Grid Search ---")
    for br, alpha in itertools.product(br_cands, alpha_cands):
        scores = []
        for bid, df in dfs.items():
            baseline     = compute_bearing_baseline(dfs, br, exclude_bid=bid)
            group_stats  = compute_group_score_stats_loo(dfs, br, exclude_bid=bid)
            hi           = v4fdr_pipeline(df, baseline, group_stats, alpha)
            scores.append(evaluate_hi(hi, df["cond"]))
        mean_score = np.mean(scores)
        print(f"  HI-A v4 (br={br}, alpha={alpha}) -> Q={mean_score:.4f}")
        if mean_score > best_a_score:
            best_a_score, best_a_params = mean_score, (br, alpha)
    print(f"\n[HI-A v4 Best] br={best_a_params[0]}, alpha={best_a_params[1]} -> Q={best_a_score:.4f}")

    # ── HI-B v4 ──────────────────────────────────────────────────
    best_b_score, best_b_params = 0.0, None
    print("\n--- V4 FDR HI-B Grid Search ---")
    for br, alpha in itertools.product(br_cands, alpha_cands):
        scores = []
        for bid, df in dfs.items():
            baseline_b    = compute_bearing_baseline_b(dfs, br, exclude_bid=bid)
            group_stats_b = compute_group_score_stats_loo_b(dfs, br, exclude_bid=bid)
            hi            = v4fdr_pipeline_b(df, baseline_b, group_stats_b, alpha)
            scores.append(evaluate_hi(hi, df["cond"]))
        mean_score = np.mean(scores)
        print(f"  HI-B v4 (br={br}, alpha={alpha}) -> Q={mean_score:.4f}")
        if mean_score > best_b_score:
            best_b_score, best_b_params = mean_score, (br, alpha)
    print(f"\n[HI-B v4 Best] br={best_b_params[0]}, alpha={best_b_params[1]} -> Q={best_b_score:.4f}")

    # ── 저장 (전역 p5/p95로 스케일링 — hi_test_v4.py와 동일 방식) ────────
    # LOO마다 기준이 달라지면 LOOCV fold 내 학습 데이터 스케일이 불일치함.
    # 저장 파일은 전체 4개 베어링 기준 단일 p5/p95를 사용해 일관성 보장.
    # (Grid search 평가는 LOO 유지, 파일 저장만 global 기준)
    V4FDR_OUT  = Path(f"{SR_BASE}/hi/output/train_fdr_v4")
    HIBV4_OUT  = Path(f"{SR_BASE}/hi/output/train_hib_v4")
    V4FDR_OUT.mkdir(parents=True, exist_ok=True)
    HIBV4_OUT.mkdir(parents=True, exist_ok=True)

    br_a, alpha_a = best_a_params
    br_b, alpha_b = best_b_params
    # exclude_bid=0 → bearing id 0은 없으므로 전체 4개 포함
    global_baseline      = compute_bearing_baseline(dfs, br_a, exclude_bid=0)
    global_group_stats   = compute_group_score_stats_loo(dfs, br_a, exclude_bid=0)
    global_baseline_b    = compute_bearing_baseline_b(dfs, br_b, exclude_bid=0)
    global_group_stats_b = compute_group_score_stats_loo_b(dfs, br_b, exclude_bid=0)

    summary_a, summary_b = [], []
    fig_a, fig_b = plt.figure(figsize=(15, 10)), plt.figure(figsize=(15, 10))

    for bid, df in dfs.items():
        # HI-A v4 (global 기준)
        hi_a = v4fdr_pipeline(df, global_baseline, global_group_stats, alpha_a)
        pd.DataFrame({"HI": hi_a}).to_csv(V4FDR_OUT / f"Bearing{bid}_best.csv", index=False)

        fig_a.add_subplot(2, 2, bid)
        plt.plot(hi_a, "o-", markersize=3, lw=1, label="HI-A v4")
        plt.title(f"Bearing {bid}  start={hi_a[0]:.3f}")
        plt.grid(True, alpha=0.3)

        for lbl, rname in [(0, "레짐1"), (1, "레짐2")]:
            idx = df["cond"] == lbl
            if sum(idx) > 1:
                summary_a.append({"bearing": bid, "regime": rname,
                                   "mon": monotonicity(hi_a[idx]),
                                   "tred": trendability(hi_a[idx])})

        # HI-B v4 (global 기준)
        hi_b = v4fdr_pipeline_b(df, global_baseline_b, global_group_stats_b, alpha_b)
        pd.DataFrame({"HI": hi_b}).to_csv(HIBV4_OUT / f"Bearing{bid}_best.csv", index=False)

        fig_b.add_subplot(2, 2, bid)
        plt.figure(fig_b.number)
        plt.plot(hi_b, "o-", markersize=3, lw=1, label="HI-B v4")
        plt.title(f"Bearing {bid}  start={hi_b[0]:.3f}")
        plt.grid(True, alpha=0.3)

        for lbl, rname in [(0, "레짐1"), (1, "레짐2")]:
            idx = df["cond"] == lbl
            if sum(idx) > 1:
                summary_b.append({"bearing": bid, "regime": rname,
                                   "mon": monotonicity(hi_b[idx]),
                                   "tred": trendability(hi_b[idx])})

    plt.figure(fig_a.number); plt.tight_layout()
    fig_a.savefig(V4FDR_OUT / "best_fdr_v4_plot.png", dpi=150); plt.close(fig_a)
    plt.figure(fig_b.number); plt.tight_layout()
    fig_b.savefig(HIBV4_OUT / "best_hib_v4_plot.png", dpi=150); plt.close(fig_b)

    pd.DataFrame(summary_a).to_csv(V4FDR_OUT / "summary_best.csv", index=False)
    pd.DataFrame(summary_b).to_csv(HIBV4_OUT / "summary_best.csv", index=False)
    print(f"\nSaved: {V4FDR_OUT}, {HIBV4_OUT}")


if __name__ == "__main__":
    run_grid_search()
    run_v4fdr_grid_search()
