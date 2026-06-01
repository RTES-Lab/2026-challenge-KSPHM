"""
Exp-K: Model Diversity + Ensemble Strategy Expansion
Based on Exp-J (0.6064) with beta=0.0 (HI-based obs_frac).
Adds: Ridge, TCN-Res, Transformer, BiLSTM, LGBM variants, DTW variants.
Tests: Bidirectional Capped, Trimmed Mean, Phase-based ensembles.
"""
from pathlib import Path
import sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from new_models import (Ridge, train_ridge, predict_ridge,
    TCNRes, MiniTransformer, BiLSTM,
    make_tabular_wide, make_tabular_hifeat,
    predict_dtw_exp, predict_dtw_adaptive_k)

BASE    = Path("/data/home/ksphm/2026-challenge-KSPHM")
HI_DIR  = BASE / "User/SP/05-26/V1b/output"
OUT     = Path(__file__).parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

BEARINGS = [1,2,3,4]; TEST_IDS = [1,2,3,4,5,6]
SEQ = 10; MATCH_LEN = 18; INTERVAL = 600
MEAN_LIFE = 116.5; HI_EOL = 0.75; BETA = 0.0
SEEDS = [42]; MIN_RUL = 1.0; EPS = 1e-8
EOL = {1:126, 2:114, 3:89, 4:137}

def load_hi(kind="train"):
    if kind == "train":
        return {b: pd.read_csv(HI_DIR/f"HI_Bearing{b}.csv")["HI"].values.astype(float) for b in BEARINGS}
    return {t: pd.read_csv(HI_DIR/"test"/f"HI_Test{t}.csv")["HI"].values.astype(float) for t in TEST_IDS}

def mnorm(x):
    x=np.asarray(x,float); return (x-x.min())/(x.max()-x.min()+EPS)
def slp(x):
    x=np.asarray(x,float); return float(np.polyfit(np.arange(len(x)),x,1)[0]) if len(x)>=2 else 0.0
def comp_score(yt,yp):
    if yt<=0: return np.nan
    er=100*(yt-yp)/yt
    return np.exp(-np.log(0.5)*er/20) if er<=0 else np.exp(np.log(0.5)*er/50)
def true_rul(n,obs): return np.maximum(n-np.asarray(obs,float),1.0)
def sc_curve(n,obs,p):
    y=true_rul(n,obs); return float(np.nanmean([comp_score(t,pp) for t,pp in zip(y,p)]))
def err_summ(n,obs,p):
    y=true_rul(n,obs); p=np.asarray(p,float)
    er=[100*(t-pp)/t for t,pp in zip(y,p) if t>0]
    return {"score":sc_curve(n,obs,p),"mean_er":float(np.nanmean(er))}

# ── Feature builders ──
def make_tab(hi, start=0):
    hi=np.asarray(hi,float); hi0=float(hi[0]); x,obs=[],[]
    for i in range(SEQ,len(hi)):
        w=hi[i-SEQ:i]; wn=mnorm(w)
        hf=float(np.clip(w[-1]/HI_EOL,0,2))
        x.append(list(wn)+[slp(wn),float(w[-1]),float(w.mean()),float(w.max()),
                 float(w.min()),float(w.std()),slp(w),float(w[-1]-w[0]),
                 float(w[-1]-hi0),hf,float(w[-1]*hf)])
        obs.append(i)
    return np.asarray(x),np.asarray(obs)

def make_seq(hi, start=0):
    hi=np.asarray(hi,float); hi0=float(hi[0]); x,obs=[],[]
    for i in range(len(hi)-SEQ):
        w=hi[i:i+SEQ]
        of=np.clip(w/HI_EOL,0,2)
        x.append(np.stack([mnorm(w),w.copy(),w-hi0,of],axis=1))
        obs.append(i+SEQ)
    return np.asarray(x),np.asarray(obs)

# ── LGBM ──
def train_lgbm(hi_tr, bids, asym=2.8):
    xs,ys=[],[]
    for b in bids:
        x,obs=make_tab(hi_tr[b]); xs.append(x); ys.append(true_rul(len(hi_tr[b]),obs))
    X,Y=np.concatenate(xs),np.concatenate(ys)
    aw=asym
    def obj(yp,ds):
        d=ds.get_label()-yp; w=np.where(d<0,aw,1.0); return -d*w, np.ones_like(d)*w
    return lgb.train({"num_leaves":15,"learning_rate":0.04,"min_child_samples":5,
        "feature_fraction":0.9,"bagging_fraction":0.9,"bagging_freq":1,"verbose":-1,
        "objective":obj}, lgb.Dataset(X,label=Y), num_boost_round=260)

def pred_lgbm(m, hi, start=0):
    x,obs=make_tab(hi,start); return obs, np.maximum(m.predict(x), MIN_RUL)

# ── LGBM variants ──
def train_lgbm_asym15(hi_tr, bids):
    return train_lgbm(hi_tr, bids, asym=1.5)

def train_lgbm_hifeat(hi_tr, bids):
    xs,ys=[],[]
    for b in bids:
        x,obs=make_tabular_hifeat(hi_tr[b]); xs.append(x); ys.append(true_rul(len(hi_tr[b]),obs))
    X,Y=np.concatenate(xs),np.concatenate(ys)
    def obj(yp,ds):
        d=ds.get_label()-yp; w=np.where(d<0,2.8,1.0); return -d*w, np.ones_like(d)*w
    return lgb.train({"num_leaves":15,"learning_rate":0.04,"min_child_samples":5,
        "feature_fraction":0.9,"bagging_fraction":0.9,"bagging_freq":1,"verbose":-1,
        "objective":obj}, lgb.Dataset(X,label=Y), num_boost_round=260)

def pred_lgbm_hifeat(m, hi, start=0):
    x,obs=make_tabular_hifeat(hi, seq=10); return obs, np.maximum(m.predict(x), MIN_RUL)

def train_ridge_model(hi_tr, bids):
    xs,ys=[],[]
    for b in bids:
        x,obs=make_tab(hi_tr[b]); xs.append(x); ys.append(true_rul(len(hi_tr[b]),obs))
    return train_ridge(np.concatenate(xs), np.concatenate(ys))

def pred_ridge(m, hi, start=0):
    x,obs=make_tab(hi,start); return obs, predict_ridge(m, x)

# ── Torch training ──
def train_nn(x_tr, y_tr, scale, ModelClass, seed, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    m=ModelClass().to(dev)
    ds=TensorDataset(torch.tensor(x_tr,dtype=torch.float32),torch.tensor(y_tr/scale,dtype=torch.float32))
    dl=DataLoader(ds,batch_size=min(32,len(ds)),shuffle=True)
    opt=torch.optim.AdamW(m.parameters(),lr=0.003,weight_decay=1e-4)
    lf=nn.SmoothL1Loss(); best_s,best_l,pat=None,np.inf,0
    for _ in range(160):
        m.train(); ls=[]
        for xb,yb in dl:
            xb,yb=xb.to(dev),yb.to(dev)
            opt.zero_grad(); l=lf(m(xb),yb); l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); ls.append(float(l.item()))
        c=float(np.mean(ls))
        if c<best_l-1e-5: best_l,pat=c,0; best_s={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        else:
            pat+=1
            if pat>=18: break
    if best_s: m.load_state_dict(best_s)
    return m

def train_nn_ens(hi_tr, bids, ModelClass, dev, seeds=SEEDS):
    xs,ys=[],[]
    for b in bids:
        x,obs=make_seq(hi_tr[b]); xs.append(x); ys.append(true_rul(len(hi_tr[b]),obs))
    X,Y=np.concatenate(xs),np.concatenate(ys)
    sc=float(max(Y.max(),1.0))
    ms=[train_nn(X,Y,sc,ModelClass,s,dev) for s in seeds]
    return ms,sc

def pred_nn(models,scale,hi,start,dev):
    x,obs=make_seq(hi,start)
    xt=torch.tensor(x,dtype=torch.float32).to(dev); ps=[]
    for m in models:
        m.eval()
        with torch.no_grad(): ps.append(np.maximum(m(xt).cpu().numpy()*scale,MIN_RUL))
    return obs, np.median(ps,axis=0)

# ── Original TCN (baseline) ──
class TCNOrig(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(4,32,3,padding=1),nn.ReLU(),nn.Dropout(0.10),
            nn.Conv1d(32,32,3,dilation=2,padding=2),nn.ReLU(),nn.Dropout(0.10),
            nn.Conv1d(32,32,3,dilation=4,padding=4),nn.ReLU())
        self.fc=nn.Sequential(nn.Linear(32,24),nn.ReLU(),nn.Linear(24,1))
    def forward(self,x):
        z=self.net(x.transpose(1,2)); return self.fc(z[:,:,-1]).squeeze(-1)

# ── DTW original ──
def seg_dist(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return (0.25*abs(a[-1]-b[-1])/0.25+0.20*abs(a.mean()-b.mean())/0.25
            +0.20*abs((a[-1]-a[0])-(b[-1]-b[0]))/0.25
            +0.15*abs(slp(a)-slp(b))/0.03+0.20*float(np.mean(np.abs(mnorm(a)-mnorm(b)))))

def pred_dtw(hi_tr,bids,hi_t,ml=18,k=6):
    hi_t=np.asarray(hi_t,float); obs=np.arange(SEQ,len(hi_t)); ps=[]
    for o in obs:
        l=min(ml,o); s=hi_t[o-l:o]; cs=[]
        for b in bids:
            hi=np.asarray(hi_tr[b],float)
            for e in range(l,len(hi)):
                cs.append((seg_dist(s,hi[e-l:e]),max(len(hi)-e,MIN_RUL)))
        top=sorted(cs,key=lambda c:c[0])[:k]
        wt=np.asarray([1/(d+EPS) for d,p in top])
        pv=np.asarray([p for d,p in top])
        ps.append(float(np.average(pv,weights=wt)))
    return obs,np.asarray(ps)

def estimate_start(hi_tr,hi_t,bids):
    t=np.asarray(hi_t,float); l=min(MATCH_LEN,len(t)); s=t[:l]; cs=[]
    for b in bids:
        hi=np.asarray(hi_tr[b],float)
        for st in range(0,max(1,len(hi)-l-3)):
            cs.append((seg_dist(s,hi[st:st+l]),st,b))
    if not cs: return 0
    top=sorted(cs,key=lambda c:c[0])[:8]
    pos=np.asarray([s for d,s,b in top],float)
    wt=np.asarray([1/(d+EPS) for d,s,b in top])
    est=int(round(np.average(pos,weights=wt)))
    g=float(t[-1]-t[0]); est=max(est,int(round(np.clip((g-0.25)/0.35,0,1)*25)))
    return int(np.clip(est,0,int(MEAN_LIFE)))

# ── Calibration ──
def calibrate(pred_bb, meta, lo=0.50, hi=1.80, step=0.01):
    best_cf,best_sc=1.0,-np.inf
    for cf in np.arange(lo,hi,step):
        sc=float(np.mean([sc_curve(meta[b]["N"],meta[b]["obs"],np.asarray(pred_bb[b])*cf) for b in BEARINGS]))
        if sc>best_sc: best_sc,best_cf=sc,float(cf)
    return best_cf,best_sc

# ── Ensemble strategies ──
def ens_capped_bidir(base, others_cal, alphas_up, caps_up, alphas_dn, caps_dn):
    """Bidirectional capped: upside AND downside from base."""
    out = np.asarray(base, float).copy()
    ref = np.maximum(out, MIN_RUL)
    for p, au, cu, ad, cd in zip(others_cal, alphas_up, caps_up, alphas_dn, caps_dn):
        p = np.asarray(p, float)
        up = au * np.clip(p - out, 0, (cu-1)*ref)
        dn = ad * np.clip(out - p, 0, (cd-1)*ref)
        out = out + up - dn
    return np.maximum(out, MIN_RUL)

def ens_trimmed_mean(preds_list, trim=1):
    """Trimmed mean: remove top/bottom trim predictions."""
    arr = np.sort(np.array(preds_list), axis=0)
    if len(arr) > 2*trim:
        return np.maximum(arr[trim:-trim].mean(axis=0), MIN_RUL)
    return np.maximum(arr.mean(axis=0), MIN_RUL)

def ens_phase(hi_vals, preds_dict, weights_early, weights_mid, weights_late):
    """Phase-based: different weights by HI level."""
    n = len(list(preds_dict.values())[0])
    out = np.zeros(n)
    for i in range(n):
        h = hi_vals[i] if i < len(hi_vals) else hi_vals[-1]
        w = weights_early if h < 0.3 else (weights_mid if h < 0.6 else weights_late)
        total = sum(w.values()) + EPS
        out[i] = sum(preds_dict[k][i] * w.get(k, 0) / total for k in preds_dict)
    return np.maximum(out, MIN_RUL)

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=== Exp-K: Model Diversity + Ensemble Strategies ===")
    hi_tr = load_hi("train"); hi_te = load_hi("test")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}\n")

    # ── Phase 1: LOOCV all models ──
    MODEL_NAMES = ["dtw","dtw_exp","dtw_adk","lgbm","lgbm_a15","lgbm_hif","ridge",
                   "tcn","tcn_res","transformer","bilstm"]
    raw = {m: {} for m in MODEL_NAMES}
    meta = {}

    for tb in BEARINGS:
        trb = [b for b in BEARINGS if b != tb]
        n = len(hi_tr[tb])
        print(f"[LOOCV] B{tb} held out (n={n})")

        # DTW variants
        obs, p = pred_dtw(hi_tr, trb, hi_tr[tb])
        raw["dtw"][tb] = p
        _, p2 = predict_dtw_exp(hi_tr, trb, hi_tr[tb])
        raw["dtw_exp"][tb] = p2
        _, p3 = predict_dtw_adaptive_k(hi_tr, trb, hi_tr[tb])
        raw["dtw_adk"][tb] = p3

        # LGBM variants
        m = train_lgbm(hi_tr, trb); _, p = pred_lgbm(m, hi_tr[tb])
        raw["lgbm"][tb] = p
        m2 = train_lgbm_asym15(hi_tr, trb); _, p2 = pred_lgbm(m2, hi_tr[tb])
        raw["lgbm_a15"][tb] = p2
        m3 = train_lgbm_hifeat(hi_tr, trb); _, p3 = pred_lgbm_hifeat(m3, hi_tr[tb])
        raw["lgbm_hif"][tb] = p3

        # Ridge
        rm = train_ridge_model(hi_tr, trb); _, pr = pred_ridge(rm, hi_tr[tb])
        raw["ridge"][tb] = pr

        # NN models
        for name, Cls in [("tcn",TCNOrig),("tcn_res",TCNRes),("transformer",MiniTransformer),("bilstm",BiLSTM)]:
            print(f"  {name}...")
            ms, sc = train_nn_ens(hi_tr, trb, Cls, dev)
            _, pp = pred_nn(ms, sc, hi_tr[tb], 0, dev)
            raw[name][tb] = pp

        meta[tb] = {"N": n, "obs": obs}
        scores = " ".join(f"{m}={sc_curve(n,obs,raw[m][tb]):.4f}" for m in MODEL_NAMES)
        print(f"  {scores}")

    # ── Calibrate all ──
    print("\n[Calibration]")
    cfs = {}
    cal = {}
    rows = []
    for m in MODEL_NAMES:
        cf, sc = calibrate(raw[m], meta)
        cfs[m] = cf
        cal[m] = {b: raw[m][b]*cf for b in BEARINGS}
        per_b = {b: sc_curve(meta[b]["N"], meta[b]["obs"], cal[m][b]) for b in BEARINGS}
        rows.append({"model":m, "cf":round(cf,2), "cal_score":round(sc,4),
                      **{f"B{b}":round(per_b[b],4) for b in BEARINGS}})
        print(f"  {m}: cf={cf:.2f} score={sc:.4f}")

    cal_df = pd.DataFrame(rows).sort_values("cal_score", ascending=False)
    cal_df.to_csv(OUT/"model_calibration.csv", index=False)
    print(f"\n{cal_df.to_string(index=False)}")

    # ── Phase 2: Ensemble strategies ──
    print("\n" + "="*60)
    print("[Ensemble Strategies]")
    ens_results = []

    # Strategy 0: Exp-J baseline reproduction (DTW + LGBM/TCN upside only)
    for al,cl,at,ct in [(0.4,2.0,1.0,2.0)]:
        blend = {}
        for b in BEARINGS:
            base = cal["dtw"][b]
            ref = np.maximum(base, MIN_RUL)
            lu = al*np.clip(cal["lgbm"][b]-base, 0, (cl-1)*ref)
            tu = at*np.clip(cal["tcn"][b]-base, 0, (ct-1)*ref)
            blend[b] = np.maximum(base+lu+tu, MIN_RUL)
        cf, sc = calibrate(blend, meta)
        ens_results.append({"strategy":"ExpJ_repro", "params":f"al={al},cl={cl},at={at},ct={ct}",
                            "cf":cf, "overall":round(sc,4)})
        print(f"  ExpJ_repro: {sc:.4f}")

    # Strategy A: Bidirectional Capped (DTW base, all models bidir)
    print("\n  Strategy A: Bidirectional Capped ...")
    best_a = -np.inf
    for al_u, cl_u, al_d, cl_d in product([0.2,0.4,0.6],[1.5,2.0],[0.1,0.2,0.3],[1.2,1.3]):
        blend = {}
        for b in BEARINGS:
            base = cal["dtw"][b]
            ref = np.maximum(base, MIN_RUL)
            out = base.copy()
            for mk in ["lgbm","tcn","lgbm_a15","tcn_res"]:
                if mk not in cal: continue
                p = cal[mk][b]
                out = out + al_u*np.clip(p-out,0,(cl_u-1)*ref) - al_d*np.clip(out-p,0,(cl_d-1)*ref)
            blend[b] = np.maximum(out, MIN_RUL)
        cf, sc = calibrate(blend, meta)
        if sc > best_a:
            best_a = sc
            best_a_params = f"au={al_u},cu={cl_u},ad={al_d},cd={cl_d}"
    ens_results.append({"strategy":"bidir_capped","params":best_a_params,"cf":0,"overall":round(best_a,4)})
    print(f"  Best bidir_capped: {best_a:.4f} ({best_a_params})")

    # Strategy B: Trimmed Mean (all models)
    print("\n  Strategy B: Trimmed Mean ...")
    for trim in [1, 2]:
        blend = {}
        for b in BEARINGS:
            plist = [cal[m][b] for m in MODEL_NAMES if m in cal]
            blend[b] = ens_trimmed_mean(plist, trim=trim)
        cf, sc = calibrate(blend, meta)
        ens_results.append({"strategy":f"trimmed_mean_t{trim}","params":f"trim={trim},n_models={len(MODEL_NAMES)}",
                            "cf":cf,"overall":round(sc,4)})
        print(f"  trimmed_mean trim={trim}: {sc:.4f}")

    # Strategy C: Phase-based (DTW-heavy late, LGBM-heavy early)
    print("\n  Strategy C: Phase-based ...")
    best_c = -np.inf
    for dw_e, dw_l in product([0.2,0.3,0.4],[0.6,0.7,0.8]):
        we = {"dtw":dw_e,"lgbm":0.5-dw_e/2,"tcn":0.5-dw_e/2}
        wm = {"dtw":0.4,"lgbm":0.35,"tcn":0.25}
        wl = {"dtw":dw_l,"lgbm":(1-dw_l)/2,"tcn":(1-dw_l)/2}
        blend = {}
        for b in BEARINGS:
            hi_obs = hi_tr[b][SEQ:]  # HI values at obs points
            pdict = {k: cal[k][b] for k in ["dtw","lgbm","tcn"]}
            blend[b] = ens_phase(hi_obs, pdict, we, wm, wl)
        cf, sc = calibrate(blend, meta)
        if sc > best_c:
            best_c = sc; best_c_params = f"dw_e={dw_e},dw_l={dw_l}"
    ens_results.append({"strategy":"phase_based","params":best_c_params,"cf":0,"overall":round(best_c,4)})
    print(f"  Best phase_based: {best_c:.4f} ({best_c_params})")

    # Strategy D: Extended capped upside (DTW base + best new models)
    print("\n  Strategy D: Extended Capped Upside (more models) ...")
    best_d = -np.inf
    # Use top calibrated non-DTW models
    top_models = cal_df[~cal_df["model"].str.startswith("dtw")].head(5)["model"].tolist()
    print(f"    Using: dtw (base) + {top_models}")
    for n_models in [2,3,4]:
        use = top_models[:n_models]
        for alpha in [0.3, 0.4, 0.5, 0.6]:
            for cap in [1.3, 1.5, 2.0]:
                blend = {}
                for b in BEARINGS:
                    base = cal["dtw"][b]; ref = np.maximum(base, MIN_RUL)
                    out = base.copy()
                    for mk in use:
                        out = out + alpha*np.clip(cal[mk][b]-out, 0, (cap-1)*ref)
                    blend[b] = np.maximum(out, MIN_RUL)
                cf, sc = calibrate(blend, meta)
                if sc > best_d:
                    best_d = sc; best_d_params = f"n={n_models},a={alpha},c={cap},models={use}"
    ens_results.append({"strategy":"ext_capped","params":str(best_d_params)[:80],"cf":0,"overall":round(best_d,4)})
    print(f"  Best ext_capped: {best_d:.4f}")

    # ── Results summary ──
    ens_df = pd.DataFrame(ens_results).sort_values("overall", ascending=False)
    ens_df.to_csv(OUT/"ensemble_comparison.csv", index=False)
    print(f"\n{'='*60}\n[FINAL SUMMARY]\n{ens_df.to_string(index=False)}")

    # ── Plot model comparison ──
    fig, axes = plt.subplots(2,2, figsize=(14,10))
    fig.suptitle("Exp-K: Per-Bearing Calibrated Scores", fontsize=12)
    for ax, b in zip(axes.flatten(), BEARINGS):
        scores = {m: sc_curve(meta[b]["N"], meta[b]["obs"], cal[m][b]) for m in MODEL_NAMES}
        colors = plt.cm.tab20(np.linspace(0,1,len(scores)))
        bars = ax.barh(list(scores.keys()), list(scores.values()), color=colors)
        ax.set_title(f"B{b}"); ax.set_xlim(0, 0.8); ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout(); plt.savefig(OUT/"model_comparison.png", dpi=150); plt.close()

    # ── Plot ensemble comparison ──
    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(ens_df["strategy"], ens_df["overall"], color="steelblue", alpha=0.8)
    ax.axvline(0.6064, color="r", ls="--", label="Exp-J=0.6064")
    ax.axvline(0.5685, color="gray", ls=":", label="Exp-E=0.5685")
    ax.set_xlabel("Overall LOOCV Score"); ax.set_title("Ensemble Strategy Comparison")
    ax.legend(); ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout(); plt.savefig(OUT/"ensemble_comparison.png", dpi=150); plt.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s → {OUT}")

if __name__ == "__main__":
    main()
