"""
Exp-K Phase 2: Fine-tune Extended Capped Upside
Best from Phase 1: DTW base + BiLSTM + TCN-Res + Transformer, alpha=0.4, cap=2.0
Now: per-model alpha/cap fine-tuning + per-bearing analysis
"""
from pathlib import Path
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import sys
sys.path.insert(0, str(Path(__file__).parent))
from new_models import TCNRes, MiniTransformer, BiLSTM

BASE   = Path("/data/home/ksphm/2026-challenge-KSPHM")
HI_DIR = BASE / "User/SP/05-26/V1b/output"
OUT    = Path(__file__).parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

BEARINGS=[1,2,3,4]; TEST_IDS=[1,2,3,4,5,6]
SEQ=10; MATCH_LEN=18; INTERVAL=600
MEAN_LIFE=116.5; HI_EOL=0.75; MIN_RUL=1.0; EPS=1e-8
SEEDS_LOO=[42]; SEEDS_FINAL=[42,123,777]
EOL={1:126,2:114,3:89,4:137}

def load_hi(k="train"):
    if k=="train": return {b:pd.read_csv(HI_DIR/f"HI_Bearing{b}.csv")["HI"].values.astype(float) for b in BEARINGS}
    return {t:pd.read_csv(HI_DIR/"test"/f"HI_Test{t}.csv")["HI"].values.astype(float) for t in TEST_IDS}

def mnorm(x):
    x=np.asarray(x,float); return (x-x.min())/(x.max()-x.min()+EPS)
def slp(x):
    x=np.asarray(x,float); return float(np.polyfit(np.arange(len(x)),x,1)[0]) if len(x)>=2 else 0.0
def comp_sc(yt,yp):
    if yt<=0: return np.nan
    er=100*(yt-yp)/yt
    return np.exp(-np.log(0.5)*er/20) if er<=0 else np.exp(np.log(0.5)*er/50)
def true_rul(n,obs): return np.maximum(n-np.asarray(obs,float),1.0)
def sc_curve(n,obs,p):
    y = true_rul(n, obs)
    p = np.asarray(p, float)
    mask = y > 0
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return 0.0
    er = 100.0 * (y - p) / y
    scores = np.where(er <= 0, np.exp(-np.log(0.5) * er / 20.0), np.exp(np.log(0.5) * er / 50.0))
    return float(np.nanmean(scores))
def err_summ(n,obs,p):
    y=true_rul(n,obs); p=np.asarray(p,float)
    er=[100*(t-pp)/t for t,pp in zip(y,p) if t>0]
    return {"score":sc_curve(n,obs,p),"mean_er":float(np.nanmean(er))}
def make_tab(hi):
    hi=np.asarray(hi,float); hi0=float(hi[0]); x,obs=[],[]
    for i in range(SEQ,len(hi)):
        w=hi[i-SEQ:i]; wn=mnorm(w); hf=float(np.clip(w[-1]/HI_EOL,0,2))
        x.append(list(wn)+[slp(wn),float(w[-1]),float(w.mean()),float(w.max()),
                 float(w.min()),float(w.std()),slp(w),float(w[-1]-w[0]),
                 float(w[-1]-hi0),hf,float(w[-1]*hf)])
        obs.append(i)
    return np.asarray(x),np.asarray(obs)
def make_seq(hi):
    hi=np.asarray(hi,float); hi0=float(hi[0]); x,obs=[],[]
    for i in range(len(hi)-SEQ):
        w=hi[i:i+SEQ]; of=np.clip(w/HI_EOL,0,2)
        x.append(np.stack([mnorm(w),w.copy(),w-hi0,of],axis=1)); obs.append(i+SEQ)
    return np.asarray(x),np.asarray(obs)

# LGBM
def train_lgbm(hi_tr,bids):
    xs,ys=[],[]
    for b in bids:
        x,obs=make_tab(hi_tr[b]); xs.append(x); ys.append(true_rul(len(hi_tr[b]),obs))
    X,Y=np.concatenate(xs),np.concatenate(ys)
    def obj(yp,ds): d=ds.get_label()-yp; w=np.where(d<0,2.8,1.0); return -d*w,np.ones_like(d)*w
    return lgb.train({"num_leaves":15,"learning_rate":0.04,"min_child_samples":5,
        "feature_fraction":0.9,"bagging_fraction":0.9,"bagging_freq":1,"verbose":-1,
        "objective":obj},lgb.Dataset(X,label=Y),num_boost_round=260)
def pred_lgbm(m,hi): x,obs=make_tab(hi); return obs,np.maximum(m.predict(x),MIN_RUL)

# NN
class AsymmetricHuberLoss(nn.Module):
    def __init__(self, over_penalty=2.8, delta=1.0):
        super().__init__()
        self.over_penalty = over_penalty
        self.delta = delta
    def forward(self, pred, target):
        diff = target - pred
        abs_diff = torch.abs(diff)
        huber = torch.where(abs_diff < self.delta, 0.5 * (diff ** 2), self.delta * (abs_diff - 0.5 * self.delta))
        loss = torch.where(diff < 0, self.over_penalty * huber, huber)
        return loss.mean()

class TCNOrig(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(4,32,3,padding=1),nn.ReLU(),nn.Dropout(0.10),
            nn.Conv1d(32,32,3,dilation=2,padding=2),nn.ReLU(),nn.Dropout(0.10),
            nn.Conv1d(32,32,3,dilation=4,padding=4),nn.ReLU())
        self.fc=nn.Sequential(nn.Linear(32,24),nn.ReLU(),nn.Linear(24,1))
    def forward(self,x): z=self.net(x.transpose(1,2)); return self.fc(z[:,:,-1]).squeeze(-1)

def train_nn(x_tr,y_tr,scale,Cls,seed,dev):
    torch.manual_seed(seed); np.random.seed(seed); m=Cls().to(dev)
    ds=TensorDataset(torch.tensor(x_tr,dtype=torch.float32),torch.tensor(y_tr/scale,dtype=torch.float32))
    dl=DataLoader(ds,batch_size=min(32,len(ds)),shuffle=True)
    opt=torch.optim.AdamW(m.parameters(),lr=0.003,weight_decay=1e-4); lf=AsymmetricHuberLoss(over_penalty=2.8)
    bs,bl,pat=None,np.inf,0
    for _ in range(160):
        m.train(); ls=[]
        for xb,yb in dl:
            xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad(); l=lf(m(xb),yb); l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); ls.append(float(l.item()))
        c=float(np.mean(ls))
        if c<bl-1e-5: bl,pat=c,0; bs={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        else:
            pat+=1
            if pat>=18: break
    if bs: m.load_state_dict(bs)
    return m
def train_ens(hi_tr,bids,Cls,dev,seeds):
    xs,ys=[],[]
    for b in bids: x,obs=make_seq(hi_tr[b]); xs.append(x); ys.append(true_rul(len(hi_tr[b]),obs))
    X,Y=np.concatenate(xs),np.concatenate(ys); sc=float(max(Y.max(),1.0))
    return [train_nn(X,Y,sc,Cls,s,dev) for s in seeds],sc
def pred_nn(ms,sc,hi,dev):
    x,obs=make_seq(hi); xt=torch.tensor(x,dtype=torch.float32).to(dev); ps=[]
    for m in ms:
        m.eval()
        with torch.no_grad(): ps.append(np.maximum(m(xt).cpu().numpy()*sc,MIN_RUL))
    return obs,np.median(ps,axis=0)

# DTW
def seg_dist(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return (0.25*abs(a[-1]-b[-1])/0.25+0.20*abs(a.mean()-b.mean())/0.25
            +0.20*abs((a[-1]-a[0])-(b[-1]-b[0]))/0.25
            +0.15*abs(slp(a)-slp(b))/0.03+0.20*float(np.mean(np.abs(mnorm(a)-mnorm(b)))))
def pred_dtw(hi_tr,bids,hi_t):
    hi_t=np.asarray(hi_t,float); obs=np.arange(SEQ,len(hi_t)); ps=[]
    for o in obs:
        l=min(MATCH_LEN,o); s=hi_t[o-l:o]; cs=[]
        for b in bids:
            hi=np.asarray(hi_tr[b],float)
            for e in range(l,len(hi)): cs.append((seg_dist(s,hi[e-l:e]),max(len(hi)-e,MIN_RUL)))
        top=sorted(cs,key=lambda c:c[0])[:6]
        wt=np.asarray([1/(d+EPS) for d,p in top]); pv=np.asarray([p for d,p in top])
        ps.append(float(np.average(pv,weights=wt)))
    return obs,np.asarray(ps)
def estimate_start(hi_tr,hi_t,bids):
    t=np.asarray(hi_t,float); l=min(MATCH_LEN,len(t)); s=t[:l]; cs=[]
    for b in bids:
        hi=np.asarray(hi_tr[b],float)
        for st in range(0,max(1,len(hi)-l-3)): cs.append((seg_dist(s,hi[st:st+l]),st,b))
    if not cs: return 0
    top=sorted(cs,key=lambda c:c[0])[:8]
    pos=np.asarray([s for d,s,b in top],float); wt=np.asarray([1/(d+EPS) for d,s,b in top])
    est=int(round(np.average(pos,weights=wt))); g=float(t[-1]-t[0])
    est=max(est,int(round(np.clip((g-0.25)/0.35,0,1)*25)))
    return int(np.clip(est,0,int(MEAN_LIFE)))
def calibrate(pbb,meta,bids,lo=0.50,hi=1.80,st=0.01,safety_margin=1.0):
    bc,bs=1.0,-np.inf
    for cf in np.arange(lo,hi,st):
        sc=float(np.mean([sc_curve(meta[b]["N"],meta[b]["obs"],np.asarray(pbb[b])*cf) for b in bids]))
        if sc>bs: bs,bc=sc,float(cf)
    return bc * safety_margin,bs

def main():
    t0=time.time()
    print("=== Exp-K Phase 2: Fine-Tune Extended Capped (Strictly Leak-free) ===")
    hi_tr=load_hi("train"); hi_te=load_hi("test")
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}\n")

    # Models: dtw, lgbm, tcn, tcn_res, transformer, bilstm
    MODELS = {"tcn":TCNOrig,"tcn_res":TCNRes,"transformer":MiniTransformer,"bilstm":BiLSTM}
    raw={m:{} for m in ["dtw","lgbm"]+list(MODELS.keys())}
    meta={}

    for tb in BEARINGS:
        trb=[b for b in BEARINGS if b!=tb]; n=len(hi_tr[tb])
        print(f"[LOOCV] B{tb}")
        obs,p=pred_dtw(hi_tr,trb,hi_tr[tb]); raw["dtw"][tb]=p
        m=train_lgbm(hi_tr,trb); _,p=pred_lgbm(m,hi_tr[tb]); raw["lgbm"][tb]=p
        for name,Cls in MODELS.items():
            ms,sc=train_ens(hi_tr,trb,Cls,dev,SEEDS_LOO); _,pp=pred_nn(ms,sc,hi_tr[tb],dev)
            raw[name][tb]=pp
        meta[tb]={"N":n,"obs":obs}

    # Leak-free LOOCV Calibration search
    local_cfs = {m: {} for m in raw}
    cal = {m: {} for m in raw}
    print("\n[Leak-Free Calibration factor selection for single models]")
    for m in raw:
        print(f"  Model: {m}")
        for tb in BEARINGS:
            trb = [b for b in BEARINGS if b!=tb]
            cf_tb_m, _ = calibrate(raw[m], meta, trb)
            local_cfs[m][tb] = cf_tb_m
            cal[m][tb] = raw[m][tb] * cf_tb_m
        # Calculate full out-of-fold LOOCV score with local calibration
        sc_m = float(np.mean([sc_curve(meta[b]["N"], meta[b]["obs"], cal[m][b]) for b in BEARINGS]))
        print(f"    Out-of-fold Leak-free LOOCV score: {sc_m:.4f}")

    # Fine-grained sweep: per-model alpha, shared cap, safety margin
    print("\n[Fine-Tune Sweep] DTW base + {bilstm, tcn_res, transformer} (100% Leak-free)")
    UP_MODELS = ["bilstm","tcn_res","transformer"]
    
    configs = []
    for a_bi in [0.4, 0.6]:
        for a_tr in [0.4, 0.6]:
            for a_tcnr in [0.4, 0.6]:
                for cap in [1.5, 1.8, 2.0]:
                    for margin in [0.90, 0.93, 0.96]:
                        alphas = {"bilstm":a_bi,"tcn_res":a_tcnr,"transformer":a_tr}
                        configs.append((UP_MODELS, alphas, cap, margin))

    print(f"  Total configs: {len(configs)}")
    best_ov, best_cfg = -np.inf, None
    
    for use, alphas, cap, margin in configs:
        fold_scores = []
        fold_ers = {b: 0.0 for b in BEARINGS}
        blend_tb_all = {}
        cf_tb_ens_all = {}
        
        for tb in BEARINGS:
            trb = [b for b in BEARINGS if b!=tb]
            
            # 1. Compute blended predictions on trb using local model cfs
            blend_trb = {}
            for b in trb:
                cf_b_m = {m: local_cfs[m][tb] for m in ["dtw"] + use}
                base = raw["dtw"][b] * cf_b_m["dtw"]
                ref = np.maximum(base, MIN_RUL)
                val = base.copy()
                for mk in use:
                    pp_cal = raw[mk][b] * cf_b_m[mk]
                    val = val + alphas[mk] * np.clip(pp_cal - base, 0, (cap-1)*ref)
                blend_trb[b] = np.maximum(val, MIN_RUL)
            
            # 2. Find best ensemble cf using trb strictly (leak-free)
            cf_tb_ens, _ = calibrate(blend_trb, meta, trb, safety_margin=margin)
            cf_tb_ens_all[tb] = cf_tb_ens
            
            # 3. Compute blended prediction on validation tb using local cfs
            base_tb = raw["dtw"][tb] * local_cfs["dtw"][tb]
            ref_tb = np.maximum(base_tb, MIN_RUL)
            val_tb = base_tb.copy()
            for mk in use:
                pp_cal_tb = raw[mk][tb] * local_cfs[mk][tb]
                val_tb = val_tb + alphas[mk] * np.clip(pp_cal_tb - base_tb, 0, (cap-1)*ref_tb)
            blend_tb = np.maximum(val_tb, MIN_RUL)
            blend_tb_all[tb] = blend_tb
            
            # 4. Score on validation tb
            sc_tb = sc_curve(meta[tb]["N"], meta[tb]["obs"], blend_tb * cf_tb_ens)
            fold_scores.append(sc_tb)
            fold_ers[tb] = err_summ(meta[tb]["N"], meta[tb]["obs"], blend_tb * cf_tb_ens)["mean_er"]
            
        sc = float(np.mean(fold_scores))
        if sc > best_ov:
            best_ov = sc
            best_cfg = (use, alphas, cap, margin, cf_tb_ens_all, blend_tb_all)
            per_b = {b: fold_scores[b-1] for b in BEARINGS}
            per_b_er = fold_ers

    use, alphas, cap, margin, cf_tb_ens_all, blend_tb_all = best_cfg
    print(f"\n[BEST] leak-free overall={best_ov:.4f} cap={cap} safety_margin={margin}")
    print(f"  models: {use}")
    print(f"  alphas: {alphas}")
    for b in BEARINGS:
        print(f"  B{b}: score={per_b[b]:.4f} mean_er={per_b_er[b]:.1f}% cf={cf_tb_ens_all[b]:.2f}")

    # Calculate global calibration factors using all 4 bearings for Test Inference
    print("\n[Global Calibration Factors for Test Inference]")
    global_cfs = {}
    for m in raw:
        cf_glob, _ = calibrate(raw[m], meta, BEARINGS)
        global_cfs[m] = cf_glob
        print(f"  {m}: global_cf={cf_glob:.2f}")

    # Compute global ensemble blend to find global ensemble cf
    global_blend = {}
    for b in BEARINGS:
        base = raw["dtw"][b] * global_cfs["dtw"]
        ref = np.maximum(base, MIN_RUL)
        val = base.copy()
        for mk in use:
            pp_cal = raw[mk][b] * global_cfs[mk]
            val = val + alphas[mk] * np.clip(pp_cal - base, 0, (cap-1)*ref)
        global_blend[b] = np.maximum(val, MIN_RUL)
    
    global_cf_ens, _ = calibrate(global_blend, meta, BEARINGS, safety_margin=margin)
    print(f"  Ensemble: global_cf={global_cf_ens:.2f}")

    # ── Test inference with 3 seeds ──
    print(f"\n[Test Inference] 3 seeds...")
    lgbm_m = train_lgbm(hi_tr, BEARINGS)
    nn_models = {}
    for name, Cls in MODELS.items():
        if name in use:
            print(f"  Training {name} (3 seeds)...")
            nn_models[name] = train_ens(hi_tr, BEARINGS, Cls, dev, SEEDS_FINAL)

    fig, axes = plt.subplots(2,3,figsize=(18,10))
    fig.suptitle(f"Exp-K Test: DTW+{'+'.join(use)} cap={cap} cf={global_cf_ens:.2f}", fontsize=10)
    test_rows = []

    for ax, tid in zip(axes.flatten(), TEST_IDS):
        hi = hi_te[tid]
        start = estimate_start(hi_tr, hi, BEARINGS)
        obs_d, p_dtw = pred_dtw(hi_tr, BEARINGS, hi)
        dtw_c = p_dtw * global_cfs["dtw"]
        
        base = dtw_c.copy()
        ref = np.maximum(base, MIN_RUL)
        preds_raw = {"dtw": dtw_c}
        
        for mk in use:
            if mk == "lgbm":
                _, pp = pred_lgbm(lgbm_m, hi)
            else:
                ms, sc = nn_models[mk]
                _, pp = pred_nn(ms, sc, hi, dev)
            pp_cal = pp * global_cfs[mk]
            preds_raw[mk] = pp_cal
            a = alphas[mk]
            base = base + a*np.clip(pp_cal-base, 0, (cap-1)*ref)
        
        final = np.maximum(base*global_cf_ens, MIN_RUL)
        hours = final * INTERVAL / 3600.0
        test_rows.append({"test":tid,"start":start,"hi_s":round(float(hi[0]),3),
                          "hi_e":round(float(hi[-1]),3),"rul_hr":round(float(hours[-1]),2)})
        print(f"  T{tid}: start={start} RUL={hours[-1]:.2f}hr")

        ax.plot(obs_d, final, "b-", lw=2, label="final")
        for k,v in preds_raw.items():
            ax.plot(obs_d, v[:len(obs_d)], lw=1, alpha=0.5, label=k)
        ax.set_title(f"T{tid} RUL={hours[-1]:.1f}hr"); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    plt.tight_layout(); plt.savefig(OUT/"test_rul_predictions_v2.png",dpi=150); plt.close()
    pd.DataFrame(test_rows).to_csv(OUT/"test_rul_results_v2.csv",index=False)

    # Save LOOCV summary
    summary = {"overall":round(best_ov,4),"cf":global_cf_ens,"cap":cap,"margin":margin,"models":str(use),"alphas":str(alphas)}
    for b in BEARINGS:
        summary[f"B{b}_score"] = round(per_b[b],4)
        summary[f"B{b}_er"] = round(per_b_er[b],1)
    pd.DataFrame([summary]).to_csv(OUT/"best_config_v2.csv",index=False)

    print(f"\nDone in {time.time()-t0:.0f}s → {OUT}")

if __name__=="__main__":
    main()
