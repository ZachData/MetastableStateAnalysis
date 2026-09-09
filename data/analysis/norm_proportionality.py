import sys; sys.path.insert(0,'/run/media/system/WDS_500/Mets')
import numpy as np, torch, json
from scipy import stats
from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import (D_MODEL,N_HEADS,D_HEAD,N_REP,VOCAB_LO,
                                            VOCAB_HI,EVAL_SEED,ov_factors,write_ov)
m,_=load_causal_lm("pythia-410m-step16000"); m.eval()
rng=np.random.default_rng(EVAL_SEED)
ids=torch.tensor(np.stack([np.concatenate([s,s]) for s in
    (rng.integers(VOCAB_LO,VOCAB_HI,size=N_REP) for _ in range(16))]),dtype=torch.long)
@torch.no_grad()
def meas():
    lc,lz,rn=[],[],[]
    for i in range(0,len(ids),16):
        o=m(ids[i:i+16],output_hidden_states=True)
        lg=o.logits[:,:-1,:].float(); t=ids[i:i+16,1:]
        z=torch.logsumexp(lg,dim=-1); c=lg.gather(-1,t.unsqueeze(-1)).squeeze(-1)
        lc.append(c[:,N_REP-1:].numpy()); lz.append(z[:,N_REP-1:].numpy())
        rn.append(o.hidden_states[-1].float().norm(dim=-1)[:,N_REP-1:].numpy()); del o,lg,z,c
    return (float(np.concatenate(lc).mean()),float(np.concatenate(lz).mean()),
            float(np.concatenate(rn).mean()))
c0,z0,r0=meas(); Z=(np.zeros((D_MODEL,1)),np.zeros((1,D_MODEL)))
# all 384 OV norms, then sample across the range + force-include the named heads
norms={}
for L in range(24):
    for h in range(N_HEADS):
        a,b=ov_factors(m,L,h); norms[(L,h)]=float(np.linalg.norm(a@b))
keys=sorted(norms,key=lambda k:norms[k])
sample=[keys[int(i)] for i in np.linspace(0,len(keys)-1,34)]
for k in [(7,8),(5,2),(11,14),(2,10)]:
    if k not in sample: sample.append(k)
print(f"baseline resid norm {r0:.2f}, logsumexp {z0:.3f}")
print(f"OV norm range over 384 heads: {norms[keys[0]]:.3f} .. {norms[keys[-1]]:.3f}\n")
print(f"{'head':>8} {'||OV||_F':>9} {'d||resid||':>11} {'dNLL':>9} {'d logsumexp':>12}")
rows=[]
for (L,h) in sample:
    a,b=ov_factors(m,L,h); write_ov(m,L,h,*Z); c,z,r=meas(); write_ov(m,L,h,a,b)
    rows.append({"head":f"L{L}H{h}","norm":norms[(L,h)],"d_resid":r-r0,
                 "dnll":-(c-z)-(-(c0-z0)),"d_lse":z-z0})
    print(f"{rows[-1]['head']:>8} {rows[-1]['norm']:>9.3f} {rows[-1]['d_resid']:>+11.4f} "
          f"{rows[-1]['dnll']:>+9.4f} {rows[-1]['d_lse']:>+12.4f}",flush=True)
x=np.array([r["norm"] for r in rows]); y=np.array([abs(r["d_resid"]) for r in rows])
n=np.array([r["dnll"] for r in rows])
print(f"\nSpearman ||OV||_F vs |d||resid|||  rho={stats.spearmanr(x,y).statistic:+.3f}")
print(f"Spearman ||OV||_F vs dNLL          rho={stats.spearmanr(x,n).statistic:+.3f}")
# residual of the named heads after regressing d_resid on norm
sl,ic,rv,_,_=stats.linregress(x,y)
print(f"linear fit |d_resid| = {sl:.4f}*||OV|| + {ic:.4f}   r^2={rv**2:.3f}")
print(f"\n{'head':>8} {'||OV||':>8} {'|d_resid|':>10} {'predicted':>10} {'excess':>9}")
for r in rows:
    if r["head"] in ("L7H8","L5H2","L11H14","L2H10"):
        pred=sl*r["norm"]+ic
        print(f"{r['head']:>8} {r['norm']:>8.3f} {abs(r['d_resid']):>10.4f} "
              f"{pred:>10.4f} {abs(r['d_resid'])-pred:>+9.4f}")
json.dump(rows,open("/run/media/system/WDS_500/Mets/data/analysis/norm_proportionality.json","w"),indent=2)
