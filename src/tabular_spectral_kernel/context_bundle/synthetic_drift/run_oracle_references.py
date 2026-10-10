"""Hindsight mixture diagnostics and full-state synthetic Bayes references.

No query labels enter deployable models. Oracle results are explicitly
information-advantaged diagnostics, with fixed support, banks and ridge.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar,brentq
from threadpoolctl import threadpool_limits
import run as R
import run_transformer as T
import run_optimized_fusion as O

ROOT=Path(__file__).parent
VARIANTS=['linear_simple','linear_stacked','linear_learned_psd','linear_transformer']


class MixtureCurve:
    """Evaluate the exact two-bank KRR curve using a generalized eigensystem."""
    def __init__(self,g,q,y,ridge):
        a=g[1]+len(y)*ridge*np.eye(len(y));delta=g[0]-g[1]
        self.e,self.v=eigh(delta,a)
        self.yp=self.v.T@y;self.op=self.v.T@np.ones(len(y))
        self.q0=q[1]@self.v;self.qd=(q[0]-q[1])@self.v

    def __call__(self,weights,index=None):
        w=np.atleast_1d(weights)
        inv=1/(1+w[:,None]*self.e)
        b=(inv*(self.yp*self.op)).sum(-1)/(inv*self.op**2).sum(-1)
        coef=(self.yp[None]-b[:,None]*self.op)*inv
        q0=self.q0 if index is None else self.q0[index:index+1]
        qd=self.qd if index is None else self.qd[index:index+1]
        return np.clip(b[:,None]+coef@q0.T+w[:,None]*(coef@qd.T),0,1)


def search_scalar(score,grid,values,extras=()):
    """Grid + every bracketed strict local minimum; no global certificate."""
    candidates=[(float(v),float(w)) for w,v in zip(grid,values)]
    candidates.extend((float(score(w)),float(w)) for w in extras)
    for i in range(1,len(grid)-1):
        if values[i]<=values[i-1] and values[i]<=values[i+1] and (values[i]<values[i-1] or values[i]<values[i+1]):
            r=minimize_scalar(score,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-10})
            candidates.append((float(r.fun),float(r.x)))
    return min(candidates)[1]


def oracle_predictions(curve,target,mean,known_weights,grid_size=501):
    grid=np.linspace(0,1,grid_size);p=curve(grid)
    cohort_loss=((p-target)**2).mean(-1)
    cw=search_scalar(lambda w:float(((curve(w)[0]-target)**2).mean()),grid,cohort_loss,known_weights.flatten())
    cohort=curve(cw)[0];n=len(target)
    outputs={'oracle_cohort_labels':cohort};weights={'oracle_cohort_labels':np.full(n,cw)}
    lows=[];highs=[]
    for i in range(n):
        pred=lambda w:float(curve(w,i)[0,0])
        extras=np.r_[known_weights[:,i],cw]
        lo=search_scalar(pred,grid,p[:,i],extras)
        hi=search_scalar(lambda w:-pred(w),grid,-p[:,i],extras)
        lows.append((lo,pred(lo)));highs.append((hi,pred(hi)))
    for name,t in [('oracle_query_labels',target),('oracle_query_mean',mean)]:
        ow=[];op=[]
        for i,z in enumerate(t):
            (wl,pl),(wh,ph)=lows[i],highs[i]
            if z<=pl:w=wl
            elif z>=ph:w=wh
            else:
                w=brentq(lambda w:float(curve(w,i)[0,0])-z,min(wl,wh),max(wl,wh),xtol=1e-12)
            ow.append(w);op.append(float(curve(w,i)[0,0]))
        outputs[name]=np.array(op);weights[name]=np.array(ow)
    assert np.mean((outputs['oracle_query_labels']-target)**2)<=np.mean((cohort-target)**2)+1e-10
    return outputs,weights,np.array([p for _,p in lows]),np.array([p for _,p in highs])


def execute(seed,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    hashes={};metrics=[];checks=[]
    def record(path):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return path
    for path in [Path(__file__),Path(R.__file__),Path(O.__file__),R.MODEL_SOURCE]:
        record(path);(out/(path.stem+'_snapshot.py')).write_bytes(path.read_bytes())
    src=ROOT/f'unemployment_results/seed{seed}';info=json.loads(record(src/'original_training.json').read_text());banks=[]
    for name in ['normal','downturn']:
        m=R.base.CanonicalSpectral(5,4,max_frequency=3.,device='cpu')
        m.load_state_dict(torch.load(record(src/f'{name}.pt'),weights_only=True,map_location='cpu'));m.eval();banks.append(m)
    for path,digest in json.loads((ROOT/'optimized_fusion_results/provenance.json').read_text()).items():
        runtime.verify_artifact(path,digest)
    record(ROOT/'optimized_fusion_results/protocol.json');record(ROOT/'optimized_fusion_results/provenance.json')
    for sc in T.C.SCENARIOS:
        with np.load(record(src/f'original_{sc}.npz')) as f:x,y,mean=f['x'],f['y'],f['mean']
        n=len(y);valid=np.arange(n)>=128;zs=[R.embedding(m,x) for m in banks];pred={};ww={}
        for v in VARIANTS:
            with np.load(record(ROOT/f'mlp_{v}_results/seed{seed}/{sc}.npz')) as f:
                pred[v]=f['transformer_krr'];ww[v]=f['transformer_alpha'][:,0]
        with np.load(record(ROOT/f'transformer_results/seed{seed}/{sc}.npz')) as f:
            pred['transformer_full']=f['transformer_krr'];ww['transformer_full']=f['transformer_alpha'][:,0]
        with np.load(record(ROOT/f'optimized_fusion_results/seed{seed}_{sc}.npz')) as f:
            for name in ['validation','convex_mkl']:pred[name]=f[name];ww[name]=f[name+'_normal_weight']
        for name in ['oracle_cohort_labels','oracle_query_labels','oracle_query_mean']:
            pred[name]=np.full(n,np.nan);ww[name]=np.full(n,np.nan)
        pred['true_conditional_mean']=mean.copy();lower=np.full(n,np.nan);upper=lower.copy()
        for origin in range(128,n,32):
            sl=slice(origin-128,origin);qs=slice(origin,origin+32)
            g=np.array([R.kernel(z[sl],z[sl],t) for z,t in zip(zs,info['temperatures'])])
            q=np.array([R.kernel(z[qs],z[sl],t) for z,t in zip(zs,info['temperatures'])])
            curve=MixtureCurve(g,q,y[sl],info['ridge'])
            known=np.array([ww[v][qs] for v in VARIANTS+['transformer_full','validation','convex_mkl']])
            # Numerical curve must match direct Cholesky at an interior weight.
            w=.371;p=R.ridge_predict(w*g[0]+(1-w)*g[1],w*q[0]+(1-w)*q[1],y[sl],info['ridge']).clip(0,1)
            error=float(np.max(np.abs(p-curve(w)[0])));assert error<1e-9
            for i,name in enumerate(VARIANTS+['transformer_full','validation','convex_mkl']):
                values=np.array([curve(known[i,j],j)[0,0] for j in range(len(p))])
                np.testing.assert_allclose(values,pred[name][qs],atol=1e-8,rtol=1e-8)
            if origin==128:
                for name in ['validation','convex_mkl']:
                    weight,_=O.optimize_weight(g,y[sl],info['ridge'],name)
                    assert abs(weight-ww[name][origin])<1e-6
            result,weights,lo,hi=oracle_predictions(curve,y[qs],mean[qs],known)
            for name,values in result.items():pred[name][qs]=values;ww[name][qs]=weights[name]
            lower[qs]=lo;upper[qs]=hi
            checks.append(dict(seed=seed,scenario=sc,origin=origin,direct_solve_max_error=error))
        for name,p in pred.items():
            mse=float(np.mean((p[valid]-y[valid])**2));clean=float(np.mean((p[valid]-mean[valid])**2))
            noise=float(np.mean(mean[valid]*(1-mean[valid])/61))
            metrics.append(dict(seed=seed,scenario=sc,method=name,rmse=mse**.5,clean_rmse=clean**.5,
                full_state_noise_rmse=noise**.5,expected_fresh_noise_rmse=(clean+noise)**.5))
        # Label oracle must dominate the explicitly included candidate gates in sample.
        oe=(pred['oracle_query_labels'][valid]-y[valid])**2
        for name in VARIANTS+['transformer_full','validation','convex_mkl']:
            assert np.all(oe<=(pred[name][valid]-y[valid])**2+1e-9)
        np.savez_compressed(out/f'{sc}.npz',**pred,**{k+'_weight':v for k,v in ww.items()},prediction_lower=lower,prediction_upper=upper)
        R.save(out/'metrics.json',metrics);print(seed,sc,{r['method']:round(100*r['rmse'],4) for r in metrics if r['scenario']==sc},flush=True)
    R.save(out/'checks.json',checks);R.save(out/'provenance.json',hashes)
    R.save(out/'protocol.json',dict(seed=seed,context=128,queries=32,banks=2,ridge='same fixed historical ridge',
        validation='Reuse verified past-only 96/32 context validation weights; refit on128',
        convex_mkl='Reuse verified past-only profiled regularized KRR weights; refit on128',
        label_oracles='Same support and KRR; query labels select shared cohort or per-query weights. Diagnostic only, not deployable.',
        mean_oracle='Per-query weights use latent-state true conditional means, not realized query labels; information-advantaged approximation reference.',
        search='501-point grid, all strict bracketed extrema/minima refined, boundaries and all reported gate/optimized weights included; numerical search, no global optimality certificate',
        bayes='True conditional mean uses latent state. Expected full-state noise MSE mean(mu*(1-mu)/61). Realized RMSE is sample dependent.',
        fresh_noise='sqrt(mean((prediction-mu)^2)+mean(mu*(1-mu)/61)); for label oracle this evaluates fixed hindsight predictions against independent new noise, not a deployable risk estimate'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):execute(a.seed,ROOT/f'oracle_reference_results/seed{a.seed}')
