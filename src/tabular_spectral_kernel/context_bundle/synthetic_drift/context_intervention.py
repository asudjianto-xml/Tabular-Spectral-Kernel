"""Frozen-model context interventions, separating encoder and KRR support effects."""
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
from threadpoolctl import threadpool_limits
import run as R
import run_transformer as T
import unemployment_data as U
from mlp_ablation import construct

ROOT=Path(__file__).parent
VARIANTS=['linear_simple','linear_stacked','linear_learned_psd','linear_transformer']
CONDITIONS=['correct','labels_masked_gate','wrong_regime_gate','query_only_gate',
            'wrong_regime_all','historical_only']+[f'shuffle_{scope}_{r}' for scope in ['gate','all'] for r in range(3)]


def query_only(model,query):
    """Bypass all context reads; retain the trained query path and head."""
    if hasattr(model,'input'):
        return model.head(model.norm(model.input(query))).softmax(-1)
    zeros=query.new_zeros(*query.shape[:-1],model.head.in_features-query.shape[-1])
    return model.head(torch.cat([query,zeros],-1)).softmax(-1)


def wrong_regime(cx,rng,localized=False):
    """Counterfactual donors: same borrowers/time layout, opposite constant macro regime.

    Select regime from observable context unemployment, never query labels or
    latent evaluation state. Donor labels are independent synthetic draws.
    """
    observed=.03+.09*(cx[:,4]+1)/2
    rate=.10 if observed.mean()<.07 else .04
    x=cx.copy();x[:,4]=2*(rate-.03)/.09-1
    mean,_=U.conditional_mean(x[:,:4],np.full(len(x),1. if rate==.10 else 0.),localized=localized)
    return x,T.C.responses(rng,mean),rate


def make_batch(banks,temps,cx,cy,qx):
    g,q=T.blocks(banks,temps,np.concatenate([cx,qx]))
    ct,qt=T.tokens(cx,cy,qx)
    return {k:torch.tensor(np.asarray(v)[None],dtype=torch.float32 if k in ['context','query'] else torch.float64)
            for k,v in dict(context=ct,query=qt,grams=g,cross=q,y=cy).items()}|{'descriptor':None}


def execute(seed,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    hashes={};rows=[];changes=[];donors=[];models={}
    def record(path):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for file in [Path(__file__),Path(T.__file__),Path(U.__file__),ROOT/'mlp_ablation.py',ROOT/'simple_spectral_gate.py',ROOT/'psd_spectral_attention.py']:
        record(file);(out/(file.stem+'_snapshot.py')).write_bytes(file.read_bytes())
    src=ROOT/f'unemployment_results/seed{seed}';banks=[]
    record(src/'original_training.json');info=json.loads((src/'original_training.json').read_text())
    for name in ['normal','downturn']:
        path=src/f'{name}.pt';record(path);m=R.base.CanonicalSpectral(5,4,max_frequency=3.,device='cpu')
        m.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'));m.eval();banks.append(m)
    for v in VARIANTS:
        path=ROOT/f'mlp_{v}_results/seed{seed}/transformer.pt';record(path)
        m=construct(v).eval();m.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'));models[v]=m
    path=src/'history.npz';record(path)
    with np.load(path) as f:hx,hy=f['x'][-128:],f['y'][-128:]
    for si,sc in enumerate(T.C.SCENARIOS):
        path=src/f'original_{sc}.npz';record(path)
        with np.load(path) as f:x,y=f['x'],f['y'];truth=f['mean']
        preds={v:{c:np.full(len(y),np.nan) for c in CONDITIONS} for v in VARIANTS}
        weights={v:{c:np.full(len(y),np.nan) for c in CONDITIONS} for v in VARIANTS}
        for origin in range(128,len(y),32):
            cx=x[origin-128:origin];cy=y[origin-128:origin];qx=x[origin:origin+32]
            rng=np.random.default_rng(800000+10000*seed+2000*si+origin)
            dx,dy,rate=wrong_regime(cx,rng,sc=='localized_downturn')
            donors.append(dict(seed=seed,scenario=sc,origin=origin,donor_rate=rate,
                               actual_context_rate=float((.03+.09*(cx[:,4]+1)/2).mean())))
            b=make_batch(banks,info['temperatures'],cx,cy,qx)
            wrong=make_batch(banks,info['temperatures'],dx,dy,qx)
            hist=make_batch(banks,info['temperatures'],hx,hy,qx)
            masked=b['context'].clone();masked[:,:,5:7]=0
            shuffled=[]
            for rep in range(3):
                perm=rng.permutation(128);c=b['context'].clone();c[:,:,5]=c[:,perm,5]
                shuffled.append(dict(b,context=c,y=b['y'][:,perm]))
            with torch.no_grad():
                for v,m in models.items():
                    correct=T.gate_weights(m,b);query=query_only(m,b['query'])
                    wa=T.gate_weights(m,wrong)
                    cases={'correct':(correct,b),'labels_masked_gate':(T.gate_weights(m,dict(b,context=masked)),b),
                           'wrong_regime_gate':(wa,b),'query_only_gate':(query,b),'wrong_regime_all':(wa,wrong),
                           'historical_only':(query,hist)}
                    for rep,s in enumerate(shuffled):
                        a=T.gate_weights(m,s);cases[f'shuffle_gate_{rep}']=(a,b);cases[f'shuffle_all_{rep}']=(a,s)
                    for condition,(a,support) in cases.items():
                        _,p=T.decode(a,support['grams'],support['cross'],support['y'],info['ridge'])
                        preds[v][condition][origin:origin+32]=p[0].clamp(0,1).numpy()
                        weights[v][condition][origin:origin+32]=a[0,:,0].numpy()
        mask=np.arange(len(y))>=128
        saved={}
        for v in VARIANTS:
            path=ROOT/f'mlp_{v}_results/seed{seed}/{sc}.npz';record(path)
            with np.load(path) as f:
                np.testing.assert_allclose(preds[v]['correct'][mask],f['transformer_krr'][mask],atol=1e-9,rtol=1e-9)
                np.testing.assert_allclose(weights[v]['correct'][mask],f['transformer_alpha'][mask,0],atol=1e-7)
            base=preds[v]['correct'][mask];alpha=weights[v]['correct'][mask]
            for c in CONDITIONS:
                p=preds[v][c][mask];w=weights[v][c][mask]
                rows.append(dict(seed=seed,scenario=sc,variant=v,condition=c,rmse=float(np.sqrt(np.mean((p-y[mask])**2))),
                    clean_rmse=float(np.sqrt(np.mean((p-truth[mask])**2))),
                    weight_abs_change=float(np.mean(np.abs(w-alpha))),prediction_rms_change=float(np.sqrt(np.mean((p-base)**2)))))
                saved[v+'__'+c+'__prediction']=preds[v][c];saved[v+'__'+c+'__normal_weight']=weights[v][c]
        np.savez_compressed(out/f'{sc}.npz',**saved)
        print('completed',seed,sc,flush=True)
        R.save(out/'metrics.json',rows)
    R.save(out/'donors.json',donors);R.save(out/'provenance.json',hashes)
    R.save(out/'protocol.json',dict(seed=seed,variants=VARIANTS,conditions=CONDITIONS,
        support='128 prior labels; next32 query features fixed; no query labels passed to gate or intervention selection',
        wrong_regime='Counterfactual synthetic donor: same borrower predictors, opposing constant unemployment .04/.10 chosen from context mean threshold .07; equilibrium exposure 0/1; independently sampled labels; actual localized/broad scenario law retained. This is an artificial intervention, not retrieved real history.',
        shuffle='3 independent permutations per origin, shared by all encoders; report mean of replicate metrics, not an ensemble prediction',
        no_context='query_only_gate bypasses all context branches but keeps real KRR support. historical_only uses the same query-only gate and last128 pre-evaluation historical rows for KRR. No empty-support KRR is claimed.',
        caveat='Inference interventions may be out of meta-training distribution. No retraining; changes measure reliance/sensitivity, not causal identification or an optimal no-context baseline.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):execute(a.seed,ROOT/f'context_intervention_results/seed{a.seed}')
