"""Validation-selected sparse linear context selectors; fresh test episodes."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run as R
import run_context_selector as S

ROOT=S.ROOT
PENALTIES=['l2','l1','elastic','group']
STRENGTHS=[1e-6,1e-5,1e-4,1e-3]
NAMES=([f'query_{i}' for i in range(5)]+[f'difference_{i}' for i in range(5)]+
       [f'squared_difference_{i}' for i in range(5)]+['kernel_mean','kernel_max','kernel_std']+
       [f'label_mean_x_query_{i}' for i in range(4)]+['macro_mean_squared_difference','macro_end_squared_difference']+
       [f'window_mean_{i}' for i in range(5)]+[f'window_std_{i}' for i in range(5)]+
       [f'window_macro_month_{i}' for i in range(4)]+[f'window_label_month_{i}' for i in range(4)]+
       ['window_label_mean','window_label_std']+[f'recent_macro_month_{i}' for i in range(4)]+
       [f'recent_label_month_{i}' for i in range(4)]+['age','macro_slope','label_slope','label_mean_minus_recent','absolute_macro_distance'])
GROUPS={'borrower':[0,1,2,3,5,6,7,8,10,11,12,13,24,25,26,27,29,30,31,32],
        'macro':[4,9,14,22,23,28,33,34,35,36,37,44,45,46,47,53,56],
        'labels':[18,19,20,21,38,39,40,41,42,43,48,49,50,51,54,55],
        'kernel':[15,16,17],'recency':[52]}


def augment(d):
    # Raw feature9 is signed query unemployment minus candidate mean.
    return dict(d,features=np.concatenate([d['features'],abs(d['features'][...,9:10])],-1))


def prox(w,step,lam,kind,active):
    z=w.clone()
    if kind=='l2':z=z/(1+step*lam)
    elif kind=='l1':z=z.sign()*(z.abs()-step*lam).clamp_min(0)
    elif kind=='elastic':z=z.sign()*(z.abs()-step*lam*.5).clamp_min(0)/(1+step*lam*.5)
    elif kind=='group':
        for ids in GROUPS.values():
            ix=[i for i in ids if active[i]]
            if not ix:continue
            norm=z[ix].norm();threshold=step*lam*np.sqrt(len(ix))
            z[ix]*=(1-threshold/norm.clamp_min(1e-30)).clamp_min(0)
    else:raise ValueError(kind)
    z[~active]=0
    return z


def hard_mse(scores,d):
    c=scores.argmax(-1);p=np.take_along_axis(d['predictions'],c[...,None],-1)[...,0]
    return float(np.mean((p-d['target'])**2))


def fit_penalty(train,val,kind,lam):
    x=train['features'];center=x.mean((0,1,2));scale=np.maximum(x.std((0,1,2)),.01)
    # Exact structural cancellation is determined from training features only.
    active_np=np.max(np.ptp(x,axis=2),axis=(0,1))>1e-7
    active=torch.tensor(active_np)
    tx=torch.tensor((x-center)/scale);vx=torch.tensor((val['features']-center)/scale)
    cost=torch.tensor((train['predictions']-train['target'][...,None])**2,dtype=torch.float32)
    w=torch.zeros(x.shape[-1],requires_grad=True);step=100.;best=np.inf;bestw=None;best_epoch=0;trace=[]
    def smooth(v):return ((tx@v).softmax(-1)*cost).sum(-1).mean()
    for epoch in range(1,301):
        loss=smooth(w);grad=torch.autograd.grad(loss,w)[0];old=float(loss.detach());trial=min(step*1.2,1000.)
        with torch.no_grad():
            for _ in range(40):
                z=prox(w-trial*grad,trial,lam,kind,active);delta=z-w
                new=float(smooth(z));bound=old+float(grad@delta)+float(delta@delta)/(2*trial)
                if new<=bound+1e-9:break
                trial*=.5
            else:raise RuntimeError('Proximal line search failed')
        w=z.detach().requires_grad_(True);step=trial
        with torch.no_grad():vm=hard_mse((vx@w).numpy(),val)
        trace.append(dict(epoch=epoch,validation_mse=vm,soft_training_mse=new,step=step,nonzero=int((w.detach().abs()>1e-8).sum())))
        if vm<best:best=vm;best_epoch=epoch;bestw=w.detach().clone()
        if epoch-best_epoch>=40:break
    model=S.Selector(center,scale)
    with torch.no_grad():model.score.weight.copy_(bestw[None])
    model.eval()
    return model,dict(kind=kind,strength=lam,validation_mse=best,best_epoch=best_epoch,trace=trace,
        nonzero=int((bestw.abs()>1e-8).sum()),active_features=int(active.sum()),
        coefficients={n:float(v) for n,v in zip(NAMES,bestw)},
        group_norms={g:float(bestw[ids].norm()) for g,ids in GROUPS.items()})


def execute(seed,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);hashes={}
    def record(p):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
    for p in [Path(__file__),Path(S.__file__),Path(R.__file__),Path(S.U.__file__),Path(S.T.C.__file__),R.MODEL_SOURCE]:
        record(p);(out/(p.stem+'_snapshot.py')).write_bytes(p.read_bytes())
    src=ROOT/f'unemployment_results/seed{seed}';info=json.loads(record(src/'original_training.json').read_text());banks=[]
    for name in ['normal','downturn']:
        m=R.base.CanonicalSpectral(5,4,max_frequency=3.,device='cpu')
        m.load_state_dict(torch.load(record(src/f'{name}.pt'),weights_only=True,map_location='cpu'));m.eval();banks.append(m)
    original=ROOT/f'context_selector_results/seed{seed}'
    protocol=json.loads(record(original/'protocol.json').read_text());weight=protocol['mixture_weight']
    tr0=S.dataset(banks,info['temperatures'],info['ridge'],weight,910000+seed,512)
    va0=S.dataset(banks,info['temperatures'],info['ridge'],weight,920000+seed,128)
    tr,va=augment(tr0),augment(va0);models={};search=[];selected={}
    state=torch.load(record(original/'selector.pt'),weights_only=True,map_location='cpu')
    old=S.Selector(state['center'].numpy(),state['scale'].numpy());old.load_state_dict(state);old.eval();models['original']=old
    adam,trace=S.train(tr,va,seed);models['augmented_adam_l2']=adam
    selected['augmented_adam_l2']=dict(best_epoch=trace['best_epoch'],trace=trace['trace'],nonzero=int((adam.score.weight.detach().abs()>1e-8).sum()))
    for kind in PENALTIES:
        best=np.inf
        for lam in STRENGTHS:
            model,recorded=fit_penalty(tr,va,kind,lam);search.append(recorded)
            if recorded['validation_mse']<best:best=recorded['validation_mse'];models[kind]=model;selected[kind]=recorded
        print(seed,kind,'lambda',selected[kind]['strength'],'nonzero',selected[kind]['nonzero'],flush=True)
    validation={}
    with torch.no_grad():
        for name,m in models.items():validation[name]=hard_mse(m(torch.tensor((va0 if name=='original' else va)['features'])).numpy(),va0)
    validation['macro_nearest']=hard_mse(-va['features'][...,-1],va)
    validation['latest']=float(np.mean((va['predictions'][...,-1]-va['target'])**2))
    winner=min(validation,key=validation.get)
    for name,m in models.items():torch.save(m.state_dict(),out/f'{name}.pt')
    R.save(out/'selection.json',dict(validation=validation,winner=winner,selected=selected,search=search))
    # All models, penalties and winner frozen before any fresh test construction.
    metrics=[]
    for i,sc in enumerate(S.SCENARIOS):
        d0=S.dataset(banks,info['temperatures'],info['ridge'],weight,1030000+seed*1000+i*100,256,sc);d=augment(d0)
        choices={'latest':np.full(d['target'].shape,5),'macro_nearest':d['macro_choice']}
        with torch.no_grad():
            for name,m in models.items():choices[name]=m(torch.tensor((d0 if name=='original' else d)['features'])).argmax(-1).numpy()
        choices['validation_selected']=choices[winner]
        result={}
        for name,c in choices.items():
            pred=np.take_along_axis(d['predictions'],c[...,None],-1)[...,0];result[name]=pred
            metrics.append(dict(seed=seed,scenario=sc,method=name,rmse=float(np.sqrt(np.mean((pred-d['target'])**2))),older_fraction=float(np.mean(c!=5))))
        np.savez_compressed(out/f'{sc}.npz',target=d['target'],candidate_predictions=d['predictions'],**result,**{k+'_choice':v for k,v in choices.items()})
        print(seed,sc,{r['method']:round(100*r['rmse'],4) for r in metrics if r['scenario']==sc},flush=True)
    R.save(out/'metrics.json',metrics);R.save(out/'provenance.json',hashes)
    R.save(out/'protocol.json',dict(seed=seed,train_seed=910000+seed,validation_seed=920000+seed,test_seed_base=1030000+seed*1000,
        train_episodes=512,validation_episodes=128,test_episodes_per_scenario=256,
        fixed_mixture=weight,fixed_ridge=info['ridge'],features=NAMES,groups=GROUPS,strengths=STRENGTHS,
        objective='Same softmax expected action error; hard argmax window at inference',
        penalties='L2 lambda||w||^2/2; L1 lambda||w||1; elastic lambda(.5||w||1+.25||w||^2); group lambda sum sqrt(active_group_size)||w_group||2',
        optimization='Fullbatch proximal gradient with smooth-majorization backtracking, zero initialization, max300 steps, hard-validation patience40. Exact thresholding; no claim of convexity or global optimization.',
        controls='Frozen original56-feature Adam/L2 selector; retrained57-feature Adam/L2 with identical old protocol; matched proximal57-feature L2 control; unemployment and latest heuristics.',
        selection='Penalty strength and checkpoint selected on same128 validation episodes, separately per penalty. Also choose overall validation winner including heuristics, frozen before new test generation.',
        limitation='Train/validation reused from earlier selector; fresh test episodes, unchanged simulator law. No adjustments after inspecting new test results.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):execute(a.seed,ROOT/f'selector_penalty_results/seed{a.seed}')
