"""Validation-driven improvements of the canonical spectral geometry.

screen: seed-zero geometry search, validation only.
confirm: fit the validation-selected recipe on seeds one and two, validation only.
evaluate: freeze all choices, refit exact heads, then score the three test splits.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from . import identifiable_spectral_california as base
from . import single_bank as fused
from .solvers import exact_path

OUT=Path.cwd()/'runs'/'spectral_competitive_california'
RIDGES=np.logspace(-7,-3,9)
CONFIGS=[dict(name='h1_warm_large_query',banks=1,atoms=4,preprocessing='standard',objective='query',steps=800,support=1536,lr=.003,reg=.0001,init='previous')]

class HousingScaler:
    def __init__(self,name):self.name=name;self.scaler=StandardScaler()
    def raw(self,x):
        z=np.array(x,dtype=float,copy=True)
        if self.name=='housing_log':z[:,[2,3,4,5]]=np.log1p(z[:,[2,3,4,5]])
        return z
    def fit(self,x):self.scaler.fit(self.raw(x));return self
    def transform(self,x):return self.scaler.transform(self.raw(x))


def periodogram_frequencies(x,y,atoms):
    # Fixed low-frequency trend coordinates plus separated supervised peaks.
    y=y-y.mean();grid=np.linspace(.25,3.,96);rows=[]
    for j in range(x.shape[1]):
        phase=2*np.pi*x[:,j,None]*grid[None]
        power=np.square(y@np.cos(phase)/len(y))+np.square(y@np.sin(phase)/len(y))
        low=[.025,.06] if atoms==4 else [.025,.06,.12,.2]
        chosen=low.copy()
        for idx in np.argsort(power)[::-1]:
            f=float(grid[idx])
            if all(abs(f-v)>.08 for v in chosen):chosen.append(f)
            if len(chosen)==atoms:break
        assert len(chosen)==atoms
        rows.append(sorted(chosen))
    return np.array(rows)


def model_for(cfg,device):
    return fused.FusedCanonicalSpectral(8,cfg['atoms'],cfg['banks'],.5,device)


def initialize(model,cfg,x,y,seed,device):
    if cfg['init']=='previous':
        old=torch.load(base.DEFAULT_OUT/f'spectral_seed{seed}.pt',map_location=device,weights_only=True)
        assert old['config']['atoms']==cfg['atoms'] and cfg['banks']==1
        model.members[0].load_state_dict({k:v for k,v in old['state_dict'].items() if k!='log_ridge'})
    else:
        omega=periodogram_frequencies(x,y,cfg['atoms'])
        rng=np.random.default_rng(seed+565)
        with torch.no_grad():
            for bank in model.members:
                frequencies=np.sort(omega*np.exp(.025*rng.normal(size=omega.shape)),axis=1)
                gaps=np.diff(frequencies,axis=1,prepend=np.zeros((8,1)))
                assert gaps.min()>0
                bank.log_gaps.copy_(base.as_tensor(np.log(gaps),device))
            if cfg['banks']==1:model.members[0].log_temperature.fill_(math.log(.5))
    # In this experiment log_ridge is log observation-noise variance, not mean-loss gamma.
    with torch.no_grad():model.log_ridge.fill_(math.log(.1))


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'training_protocol.json'
    protocol=dict(configs=CONFIGS,selection='Search seed-zero validation; lock recipe; tune its checkpoints and ridge on validation of seeds one/two; test evaluation only after all choices freeze',
      geometry='One canonical trace-one matrix-exponential metric; four ordered frequencies per feature; Laplace kernel and one ridge head',
      initialization='Published inner-trained four-frequency standard-scaled warm starts, or matching initialization-search selections',
      preprocessing='Standardization or log1p of AveRooms/AveBedrms/Population/AveOccup followed by standardization; fit on inner-training only',
      objective='Fixed-subset dense NLML or larger-support episode query loss, depending on declared candidate',
      lrs='Diagonal metric and bandwidth/weights base LR; off-diagonal metric .2*LR; frequency .2*LR; cosine decay to 20% of starting LR',
      bounds='Log frequency gaps [-9,2]; log bandwidth [-5,3]; observation noise [1e-4,1]; metric spectral spread constrained to at most 20 after optimizer steps',
      validation_landmarks=2048,validation_interval=200,checkpoint_selection='Nystrom validation followed by exact validation ridge selection for each candidate winner',
      exact_ridges=RIDGES.tolist(),clip='Select raw prediction or clipping to fitting-label min/max on validation',
      baselines='Unchanged prior tuned CatBoost/LightGBM predictions; identical splits',
      scope='Exploratory improvement on repeatedly inspected holdouts, no independent confirmation claim')
    if p.exists():assert json.loads(p.read_text())==protocol
    else:base.save(p,protocol)
    for source in [Path(__file__),Path(base.__file__),Path(fused.__file__),Path(__file__).with_name('solvers.py')]:
        dest=OUT/source.name
        if dest.exists():assert dest.read_bytes()==source.read_bytes()
        else:dest.write_bytes(source.read_bytes())


def choose(pred,yval,low,high,ridges):
    trials=[]
    for clip in [False,True]:
        pp=np.clip(pred,low,high) if clip else pred
        err=np.sqrt(np.square(pp-yval[:,None]).mean(0))
        for j,ridge in enumerate(ridges):trials.append(dict(ridge=float(ridge),clip=clip,validation_rmse=float(err[j])))
    return min(trials,key=lambda r:r['validation_rmse']),trials


@torch.no_grad()
def constrain(model):
    for member in model.members:
        member.log_gaps.clamp_(-9,2);member.log_temperature.clamp_(-5,3)
        # A homogeneous constraint in the unique trace-zero coordinates.
        eig=torch.linalg.eigvalsh(member.generator())
        spread=float(eig[-1]-eig[0])
        if spread>20:
            member.offdiag.mul_(20/spread);member.diagonal.mul_(20/spread)
    model.log_ridge.clamp_(math.log(1e-4),0)


def train(seed,cfg,device):
    folder=OUT/f'{cfg["name"]}_seed{seed}';folder.mkdir(exist_ok=True)
    if (folder/'selection.json').exists():return json.loads((folder/'selection.json').read_text())
    started=time.perf_counter();torch.manual_seed(778+seed)
    X,y,tr,va,_,_=base.data_split(seed)
    scaler=HousingScaler(cfg['preprocessing']).fit(X[tr]);raw=scaler.transform(X[tr])
    xt=base.as_tensor(raw,device);xv=base.as_tensor(scaler.transform(X[va]),device)
    ym,ys=float(y[tr].mean()),float(y[tr].std());yt=base.as_tensor((y[tr]-ym)/ys,device)
    model=model_for(cfg,device);initialize(model,cfg,raw,(y[tr]-ym)/ys,seed,device)
    optimizer=torch.optim.Adam([
      dict(params=[m.diagonal for m in model.members],lr=cfg['lr']),
      dict(params=[m.offdiag for m in model.members],lr=.2*cfg['lr']),
      dict(params=[m.log_gaps for m in model.members],lr=.2*cfg['lr']),
      dict(params=[m.log_temperature for m in model.members]+[model.log_ridge,model.weight_coordinates],lr=cfg['lr'])])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['steps'],eta_min=0)
    initial_lrs=[g['lr'] for g in optimizer.param_groups]
    gen=torch.Generator(device=device).manual_seed(seed+887)
    fixed=torch.randperm(len(xt),generator=gen,device=device)[:cfg['support']]
    landmarks=torch.tensor(np.random.default_rng(seed+77).permutation(len(tr))[:2048],device=device)
    best=(float('inf'),None,None);trace=[]
    for step in range(cfg['steps']+1):
        if step%200==0 or step==cfg['steps']:
            pred=base.nystrom_path(model,xt,yt,xv,landmarks,RIDGES).cpu().numpy()*ys+ym
            choice,_=choose(pred,y[va],y[tr].min(),y[tr].max(),RIDGES)
            trace.append(dict(step=step,**choice,noise=float(model.log_ridge.exp().detach()),weights=model.weights().detach().cpu().tolist(),seconds=time.perf_counter()-started))
            print(cfg['name'],'seed',seed,'step',step,'val',choice['validation_rmse'],flush=True)
            base.save(folder/'trace.json',trace)
            if choice['validation_rmse']<best[0]:best=(choice['validation_rmse'],copy.deepcopy(model.state_dict()),step)
        if step==cfg['steps']:break
        optimizer.zero_grad(set_to_none=True)
        if cfg['objective']=='nlml':
            xs=xt[fixed];ysub=yt[fixed]-yt[fixed].mean();z=model.embedding(xs)
            k=model.kernel(z,z);n=len(fixed)
            l=torch.linalg.cholesky(k+model.log_ridge.exp()*torch.eye(n,dtype=base.DTYPE,device=device))
            coef=torch.cholesky_solve(ysub[:,None],l)[:,0]
            loss=(.5*(ysub*coef).sum()+l.diag().log().sum())/n
        else:
            idx=torch.randperm(len(xt),generator=gen,device=device)[:cfg['support']+512]
            ns=cfg['support'];s,q=idx[:ns],idx[ns:];z=model.embedding(xt[idx]);zs,zq=z[:ns],z[ns:]
            k=model.kernel(zs,zs);mu=yt[s].mean()
            l=torch.linalg.cholesky(k+model.log_ridge.exp()*torch.eye(ns,dtype=base.DTYPE,device=device))
            coef=torch.cholesky_solve((yt[s]-mu)[:,None],l)[:,0]
            loss=(model.kernel(zq,zs)@coef+mu-yt[q]).square().mean()
        penalty=cfg['reg']*sum(m.offdiag.square().sum()+m.diagonal.square().sum() for m in model.members)/(model.banks*model.p)
        (loss+penalty).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step();constrain(model)
        # Explicit cosine schedule retains 20% of each group's initial rate.
        factor=.2+.8*.5*(1+math.cos(math.pi*(step+1)/cfg['steps']))
        for group,start_lr in zip(optimizer.param_groups,initial_lrs):group['lr']=start_lr*factor
    model.load_state_dict(best[1]);model.eval()
    t=time.perf_counter();pred,residuals=exact_path(model,xt,yt,xv,RIDGES)
    pred=pred.cpu().numpy()*ys+ym
    choice,candidates=choose(pred,y[va],y[tr].min(),y[tr].max(),RIDGES)
    diag=fused.diagnostics(model)
    record=dict(seed=seed,config=cfg,selected_step=best[2],nystrom_validation_rmse=best[0],selected=choice,
      exact_candidates=candidates,relative_solve_residuals=residuals,exact_selection_seconds=time.perf_counter()-t,
      total_seconds=time.perf_counter()-started,diagnostics=diag)
    torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},config=cfg,target_mean=ym,target_std=ys),folder/'model.pt')
    np.savez_compressed(folder/'validation_predictions.npz',validation_indices=va,y_validation=y[va],ridges=RIDGES,predictions=pred)
    base.save(folder/'selection.json',record)
    print('EXACT SELECTED',cfg['name'],'seed',seed,choice,flush=True)
    del model,xt,xv,yt;torch.cuda.empty_cache()
    return record


def screen(args):
    prepare();rows=[]
    for cfg in CONFIGS:rows.append(train(0,cfg,args.device))
    winner=min(rows,key=lambda r:r['selected']['validation_rmse'])
    base.save(OUT/'recipe_frozen.json',dict(config=winner['config'],validation_rmse=winner['selected']['validation_rmse'],selection_seed=0))
    print('FROZEN RECIPE',winner['config'],winner['selected'],flush=True)


def confirm(args):
    prepare();recipe=json.loads((OUT/'recipe_frozen.json').read_text())['config']
    rows=[json.loads((OUT/f'{recipe["name"]}_seed0/selection.json').read_text())]
    for seed in [1,2]:rows.append(train(seed,recipe,args.device))
    base.save(OUT/'all_selections_frozen.json',rows)


def evaluate(args):
    rows=json.loads((OUT/'all_selections_frozen.json').read_text());results=[]
    for row in rows:
        seed=row['seed'];cfg=row['config'];folder=OUT/f'{cfg["name"]}_seed{seed}'
        if (folder/'result.json').exists():results.append(json.loads((folder/'result.json').read_text()));continue
        model=model_for(cfg,args.device);ckpt=torch.load(folder/'model.pt',map_location=args.device,weights_only=True)
        model.load_state_dict(ckpt['state_dict']);model.eval()
        X,y,tr,va,outer,te=base.data_split(seed);scaler=HousingScaler(cfg['preprocessing']).fit(X[tr])
        ym,ys=ckpt['target_mean'],ckpt['target_std'];t=time.perf_counter()
        xf=base.as_tensor(scaler.transform(X[outer]),args.device);xt=base.as_tensor(scaler.transform(X[te]),args.device)
        yf=base.as_tensor((y[outer]-ym)/ys,args.device)
        pp,residuals=exact_path(model,xf,yf,xt,[row['selected']['ridge']]);raw=pp[:,0].cpu().numpy()*ys+ym
        pred=np.clip(raw,y[outer].min(),y[outer].max()) if row['selected']['clip'] else raw
        result=dict(seed=seed,config=cfg,selected=row['selected'],test=base.scores(y[te],pred),raw_test=base.scores(y[te],raw),
          final_seconds=time.perf_counter()-t,training_selection_seconds=row['total_seconds'],relative_solve_residuals=residuals)
        base.save(folder/'result.json',result);results.append(result)
        np.savez_compressed(folder/'test_predictions.npz',train_indices=tr,validation_indices=va,outer_train_indices=outer,
          test_indices=te,y_test=y[te],prediction=pred,raw_prediction=raw)
        print('TEST RESULT',result,flush=True)
        del model,xf,xt,yf;torch.cuda.empty_cache()
    models={'improved_spectral':results}
    for name in ['catboost','lightgbm','spectral']:
        models[name]=[json.loads((base.DEFAULT_OUT/f'{name}_seed{s}.json').read_text()) for s in [0,1,2]]
    models['spectral_exact_old']=[json.loads((base.ROOT/'spectral_exact_california'/f'result_seed{s}.json').read_text()) for s in [0,1,2]]
    summary={name:{metric:dict(mean=float(np.mean([r['test'][metric] for r in vals])),std=float(np.std([r['test'][metric] for r in vals],ddof=1))) for metric in ['rmse_dollars','r2']} for name,vals in models.items()}
    base.save(OUT/'summary.json',dict(summary=summary,results=models));print('SUMMARY',json.dumps(summary),flush=True)


def check():
    from scipy.linalg import expm
    torch.manual_seed(0);m=fused.FusedCanonicalSpectral(2,2,2,device='cpu')
    x=torch.randn(12,2,dtype=base.DTYPE);xq=torch.randn(5,2,dtype=base.DTYPE);y=torch.randn(12,dtype=base.DTYPE)
    pp,res=exact_path(m,x,y,xq,[1e-3])
    pn=base.nystrom_path(m,x,y,xq,torch.arange(len(x)),[1e-3])
    assert (pp-pn).abs().max()<2e-7 and max(res)<1e-10
    omega=periodogram_frequencies(np.random.default_rng(0).normal(size=(100,8)),np.arange(100),8)
    assert (omega>0).all() and (np.diff(omega,axis=1)>0).all()
    scaler=HousingScaler('housing_log').fit(np.ones((10,8)));assert np.isfinite(scaler.transform(np.ones((2,8)))).all()
    print('PASS exact intercept solve/Nystrom equivalence, residual, ordered spectral initialization, log preprocessing.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['check','screen','confirm','evaluate']);p.add_argument('--device',default='cuda');args=p.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(4):
        if args.command=='check':check()
        else:globals()[args.command](args)
