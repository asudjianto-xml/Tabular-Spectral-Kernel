"""Refit all spectral parameters after validation selections freeze; exact ridge.

Primary endpoint is full-geometry refit. Frozen-geometry predictions are a
predeclared ablation, never a test-based model-selection option.
"""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from . import spectral_competitive_train as e

OUT=e.OUT/'full_refit'

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(primary='Refit geometry and exact ridge on all outer-training rows',
      ablation='Validation-trained frozen geometry with exact ridge refit on all outer-training rows',
      recipe='Locked seed-zero validation recipe; per-seed checkpoint length, gamma and clipping chosen on inner validation',
      initialization='Same previous inner-trained one-bank canonical checkpoint as refinement selection',
      preprocessing='Keep inner-training-fitted scaling and target normalization for warm-start coordinate consistency',
      training='Same optimizer, constraints, episode size and schedule as validation training; selected number of steps; episodes now draw from all outer-training rows',
      schedule='Use original planned 800-step cosine denominator, stopping at selected step',
      randomness='Same generator seed and initial fixed-subset draw as validation training, now on outer-training pool',
      evaluation='Save all three refit models before any test evaluation; no further test-based choices',
      scope='Exploratory reused holdouts, not independent equivalence evidence')
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:e.base.save(p,protocol)
    dest=OUT/Path(__file__).name
    if dest.exists():assert dest.read_bytes()==Path(__file__).read_bytes()
    else:dest.write_bytes(Path(__file__).read_bytes())


def refit(row,device):
    seed=row['seed'];cfg=row['config'];dest=OUT/f'refit_seed{seed}.pt'
    if dest.exists():return
    assert cfg['init']=='previous' and cfg['objective']=='query'
    X,y,tr,va,outer,te=e.base.data_split(seed)
    scaler=e.HousingScaler(cfg['preprocessing']).fit(X[tr])
    raw=scaler.transform(X[outer]);x=e.base.as_tensor(raw,device)
    ym,ys=float(y[tr].mean()),float(y[tr].std());yt=e.base.as_tensor((y[outer]-ym)/ys,device)
    torch.manual_seed(778+seed);model=e.model_for(cfg,device)
    e.initialize(model,cfg,raw,(y[outer]-ym)/ys,seed,device)
    optimizer=torch.optim.Adam([
      dict(params=[m.diagonal for m in model.members],lr=cfg['lr']),
      dict(params=[m.offdiag for m in model.members],lr=.2*cfg['lr']),
      dict(params=[m.log_gaps for m in model.members],lr=.2*cfg['lr']),
      dict(params=[m.log_temperature for m in model.members]+[model.log_ridge,model.weight_coordinates],lr=cfg['lr'])])
    initial_lrs=[g['lr'] for g in optimizer.param_groups]
    gen=torch.Generator(device=device).manual_seed(seed+887)
    torch.randperm(len(x),generator=gen,device=device) # Match validation training RNG warm-up.
    started=time.perf_counter();trace=[]
    for step in range(row['selected_step']):
        optimizer.zero_grad(set_to_none=True)
        ns=cfg['support'];idx=torch.randperm(len(x),generator=gen,device=device)[:ns+512]
        s,q=idx[:ns],idx[ns:];z=model.embedding(x[idx]);zs,zq=z[:ns],z[ns:]
        k=model.kernel(zs,zs);mu=yt[s].mean()
        l=torch.linalg.cholesky(k+model.log_ridge.exp()*torch.eye(ns,dtype=e.base.DTYPE,device=device))
        coef=torch.cholesky_solve((yt[s]-mu)[:,None],l)[:,0]
        loss=(model.kernel(zq,zs)@coef+mu-yt[q]).square().mean()
        penalty=cfg['reg']*sum(m.offdiag.square().sum()+m.diagonal.square().sum() for m in model.members)/(model.banks*model.p)
        (loss+penalty).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step();e.constrain(model)
        factor=.2+.8*.5*(1+math.cos(math.pi*(step+1)/cfg['steps']))
        for group,start_lr in zip(optimizer.param_groups,initial_lrs):group['lr']=start_lr*factor
        if (step+1)%200==0:
            trace.append(dict(step=step+1,episode_loss=float(loss.detach()),seconds=time.perf_counter()-started))
            print('REFIT',seed,trace[-1],flush=True)
    diag=e.fused.diagnostics(model)
    torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},config=cfg,target_mean=ym,target_std=ys,
      scaler_mean=scaler.scaler.mean_.tolist(),scaler_scale=scaler.scaler.scale_.tolist(),selected_step=row['selected_step']),dest)
    e.base.save(OUT/f'refit_seed{seed}.json',dict(seed=seed,seconds=time.perf_counter()-started,training_rows=len(outer),trace=trace,diagnostics=diag))
    del model,x,yt;torch.cuda.empty_cache()


def evaluate(rows,device):
    assert all((OUT/f'refit_seed{r["seed"]}.pt').exists() for r in rows)
    results={name:[] for name in ['spectral_full_refit','spectral_frozen_refined']}
    for row in rows:
        seed=row['seed'];cfg=row['config'];X,y,tr,va,outer,te=e.base.data_split(seed)
        scaler=e.HousingScaler(cfg['preprocessing']).fit(X[tr])
        for name,source in [('spectral_full_refit',OUT/f'refit_seed{seed}.pt'),('spectral_frozen_refined',e.OUT/f'{cfg["name"]}_seed{seed}'/'model.pt')]:
            dest=OUT/f'{name}_seed{seed}.json'
            if dest.exists():results[name].append(json.loads(dest.read_text()));continue
            checkpoint=torch.load(source,map_location=device,weights_only=True)
            model=e.model_for(cfg,device);model.load_state_dict(checkpoint['state_dict']);model.eval()
            ym,ys=checkpoint['target_mean'],checkpoint['target_std'];started=time.perf_counter()
            xf=e.base.as_tensor(scaler.transform(X[outer]),device);xt=e.base.as_tensor(scaler.transform(X[te]),device)
            yf=e.base.as_tensor((y[outer]-ym)/ys,device)
            pp,residual=e.exact_path(model,xf,yf,xt,[row['selected']['ridge']]);raw=pp[:,0].cpu().numpy()*ys+ym
            pred=np.clip(raw,y[outer].min(),y[outer].max()) if row['selected']['clip'] else raw
            record=dict(seed=seed,model=name,config=cfg,selected_step=row['selected_step'],selected=row['selected'],test=e.base.scores(y[te],pred),
              raw_test=e.base.scores(y[te],raw),exact_head_seconds=time.perf_counter()-started,relative_solve_residuals=residual)
            e.base.save(dest,record);results[name].append(record)
            np.savez_compressed(OUT/f'{name}_seed{seed}_predictions.npz',train_indices=tr,validation_indices=va,outer_train_indices=outer,test_indices=te,y_test=y[te],prediction=pred,raw_prediction=raw)
            print('TEST',name,seed,record['test'],flush=True)
            del model,xf,xt,yf;torch.cuda.empty_cache()
    for name in ['catboost','lightgbm','spectral']:
        results[name]=[json.loads((e.base.DEFAULT_OUT/f'{name}_seed{s}.json').read_text()) for s in range(3)]
    results['spectral_exact_old']=[json.loads((e.base.ROOT/'spectral_exact_california'/f'result_seed{s}.json').read_text()) for s in range(3)]
    summary={name:{metric:dict(mean=float(np.mean([r['test'][metric] for r in vals])),std=float(np.std([r['test'][metric] for r in vals],ddof=1))) for metric in ['rmse_dollars','r2']} for name,vals in results.items()}
    e.base.save(OUT/'summary.json',dict(summary=summary,results=results));print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');args=p.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(4):
        prepare();rows=json.loads((e.OUT/'all_selections_frozen.json').read_text())
        for row in rows:refit(row,args.device)
        evaluate(rows,args.device)
