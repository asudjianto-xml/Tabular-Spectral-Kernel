"""Restore exact decoding of frozen identifiable spectral geometry.

All three ridge/postprocessing selections finish before any test evaluation.
This is a follow-up on previously inspected holdouts, not independent confirmation.
"""
from pathlib import Path
import argparse,hashlib,json,time,shutil
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from . import identifiable_spectral_california as base
from .solvers import exact_path

OUT=Path.cwd()/'runs'/'spectral_exact_california'
RIDGES=np.logspace(-7,-4,7)


def load(seed,device):
    checkpoint=base.DEFAULT_OUT/f'spectral_seed{seed}.pt'
    ckpt=torch.load(checkpoint,map_location=device,weights_only=True)
    cfg=ckpt['config'];model=base.CanonicalSpectral(8,cfg['atoms'],cfg['max_frequency'],device)
    model.load_state_dict(ckpt['state_dict']);model.eval()
    X,y,tr,va,outer,te=base.data_split(seed)
    scaler=base.preprocess(cfg['preprocessing'],X[tr],seed)
    ym,ys=y[tr].mean(),y[tr].std()
    return model,X,y,tr,va,outer,te,scaler,ym,ys


def main(args):
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(seeds=[0,1,2],geometry='Frozen validation-selected canonical one-bank model from prior benchmark; no parameter changes',
      decoder='Exact full-data Laplace kernel ridge with unpenalized intercept; no Nystrom approximation',
      ridge_grid=RIDGES.tolist(),postprocessing_candidates=['raw','clip to fitting-label min and max'],
      selection='Validation only; all seed selections completed before any new test predictions',
      refit='All 16512 outer-training rows; same frozen preprocessing and geometry',
      comparison='Prior tuned CatBoost/LightGBM on exactly the same splits',
      scope='Exploratory repeated-holdout follow-up; practical competitiveness is descriptive, not statistical equivalence')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:base.save(path,protocol)
    for source in [Path(__file__),Path(base.__file__),Path(__file__).with_name('solvers.py')]:
        dest=OUT/source.name
        if dest.exists():assert dest.read_bytes()==source.read_bytes()
        else:dest.write_bytes(source.read_bytes())
    selection=[]
    for seed in [0,1,2]:
        path=OUT/f'selection_seed{seed}.json'
        if path.exists():selection.append(json.loads(path.read_text()));continue
        t=time.perf_counter();model,X,y,tr,va,outer,te,scaler,ym,ys=load(seed,args.device)
        x=base.as_tensor(scaler.transform(X[tr]),args.device);xv=base.as_tensor(scaler.transform(X[va]),args.device)
        yt=base.as_tensor((y[tr]-ym)/ys,args.device)
        pred,residual=exact_path(model,x,yt,xv,RIDGES);pred=pred.cpu().numpy()*ys+ym
        records=[]
        for clip in [False,True]:
            p=np.clip(pred,y[tr].min(),y[tr].max()) if clip else pred
            errors=np.sqrt(np.square(p-y[va,None]).mean(0))
            for j,ridge in enumerate(RIDGES):records.append(dict(ridge=float(ridge),clip=clip,validation_rmse=float(errors[j])))
        best=min(records,key=lambda r:r['validation_rmse'])
        row=dict(seed=seed,selected=best,candidates=records,relative_solve_residuals=residual,seconds=time.perf_counter()-t)
        selection.append(row);base.save(path,row)
        print('SELECTION',row,flush=True)
        del model,x,xv,yt,pred;torch.cuda.empty_cache()
    base.save(OUT/'selection_frozen.json',selection)
    results=[]
    for s in selection:
        seed=s['seed'];path=OUT/f'result_seed{seed}.json'
        if path.exists():results.append(json.loads(path.read_text()));continue
        t=time.perf_counter();model,X,y,tr,va,outer,te,scaler,ym,ys=load(seed,args.device)
        xf=base.as_tensor(scaler.transform(X[outer]),args.device);xt=base.as_tensor(scaler.transform(X[te]),args.device)
        yf=base.as_tensor((y[outer]-ym)/ys,args.device)
        pred,residual=exact_path(model,xf,yf,xt,[s['selected']['ridge']]);raw=pred[:,0].cpu().numpy()*ys+ym
        pred=np.clip(raw,y[outer].min(),y[outer].max()) if s['selected']['clip'] else raw
        result=dict(seed=seed,model='exact_identifiable_spectral',selected=s['selected'],test=base.scores(y[te],pred),
            raw_test=base.scores(y[te],raw),selection_seconds=s['seconds'],refit_and_prediction_seconds=time.perf_counter()-t,
            relative_solve_residuals=residual,clipped_predictions=int(np.count_nonzero(pred!=raw)),
            geometry_checkpoint_sha256=hashlib.sha256((base.DEFAULT_OUT/f'spectral_seed{seed}.pt').read_bytes()).hexdigest())
        base.save(path,result);results.append(result)
        np.savez_compressed(OUT/f'seed{seed}_predictions.npz',train_indices=tr,validation_indices=va,
            outer_train_indices=outer,test_indices=te,y_test=y[te],prediction=pred,raw_prediction=raw)
        print('RESULT',result,flush=True)
        del model,xf,xt,yf;torch.cuda.empty_cache()
    models={'spectral_exact':results}
    for name in ['spectral','catboost','lightgbm']:
        models[name]=[json.loads((base.DEFAULT_OUT/f'{name}_seed{s}.json').read_text()) for s in [0,1,2]]
    summary={name:{metric:dict(mean=float(np.mean([r['test'][metric] for r in rows])),std=float(np.std([r['test'][metric] for r in rows],ddof=1))) for metric in ['rmse_dollars','r2']} for name,rows in models.items()}
    base.save(OUT/'summary.json',dict(summary=summary,results=models))
    lines=['# Exact decoding of identifiable spectral geometry','',
      'Same frozen one-bank geometry, same data and preprocessing; replace the 2,048-landmark approximation with exact kernel ridge. Validation chooses ridge and optional clipping to the fitting-label range. All three selections were frozen before the new test evaluations.','',
      '| Model | Test RMSE dollars mean ± SD | Test R² mean ± SD |','|---|---:|---:|']
    for name,r in summary.items():lines.append(f'| {name} | {r["rmse_dollars"]["mean"]:,.0f} ± {r["rmse_dollars"]["std"]:,.0f} | {r["r2"]["mean"]:.4f} ± {r["r2"]["std"]:.4f} |')
    lines+=['','## Interpretation','',
      'The geometry remains the previous validation-selected one-bank model with four frequencies per feature and a 64-dimensional canonical metric. These results do not establish success of the four-bank geometry. The spectral-only predictor contains no tree predictions or tree-derived features. Its geometry was already trained in the earlier experiment; reported additional times cover decoder selection and exact final fitting, not that prior training.','',
      'The same holdouts have been inspected in earlier experiments, so the comparison is exploratory. Random-split SD is descriptive and does not establish statistical equivalence. Exact decoding removes the earlier approximation but requires quadratic kernel storage and cubic factorization; this is a compute/accuracy tradeoff, not a scalable production solution.','',
      '## Per split','', '| Seed | Selected ridge | Clip | Validation RMSE | Test RMSE dollars | Raw test RMSE dollars | Final solve seconds |', '|---|---:|---|---:|---:|---:|---:|']
    for r in results:lines.append(f'| {r["seed"]} | {r["selected"]["ridge"]:.2g} | {r["selected"]["clip"]} | {r["selected"]["validation_rmse"]:.5f} | {r["test"]["rmse_dollars"]:,.0f} | {r["raw_test"]["rmse_dollars"]:,.0f} | {r["refit_and_prediction_seconds"]:.1f} |')
    lines+=['','## Reproduction','',
      'Run `spectral_exact_california.py` in the benchmark environment. See the protocol, frozen selection file, source snapshots, checkpoint hashes, prediction arrays and solver residuals for provenance.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines));print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');args=p.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(4):main(args)
