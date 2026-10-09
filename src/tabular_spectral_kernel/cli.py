"""Command-line reproduction of the paper; paths are supplied by the caller."""
import argparse,hashlib,json,math,platform,shutil,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from . import datasets
from ._paper import identifiable_spectral_california as base
from ._paper import spectral_stage1_extended as ext
from ._paper import spectral_competitive_train as refinement
from ._paper import spectral_competitive_finalize as final
from ._paper import spectral_exact_california as original_exact
from ._paper.solvers import exact_path

RECORDED=Path(__file__).resolve().parent/'recorded'

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')

def read(path):return json.loads(Path(path).read_text())

def configure(args):
    output=Path(args.output).expanduser().resolve();data=Path(args.data_dir).expanduser().resolve()
    device=args.device
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable. Install a suitable PyTorch build or use --device cpu.')
    torch.set_num_threads(args.threads);ext.DEVICE=device
    base.ROOT=output;base.DEFAULT_OUT=output/'identifiable_spectral_california';base.DATA_HOME=data/'california'
    ext.OUT=output/'spectral_stage1_extended';ext.DATA=data
    refinement.OUT=output/'spectral_competitive_california';final.OUT=refinement.OUT/'full_refit';original_exact.OUT=output/'spectral_exact_california'
    return output,data,device

def lock_run(output,args,device):
    output.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ['output','data_dir']}
    config['device']=device;path=output/'run_config.json'
    if path.exists() and read(path)!=config:raise ValueError('Output directory contains a different run configuration; choose a new --output directory.')
    save(path,config)
    import importlib.metadata as md
    save(output/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),device=device,packages={p:md.version(p) for p in ['torch','numpy','scipy','scikit-learn','pandas','catboost','lightgbm']}))

def verify(output):
    records=[]
    def check(path,array_path,classification=False):
        row=read(path);z=np.load(array_path);truth=z['truth'] if 'truth' in z else z['y_test'];pred=z['prediction']
        observed=ext.metrics(dict(classification=classification),truth,pred) if 'truth' in z else base.scores(truth,pred)
        for key,value in row['test'].items():
            if key in observed:np.testing.assert_allclose(value,observed[key],atol=1e-10,rtol=1e-10)
        keys=[k for k in ['train_indices','validation_indices','calibration_indices','test_indices'] if k in z]
        for i,a in enumerate(keys):
            for b in keys[i+1:]:assert not np.intersect1d(z[a],z[b]).size,(path,a,b)
        assert np.isfinite(pred).all()
        records.append(dict(file=str(path.relative_to(RECORDED)),metrics=observed))
    initial=RECORDED/'identifiable_spectral_california'
    for s in range(3):
        for family in ['spectral','catboost','lightgbm']:check(initial/f'{family}_seed{s}.json',initial/f'{family}_seed{s}_predictions.npz')
        exact=RECORDED/'spectral_exact_california';check(exact/f'result_seed{s}.json',exact/f'seed{s}_predictions.npz')
        for family in ['spectral_full_refit','spectral_frozen_refined']:
            folder=RECORDED/'spectral_competitive_california/full_refit';check(folder/f'{family}_seed{s}.json',folder/f'{family}_seed{s}_predictions.npz')
    for name in ['bike','taiwan','adult']:
        for s in range(3):
            folder=RECORDED/'spectral_stage1_extended'/name/f'seed{s}'
            for family in ['spectral','catboost']:check(folder/f'{family}_test.json',folder/f'{family}_test.npz',name!='bike')
    manifest=read(RECORDED/'manifest.json')
    for name,digest in manifest.items():assert hashlib.sha256((RECORDED/name).read_bytes()).hexdigest()==digest,name
    import csv
    groups={}
    for record in records:
        parts=Path(record['file']).parts
        dataset=parts[1] if parts[0]=='spectral_stage1_extended' else 'california'
        model=Path(parts[-1]).stem.split('_test')[0].split('_seed')[0]
        if parts[0]=='spectral_exact_california':model='spectral_exact_initial'
        for metric,value in record['metrics'].items():groups.setdefault((dataset,model,metric),[]).append(value)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'paper_results.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['dataset','model','metric','mean','std','runs'])
        for key,values in sorted(groups.items()):writer.writerow([*key,float(np.mean(values)),float(np.std(values,ddof=1)),len(values)])
    result=dict(passed=True,prediction_sets=len(records),artifact_hashes=len(manifest),results=records)
    save(output/'verification.json',result);print(f'PASS: {len(records)} prediction sets and {len(manifest)} artifact hashes; {output / "verification.json"}')
    return result

def smoke(output,device,steps=12):
    import pandas as pd
    rng=np.random.default_rng(17);x=rng.normal(size=(128,3));category=np.where(x[:,0]>0,'A','B')
    df=pd.DataFrame(x,columns=['x1','x2','x3']);df['category']=category
    y=np.sin(x[:,0])+x[:,1]*.4+(category=='A')*.2+rng.normal(0,.05,128)
    d=dict(df=df,nums=['x1','x2','x3'],cats=['category'],tr=np.arange(96),va=np.arange(96,128))
    prep=ext.MixedPreprocessor('standard').fit(d);xt=ext.tensor(prep.transform(df.iloc[:96]));xv=ext.tensor(prep.transform(df.iloc[96:]));yt=ext.tensor(y[:96])
    torch.manual_seed(17);m=ext.MixedSpectral(3,prep.q);opt=torch.optim.Adam(m.parameters(),lr=.003);losses=[]
    for _ in range(steps):
        ids=torch.randperm(96,device=device)[:48];s,q=ids[:32],ids[32:];opt.zero_grad();z=m.embedding(xt[ids]);K=m.kernel(z[:32],z[:32]);mu=yt[s].mean()
        L=torch.linalg.cholesky(K+m.log_ridge.exp()*torch.eye(32,dtype=torch.float64,device=device));a=torch.cholesky_solve((yt[s]-mu)[:,None],L)[:,0]
        loss=(m.kernel(z[32:],z[:32])@a+mu-yt[q]).square().mean();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step();ext.constrain(m);losses.append(float(loss.detach()))
    h=ext.head(m,xt,yt,1e-3);pred=ext.head_predict(m,h,xv)
    assert h['residual']<1e-8 and np.isfinite(pred).all()
    result=dict(mode='synthetic smoke test, not a paper benchmark',device=device,steps=steps,rmse=float(np.sqrt(np.mean((pred-y[96:])**2))),ridge_residual=h['residual'],losses=losses)
    save(output/'smoke.json',result);torch.save(dict(state_dict={k:v.cpu() for k,v in m.state_dict().items()}),output/'smoke_model.pt');print(json.dumps(result,indent=2))
    return result

def replay_extended(name,seeds,output,device):
    from catboost import CatBoostClassifier,CatBoostRegressor
    results=[]
    for seed in seeds:
        d=ext.load(name,seed);folder=RECORDED/'spectral_stage1_extended'/name/f'seed{seed}'
        selection=read(folder/'spectral_selection.json');prep=ext.MixedPreprocessor(selection['kind']).fit(d)
        m=ext.MixedSpectral(len(d['nums']),prep.q);cp=torch.load(folder/selection['kind']/'checkpoint.pt',map_location=device,weights_only=True);m.load_state_dict(cp['state_dict']);m.eval()
        x=ext.tensor(prep.transform(d['df'].iloc[d['tr']]));y=ext.tensor((d['y'][d['tr']]-cp['mu'])/cp['sd'])
        h=ext.head(m,x,y,selection['selected']['ridge']);raw=ext.head_predict(m,h,ext.tensor(prep.transform(d['df'].iloc[d['te']])))*cp['sd']+cp['mu']
        for family in ['spectral','catboost']:
            sel=read(folder/f'{family}_selection.json')
            if family=='catboost':
                model=(CatBoostClassifier if d['classification'] else CatBoostRegressor)();model.load_model(str(folder/'catboost.cbm'));raw=model.predict(d['df'].iloc[d['te']],prediction_type='RawFormulaVal',thread_count=4)
            pred=ext.apply_cal(raw,sel['calibration']) if d['classification'] else (np.clip(raw,0,d['y'][d['tr']].max()) if family=='spectral' and sel['selected']['clip'] else raw)
            stored=np.load(folder/f'{family}_test.npz');np.testing.assert_array_equal(stored['test_indices'],d['te']);np.testing.assert_array_equal(stored['truth'],d['y'][d['te']])
            error=float(np.max(np.abs(pred-stored['prediction'])))
            np.testing.assert_allclose(pred,stored['prediction'],atol=2e-6,rtol=1e-7)
            row=dict(dataset=name,seed=seed,family=family,test=ext.metrics(d,d['y'][d['te']],pred),max_replay_error=error);results.append(row)
            save(output/name/f'seed{seed}'/f'{family}.json',row);print(json.dumps(row),flush=True)
        del m,x,y,h;torch.cuda.empty_cache()
    save(output/name/'replay.json',dict(results=results))

def replay_california(seeds,output,device):
    results=[]
    for seed in seeds:
        folder=RECORDED/'spectral_competitive_california/full_refit';row=read(folder/f'spectral_full_refit_seed{seed}.json')
        cp=torch.load(folder/f'refit_seed{seed}.pt',map_location=device,weights_only=True);m=refinement.model_for(cp['config'],device);m.load_state_dict(cp['state_dict']);m.eval()
        X,y,tr,va,outer,te=base.data_split(seed);scaler=refinement.HousingScaler('standard').fit(X[tr]);mu,sd=cp['target_mean'],cp['target_std']
        pred,res=exact_path(m,base.as_tensor(scaler.transform(X[outer]),device),base.as_tensor((y[outer]-mu)/sd,device),base.as_tensor(scaler.transform(X[te]),device),[row['selected']['ridge']])
        pred=pred[:,0].cpu().numpy()*sd+mu
        if row['selected']['clip']:pred=np.clip(pred,y[outer].min(),y[outer].max())
        saved=np.load(folder/f'spectral_full_refit_seed{seed}_predictions.npz');np.testing.assert_array_equal(saved['test_indices'],te);np.testing.assert_array_equal(saved['y_test'],y[te]);np.testing.assert_allclose(pred,saved['prediction'],atol=2e-6,rtol=1e-7)
        result=dict(dataset='california',seed=seed,test=base.scores(y[te],pred),residual=max(res),max_replay_error=float(np.max(np.abs(pred-saved['prediction']))));results.append(result);save(output/'california'/f'seed{seed}.json',result);print(json.dumps(result),flush=True)
    save(output/'california/replay.json',dict(results=results))

def train_extended(name,device):
    ext.DATASETS=[name];ext.SEEDS=[0,1,2];ext.prepare()
    fixed=None
    for seed in ext.SEEDS:fixed=ext.spectral_select(name,seed,fixed)['kind']
    fixed=None
    for seed in ext.SEEDS:fixed=ext.cat_select(name,seed,fixed)['selected']['config']
    ext.evaluate()

def train_california(device,initialization):
    args=SimpleNamespace(out=base.DEFAULT_OUT,seeds=[0,1,2],steps=300,trials=16,landmarks=512,device=device)
    base.DEFAULT_OUT.mkdir(parents=True,exist_ok=True)
    if initialization=='search':base.spectral(args)
    else:
        # These are the original inner-trained warm starts, never test-fitted models.
        for seed in args.seeds:
            for suffix in ['.pt','.json','_predictions.npz']:
                name=f'spectral_seed{seed}{suffix}';shutil.copy2(RECORDED/'identifiable_spectral_california'/name,base.DEFAULT_OUT/name)
    for seed in args.seeds:
        cfg=torch.load(base.DEFAULT_OUT/f'spectral_seed{seed}.pt',map_location='cpu',weights_only=True)['config']
        if cfg['atoms']!=4 or cfg['preprocessing']!='standard':
            raise ValueError('The initialization search selected a different architecture or preprocessing from the paper warm start. Use --initialization recorded with a new output directory to run the published refinement recipe.')
    base.trees(args)
    original_exact.main(SimpleNamespace(device=device))
    refinement.prepare();recipe=read(RECORDED/'spectral_competitive_california/recipe_frozen.json')
    save(refinement.OUT/'recipe_frozen.json',recipe)
    rows=[refinement.train(s,recipe['config'],device) for s in args.seeds]
    save(refinement.OUT/'all_selections_frozen.json',rows)
    final.prepare()
    for row in rows:final.refit(row,device)
    final.evaluate(rows,device)

def main(argv=None):
    parser=argparse.ArgumentParser(description='Reproduce Learning Spectral Geometry for Kernel Regression')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['smoke','verify','assets','download','reproduce']:
        p=sub.add_parser(name);p.add_argument('--output',default='runs');p.add_argument('--data-dir',default='data');p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--threads',type=int,default=4)
        if name in ['download','reproduce']:p.add_argument('--dataset',choices=['california','bike','taiwan','adult','all'],default='all')
        if name=='reproduce':
            p.add_argument('--mode',choices=['replay','train'],default='replay');p.add_argument('--seeds',nargs='+',type=int,choices=[0,1,2],default=[0,1,2]);p.add_argument('--initialization',choices=['recorded','search'],default='recorded')
    args=parser.parse_args(argv)
    if args.threads<1:parser.error('--threads must be positive')
    output,data,device=configure(args)
    with threadpool_limits(args.threads):
        if args.command=='verify':verify(output);return
        if args.command=='assets':
            verify(output)
            from .assets import build
            build(output);return
        if args.command=='smoke':lock_run(output,args,device);smoke(output,device);return
        names=['california','bike','taiwan','adult'] if args.dataset=='all' else [args.dataset]
        if args.command=='reproduce':
            if args.mode=='train' and args.seeds!=[0,1,2]:parser.error('Paper training requires --seeds 0 1 2; use replay for a subset.')
            lock_run(output,args,device)
        for name in names:
            datasets.prepare(name,data)
            if args.command=='download':print(f'{name}: verified inputs in {data}');continue
            if args.mode=='replay':
                if name=='california':replay_california(args.seeds,output,device)
                else:replay_extended(name,args.seeds,output,device)
            elif name=='california':train_california(device,args.initialization)
            else:
                ext.OUT=output/f'{name}_training';train_extended(name,device)

if __name__=='__main__':main()
