"""Predeclared full-data benchmark of the canonical matrix-exponential kernel.

Use the installed tsk command; see README.md.
Subcommands: check, spectral, trees, report. No test data are used for selection.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path.cwd() / 'runs'
DATA_HOME = Path.cwd() / 'data' / 'california'
DEFAULT_OUT = ROOT / 'identifiable_spectral_california'
DTYPE = torch.float64
RIDGES = np.logspace(-7, -2, 11)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def scores(y, pred):
    assert y.shape == pred.shape and np.isfinite(pred).all()
    return dict(rmse=float(np.sqrt(mean_squared_error(y, pred))),
                rmse_dollars=float(np.sqrt(mean_squared_error(y, pred)) * 100000),
                mae_dollars=float(mean_absolute_error(y, pred)*100000), r2=float(r2_score(y, pred)))


def data_split(seed):
    data = fetch_california_housing(data_home=DATA_HOME, download_if_missing=True)
    outer, test = train_test_split(np.arange(len(data.target)), test_size=.2, random_state=seed)
    train, val = train_test_split(outer, test_size=.2, random_state=1000 + seed)
    assert not set(train) & set(val) and not set(outer) & set(test)
    return data.data, data.target, train, val, outer, test


class CanonicalSpectral(torch.nn.Module):
    """Only ordered frequencies, unique trace-zero coordinates, log T and ridge."""
    def __init__(self, d, atoms, max_frequency=1., device='cpu'):
        super().__init__()
        self.d, self.atoms = d, atoms
        self.p = 2*d*atoms
        p = self.p
        ij = torch.triu_indices(p, p, offset=1, device=device)
        self.register_buffer('ii', ij[0]); self.register_buffer('jj', ij[1])
        # Orthonormal Helmert contrasts: an actual p-1 dimensional trace-zero basis.
        h = torch.zeros(p-1, p, dtype=DTYPE, device=device)
        for r in range(1, p):
            h[r-1, :r] = 1/math.sqrt(r*(r+1))
            h[r-1, r] = -r/math.sqrt(r*(r+1))
        self.register_buffer('helmert', h)
        self.offdiag = torch.nn.Parameter(torch.zeros(len(ij[0]), dtype=DTYPE, device=device))
        self.diagonal = torch.nn.Parameter(torch.zeros(p-1, dtype=DTYPE, device=device))
        omega = torch.logspace(math.log10(.025), math.log10(max_frequency), atoms, dtype=DTYPE, device=device)
        gaps = torch.diff(omega, prepend=torch.zeros(1,dtype=DTYPE,device=device))
        self.log_gaps = torch.nn.Parameter(gaps.log().repeat(d,1))
        self.log_temperature = torch.nn.Parameter(torch.tensor(math.log(.5),dtype=DTYPE,device=device))
        self.log_ridge = torch.nn.Parameter(torch.tensor(math.log(3e-4),dtype=DTYPE,device=device))

    def generator(self):
        a = torch.diag(self.diagonal @ self.helmert)
        a = a.index_put((self.ii, self.jj), self.offdiag/math.sqrt(2))
        return a+a.T-torch.diag(torch.diagonal(a))

    def normalized_root(self):
        # exp(A/2)^2=exp(A); ||exp(A/2)||_F^2=tr exp(A) for symmetric A.
        root = torch.matrix_exp(.5*self.generator())
        return root / torch.linalg.vector_norm(root)

    def frequencies(self):
        return self.log_gaps.exp().cumsum(-1)

    def embedding(self, x, root=None):
        if root is None: root = self.normalized_root()
        phase = 2*math.pi*x[:,:,None]*self.frequencies()[None]
        raw = torch.stack((phase.cos(), phase.sin()), dim=-1).flatten(1)
        return raw @ root

    def kernel(self, za, zb):
        # Direct cdist avoids sqrt-of-rounded-negative quadratic distances.
        dist = torch.cdist(za, zb, compute_mode='donot_use_mm_for_euclid_dist')
        return torch.exp(-dist/self.log_temperature.exp())


def as_tensor(a, device):
    return torch.as_tensor(a,dtype=DTYPE,device=device)


def kernel_blocks(model, za, zb, chunk=1024):
    return torch.cat([model.kernel(za[i:i+chunk],zb) for i in range(0,len(za),chunk)])


@torch.no_grad()
def nystrom_path(model, x, y, xq, landmark_indices, ridges):
    root = model.normalized_root()
    z = model.embedding(x,root); zq = model.embedding(xq,root)
    landmarks = z[landmark_indices]
    kmm = model.kernel(landmarks,landmarks)
    chol = torch.linalg.cholesky(kmm + 1e-8*torch.eye(len(landmarks),dtype=DTYPE,device=x.device))
    phi = torch.linalg.solve_triangular(chol,kernel_blocks(model,z,landmarks).T,upper=False).T
    phiq = torch.linalg.solve_triangular(chol,kernel_blocks(model,zq,landmarks).T,upper=False).T
    pm = phi.mean(0); ym = y.mean()
    a = phi-pm; b = phiq-pm
    cov = a.T@a/len(y)
    eig,u = torch.linalg.eigh((cov+cov.T)*.5)
    rhs = u.T @ (a.T@(y-ym)/len(y))
    rr = as_tensor(ridges,x.device)
    coefs = u @ (rhs[:,None]/(eig.clamp_min(0)[:,None]+rr[None]))
    predictions = b@coefs+ym
    return predictions


def preprocess(name, x, seed):
    scaler = (StandardScaler() if name=='standard' else
              QuantileTransformer(n_quantiles=min(1000,len(x)),output_distribution='normal',random_state=seed))
    scaler.fit(x)
    return scaler


def spectral_configs():
    return [dict(preprocessing=prep,atoms=k,lr=lr,max_frequency=mx)
            for prep in ['standard','quantile']
            for k in [4,8]
            for lr,mx in [(.02,.5),(.04,1.5)]]


def protocol(args):
    return dict(dataset='California Housing', samples=20640, features=8,
      seeds=args.seeds, outer_test_fraction=.2, inner_validation_fraction=.2,
      split='sklearn train_test_split: outer random_state=seed; inner random_state=1000+seed',
      spectral=dict(parameterization='ordered cumulative exp frequencies; orthonormal trace-zero symmetric A; exact torch.matrix_exp(A/2) normalized root; log bandwidth',
        banks=1,configs=spectral_configs(),steps=args.steps,support_batch=384,query_batch=256,
        objective='MSE on disjoint support/query episodes sampled from inner training only; differentiate exact support KRR',
        validation='All inner-training rows, uniform training landmarks, centered Nystrom ridge path; every 50 steps and initialization',
        validation_landmarks=args.landmarks,final_rank_candidates=[args.landmarks,1024,2048],
        ridge_mean_loss_grid=RIDGES.tolist(),final_refit='Freeze geometry and preprocessing; select rank/ridge on validation; fit centered Nystrom ridge on all outer-training rows'),
      trees=dict(trials=args.trials,search='Fixed scrambled Sobol design plus a default-like first candidate',
        max_iterations=4000,early_stopping_rounds=150,
        lightgbm_space=dict(num_leaves=[15,255],max_depth=[-1,4,6,8,10],learning_rate=[.015,.15],
          min_child_samples=[5,100],colsample_bytree=[.65,1],subsample=[.65,1],reg_alpha=[1e-5,10],reg_lambda=[1e-4,30]),
        catboost_space=dict(depth=[4,10],learning_rate=[.02,.15],l2_leaf_reg=[1,30],random_strength=[.01,3],bagging_temperature=[0,2]),
        final_refit='Selected hyperparameters and iteration count, all outer-training rows'),
      comparison='Same row splits and target labels; different search counts/compute are reported, not equal-compute HPO',
      scope='Single-table regression test of the identifiable geometry, not pretrained cross-table transfer',
      evaluation='No outer-test labels used for model selection; no post-test hyperparameter adaptation')


def init_output(args):
    args.out.mkdir(parents=True,exist_ok=True)
    desired=protocol(args)
    path=args.out/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())==desired, 'Existing protocol differs; use a new output directory.'
    else: save(path,desired)
    snap=args.out/'executed_source.py'
    current=Path(__file__).read_bytes()
    if snap.exists(): assert snap.read_bytes()==current, 'Source changed; use a new output directory.'
    else: snap.write_bytes(current)


def spectral(args):
    init_output(args)
    device=torch.device(args.device)
    for seed in args.seeds:
        if (args.out/f'spectral_seed{seed}.json').exists(): continue
        started=time.perf_counter(); X,y,tr,va,outer,te=data_split(seed)
        ym,ys=float(y[tr].mean()),float(y[tr].std())
        yt=as_tensor((y[tr]-ym)/ys,device); yv=as_tensor((y[va]-ym)/ys,device)
        rng=np.random.default_rng(seed+77)
        landmark_order=rng.permutation(len(tr))
        train_indices=torch.tensor(landmark_order[:args.landmarks],device=device)
        candidates=[]; winner=None
        for trial,cfg in enumerate(spectral_configs()):
            torch.manual_seed(2000+seed)
            scaler=preprocess(cfg['preprocessing'],X[tr],seed)
            xt=as_tensor(scaler.transform(X[tr]),device); xv=as_tensor(scaler.transform(X[va]),device)
            model=CanonicalSpectral(8,cfg['atoms'],cfg['max_frequency'],device)
            opt=torch.optim.Adam([
              {'params':[model.offdiag,model.diagonal], 'lr':cfg['lr']},
              {'params':[model.log_gaps], 'lr':cfg['lr']*.2},
              {'params':[model.log_temperature,model.log_ridge], 'lr':cfg['lr']}])
            trace=[]; best=(float('inf'),None,None,None)
            gen=torch.Generator(device=device).manual_seed(3000+seed)
            trial_start=time.perf_counter()
            for step in range(args.steps+1):
                if step%50==0 or step==args.steps:
                    pred=nystrom_path(model,xt,yt,xv,train_indices,RIDGES)
                    errors=(pred-yv[:,None]).square().mean(0)
                    j=int(errors.argmin()); loss=float(errors[j])
                    row=dict(step=step,validation_rmse=math.sqrt(loss)*ys,
                             ridge=float(RIDGES[j]),temperature=float(model.log_temperature.exp().detach()),
                             elapsed_seconds=time.perf_counter()-trial_start)
                    trace.append(row)
                    print(f'spectral seed={seed} trial={trial} step={step} val_RMSE={row["validation_rmse"]:.5f}',flush=True)
                    if loss<best[0]: best=(loss,copy.deepcopy(model.state_dict()),step,float(RIDGES[j]))
                if step==args.steps: break
                idx=torch.randperm(len(xt),generator=gen,device=device)[:640]
                s,q=idx[:384],idx[384:]
                opt.zero_grad(set_to_none=True)
                root=model.normalized_root()
                z=model.embedding(xt[idx],root); zs,zq=z[:384],z[384:]
                kss=model.kernel(zs,zs); kqs=model.kernel(zq,zs)
                center=yt[s].mean(); target=yt[s]-center
                ridge=model.log_ridge.exp()
                l=torch.linalg.cholesky(kss+len(s)*ridge*torch.eye(len(s),dtype=DTYPE,device=device))
                coef=torch.cholesky_solve(target[:,None],l)[:,0]
                err=(kqs@coef+center-yt[q]).square().mean()
                # Modest fixed shape regularizer, declared independently of test outcomes.
                regularizer=1e-4*(model.offdiag.square().sum()+model.diagonal.square().sum())/model.p
                (err+regularizer).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                opt.step()
                with torch.no_grad():
                    model.log_gaps.clamp_(-9,2)
                    model.log_temperature.clamp_(-5,3)
                    model.log_ridge.clamp_(math.log(1e-7),math.log(.01))
            record=dict(trial=trial,config=cfg,selected_step=best[2],validation_rmse=math.sqrt(best[0])*ys,
                        ridge=best[3],seconds=time.perf_counter()-trial_start,trace=trace)
            candidates.append(record)
            save(args.out/f'spectral_seed{seed}_trials.json',candidates)
            if winner is None or best[0]<winner['loss']:
                winner=dict(loss=best[0],state=best[1],config=cfg,trial=trial,step=best[2])
        # Geometry is now selected. Rank and ridge refinement uses validation only.
        cfg=winner['config']; scaler=preprocess(cfg['preprocessing'],X[tr],seed)
        xt=as_tensor(scaler.transform(X[tr]),device); xv=as_tensor(scaler.transform(X[va]),device)
        model=CanonicalSpectral(8,cfg['atoms'],cfg['max_frequency'],device)
        model.load_state_dict(winner['state']); model.eval()
        rank_trials=[]; rank_best=(float('inf'),None,None)
        for rank in sorted(set([args.landmarks,1024,2048])):
            pred=nystrom_path(model,xt,yt,xv,torch.tensor(landmark_order[:rank],device=device),RIDGES)
            errors=(pred-yv[:,None]).square().mean(0); j=int(errors.argmin()); loss=float(errors[j])
            rank_trials.append(dict(rank=rank,ridge=float(RIDGES[j]),validation_rmse=math.sqrt(loss)*ys))
            print(f'spectral seed={seed} rank={rank} val_RMSE={math.sqrt(loss)*ys:.5f}',flush=True)
            if loss<rank_best[0]:rank_best=(loss,rank,float(RIDGES[j]))
        selection_seconds=time.perf_counter()-started
        # First use of outer test features; all choices are frozen.
        xf=as_tensor(scaler.transform(X[outer]),device); xte=as_tensor(scaler.transform(X[te]),device)
        yf=as_tensor((y[outer]-ym)/ys,device)
        final_landmarks=np.random.default_rng(seed+88).permutation(len(outer))[:rank_best[1]]
        final_start=time.perf_counter()
        pred=nystrom_path(model,xf,yf,xte,torch.tensor(final_landmarks,device=device),[rank_best[2]])[:,0].cpu().numpy()*ys+ym
        final_seconds=time.perf_counter()-final_start
        with torch.no_grad():
            A=model.generator(); root=model.normalized_root(); B=root@root.T
            eigen=torch.linalg.eigvalsh(B)
            w=B.diagonal().reshape(8,cfg['atoms'],2).sum(-1)
            importance=w.sum(-1); amplitude=(w/importance[:,None]).sqrt()
            diagnostics=dict(generator_trace=float(A.trace()),metric_trace=float(B.trace()),
                metric_min_eigenvalue=float(eigen.min()),metric_max_eigenvalue=float(eigen.max()),
                metric_condition=float(eigen.max()/eigen.min()),importance=importance.cpu().tolist(),
                amplitudes=amplitude.cpu().tolist(),frequencies=model.frequencies().cpu().tolist(),
                temperature=float(model.log_temperature.exp()),matrix_dimension=model.p,
                metric_parameters=model.p*(model.p+1)//2-1)
        result=dict(seed=seed,model='identifiable_spectral_laplace',config=cfg,
            selected_trial=winner['trial'],selected_step=winner['step'],rank_trials=rank_trials,
            selected_rank=rank_best[1],selected_ridge=rank_best[2],validation_rmse=math.sqrt(rank_best[0])*ys,
            test=scores(y[te],pred),selection_seconds=selection_seconds,final_fit_and_predict_seconds=final_seconds,
            total_seconds=time.perf_counter()-started,diagnostics=diagnostics,
            versions=dict(torch=torch.__version__,numpy=np.__version__),hardware=str(device))
        torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},config=cfg,
                        rank=rank_best[1],ridge=rank_best[2]),args.out/f'spectral_seed{seed}.pt')
        np.savez_compressed(args.out/f'spectral_seed{seed}_predictions.npz',train_indices=tr,validation_indices=va,
                            outer_train_indices=outer,test_indices=te,landmark_positions=final_landmarks,y_test=y[te],prediction=pred)
        save(args.out/f'spectral_seed{seed}.json',result)
        print(f'SPECTRAL FINISHED seed={seed}: {result["test"]}',flush=True)
        del model,xt,xv,xf,xte
        if device.type=='cuda':torch.cuda.empty_cache()


def tree_configs(count):
    points=qmc.Sobol(8,scramble=True,seed=913).random_base2(int(math.ceil(math.log2(count))))[:count]
    lgb=[]; cat=[]
    log=lambda u,lo,hi: float(np.exp(np.log(lo)+u*np.log(hi/lo)))
    for u in points:
        lgb.append(dict(num_leaves=int(round(log(u[0],15,255))),max_depth=[-1,4,6,8,10][min(int(u[1]*5),4)],
          learning_rate=log(u[2],.015,.15),min_child_samples=int(5+u[3]*95),colsample_bytree=float(.65+.35*u[4]),
          subsample=float(.65+.35*u[5]),subsample_freq=1,reg_alpha=log(u[6],1e-5,10),reg_lambda=log(u[7],1e-4,30)))
        cat.append(dict(depth=int(4+min(int(u[0]*7),6)),learning_rate=log(u[1],.02,.15),
          l2_leaf_reg=log(u[2],1,30),random_strength=log(u[3],.01,3),bagging_temperature=float(2*u[4])))
    lgb[0]=dict(num_leaves=31,max_depth=-1,learning_rate=.05,min_child_samples=20,colsample_bytree=1.,subsample=1.,subsample_freq=1,reg_alpha=0.,reg_lambda=1.)
    cat[0]=dict(depth=6,learning_rate=.05,l2_leaf_reg=3.,random_strength=1.,bagging_temperature=1.)
    return dict(lightgbm=lgb,catboost=cat)


def trees(args):
    import lightgbm as lgb
    import catboost
    init_output(args)
    configs=tree_configs(args.trials)
    save(args.out/'tree_search_configs.json',configs)
    for seed in args.seeds:
        X,y,tr,va,outer,te=data_split(seed)
        for name,candidates in configs.items():
            if (args.out/f'{name}_seed{seed}.json').exists():continue
            start=time.perf_counter(); trials=[]; best=(float('inf'),None,None)
            for trial,cfg in enumerate(candidates):
                t=time.perf_counter()
                if name=='lightgbm':
                    fit=lgb.LGBMRegressor(**cfg,n_estimators=4000,random_state=seed,n_jobs=4,verbosity=-1)
                    fit.fit(X[tr],y[tr],eval_set=[(X[va],y[va])],callbacks=[lgb.early_stopping(150,verbose=False)])
                    rounds=fit.best_iteration_
                else:
                    fit=catboost.CatBoostRegressor(**cfg,iterations=4000,loss_function='RMSE',random_seed=seed,
                        thread_count=4,verbose=False,allow_writing_files=False,bootstrap_type='Bayesian')
                    fit.fit(X[tr],y[tr],eval_set=(X[va],y[va]),early_stopping_rounds=150,verbose=False)
                    rounds=fit.get_best_iteration()+1
                pred=fit.predict(X[va]); error=float(np.sqrt(mean_squared_error(y[va],pred)))
                trials.append(dict(trial=trial,config=cfg,validation_rmse=error,iterations=rounds,seconds=time.perf_counter()-t))
                save(args.out/f'{name}_seed{seed}_trials.json',trials)
                print(f'{name} seed={seed} trial={trial} val_RMSE={error:.5f} rounds={rounds}',flush=True)
                if error<best[0]:best=(error,cfg,rounds)
            selection_seconds=time.perf_counter()-start; t=time.perf_counter()
            if name=='lightgbm':
                final=lgb.LGBMRegressor(**best[1],n_estimators=best[2],random_state=seed,n_jobs=4,verbosity=-1)
            else:
                final=catboost.CatBoostRegressor(**best[1],iterations=best[2],loss_function='RMSE',random_seed=seed,
                    thread_count=4,verbose=False,allow_writing_files=False,bootstrap_type='Bayesian')
            final.fit(X[outer],y[outer]); pred=final.predict(X[te])
            result=dict(seed=seed,model=name,config=best[1],selected_iterations=best[2],validation_rmse=best[0],
              test=scores(y[te],pred),selection_seconds=selection_seconds,final_fit_and_predict_seconds=time.perf_counter()-t,
              total_seconds=time.perf_counter()-start,version=lgb.__version__ if name=='lightgbm' else catboost.__version__)
            save(args.out/f'{name}_seed{seed}.json',result)
            np.savez_compressed(args.out/f'{name}_seed{seed}_predictions.npz',train_indices=tr,validation_indices=va,
              outer_train_indices=outer,test_indices=te,y_test=y[te],prediction=pred)
            if name=='lightgbm':final.booster_.save_model(str(args.out/f'{name}_seed{seed}.txt'))
            else:final.save_model(str(args.out/f'{name}_seed{seed}.cbm'))
            print(f'{name.upper()} FINISHED seed={seed}: {result["test"]}',flush=True)


def checks():
    torch.manual_seed(41)
    model=CanonicalSpectral(2,2)
    with torch.no_grad():model.offdiag.normal_(0,.15);model.diagonal.normal_(0,.1)
    a=model.generator();r=model.normalized_root();b=r@r.T
    assert abs(float(a.trace().detach()))<1e-12
    assert abs(float(b.trace().detach())-1)<1e-12
    eigen,u=torch.linalg.eigh(b)
    logb=(u*eigen.log())@u.T
    inverse=logb-logb.trace()/model.p*torch.eye(model.p,dtype=DTYPE)
    assert torch.allclose(inverse,a,atol=1e-12)
    w=b.diagonal().reshape(-1,2).sum(-1)
    d=(w/2).sqrt().repeat_interleave(2)
    c=b/d[:,None]/d[None,:]
    assert torch.allclose(c.diagonal().reshape(-1,2).sum(-1),torch.full_like(w,2))
    x=torch.randn(12,2,dtype=DTYPE);z=model.embedding(x)
    k=model.kernel(z,z)
    assert torch.allclose(k.diag(),torch.ones(12,dtype=DTYPE))
    assert float(torch.linalg.eigvalsh(k).min().detach())>-1e-10
    p=torch.randperm(model.p)
    ap=a[p][:,p]; ep=torch.matrix_exp(ap);bp=ep/ep.trace()
    assert torch.allclose(bp,b[p][:,p],atol=1e-12)
    def exponential_map(theta):
        aa=torch.tensor([[1.,0.],[0.,-1.]],dtype=DTYPE)*theta[0]
        aa=aa+torch.tensor([[0.,1.],[1.,0.]],dtype=DTYPE)*theta[1]
        rr=torch.matrix_exp(.5*aa); return rr/torch.linalg.vector_norm(rr)
    assert torch.autograd.gradcheck(exponential_map,(torch.tensor([.1,.2],dtype=DTYPE,requires_grad=True),))
    y=torch.randn(len(x),dtype=DTYPE);xq=torch.randn(5,2,dtype=DTYPE)
    pred=nystrom_path(model,x,y,xq,torch.arange(len(x)),[1e-3])[:,0]
    # Independent centered dense kernel-ridge calculation.
    with torch.no_grad():
        kq=model.kernel(model.embedding(xq),z); n=len(x)
        h=torch.eye(n,dtype=DTYPE)-torch.ones(n,n,dtype=DTYPE)/n
        kc=h@k@h
        kqc=kq-kq.mean(1,keepdim=True)-k.mean(0,keepdim=True)+k.mean()
        exact=kqc@torch.linalg.solve(kc+n*1e-3*torch.eye(n,dtype=DTYPE),y-y.mean())+y.mean()
        assert torch.max(torch.abs(pred-exact))<2e-7
    # A complete training backward pass, including matrix exp, cdist and solve.
    loss=k.square().sum();loss.backward()
    for name in ['offdiag','diagonal','log_gaps','log_temperature']:
        assert torch.isfinite(getattr(model,name).grad).all(),name
    print('PASS: unique metric inverse, mass/shape decomposition, PSD/unit diagonal, orthogonal equivariance, matrix-exp gradcheck, full-landmark Nystrom/dense ridge agreement, training gradients.',flush=True)


def report(args):
    results=[]
    for seed in args.seeds:
        reference=None
        for name in ['spectral','lightgbm','catboost']:
            path=args.out/f'{name}_seed{seed}.json'
            if not path.exists():raise RuntimeError(f'Missing {path}')
            record=json.loads(path.read_text());results.append(record)
            a=np.load(args.out/f'{name}_seed{seed}_predictions.npz')
            if reference is None:reference={k:a[k] for k in ['train_indices','validation_indices','outer_train_indices','test_indices','y_test']}
            else:
                for k,v in reference.items():assert np.array_equal(v,a[k]),(name,seed,k)
            assert np.isclose(scores(a['y_test'],a['prediction'])['rmse'],record['test']['rmse'])
    summary={}
    for name in ['identifiable_spectral_laplace','lightgbm','catboost']:
        rows=[r for r in results if r['model']==name]
        summary[name]={key:dict(mean=float(np.mean([r['test'][key] for r in rows])),std=float(np.std([r['test'][key] for r in rows],ddof=1))) for key in ['rmse_dollars','mae_dollars','r2']}
        summary[name]['total_seconds']=sum(r['total_seconds'] for r in rows)
    save(args.out/'summary.json',dict(summary=summary,results=results,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    lines=['# Identifiable spectral kernel on California Housing','',
      'Three full-data random train/test splits; all models use the same rows. Each split has 13,209 inner-training rows, 3,303 validation rows and 4,128 test rows. Selected models refit their prediction heads on all 16,512 outer-training rows. Test labels do not select configurations.','',
      '| Model | Test RMSE in dollars, mean ± SD | Test R², mean ± SD | Total seconds |','|---|---:|---:|---:|']
    for name,s in summary.items():
        lines.append(f'| {name} | {s["rmse_dollars"]["mean"]:,.0f} ± {s["rmse_dollars"]["std"]:,.0f} | {s["r2"]["mean"]:.4f} ± {s["r2"]["std"]:.4f} | {s["total_seconds"]:.1f} |')
    lines += ['', '## Protocol and limitations','',
      f'The spectral model has one bank, eight configurations, {args.steps} episode-training steps per configuration and checkpoint/ridge selection on validation data. Its full normalized metric uses the exact matrix exponential. Large-data prediction uses an explicitly approximate centered Nyström ridge head; landmark count is selected from {sorted(set([args.landmarks,1024,2048]))}.', '',
      f'LightGBM and CatBoost each receive {args.trials} configurations, up to 4,000 iterations and validation early stopping. Search counts and computation differ; this is not an equal-compute comparison. CPU baselines and GPU spectral training may run concurrently, so wall times are not isolated hardware benchmarks.', '',
      'Results describe single-table regression with this parameterization and search budget. They do not establish pretrained cross-table transfer or a general ranking of model families. Three random splits are descriptive repeats, not independent population-level significance tests.', '',
      '## Per split results','', '| Seed | Model | Test RMSE dollars | Test R² |', '|---|---|---:|---:|']
    for r in results:lines.append(f'| {r["seed"]} | {r["model"]} | {r["test"]["rmse_dollars"]:,.0f} | {r["test"]["r2"]:.4f} |')
    lines += ['', '## Reproduction','', 'See `protocol.json`, `executed_source.py`, per-trial JSON files, selected models and prediction arrays. Run the script with `check`, then `spectral` and `trees`, followed by `report`, using the same CLI options.','']
    (args.out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','spectral','trees','report'])
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--trials',type=int,default=16)
    parser.add_argument('--landmarks',type=int,default=512)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(4):
        if args.command=='check':checks()
        else:globals()[args.command](args)


if __name__=='__main__':main()
