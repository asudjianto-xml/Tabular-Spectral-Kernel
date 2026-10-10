"""Chronological real-data test of spectral retrieval and context-conditioned PSD fusion.
No test-label tuning. Existing outputs are never overwritten. See protocol.json.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import argparse, copy, hashlib, importlib.util, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
DATA=runtime.bike_path()
from tabular_spectral_kernel._paper import identifiable_spectral_california as base
SOURCE=Path(base.__file__)
DEVICE=runtime.device()
N=128; LAGS=[0,7,28,56,91,182]; RIDGES=[1e-5,1e-4,1e-3,1e-2]
FEATURES=['hour_sin','hour_cos','weekday_sin','weekday_cos','year_sin','year_cos',
          'temp','hum','windspeed','workingday','holiday','weather1','weather2','weather3','weather4']

def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def t(x):return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)
def clone(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

def load():
    runtime.require_bike(DATA)
    df=pd.read_csv(DATA).sort_values(['dteday','hr']).reset_index(drop=True)
    dt=pd.to_datetime(df.dteday)+pd.to_timedelta(df.hr,unit='h')
    assert dt.is_unique and dt.is_monotonic_increasing and not df.isna().any().any()
    doy=pd.to_datetime(df.dteday).dt.dayofyear.to_numpy()
    x=np.column_stack([np.sin(2*np.pi*df.hr/24),np.cos(2*np.pi*df.hr/24),
        np.sin(2*np.pi*df.weekday/7),np.cos(2*np.pi*df.weekday/7),
        np.sin(2*np.pi*(doy-1)/365.25),np.cos(2*np.pi*(doy-1)/365.25),
        df.temp,df.hum,df.windspeed,df.workingday,df.holiday,
        *[(df.weathersit==i).astype(float) for i in range(1,5)]]).astype(float)
    early=(dt<np.datetime64('2011-07-01')).to_numpy()
    center=x[early].mean(0);scale=x[early].std(0);scale[scale<.1]=1
    # Shared scaling fitted before all adaptation episodes. The categorical vocabulary is specified, not inferred from test.
    x=(x-center)/scale
    y=df.cnt.to_numpy(float); ys=float(y[early].std())
    return dict(df=df,dt=dt.to_numpy(),x=x,y=y,y_scaled=y/ys,ys=ys,center=center,scale=scale)

def solve(k,cross,y,ridge):
    """Batched exact KRR; k [...,n,n], cross [...,n], y [...,n]."""
    n=k.shape[-1]; a=k+n*ridge*torch.eye(n,dtype=k.dtype,device=k.device)
    chol=torch.linalg.cholesky(a)
    rhs=torch.stack([y,torch.ones_like(y)],-1)
    sol=torch.cholesky_solve(rhs,chol)
    b=sol[...,0].sum(-1)/sol[...,1].sum(-1)
    coef=sol[...,0]-b[...,None]*sol[...,1]
    return (cross*coef).sum(-1)+b

def bank_train(d,seed,out,steps):
    records=[];zs=[];temps=[]
    for j,(start,end) in enumerate([('2011-01-01','2011-03-01'),('2011-03-01','2011-05-01'),('2011-05-01','2011-07-01')]):
        ix=np.flatnonzero((d['dt']>=np.datetime64(start))&(d['dt']<np.datetime64(end)))
        cut=int(.8*len(ix)); train,val=ix[:cut],ix[cut:]
        torch.manual_seed(1000+seed*10+j)
        m=base.CanonicalSpectral(len(FEATURES),4,max_frequency=1.,device=DEVICE)
        x=torch.as_tensor(d['x'],dtype=torch.float64,device=DEVICE)
        y=torch.as_tensor(d['y_scaled'],dtype=torch.float64,device=DEVICE)
        opt=torch.optim.Adam([{'params':[m.diagonal,m.log_temperature,m.log_ridge],'lr':.01},
                              {'params':[m.offdiag,m.log_gaps],'lr':.002}])
        rng=np.random.default_rng(2000+seed*10+j); trace=[];best=float('inf');state=None
        # A fixed within-window support is used for checkpoint validation; no future bank or adaptation labels.
        sv=train[-256:]; vv=val
        for step in range(steps+1):
            if step%50==0 or step==steps:
                with torch.no_grad():
                    zz=m.embedding(x[np.r_[sv,vv]]);k=m.kernel(zz[:len(sv)],zz[:len(sv)])
                    q=m.kernel(zz[len(sv):],zz[:len(sv)])
                    pred=solve(k.expand(len(vv),-1,-1),q,y[sv].expand(len(vv),-1),1e-3)
                    loss=float((pred-y[vv]).square().mean())
                trace.append(dict(step=step,validation_mse_scaled=loss))
                if loss<best:best=loss;state=clone(m);beststep=step
            if step==steps:break
            ids=rng.choice(train,160,replace=False);z=m.embedding(x[ids]);k=m.kernel(z[:128],z[:128]);q=m.kernel(z[128:],z[:128])
            # Share a factorization over the geometry training query minibatch.
            A=k+128*m.log_ridge.exp()*torch.eye(128,dtype=k.dtype,device=DEVICE)
            sol=torch.cholesky_solve(torch.stack([y[ids[:128]],torch.ones(128,dtype=k.dtype,device=DEVICE)],-1),torch.linalg.cholesky(A))
            b=sol[:,0].sum()/sol[:,1].sum();pr=q@(sol[:,0]-b*sol[:,1])+b
            loss=(pr-y[ids[128:]]).square().mean()+1e-5*(m.diagonal.square().mean()+m.offdiag.square().mean())
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
            with torch.no_grad():
                m.log_temperature.clamp_(math.log(.03),math.log(10));m.log_ridge.clamp_(math.log(1e-5),math.log(.1));m.log_gaps.clamp_(-8,1)
        m.load_state_dict(state);m.eval()
        with torch.no_grad():zs.append(m.embedding(x).float());temps.append(float(m.log_temperature.exp()))
        torch.save(dict(state_dict=state,center=d['center'],scale=d['scale']),out/f'bank{j}.pt')
        records.append(dict(bank=j,start=start,end_exclusive=end,fit_rows=len(train),validation_rows=len(val),selected_step=beststep,trace=trace))
        print('BANK',seed,j,'selected',beststep,'count_rmse',round(math.sqrt(best)*d['ys'],3),flush=True)
    save(out/'banks.json',records)
    return zs,temps


def indices(d,day):
    origin=np.datetime64(day,'ns');q=np.flatnonzero((d['dt']>=origin)&(d['dt']<origin+np.timedelta64(1,'D')))
    supports=[];valid=[]
    for lag in LAGS:
        stop=np.searchsorted(d['dt'],origin-np.timedelta64(lag,'D'))
        valid.append(stop>=N);supports.append(np.arange(max(N,stop)-N,max(N,stop)))
    sp=np.array(supports);mask=np.array(valid)
    assert len(q)>0 and mask[0] and np.all(d['dt'][sp[mask]]<origin)
    return q,sp,mask

@torch.no_grad()
def episode(d,zs,temps,day):
    qi,sp,mask=indices(d,day);qnum=len(qi)
    # Invalid candidates are masked and represented by the recent window.
    sp[~mask]=sp[0]
    grams=[];cross=[]
    for z,temp in zip(zs,temps):
        a=z[sp];b=z[qi]
        grams.append(torch.exp(-torch.cdist(a,a,compute_mode="donot_use_mm_for_euclid_dist")/temp))
        cross.append(torch.exp(-torch.cdist(b[None].expand(len(LAGS),-1,-1),a,compute_mode="donot_use_mm_for_euclid_dist")/temp))
    K=torch.stack(grams,1);Q=torch.stack(cross,2) # [L,M,N,N], [L,Q,M,N]
    y=t(d['y_scaled'][sp]);xx=d['x'][sp];xq=d['x'][qi]
    mean=xx.mean(1);sd=xx.std(1);ym=d['y_scaled'][sp].mean(1);ystd=d['y_scaled'][sp].std(1)
    age=((np.datetime64(day)-d['dt'][sp[:,-1]])/np.timedelta64(1,'D')).astype(float)
    # Covariate and response descriptors; no query response enters the feature map.
    feats=[]
    for l in range(len(LAGS)):
        dif=xq-mean[l]
        f=np.column_stack([xq,np.broadcast_to(mean[l],xq.shape),dif,dif*dif,
             np.broadcast_to(sd[l],xq.shape),
             np.full(qnum,ym[l]),np.full(qnum,ystd[l]),np.full(qnum,age[l]/365),
             np.full(qnum,d['y_scaled'][sp[l,-32:]].mean()-d['y_scaled'][sp[l,:32]].mean()),
             np.full(qnum,d['y_scaled'][sp[0]].mean()),
             Q[l].mean(-1).cpu().numpy(),Q[l].amax(-1).cpu().numpy()])
        feats.append(f)
    return dict(day=str(day),qi=qi,sp=sp,mask=mask,K=K.cpu(),Q=Q.cpu(),y=y.cpu(),
                f=torch.tensor(np.array(feats),dtype=torch.float32),target=t(d['y_scaled'][qi]).cpu())


def episodes(d,zs,temps,start,end):
    days=np.unique(d['dt'].astype('datetime64[D]'))
    days=days[(days>=np.datetime64(start))&(days<np.datetime64(end))]
    return [episode(d,zs,temps,day) for day in days]

@torch.no_grad()
def fixed_predictions(e,alpha,ridge):
    k=torch.einsum('lmij,m->lij',e['K'].to(DEVICE),t(alpha))
    q=torch.einsum('lqmi,m->lqi',e['Q'].to(DEVICE),t(alpha))
    n=q.shape[1];return solve(k[:,None].expand(-1,n,-1,-1),q,e['y'].to(DEVICE)[:,None].expand(-1,n,-1),ridge)

class SpectralRule(nn.Module):
    def __init__(self,center,scale,outputs=1,atoms=32):
        super().__init__();self.register_buffer('center',center);self.register_buffer('scale',scale)
        self.frequency=nn.Parameter(torch.randn(atoms,len(center),device=center.device)/math.sqrt(len(center)))
        self.head=nn.Parameter(torch.zeros(2*atoms,outputs,device=center.device))
        self.bias=nn.Parameter(torch.zeros(outputs,device=center.device));self.atoms=atoms
    def phi(self,f):
        phase=((f-self.center)/self.scale).clamp(-20,20)@self.frequency.T
        return torch.cat([phase.cos(),phase.sin()],-1)/math.sqrt(self.atoms)
    def forward(self,f):return self.phi(f)@self.head+self.bias


def normalization(train):
    f=torch.cat([e['f'][e['mask']].flatten(0,1) for e in train],0).to(DEVICE)
    return f.mean(0),f.std(0).clamp_min(.1)

def gated_predictions(e,gate,ridge,subsample=None):
    f=e['f'].to(DEVICE);k=e['K'].to(DEVICE);q=e['Q'].to(DEVICE);y=e['y'].to(DEVICE)
    if subsample is not None:
        li,qi=subsample
        alpha=gate(f[li,qi]).softmax(-1)
        kk=torch.einsum('bm,bmij->bij',alpha,k[li]);qq=torch.einsum('bm,bmi->bi',alpha,q[li,qi])
        return solve(kk,qq,y[li],ridge),alpha
    alpha=gate(f).softmax(-1)
    kk=torch.einsum('lqm,lmij->lqij',alpha,k);qq=torch.einsum('lqm,lqmi->lqi',alpha,q)
    return solve(kk,qq,y[:,None].expand(-1,q.shape[1],-1),ridge),alpha

@torch.no_grad()
def gate_validation(model,data,ridge):
    # Recent support validation matches one deployment arm; selected support is validated separately later.
    s=0;n=0
    for e in data:
        qi=np.arange(len(e['qi']));pr,_=gated_predictions(e,model,ridge,(np.zeros(len(qi),dtype=int),qi))
        s+=float((pr.clamp_min(0)-e['target'].to(DEVICE)).square().sum());n+=len(qi)
    return s/n

def train_gate(train,val,norm,alpha,ridge,seed,out,epochs):
    torch.manual_seed(5000+seed);m=SpectralRule(*norm,outputs=3).to(DEVICE)
    with torch.no_grad():m.bias.copy_(t(np.log(np.maximum(alpha,.01))))
    opt=torch.optim.Adam([{'params':[m.head,m.bias],'lr':.02},{'params':[m.frequency],'lr':.002}])
    rng=np.random.default_rng(6000+seed);best=gate_validation(m,val,ridge);state=clone(m);bestepoch=0;trace=[]
    for epoch in range(1,epochs+1):
        for ei in rng.permutation(len(train)):
            e=train[ei];qi=rng.integers(len(e['qi']),size=16);li=rng.choice(np.flatnonzero(e['mask']),size=16)
            pred,_=gated_predictions(e,m,ridge,(li,qi))
            loss=(pred.clamp_min(0)-e['target'].to(DEVICE)[qi]).square().mean()+1e-4*m.head.square().mean()+1e-6*m.frequency.square().mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        if epoch%5==0 or epoch==epochs:
            v=gate_validation(m,val,ridge);trace.append(dict(epoch=epoch,mse=v))
            if v<best:best=v;state=clone(m);bestepoch=epoch
            print('GATE',seed,epoch,'validation_scaled_mse',round(v,5),flush=True)
    m.load_state_dict(state);m.eval();torch.save(clone(m),out/'gate.pt')
    return m,dict(best_epoch=bestepoch,validation_mse=best,trace=trace)

@torch.no_grad()
def cache_predictions(data,gate,alpha,ridge):
    fixed=[];adaptive=[]
    for e in data:
        fixed.append(fixed_predictions(e,alpha,ridge).clamp_min(0).cpu())
        adaptive.append(gated_predictions(e,gate,ridge)[0].clamp_min(0).cpu())
    return fixed,adaptive


def train_selector(train,val,train_pred,val_pred,norm,seed,out,name,epochs=100):
    torch.manual_seed(7000+seed);m=SpectralRule(*norm).to(DEVICE)
    opt=torch.optim.Adam([{'params':[m.head,m.bias],'lr':.03},{'params':[m.frequency],'lr':.003}])
    # Flatten query rows, retaining the six actions for each query; masks exclude unavailable windows.
    def assemble(data,preds):
        f=torch.cat([e['f'].transpose(0,1) for e in data]).to(DEVICE)
        costs=torch.cat([(p-e['target'][None]).square().T for e,p in zip(data,preds)]).to(DEVICE)
        masks=torch.cat([torch.tensor(e['mask'])[None].expand(len(e['qi']),-1) for e in data]).to(DEVICE)
        return f,costs,masks
    xf,xc,xm=assemble(train,train_pred);vf,vc,vm=assemble(val,val_pred)
    def score():
        with torch.no_grad():
            choice=m(vf)[...,0].masked_fill(~vm,-torch.inf).argmax(-1)
            return float(vc.gather(1,choice[:,None]).mean())
    best=score();state=clone(m);bestepoch=0;trace=[];rng=np.random.default_rng(8000+seed)
    for epoch in range(1,epochs+1):
        for ix in np.array_split(rng.permutation(len(xf)),16):
            prob=m(xf[ix])[...,0].masked_fill(~xm[ix],-torch.inf).softmax(-1)
            loss=(prob*xc[ix]).sum(-1).mean()+1e-4*m.head.square().mean()+1e-6*m.frequency.square().mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        v=score();trace.append(dict(epoch=epoch,mse=v))
        if v<best:best=v;state=clone(m);bestepoch=epoch
        if epoch-bestepoch>=20:break
    m.load_state_dict(state);m.eval();torch.save(clone(m),out/f'{name}.pt')
    print('SELECTOR',seed,name,bestepoch,'validation_scaled_mse',round(best,5),flush=True)
    return m,dict(best_epoch=bestepoch,validation_mse=best,trace=trace)


def mixture_grid():
    return [np.array([a,b,4-a-b])/4 for a in range(5) for b in range(5-a)]

@torch.no_grad()
def calibrate(val,out):
    records=[]
    for ridge in RIDGES:
        for alpha in mixture_grid()+[np.ones(3)/3]:
            err=0;n=0
            for e in val:
                ee=dict(e,K=e['K'][:1],Q=e['Q'][:1],y=e['y'][:1])
                p=fixed_predictions(ee,alpha,ridge)[0].clamp_min(0)
                err+=float((p-e['target'].to(DEVICE)).square().sum());n+=len(p)
            records.append(dict(alpha=alpha.tolist(),ridge=ridge,mse=err/n))
    save(out/'calibration.json',records)
    best=min(records,key=lambda z:z['mse'])
    latest=min([r for r in records if r['alpha']==[0.,0.,1.]],key=lambda z:z['mse'])
    single=min([r for r in records if max(r['alpha'])==1],key=lambda z:z['mse'])
    uniform=min([r for r in records if r['alpha']==[1/3]*3],key=lambda z:z['mse'])
    return dict(fixed=best,latest=latest,single=single,uniform=uniform)

@torch.no_grad()
def choice(model,e):
    scores=model(e['f'].to(DEVICE))[...,0].masked_fill(~torch.tensor(e['mask'],device=DEVICE)[:,None],-torch.inf)
    return scores.argmax(0)


def metrics(y,p):return dict(rmse=float(np.sqrt(np.mean((p-y)**2))),mae=float(np.mean(np.abs(p-y))),r2=float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)))

@torch.no_grad()
def evaluate(d,zs,temps,cal,gate,sf,sg,out):
    rows=[];alphas=[];choices=[];audits=[]
    days=np.unique(d['dt'].astype('datetime64[D]'));days=days[days>=np.datetime64('2012-01-01')]
    for number,day in enumerate(days):
        e=episode(d,zs,temps,day);q=torch.arange(len(e['qi']),device=DEVICE)
        a=cal['fixed']['alpha'];r=cal['fixed']['ridge']
        fp=fixed_predictions(e,a,r).clamp_min(0);gp,alpha=gated_predictions(e,gate,r);gp=gp.clamp_min(0)
        fi=choice(sf,e);gi=choice(sg,e)
        # Temperature/humidity/wind proximity is a label-free retrieval reference using the identical candidate pool.
        raw=d['x'];means=raw[e['sp']].mean(1)
        dist=((raw[e['qi']][None,:,6:9]-means[:,None,6:9])**2).sum(-1);dist[~e['mask']]=np.inf
        wi=t(dist.argmin(0)).long()
        pred={'recent_fixed_fusion':fp[0],'selected_fixed_fusion':fp[fi,q],
              'recent_adaptive_fusion':gp[0],'selected_adaptive_fusion':gp[gi,q],
              'weather_selected_fixed_fusion':fp[wi,q]}
        for name,key in [('recent_latest_kernel','latest'),('recent_best_single','single'),('recent_uniform_fusion','uniform')]:
            ee=dict(e,K=e['K'][:1],Q=e['Q'][:1],y=e['y'][:1]);conf=cal[key]
            pred[name]=fixed_predictions(ee,conf['alpha'],conf['ridge'])[0].clamp_min(0)
        # Same learned weights, same ridge, separate ridge solves: mechanistic averaging comparator.
        members=torch.stack([fixed_predictions(dict(e,K=e['K'][:1],Q=e['Q'][:1],y=e['y'][:1]),np.eye(3)[j],r)[0] for j in range(3)],-1)
        pred['recent_prediction_average']=(members*alpha[0]).sum(-1).clamp_min(0)
        # Seasonal naive reference: previous available matching weekday/hour; no current-day labels.
        naive=[]
        for ix in e['qi']:
            past=np.flatnonzero((d['dt']<np.datetime64(day))&(d['df'].hr.to_numpy()==d['df'].hr.iloc[ix])&(d['df'].weekday.to_numpy()==d['df'].weekday.iloc[ix]))
            naive.append(d['y'][past[-1]])
        frame=pd.DataFrame({'row':e['qi'],'timestamp':d['dt'][e['qi']],'target':d['y'][e['qi']]})
        for name,p in pred.items():frame[name]=p.cpu().numpy()*d['ys']
        frame['seasonal_naive']=naive;rows.append(frame)
        choices.append(pd.DataFrame({'row':e['qi'],'fixed_context':fi.cpu().numpy(),'adaptive_context':gi.cpu().numpy(),'weather_context':wi.cpu().numpy()}))
        alphas.append(pd.DataFrame(alpha[gi,q].cpu().numpy(),columns=['weight0','weight1','weight2']).assign(row=e['qi']))
        if number in [0,90,180,270,365]:
            l=int(gi[0]);w=alpha[l,0];k=torch.einsum('m,mij->ij',w,e['K'][l].to(DEVICE))
            eig=float(torch.linalg.eigvalsh(k.double()).min())
            pr64=solve(k.double(),torch.einsum('m,mi->i',w,e['Q'][l,0].to(DEVICE)).double(),e['y'][l].to(DEVICE).double(),r)
            discrepancy=abs(float(pr64.clamp_min(0)-gp[l,0]))*d['ys']
            assert eig>-1e-4 and discrepancy<.05,(eig,discrepancy)
            audits.append(dict(day=str(day),min_eigenvalue=eig,float32_vs_float64_count_difference=discrepancy))
        if number%60==0:print('TEST progress',number,'/',len(days),flush=True)
    df=pd.concat(rows,ignore_index=True);df.to_csv(out/'predictions.csv',index=False)
    pd.concat(choices,ignore_index=True).to_csv(out/'choices.csv',index=False)
    pd.concat(alphas,ignore_index=True).to_csv(out/'weights.csv',index=False)
    methods=[c for c in df if c not in ['row','timestamp','target']]
    result={name:metrics(df.target.to_numpy(),df[name].to_numpy()) for name in methods}
    monthly=[]
    for month,g in df.groupby(df.timestamp.dt.to_period('M')):
        monthly.extend([dict(month=str(month),method=name,**metrics(g.target.to_numpy(),g[name].to_numpy())) for name in methods])
    save(out/'metrics.json',result);save(out/'monthly.json',monthly);save(out/'psd_audit.json',audits)
    print('RESULT',out.name,{k:round(v['rmse'],3) for k,v in result.items()},flush=True)


def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    protocol=dict(data=str(DATA),data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),features=FEATURES,
      target='cnt (raw counts scaled by pre-July2011 standard deviation); no casual/registered inputs',
      bank_windows=['Jan-Feb2011','Mar-Apr2011','May-Jun2011'],bank_internal_split='first80% fit; last20% checkpoint validation',
      scaling='shared input center/scale and response scale fitted Jan-Jun2011 only',
      meta_train='2011-07-01 through 2011-09-30',validation='2011-10-01 through 2011-12-31',test='all observed hours in 2012',
      context_size=N,context_lags_days=LAGS,context='last128 observed rows strictly before daily origin minus lag; unavailable windows masked',
      horizon='all observed hours of each day; no same-day labels; previous test-day outcomes become available at next origin',
      weather='observed target-hour weather: conditional prediction, not an ex-ante weather forecast',
      geometry='three frozen canonical learned spectral Laplace kernels, four frequencies per covariate, full trace-one SPD metric',
      selection='spectral scorer on query/context descriptors; expected candidate error training, hard argmax deployment',
      fusion='spectral feature encoder with affine three-logit head and softmax; one weight per query/context used consistently in full local Gram',
      training='two-stage: gate trained over all eligible context types, then separate selectors learn fixed-fusion or gated-fusion costs; no joint end-to-end retraining',
      gate_epochs=args.gate_epochs,geometry_steps=args.steps,seeds=args.seeds,ridge_grid=RIDGES,
      fixed_calibration='validation grid: three-kernel simplex quarters plus uniform; latest/best single/uniform have separately validation-selected ridge',
      gate_checkpoint='validation recent-context count MSE, every5epochs and initialization; selector checkpoint hard-selection validation MSE',
      evaluation='No 2012 labels used to tune persistent parameters, hyperparameters, or checkpoints; prior observed test labels allowed only in rolling contexts',
      caveats=['One observed series; seeds measure optimization variability, not independent datasets','Small 2011 meta-training period and restricted historical bank','Real-data temporal shift is not proof of isolated conditional concept drift'])
    save(args.out/'protocol.json',protocol)
    (args.out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'canonical_source.py').write_bytes(SOURCE.read_bytes())
    d=load();save(args.out/'data_audit.json',dict(rows=len(d['y']),start=str(d['dt'][0]),end=str(d['dt'][-1]),missing_clock_hours=int((d['dt'][-1]-d['dt'][0])/np.timedelta64(1,'h'))+1-len(d['dt']),target_equals_components=bool((d['df'].cnt==d['df'].casual+d['df'].registered).all()),test_rows=int((d['dt']>=np.datetime64('2012-01-01')).sum()),test_days=int(len(np.unique(d['dt'][d['dt']>=np.datetime64('2012-01-01')].astype('datetime64[D]'))))))
    for seed in args.seeds:
        start=time.time();out=args.out/f'seed{seed}';out.mkdir()
        zs,temps=bank_train(d,seed,out,args.steps)
        train=episodes(d,zs,temps,'2011-07-01','2011-10-01');val=episodes(d,zs,temps,'2011-10-01','2012-01-01')
        cal=calibrate(val,out);save(out/'selected_calibration.json',cal);print('CALIBRATION',seed,cal,flush=True)
        norm=normalization(train)
        gate,gr=train_gate(train,val,norm,cal['fixed']['alpha'],cal['fixed']['ridge'],seed,out,args.gate_epochs)
        tf,tg=cache_predictions(train,gate,cal['fixed']['alpha'],cal['fixed']['ridge'])
        vf,vg=cache_predictions(val,gate,cal['fixed']['alpha'],cal['fixed']['ridge'])
        sf,fr=train_selector(train,val,tf,vf,norm,seed,out,'selector_fixed')
        sg,sr=train_selector(train,val,tg,vg,norm,seed,out,'selector_adaptive')
        save(out/'training.json',dict(gate=gr,selector_fixed=fr,selector_adaptive=sr))
        # Persist all selected rules before the first test error is computed.
        save(out/'frozen_before_test.json',dict(checkpoints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.pt')},calibration=cal))
        del train,val,tf,tg,vf,vg
        evaluate(d,zs,temps,cal,gate,sf,sg,out)
        save(out/'runtime.json',dict(seconds=time.time()-start,device=DEVICE))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).parent/'results');p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--steps',type=int,default=250);p.add_argument('--gate-epochs',type=int,default=25);a=p.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):run(a)
