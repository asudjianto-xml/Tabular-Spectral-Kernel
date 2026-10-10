"""Train a linear query-conditioned window selector on independent synthetic episodes."""
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
from torch import nn
from threadpoolctl import threadpool_limits
import run as R
import run_transformer as T
import unemployment_data as U
import run_optimized_fusion as O

ROOT=Path(__file__).parent
SCENARIOS=['recurring','gradual','stable','localized']


def episode(rng,scenario):
    # Eight burn-in months, 24 observed historical months, then one query month.
    t=np.arange(33);phase=rng.uniform(0,2*np.pi)
    low=rng.uniform(.035,.045);high=rng.uniform(.085,.105)
    if scenario=='stable':rates=np.full(33,low)
    elif scenario=='gradual':
        start,end=(low,high) if rng.random()<.5 else (high,low)
        rates=np.linspace(start,end,33)
    else:
        period=rng.uniform(10,20)
        rates=low+(high-low)*(.5+.5*np.sin(2*np.pi*t/period+phase))
    rates=np.clip(rates+rng.normal(0,.001,33),.03,.12)
    stress,_=U.exposure(rates);rates=rates[8:];stress=stress[8:]
    bx,_=T.C.covariates(rng,800)
    mean,_=U.conditional_mean(bx,np.repeat(stress,32),localized=scenario=='localized')
    return U.feature_matrix(bx,np.repeat(rates,32)),T.C.responses(rng,mean),mean


def descriptors(cx,cy,qx,cross):
    """Only available labeled history/query features; no outcome-to-select input."""
    # cx: six chronologically ordered windows, each128 rows. Four months/window.
    latest_rates=cx[-1,:,4].reshape(4,32).mean(1)
    latest_labels=cy[-1].reshape(4,32).mean(1)
    out=[]
    for j in range(6):
        x=cx[j];y=cy[j];mx=x.mean(0);sx=x.std(0)
        wr=x[:,4].reshape(4,32).mean(1);wy=y.reshape(4,32).mean(1)
        for_query=np.column_stack([qx,qx-mx,(qx-mx)**2,
            cross[j].mean(1),cross[j].max(1),cross[j].std(1),
            qx[:,:4]*y.mean(),(qx[:,4]-wr.mean())**2,
            (qx[:,4]-wr[-1])**2])
        common=np.r_[mx,sx,wr,wy,y.mean(),y.std(),latest_rates,latest_labels,(5-j)/5,
                     wr[-1]-wr[0],wy[-1]-wy[0],y.mean()-latest_labels.mean()]
        out.append(np.column_stack([for_query,np.broadcast_to(common,(len(qx),len(common)))]))
    return np.stack(out,axis=1)


def dataset(banks,temps,ridge,weight,seed,count,scenario=None):
    rng=np.random.default_rng(seed);features=[];predictions=[];targets=[];means=[];raw_choices=[]
    for i in range(count):
        sc=scenario if scenario else SCENARIOS[int(rng.integers(4))]
        x,y,mu=episode(rng,sc);z=[R.embedding(m,x) for m in banks]
        preds=[];cross=[]
        for j in range(6):
            sl=slice(j*128,(j+1)*128)
            g=sum(w*R.kernel(zz[sl],zz[sl],temp) for w,zz,temp in zip([weight,1-weight],z,temps))
            q=sum(w*R.kernel(zz[768:],zz[sl],temp) for w,zz,temp in zip([weight,1-weight],z,temps))
            preds.append(R.ridge_predict(g,q,y[sl],ridge).clip(0,1));cross.append(q)
        cx=x[:768].reshape(6,128,5);cy=y[:768].reshape(6,128)
        features.append(descriptors(cx,cy,x[768:],np.array(cross)))
        predictions.append(np.stack(preds,axis=1));targets.append(y[768:]);means.append(mu[768:])
        raw_choices.append(np.argmin(abs(cx[:,:,4].mean(1)[None]-x[768:,4,None]),axis=1))
    return dict(features=np.array(features,dtype=np.float32),predictions=np.array(predictions),
                target=np.array(targets),mean=np.array(means),macro_choice=np.array(raw_choices))


class Selector(nn.Module):
    def __init__(self,center,scale):
        super().__init__();self.register_buffer('center',torch.tensor(center));self.register_buffer('scale',torch.tensor(scale))
        self.score=nn.Linear(len(center),1,bias=False)
        nn.init.zeros_(self.score.weight)

    def forward(self,x):return self.score((x-self.center)/self.scale).squeeze(-1)


def train(data,validation,seed):
    x=data['features'];center=x.mean((0,1,2));scale=np.maximum(x.std((0,1,2)),.01)
    torch.manual_seed(970000+seed);m=Selector(center,scale)
    opt=torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-4)
    tx=torch.tensor(x);cost=torch.tensor((data['predictions']-data['target'][...,None])**2,dtype=torch.float32)
    vx=torch.tensor(validation['features']);vp=validation['predictions'];vy=validation['target']
    rng=np.random.default_rng(980000+seed);best=np.inf;state=None;trace=[];best_epoch=0
    for epoch in range(100):
        for ix in np.array_split(rng.permutation(len(x)),max(1,len(x)//32)):
            prob=m(tx[ix]).softmax(-1)
            # Expected error of selecting one window, not an averaged prediction.
            loss=(prob*cost[ix]).sum(-1).mean()
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():choice=m(vx).argmax(-1).numpy()
        pred=np.take_along_axis(vp,choice[...,None],-1)[...,0];vl=float(np.mean((pred-vy)**2))
        trace.append(dict(epoch=epoch+1,validation_mse=vl))
        if vl<best:best=vl;best_epoch=epoch+1;state=copy.deepcopy(m.state_dict())
        if epoch+1-best_epoch>=15:break
    m.load_state_dict(state);m.eval()
    return m,dict(trace=trace,best_epoch=best_epoch,parameters=sum(p.numel() for p in m.parameters()),
                  training_oracle_window_frequency=np.bincount(cost.argmin(-1).numpy().ravel(),minlength=6).tolist())


def execute(seed,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);hashes={}
    def record(p):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
    for p in [Path(__file__),Path(R.__file__),Path(U.__file__),Path(T.C.__file__),Path(O.__file__),R.MODEL_SOURCE]:
        record(p);(out/(p.stem+'_snapshot.py')).write_bytes(p.read_bytes())
    src=ROOT/f'unemployment_results/seed{seed}';info=json.loads(record(src/'original_training.json').read_text());banks=[]
    for name in ['normal','downturn']:
        m=R.base.CanonicalSpectral(5,4,max_frequency=3.,device='cpu')
        m.load_state_dict(torch.load(record(src/f'{name}.pt'),weights_only=True,map_location='cpu'));m.eval();banks.append(m)
    with np.load(record(src/'history.npz')) as f:hx,hy=f['x'][-128:],f['y'][-128:]
    hz=[R.embedding(m,hx) for m in banks];hg=np.array([R.kernel(z,z,t) for z,t in zip(hz,info['temperatures'])])
    weight,cal=O.optimize_weight(hg,hy,info['ridge'],'convex_mkl')
    seeds=dict(train=910000+seed,validation=920000+seed,**{sc:930000+seed*1000+i*100 for i,sc in enumerate(SCENARIOS)})
    tr=dataset(banks,info['temperatures'],info['ridge'],weight,seeds['train'],512)
    va=dataset(banks,info['temperatures'],info['ridge'],weight,seeds['validation'],128)
    print('datasets',seed,flush=True);m,training=train(tr,va,seed)
    torch.save(m.state_dict(),out/'selector.pt');R.save(out/'training.json',training)
    # Freeze before constructing any fresh evaluation episode.
    metrics=[]
    for sc in SCENARIOS:
        d=dataset(banks,info['temperatures'],info['ridge'],weight,seeds[sc],256,sc)
        with torch.no_grad():scores=m(torch.tensor(d['features'])).numpy()
        p=d['predictions'];y=d['target'];mu=d['mean']
        choices=dict(latest=np.full(y.shape,5),learned=scores.argmax(-1),macro_nearest=d['macro_choice'],
            oracle_labels=((p-y[...,None])**2).argmin(-1),oracle_mean=((p-mu[...,None])**2).argmin(-1))
        result={};losses={}
        for name,c in choices.items():
            pred=np.take_along_axis(p,c[...,None],-1)[...,0];result[name]=pred
            losses[name]=((pred-y)**2).mean(1)
            metrics.append(dict(seed=seed,scenario=sc,method=name,rmse=float(np.sqrt(np.mean((pred-y)**2))),
                clean_rmse=float(np.sqrt(np.mean((pred-mu)**2))),older_fraction=float(np.mean(c!=5)),
                mean_window_age_months=float(np.mean((5-c)*4)),
                selection_frequency=np.bincount(c.ravel(),minlength=6).tolist()))
        np.savez_compressed(out/f'{sc}.npz',target=y,mean=mu,candidate_predictions=p,scores=scores,
                            **result,**{k+'_choice':v for k,v in choices.items()},**{k+'_episode_mse':v for k,v in losses.items()})
        print(seed,sc,{r['method']:round(r['rmse']*100,4) for r in metrics if r['scenario']==sc},flush=True)
    R.save(out/'metrics.json',metrics);R.save(out/'provenance.json',hashes)
    R.save(out/'protocol.json',dict(seed=seed,episode_seeds=seeds,train_episodes=512,validation_episodes=128,test_episodes_per_scenario=256,
        history='24 observed months x32 borrowers; six nonoverlapping four-month windows of128; predict next32; eight macro burn-in months',
        mixture_weight=weight,mixture_calibration='Once per bank seed: convex MKL on final128 pre-evaluation historical rows; frozen for every window/query/episode',
        ridge=info['ridge'],selector='Shared linear scorer on standardized query/window descriptors, no MLP. Softmax expected action squared error during training; hard argmax window at inference.',
        supervision='Exhaustively evaluate all6 candidate KRR windows on each training query label. Train cost-sensitive selector; no oracle labels used at inference. Features exclude query targets and latent state.',
        selection='Best hard-selection MSE on independent128 validation episodes; Adam lr.01 weight_decay1e-4, max100 epochs patience15; no test tuning',
        features='Observable query and window means/std, four monthly macro/label means, latest-window summaries, relative age, query-window differences/squares, kernel affinities, label-mean/query interactions; no candidate prediction as feature',
        inference='Descriptor computation uses candidate features/labels/kernel affinities; one selected KRR support per query; benchmark computes all candidates for scoring/oracles, no latency claim',
        novelty='Fresh independent evaluation episodes and new randomized paths; simulator response law shared with training, not an unseen task family',
        caveat='Label oracle fits evaluation noise; mean oracle has privileged latent information. Window subsets preserve local PSD; query-specific choices do not define one global Gram.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):execute(a.seed,ROOT/f'context_selector_results/seed{a.seed}')
