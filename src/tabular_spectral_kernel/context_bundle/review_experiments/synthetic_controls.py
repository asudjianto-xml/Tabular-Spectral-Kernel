"""Matched EWMA-input KRR controls, nonlinear selector control and same-episode oracles."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import sys,json,copy,hashlib
from pathlib import Path
import numpy as np,torch
from torch import nn
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'synthetic_drift'))
import run as R,run_context_selector as S,run_selector_penalties as P
OUT=runtime.output_path(__file__, Path(__file__).parent/'synthetic_results');DEV=runtime.device();DT=torch.float64

def raw_episode(rng,scenario):
    time=np.arange(33);phase=rng.uniform(0,2*np.pi);low=rng.uniform(.035,.045);high=rng.uniform(.085,.105)
    if scenario=='stable':rates=np.full(33,low)
    elif scenario=='gradual':
        start,end=(low,high) if rng.random()<.5 else (high,low);rates=np.linspace(start,end,33)
    else:
        period=rng.uniform(10,20);rates=low+(high-low)*(.5+.5*np.sin(2*np.pi*time/period+phase))
    rates=np.clip(rates+rng.normal(0,.001,33),.03,.12)
    stress,memory=S.U.exposure(rates);bx,_=S.T.C.covariates(rng,800)
    mean,_=S.U.conditional_mean(bx,np.repeat(stress[8:],32),localized=scenario=='localized')
    x=S.U.feature_matrix(bx,np.repeat(rates[8:],32));y=S.T.C.responses(rng,mean)
    # Initial state and eight macro-only burn-in months are available to this explicitly augmented baseline.
    return np.column_stack([x,2*np.repeat(memory[8:],32)-1]),y,mean

def raw_data(seed,count,scenario=None):
    rng=np.random.default_rng(seed);x=[];y=[];mu=[]
    for _ in range(count):
        sc=scenario if scenario else S.SCENARIOS[int(rng.integers(4))]
        a,b,c=raw_episode(rng,sc);x.append(a);y.append(b);mu.append(c)
    return np.array(x),np.array(y),np.array(mu)

@torch.no_grad()
def rbf_path(raw,dim,width,ridges):
    x,y,_=raw;cx=torch.tensor(x[:,640:768,:dim],dtype=DT,device=DEV);qx=torch.tensor(x[:,768:,:dim],dtype=DT,device=DEV)
    ys=torch.tensor(y[:,640:768],dtype=DT,device=DEV)
    k=torch.exp(-torch.cdist(cx,cx).square()/(2*width**2));q=torch.exp(-torch.cdist(qx,cx).square()/(2*width**2));pred={}
    for ridge in ridges:
        ch=torch.linalg.cholesky(k+128*ridge*torch.eye(128,device=DEV,dtype=DT))
        sol=torch.cholesky_solve(torch.stack([ys,torch.ones_like(ys)],-1),ch);b=sol[...,0].sum(-1)/sol[...,1].sum(-1)
        p=(q@(sol[...,0]-b[:,None]*sol[...,1])[...,None])[...,0]+b[:,None]
        pred[ridge]=p.clamp(0,1).cpu().numpy()
    return pred

class MLP(nn.Module):
    def __init__(self,center,scale,hidden):
        super().__init__();self.register_buffer('center',center);self.register_buffer('scale',scale)
        self.hidden=nn.Linear(len(center),hidden);self.head=nn.Linear(hidden,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):return self.head(torch.relu(self.hidden((x-self.center)/self.scale)))[...,0]

def fit_mlp(tr,va,center,scale,seed,cfg):
    hidden,penalty=cfg;torch.manual_seed(1130000+seed)
    m=MLP(center,scale,hidden).to(DEV);opt=torch.optim.Adam([dict(params=m.hidden.parameters(),lr=.002),dict(params=m.head.parameters(),lr=.02)])
    tx=torch.tensor(tr['features'],device=DEV);vx=torch.tensor(va['features'],device=DEV)
    cost=torch.tensor((tr['predictions']-tr['target'][...,None])**2,dtype=torch.float32,device=DEV)
    def score():
        with torch.no_grad():return P.hard_mse(m(vx).cpu().numpy(),va)
    best=score();state=copy.deepcopy(m.state_dict());bestepoch=0;trace=[];rng=np.random.default_rng(1140000+seed)
    for epoch in range(1,51):
        for ix in np.array_split(rng.permutation(len(tx)),16):
            risk=(m(tx[ix]).softmax(-1)*cost[ix]).sum(-1).mean()
            loss=risk+penalty*.5*m.head.weight.square().sum()+1e-6*m.hidden.weight.square().mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        v=score();trace.append(dict(epoch=epoch,mse=v))
        if v<best:best=v;state=copy.deepcopy(m.state_dict());bestepoch=epoch
        if epoch-bestepoch>=8:break
    m.load_state_dict(state);m.eval();return m,dict(hidden=hidden,penalty=penalty,validation_mse=best,best_epoch=bestepoch,parameters=sum(p.numel() for p in m.parameters()),trace=trace)


def main():
    OUT.mkdir(exist_ok=False);(OUT/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    R.save(OUT/'protocol.json',dict(seeds=[0,1,2],episodes='same512training,128validation and256test per scenario/seed as final spectral-selector study',ewma='exact causal h from all observed macro history including eight burn-in months, known simulator initial h=0; no future unemployment or response labels',baseline='matched Gaussian KRR on latest128, five snapshot predictors vs six including scaled h; bandwidth/ridge tuned independently on same mixed validation episodes',widths=[.5,1,2,4],ridges=[1e-6,1e-5,1e-4,1e-3,1e-2],mlp='one ReLU hidden layer32 or64, matching spectral parameter counts to within1; same57descriptors, costs, epochs, patience, and three configurations',review_stage=True,test_previously_inspected=True))
    for seed in [0,1,2]:
        out=OUT/f'seed{seed}';out.mkdir()
        # Exact common-random-number identity with the existing generator.
        for sc in S.SCENARIOS:
            a,b,c=raw_episode(np.random.default_rng(77),sc);aa,bb,cc=S.episode(np.random.default_rng(77),sc)
            np.testing.assert_array_equal(a[:,:5],aa);np.testing.assert_array_equal(b,bb);np.testing.assert_array_equal(c,cc)
        va_raw=raw_data(920000+seed,128);rbf={};search=[]
        for dim in [5,6]:
            best=None
            for width in [.5,1,2,4]:
                for ridge,pred in rbf_path(va_raw,dim,width,[1e-6,1e-5,1e-4,1e-3,1e-2]).items():
                    rec=dict(dim=dim,width=width,ridge=ridge,validation_mse=float(np.mean((pred-va_raw[1][:,768:])**2)));search.append(rec)
                    if best is None or rec['validation_mse']<best['validation_mse']:best=rec
            rbf[dim]=best
        R.save(out/'rbf_selection.json',dict(chosen=rbf,search=search));print('EWMA VALIDATION',seed,rbf,flush=True)
        src=ROOT/f'synthetic_drift/unemployment_results/seed{seed}';info=json.loads((src/'original_training.json').read_text());banks=[]
        for name in ['normal','downturn']:
            m=R.base.CanonicalSpectral(5,4,max_frequency=3.,device='cpu');m.load_state_dict(torch.load(src/f'{name}.pt',weights_only=True, map_location='cpu'));banks.append(m)
        weight=json.loads((ROOT/f'synthetic_drift/context_selector_results/seed{seed}/protocol.json').read_text())['mixture_weight']
        tr=P.augment(S.dataset(banks,info['temperatures'],info['ridge'],weight,910000+seed,512));va=P.augment(S.dataset(banks,info['temperatures'],info['ridge'],weight,920000+seed,128))
        st=torch.load(ROOT/f'synthetic_drift/spectral_selector_results/seed{seed}/spectral_all.pt',weights_only=True, map_location='cpu');center=st['center'];scale=st['scale']
        searches=[];winner=None
        for cfg in [(32,1e-5),(32,1e-4),(64,1e-4)]:
            m,rec=fit_mlp(tr,va,center,scale,seed,cfg);searches.append(rec)
            if winner is None or rec['validation_mse']<winner[0]:winner=(rec['validation_mse'],m,rec)
            print('MLP VALIDATION',seed,cfg,rec['validation_mse'],rec['best_epoch'],flush=True)
        model=winner[1];torch.save({k:v.cpu() for k,v in model.state_dict().items()},out/'mlp.pt');R.save(out/'mlp_selection.json',dict(search=searches,chosen=winner[2]))
        records=[]
        for si,sc in enumerate(S.SCENARIOS):
            testseed=1230000+seed*1000+si*100;raw=raw_data(testseed,256,sc);d=P.augment(S.dataset(banks,info['temperatures'],info['ridge'],weight,testseed,256,sc))
            old=np.load(ROOT/f'synthetic_drift/spectral_selector_results/seed{seed}/{sc}.npz');np.testing.assert_allclose(d['target'],old['target'],atol=0,rtol=0);np.testing.assert_allclose(d['predictions'],old['candidate_predictions'],atol=1e-12)
            np.testing.assert_array_equal(raw[1][:,768:],d['target']);mu=raw[2][:,768:];y=d['target'];p=d['predictions']
            with torch.no_grad():choice=model(torch.tensor(d['features'],device=DEV)).argmax(-1).cpu().numpy()
            preds={'mlp_selector':np.take_along_axis(p,choice[...,None],-1)[...,0],
                   'label_context_oracle':np.take_along_axis(p,((p-y[...,None])**2).argmin(-1)[...,None],-1)[...,0],
                   'mean_context_oracle':np.take_along_axis(p,((p-mu[...,None])**2).argmin(-1)[...,None],-1)[...,0],
                   'conditional_mean':mu}
            for dim,cfg in rbf.items():preds['rbf_snapshot' if dim==5 else 'rbf_ewma']=rbf_path(raw,dim,cfg['width'],[cfg['ridge']])[cfg['ridge']]
            for name,pred in preds.items():records.append(dict(seed=seed,scenario=sc,method=name,rmse_pp=float(100*np.sqrt(np.mean((pred-y)**2))),clean_rmse_pp=float(100*np.sqrt(np.mean((pred-mu)**2)))))
            np.savez_compressed(out/f'{sc}.npz',target=y,mean=mu,**preds)
            print('SYNTHETIC',seed,sc,{r['method']:round(r['rmse_pp'],4) for r in records if r['scenario']==sc},flush=True)
        R.save(out/'metrics.json',records)
if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
