"""Learned finite-spectral PSD relevance kernels; no MLP and unchanged KRR stage."""
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
import run_context_selector as S
import run_selector_penalties as P

ROOT=S.ROOT
CONFIGS=[dict(subset=subset,atoms=atoms,penalty=penalty)
         for subset in ['sparse','all'] for atoms,penalty in [(32,1e-5),(32,1e-4),(64,1e-4)]]


class SpectralSelector(nn.Module):
    def __init__(self,center,scale,indices,base_weight,atoms=32,hybrid=False):
        super().__init__()
        self.register_buffer('center',center.clone());self.register_buffer('scale',scale.clone())
        self.register_buffer('indices',torch.tensor(indices,dtype=torch.long))
        self.register_buffer('base_weight',base_weight.clone() if hybrid else torch.zeros_like(base_weight))
        self.atoms=atoms
        self.frequency=nn.Parameter(torch.randn(atoms,len(indices))/np.sqrt(len(indices)))
        self.coefficient=nn.Parameter(torch.zeros(2*atoms))

    def spectral_features(self,x):
        z=((x-self.center)/self.scale)[...,self.indices]
        phase=z@self.frequency.T
        return torch.cat([phase.cos(),phase.sin()],-1)/np.sqrt(self.atoms)

    def kernel(self,x,y):return self.spectral_features(x)@self.spectral_features(y).transpose(-1,-2)

    def forward(self,x):
        z=(x-self.center)/self.scale
        return (z*self.base_weight).sum(-1)+self.spectral_features(x)@self.coefficient


def fit(train,val,base,indices,config,hybrid,seed):
    torch.manual_seed(1130000+seed)
    model=SpectralSelector(base.center,base.scale,indices,base.score.weight.detach()[0],config['atoms'],hybrid)
    opt=torch.optim.Adam([{'params':[model.coefficient],'lr':.02},
                          {'params':[model.frequency],'lr':.002}])
    tx=torch.tensor(train['features']);vx=torch.tensor(val['features'])
    cost=torch.tensor((train['predictions']-train['target'][...,None])**2,dtype=torch.float32)
    with torch.no_grad():initial=P.hard_mse(model(vx).numpy(),val)
    best=initial;state=copy.deepcopy(model.state_dict());best_epoch=0;trace=[]
    rng=np.random.default_rng(1140000+seed)
    for epoch in range(1,51):
        for ix in np.array_split(rng.permutation(len(tx)),16):
            risk=(model(tx[ix]).softmax(-1)*cost[ix]).sum(-1).mean()
            loss=risk+config['penalty']*.5*model.coefficient.square().sum()+1e-6*model.frequency.square().mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        with torch.no_grad():v=P.hard_mse(model(vx).numpy(),val)
        trace.append(dict(epoch=epoch,validation_mse=v))
        if v<best:best=v;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
        if epoch-best_epoch>=8:break
    model.load_state_dict(state);model.eval()
    return model,dict(config=config,hybrid=hybrid,validation_mse=best,initial_validation_mse=initial,
                     best_epoch=best_epoch,trace=trace,input_descriptors=len(indices),parameters=sum(p.numel() for p in model.parameters()))


def execute(seed,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);hashes={}
    def record(p):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
    for p in [Path(__file__),Path(S.__file__),Path(P.__file__),Path(R.__file__),Path(S.U.__file__),Path(S.T.C.__file__),R.MODEL_SOURCE]:
        record(p);(out/(p.stem+'_snapshot.py')).write_bytes(p.read_bytes())
    src=ROOT/f'unemployment_results/seed{seed}';info=json.loads(record(src/'original_training.json').read_text());banks=[]
    for name in ['normal','downturn']:
        m=R.base.CanonicalSpectral(5,4,max_frequency=3.,device='cpu')
        m.load_state_dict(torch.load(record(src/f'{name}.pt'),weights_only=True,map_location='cpu'));m.eval();banks.append(m)
    original=ROOT/f'context_selector_results/seed{seed}'
    weight=json.loads(record(original/'protocol.json').read_text())['mixture_weight']
    previous=ROOT/f'selector_penalty_results/seed{seed}'
    state=torch.load(record(previous/'l1.pt'),weights_only=True,map_location='cpu')
    base=S.Selector(state['center'].numpy(),state['scale'].numpy());base.load_state_dict(state);base.eval()
    indices=torch.where(base.score.weight.detach()[0].abs()>1e-8)[0].tolist()
    tr=P.augment(S.dataset(banks,info['temperatures'],info['ridge'],weight,910000+seed,512))
    va=P.augment(S.dataset(banks,info['temperatures'],info['ridge'],weight,920000+seed,128))
    search=[];models={'l1':base};selected={};best={}
    for hybrid in [False,True]:
        kind='hybrid' if hybrid else 'spectral'
        for config in CONFIGS:
            ix=indices if config['subset']=='sparse' else list(range(57))
            m,rec=fit(tr,va,base,ix,config,hybrid,seed);search.append(rec)
            key=kind+'_'+config['subset']
            if key not in best or rec['validation_mse']<best[key]:
                models[key]=m;best[key]=rec['validation_mse'];selected[key]=rec
            print(seed,kind,config,'validation',round(rec['validation_mse'],6),'epoch',rec['best_epoch'],flush=True)
        winner=min([kind+'_sparse',kind+'_all'],key=lambda k:best[k])
        models[kind]=models[winner];selected[kind]=dict(selected[winner],selected_variant=winner)
    with torch.no_grad():
        validation={name:P.hard_mse(m(torch.tensor(va['features'])).numpy(),va) for name,m in models.items()}
    validation['macro_nearest']=P.hard_mse(-va['features'][...,-1],va)
    validation['latest']=float(np.mean((va['predictions'][...,-1]-va['target'])**2))
    overall=min(['l1','spectral','hybrid','macro_nearest','latest'],key=validation.get)
    for name,m in models.items():torch.save(m.state_dict(),out/f'{name}.pt')
    R.save(out/'selection.json',dict(search=search,selected=selected,validation=validation,overall=overall,sparse_indices=indices))
    # No new test episode exists before every selector/checkpoint is frozen.
    metrics=[];audits=[]
    for i,sc in enumerate(S.SCENARIOS):
        d=P.augment(S.dataset(banks,info['temperatures'],info['ridge'],weight,1230000+seed*1000+i*100,256,sc))
        tx=torch.tensor(d['features']);choices={'latest':np.full(d['target'].shape,5),'macro_nearest':d['macro_choice']}
        with torch.no_grad():
            for name,m in models.items():
                choices[name]=m(tx).argmax(-1).numpy()
                if name in ['spectral_sparse','spectral_all','hybrid_sparse','hybrid_all']:
                    for episode in [0,127,255]:
                        z=tx[episode].reshape(-1,57).double();mm=copy.deepcopy(m).double()
                        k=mm.kernel(z,z);eig=float(torch.linalg.eigvalsh(k).min());sym=float((k-k.T).abs().max())
                        assert eig>-1e-9 and sym<1e-12
                        # Scores for a query/candidate do not depend on other query rows.
                        torch.testing.assert_close(m(tx[episode:episode+1,:1]),m(tx[episode:episode+1])[:,:1])
                        audits.append(dict(seed=seed,scenario=sc,method=name,episode=episode,min_eigenvalue=eig,symmetry_error=sym))
        choices['validation_selected']=choices[overall];result={}
        for name,c in choices.items():
            pred=np.take_along_axis(d['predictions'],c[...,None],-1)[...,0];result[name]=pred
            metrics.append(dict(seed=seed,scenario=sc,method=name,rmse=float(np.sqrt(np.mean((pred-d['target'])**2))),older_fraction=float(np.mean(c!=5))))
        np.savez_compressed(out/f'{sc}.npz',target=d['target'],candidate_predictions=d['predictions'],**result,**{k+'_choice':v for k,v in choices.items()})
        print('TEST',seed,sc,{r['method']:round(100*r['rmse'],4) for r in metrics if r['scenario']==sc},flush=True)
    R.save(out/'metrics.json',metrics);R.save(out/'psd_audit.json',audits);R.save(out/'provenance.json',hashes)
    R.save(out/'protocol.json',dict(seed=seed,configs=CONFIGS,train_seed=910000+seed,validation_seed=920000+seed,test_seed_base=1230000+seed*1000,
        train_episodes=512,validation_episodes=128,test_episodes_per_scenario=256,fixed_mixture=weight,fixed_ridge=info['ridge'],
        kernel='phi(z)=[cos(Omega*z),sin(Omega*z)]/sqrt(M), k=phi(z)^T phi(zprime); frequencies learned globally; no MLP',
        scorer='Pure: beta^T phi. Hybrid: frozen L1 linear score plus beta^T phi, representable in sum of linear and spectral RKHSs.',
        features='57 standardized descriptors; sparse subset from earlier validation-selected L1, or all57 (including query-invariant features, which can enter nonlinear interactions)',
        training='Same expected action error; Adam head .02/frequencies .002, coefficient L2 lambda/2, frequency mean-square penalty1e-6; max50 epochs patience8; epoch0 eligible',
        selection='Within each subset choose atoms/penalty/checkpoint by validation hard-selection error; choose subset separately for pure and hybrid; freeze before fresh test generation',
        caveat='Finite-dimensional spectral approximation, different capacity and optimization from linear baseline. Descriptor selection/validation reused; new test episodes share simulator law. No test retuning or inference latency claim.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):execute(a.seed,ROOT/f'spectral_selector_results/seed{a.seed}')
