"""Review-stage validation-selected ridge expansion and oracle diagnosis.
Original artifacts are read-only. The 2012 test series was inspected previously;
this is a declared review-stage sensitivity, not a new untouched test.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import sys,importlib.util,json,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bike_base',ROOT/'bikeshare_context_fusion/run.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
OUT=runtime.output_path(__file__, Path(__file__).parent/'bike_results')

def load_bank(d,seed):
    zs=[];temps=[]
    for j in range(3):
        cp=torch.load(ROOT/f'bikeshare_context_fusion/results/seed{seed}/bank{j}.pt',map_location='cpu',weights_only=False)
        m=B.base.CanonicalSpectral(len(B.FEATURES),4,max_frequency=1.,device=B.DEVICE);m.load_state_dict(cp['state_dict'])
        with torch.no_grad():zs.append(m.embedding(torch.tensor(d['x'],dtype=torch.float64,device=B.DEVICE)).float());temps.append(float(m.log_temperature.exp()))
    return zs,temps

def rule(path):
    st=torch.load(path,map_location=B.DEVICE,weights_only=True)
    m=B.SpectralRule(st['center'],st['scale'],outputs=st['bias'].numel());m.load_state_dict(st);m.eval();return m

@torch.no_grad()
def oracle(d,zs,temps,gate,cal,seed,folder,start,end,label):
    errors={};truth=[];orig=[];fixed_c=[];gate_c=[];choicecounts={};dates=[]
    for day in np.unique(d['dt'].astype('datetime64[D]')):
        if not np.datetime64(start)<=day<np.datetime64(end):continue
        e=B.episode(d,zs,temps,day);target=e['target'].numpy();truth.extend(target);dates.extend([str(day)]*len(target))
        fixed=B.fixed_predictions(e,cal['fixed']['alpha'],cal['fixed']['ridge']).clamp_min(0).cpu().numpy()
        gated=B.gated_predictions(e,gate,cal['fixed']['ridge'])[0].clamp_min(0).cpu().numpy()
        fixed_c.append(fixed.T);gate_c.append(gated.T)
        for kind,p in [('fixed',fixed),('gated',gated)]:
            costs=(p-target[None])**2;costs[~e['mask']]=np.inf
            ix=costs.argmin(0);dayix=costs.mean(1).argmin()
            for name,pr in [('recent',p[0]),('query_oracle',p[ix,np.arange(len(target))]),('day_oracle',p[dayix])]:
                errors.setdefault(kind+'_'+name,[]).extend(pr)
            choicecounts.setdefault(kind,[]).extend(ix)
    y=np.array(truth)*d['ys'];record={k:B.metrics(y,np.array(v)*d['ys']) for k,v in errors.items()}
    for kind,c in choicecounts.items():record[kind+'_older_fraction']=float(np.mean(np.array(c)!=0))
    np.savez_compressed(folder/f'{label}_candidates.npz',target=y,fixed=np.concatenate(fixed_c)*d['ys'],gated=np.concatenate(gate_c)*d['ys'],day=dates)
    B.save(folder/f'{label}_oracles.json',record);print('ORACLE',seed,label,{k:round(v['rmse'],3) for k,v in record.items() if isinstance(v,dict)},flush=True)

@torch.no_grad()
def large_support(d,zs,temps,cal,out):
    alpha=B.t(cal['fixed']['alpha']);record=[];predictions={}
    for n in [256,512]:
        vals={r:[] for r in B.RIDGES};vy=[];test=[];testy=[];testrows=[]
        def pred(day,ridges):
            stop=np.searchsorted(d['dt'],np.datetime64(day));sp=np.arange(stop-n,stop)
            qi=np.flatnonzero((d['dt']>=day)&(d['dt']<day+np.timedelta64(1,'D')))
            assert d['dt'][sp[-1]]<day
            k=sum(w*torch.exp(-torch.cdist(z[sp],z[sp],compute_mode='donot_use_mm_for_euclid_dist')/temp) for w,z,temp in zip(alpha,zs,temps))
            q=sum(w*torch.exp(-torch.cdist(z[qi],z[sp],compute_mode='donot_use_mm_for_euclid_dist')/temp) for w,z,temp in zip(alpha,zs,temps))
            y=B.t(d['y_scaled'][sp]);p={}
            for r in ridges:
                # One factorization serves every query in the day.
                sol=torch.cholesky_solve(torch.stack([y,torch.ones_like(y)],1),torch.linalg.cholesky(k+n*r*torch.eye(n,device=B.DEVICE)))
                b=sol[:,0].sum()/sol[:,1].sum();p[r]=(q@(sol[:,0]-b*sol[:,1])+b).clamp_min(0).cpu().numpy()*d['ys']
            return qi,p
        for day in np.unique(d['dt'].astype('datetime64[D]')):
            if np.datetime64('2011-10-01')<=day<np.datetime64('2012-01-01'):
                qi,p=pred(day,B.RIDGES);vy.extend(d['y'][qi])
                for r in B.RIDGES:vals[r].extend(p[r])
        losses={r:float(np.mean((np.array(p)-vy)**2)) for r,p in vals.items()};best=min(losses,key=losses.get)
        B.save(out/f'support{n}_validation.json',dict(losses=losses,chosen=best))
        for day in np.unique(d['dt'].astype('datetime64[D]')):
            if day>=np.datetime64('2012-01-01'):
                qi,p=pred(day,[best]);test.extend(p[best]);testy.extend(d['y'][qi]);testrows.extend(qi)
        record.append(dict(n=n,ridge=best,**B.metrics(np.array(testy),np.array(test))))
        predictions[f'n{n}']=test;print('SUPPORT',out.name,record[-1],flush=True)
    np.savez_compressed(out/'support_predictions.npz',target=testy,row=testrows,**predictions);B.save(out/'support_metrics.json',record)

def main():
    OUT.mkdir(exist_ok=False);B.RIDGES=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]
    B.save(OUT/'protocol.json',dict(seeds=[0,1,2],ridge_grid=B.RIDGES,review_stage=True,test_previously_inspected=True,selection='2011 validation only; bank checkpoints unchanged; gates and selectors retrained with chosen ridge before replaying 2012',oracle='query-label and whole-day-label best of six candidates; retrospective optimistic diagnostics, not deployable',large_context='256/512 support; weights from calibrated128; ridge selected separately on validation'))
    (OUT/'executed_source.py').write_bytes(Path(__file__).read_bytes());d=B.load()
    for seed in [0,1,2]:
        out=OUT/f'seed{seed}';out.mkdir();zs,temps=load_bank(d,seed)
        old=ROOT/f'bikeshare_context_fusion/results/seed{seed}';oldcal=json.loads((old/'selected_calibration.json').read_text());oldgate=rule(old/'gate.pt')
        oracle(d,zs,temps,oldgate,oldcal,seed,out,'2011-10-01','2012-01-01','original_validation')
        oracle(d,zs,temps,oldgate,oldcal,seed,out,'2012-01-01','2013-01-01','original_test')
        tr=B.episodes(d,zs,temps,'2011-07-01','2011-10-01');va=B.episodes(d,zs,temps,'2011-10-01','2012-01-01')
        cal=B.calibrate(va,out);B.save(out/'selected_calibration.json',cal);print('EXTENDED',seed,cal,flush=True)
        norm=B.normalization(tr);gate,gr=B.train_gate(tr,va,norm,cal['fixed']['alpha'],cal['fixed']['ridge'],seed,out,25)
        tf,tg=B.cache_predictions(tr,gate,cal['fixed']['alpha'],cal['fixed']['ridge']);vf,vg=B.cache_predictions(va,gate,cal['fixed']['alpha'],cal['fixed']['ridge'])
        sf,fr=B.train_selector(tr,va,tf,vf,norm,seed,out,'selector_fixed');sg,sr=B.train_selector(tr,va,tg,vg,norm,seed,out,'selector_adaptive')
        B.save(out/'training.json',dict(gate=gr,selector_fixed=fr,selector_adaptive=sr));B.save(out/'frozen_before_test.json',dict(calibration=cal,hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.pt')}))
        B.evaluate(d,zs,temps,cal,gate,sf,sg,out)
        oracle(d,zs,temps,gate,cal,seed,out,'2011-10-01','2012-01-01','revised_validation')
        oracle(d,zs,temps,gate,cal,seed,out,'2012-01-01','2013-01-01','revised_test')
        del tr,va,tf,tg,vf,vg
        large_support(d,zs,temps,cal,out)

if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
