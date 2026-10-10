"""Validation-only check of whether the expanded ridge grid has reached its interpolation limit."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import json
from pathlib import Path
import numpy as np,torch
from threadpoolctl import threadpool_limits
from bike_revision import B,load_bank,ROOT
OUT=runtime.output_path(__file__, Path(__file__).parent/'bike_results')
@torch.no_grad()
def main():
    d=B.load();records=[]
    for seed in [0,1,2]:
        zs,temps=load_bank(d,seed);cal=json.loads((OUT/f'seed{seed}/selected_calibration.json').read_text())
        errors={key:{r:[] for r in [0.,1e-9,1e-8,1e-7]} for key in ['fixed','latest','single','uniform']};y=[]
        for day in np.unique(d['dt'].astype('datetime64[D]')):
            if not np.datetime64('2011-10-01')<=day<np.datetime64('2012-01-01'):continue
            e=B.episode(d,zs,temps,day);e=dict(e,K=e['K'][:1].double(),Q=e['Q'][:1].double(),y=e['y'][:1].double())
            y.extend(e['target'].numpy()*d['ys'])
            for key in errors:
                a=torch.tensor(cal[key]['alpha'],dtype=torch.float64,device=B.DEVICE)
                k=torch.einsum('lmij,m->lij',e['K'].to(B.DEVICE),a)[0];q=torch.einsum('lqmi,m->lqi',e['Q'].to(B.DEVICE),a)[0];ys=e['y'][0].to(B.DEVICE)
                for ridge in errors[key]:
                    # Finite frozen Gram is strictly positive definite here; lambda=0 is a diagnostic only.
                    sol=torch.cholesky_solve(torch.stack([ys,torch.ones_like(ys)],-1),torch.linalg.cholesky(k+128*ridge*torch.eye(128,dtype=k.dtype,device=k.device)))
                    b=sol[:,0].sum()/sol[:,1].sum();p=(q@(sol[:,0]-b*sol[:,1])+b).clamp_min(0).cpu().numpy()*d['ys'];errors[key][ridge].extend(p)
        for key,paths in errors.items():
            vals={str(r):float(np.sqrt(np.mean((np.array(p)-y)**2))) for r,p in paths.items()}
            records.append(dict(seed=seed,method=key,validation_rmse=vals,delta_1e7_minus_zero=vals['1e-07']-vals['0.0']))
    B.save(OUT/'ridge_limit_validation.json',records);print('Max validation RMSE difference, 1e-7 vs0:',max(abs(r['delta_1e7_minus_zero']) for r in records))
if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
