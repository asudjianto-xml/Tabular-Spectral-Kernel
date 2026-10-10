"""Additional exactly matched-width32 MLP control. Selection uses old validation records only."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import json
from pathlib import Path
import numpy as np,torch
from threadpoolctl import threadpool_limits
import synthetic_controls as C
OUT=runtime.output_path(__file__, Path(__file__).parent/'mlp32_results')
def main():
    OUT.mkdir(exist_ok=False);(OUT/'executed_source.py').write_bytes(Path(__file__).read_bytes());records=[]
    for seed in range(3):
        source=C.OUT/f'seed{seed}';trials=json.loads((source/'mlp_selection.json').read_text())['search'];chosen=min([r for r in trials if r['hidden']==32],key=lambda r:r['validation_mse'])
        info=json.loads((C.ROOT/f'synthetic_drift/unemployment_results/seed{seed}/original_training.json').read_text());weight=json.loads((C.ROOT/f'synthetic_drift/context_selector_results/seed{seed}/protocol.json').read_text())['mixture_weight'];banks=[]
        for name in ['normal','downturn']:
            m=C.R.base.CanonicalSpectral(5,4,max_frequency=3.,device='cpu');m.load_state_dict(torch.load(C.ROOT/f'synthetic_drift/unemployment_results/seed{seed}/{name}.pt',weights_only=True, map_location='cpu'));banks.append(m)
        st=torch.load(source/'mlp.pt',weights_only=True, map_location='cpu')
        if st['hidden.weight'].shape[0]==32:
            m=C.MLP(st['center'],st['scale'],32).to(C.DEV);m.load_state_dict(st)
        else:
            tr=C.P.augment(C.S.dataset(banks,info['temperatures'],info['ridge'],weight,910000+seed,512));va=C.P.augment(C.S.dataset(banks,info['temperatures'],info['ridge'],weight,920000+seed,128))
            m,rec=C.fit_mlp(tr,va,st['center'],st['scale'],seed,(32,chosen['penalty']))
            assert abs(rec['validation_mse']-chosen['validation_mse'])<1e-8
        torch.save({k:v.cpu() for k,v in m.state_dict().items()},OUT/f'seed{seed}.pt')
        for si,sc in enumerate(C.S.SCENARIOS):
            d=C.P.augment(C.S.dataset(banks,info['temperatures'],info['ridge'],weight,1230000+seed*1000+si*100,256,sc))
            with torch.no_grad():ix=m(torch.tensor(d['features'],device=C.DEV)).argmax(-1).cpu().numpy()
            p=np.take_along_axis(d['predictions'],ix[...,None],-1)[...,0]
            records.append(dict(seed=seed,scenario=sc,method='mlp32',parameters=1889,rmse_pp=float(100*np.sqrt(np.mean((p-d['target'])**2)))))
            np.savez_compressed(OUT/f'seed{seed}_{sc}.npz',prediction=p,target=d['target'])
        print('EXACT WIDTH',seed,records[-4:],flush=True)
    C.R.save(OUT/'metrics.json',records)
if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
