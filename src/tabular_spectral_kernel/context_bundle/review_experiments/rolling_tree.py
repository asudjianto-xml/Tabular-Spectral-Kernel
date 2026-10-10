"""Rolling expanding-history CatBoost anchor, validation-selected depth."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import importlib.util,json,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bike_base',ROOT/'bikeshare_context_fusion/run.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
OUT=runtime.output_path(__file__, Path(__file__).parent/'tree_results')

def main():
    OUT.mkdir(exist_ok=False);d=B.load();days=np.unique(d['dt'].astype('datetime64[D]'))
    configs=[dict(depth=k,iterations=200,learning_rate=.08,l2_leaf_reg=3) for k in [4,6]]
    B.save(OUT/'protocol.json',dict(configs=configs,seed=0,validation='Oct-Dec2011',test='all2012',support='all observed labels strictly before each daily origin',inputs=B.FEATURES,target='raw cnt; clipped at0',note='One deterministic optimization seed; review-stage validation-selected anchor; same observed-weather task.'))
    (OUT/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    def fit(day,cfg):
        sp=np.flatnonzero(d['dt']<day);qi=np.flatnonzero((d['dt']>=day)&(d['dt']<day+np.timedelta64(1,'D')))
        assert d['dt'][sp[-1]]<d['dt'][qi[0]]
        m=CatBoostRegressor(**cfg,loss_function='RMSE',random_seed=0,thread_count=2,allow_writing_files=False,verbose=False)
        m.fit(d['x'][sp],d['y'][sp]);return qi,np.maximum(m.predict(d['x'][qi]),0)
    records=[]
    for cfg in configs:
        y=[];p=[]
        for day in days[(days>=np.datetime64('2011-10-01'))&(days<np.datetime64('2012-01-01'))]:
            qi,pred=fit(day,cfg);y.extend(d['y'][qi]);p.extend(pred)
        rec=dict(config=cfg,**B.metrics(np.array(y),np.array(p)));records.append(rec);print('VALIDATION',rec,flush=True)
    winner=min(records,key=lambda r:r['rmse']);B.save(OUT/'selection.json',dict(records=records,chosen=winner))
    frames=[]
    for j,day in enumerate(days[days>=np.datetime64('2012-01-01')]):
        qi,p=fit(day,winner['config']);frames.append(pd.DataFrame(dict(row=qi,timestamp=d['dt'][qi],target=d['y'][qi],prediction=p)))
        if j%60==0:print('TEST',j,flush=True)
    f=pd.concat(frames);f.to_csv(OUT/'predictions.csv',index=False);m=B.metrics(f.target.to_numpy(),f.prediction.to_numpy());B.save(OUT/'metrics.json',m);print('RESULT',m,flush=True)
if __name__=='__main__':main()
