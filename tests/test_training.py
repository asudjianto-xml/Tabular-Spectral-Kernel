"""Exercise selection, exact fitting and calibration on a small independent fixture."""
import numpy as np
import pandas as pd
import pytest
from tabular_spectral_kernel._paper import spectral_stage1_extended as ext

@pytest.mark.parametrize('classification',[False,True])
def test_training_selection_pipeline(tmp_path,monkeypatch,classification):
    rng=np.random.default_rng(41)
    x=rng.normal(size=(120,2));y=(x[:,0]>.1).astype(float) if classification else np.sin(x[:,0])+.2*x[:,1]
    df=pd.DataFrame(x,columns=['a','b']);df['category']=np.where(x[:,1]>0,'left','right')
    d=dict(name='fixture',seed=0,df=df,y=y,nums=['a','b'],cats=['category'],tr=np.arange(72),va=np.arange(72,88),cal=np.arange(88,104),te=np.arange(104,120),classification=classification,split_info={})
    for key,value in dict(OUT=tmp_path/'new output',DEVICE='cpu',STEPS=2,SUPPORT=24,QUERY=12,RIDGES=np.array([1e-4,1e-3,1e-2])).items():monkeypatch.setattr(ext,key,value)
    monkeypatch.setattr(ext,'load',lambda name,seed:d)
    result=ext.spectral_select('fixture',0,None)
    assert result['max_relative_solve_residual']<1e-8
    assert result['min_metric_eigenvalue']>0
    assert result['trace_error']<1e-10
    assert (result['calibration'] is not None)==classification
    assert (ext.OUT/'fixture/seed0/head.pt').exists()
