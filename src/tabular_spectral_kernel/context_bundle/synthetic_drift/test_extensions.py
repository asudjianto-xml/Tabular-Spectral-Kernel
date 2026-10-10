import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).parent))
import run_localized as L
import run_meta as M


def test_selective_correction_preserves_inactive_population_and_is_psd():
    base=np.array([.2,.7,.9,.4]);groups=np.array([0,1,0,1])
    prediction=L.apply_local(base,groups,{1:np.array([.5,-.1])})
    np.testing.assert_array_equal(prediction[groups==0],base[groups==0])
    rng=np.random.default_rng(0);features=rng.normal(size=(4,3));k=features@features.T
    gate=(groups==1).astype(float);local=gate[:,None]*k*gate[None,:]
    assert np.linalg.eigvalsh(local).min()>-1e-12
    np.testing.assert_array_equal(local[groups==0],0)


def test_residual_monitor_detects_only_changed_group():
    groups=np.repeat([0,1],64);residual=np.r_[np.full(64,.1),np.ones(64)]
    flags,_=L.detect(residual,groups,[.01,.01])
    assert flags==[False,True]


def test_meta_descriptors_and_predictions_are_context_permutation_invariant():
    torch.manual_seed(8);rng=np.random.default_rng(8)
    x=rng.normal(size=(20,3));q=rng.normal(size=(7,3));y=rng.normal(size=20)
    grams=[np.exp(-M.R.cdist(x[:,j:j+1],x[:,j:j+1])) for j in [0,1]]
    cross=[np.exp(-M.R.cdist(q[:,j:j+1],x[:,j:j+1])) for j in [0,1]]
    gate=M.ContextGate();p=rng.permutation(len(y))
    before=M.forward(gate,grams,cross,y,.01)
    after=M.forward(gate,[k[np.ix_(p,p)] for k in grams],[k[:,p] for k in cross],y[p],.01)
    for a,b in zip(before,after):np.testing.assert_allclose(a,b,atol=1e-12)
    assert np.all(before[0]>=0);np.testing.assert_allclose(before[0].sum(),1.)
    assert np.all(before[1]>=y.min()) and np.all(before[1]<=y.max())
    # Query covariates affect predictions but never the encoder weights.
    moved=M.forward(gate,grams,[k[::-1] for k in cross],y,.01)
    np.testing.assert_array_equal(before[0],moved[0])


def test_nw_preserves_constant_labels():
    q=np.array([[.1,.2,.5],[.8,.1,.2]])
    np.testing.assert_allclose(M.nw_predict([q,q*.5],np.full(3,7.),[.3,.7]),7.)
