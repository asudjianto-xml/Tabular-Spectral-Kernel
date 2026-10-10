"""Numerical and leakage checks for the synthetic drift experiment."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("synthetic_drift_run", Path(__file__).with_name("run.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def test_controlled_drift_has_identical_inputs_and_noise():
    stationary=R.generate(4,"stationary")
    recurring=R.generate(4,"recurring")
    np.testing.assert_array_equal(stationary["x"],recurring["x"])
    np.testing.assert_allclose(stationary["y"]-stationary["mean"],
                               recurring["y"]-recurring["mean"],atol=1e-15)
    assert not np.allclose(stationary["mean"][384:768],recurring["mean"][384:768])
    np.testing.assert_array_equal(stationary["mean"][768:1152],recurring["mean"][768:1152])


def test_psd_for_joint_context_query_and_fusion():
    torch.manual_seed(3)
    model=R.base.CanonicalSpectral(3,4,max_frequency=3.)
    x=np.random.default_rng(3).uniform(-1,1,(24,3))
    z=R.embedding(model,x)
    k=R.kernel(z,z,float(model.log_temperature.detach().exp()))
    for gram in [k,.3*k+.7*np.eye(len(k))]:
        np.testing.assert_allclose(np.diag(gram),1,atol=1e-12)
        np.testing.assert_allclose(gram,gram.T,atol=1e-12)
        assert np.linalg.eigvalsh(gram).min()>-1e-10


def test_ridge_matches_augmented_intercept_equations():
    rng=np.random.default_rng(8);x=rng.normal(size=(12,4));k=x@x.T;y=rng.normal(size=12)
    c,b=R.ridge_fit(k,y,.02)
    system=np.block([[k+12*.02*np.eye(12),np.ones((12,1))],
                     [np.ones((1,12)),np.zeros((1,1))]])
    expected=np.linalg.solve(system,np.r_[y,0.])
    np.testing.assert_allclose(np.r_[c,b],expected,atol=1e-11)


def test_fusion_differs_from_average_and_reduces_at_vertices():
    y=np.array([1.,-1.]);a=np.eye(2);b=np.ones((2,2));mixed=(a+b)/2
    fused=R.ridge_predict(mixed,mixed,y,.5)
    average=(R.ridge_predict(a,a,y,.5)+R.ridge_predict(b,b,y,.5))/2
    np.testing.assert_allclose(fused,y/3)
    np.testing.assert_allclose(average,y/4)
    grams=[a,b,a];selection=dict(kernel_fusion=dict(weight=1.,ridge=.5),
        uniform_kernel=dict(weight=.5,ridge=.5),best_single=dict(kernel=0,ridge=.5),
        pooled_kernel=dict(kernel=2,ridge=.5),latest_kernel=dict(kernel=1,ridge=.5),
        prediction_average=dict(weight=1.,ridges=[.5,.5]))
    preds=R.predict_heads(grams,grams,y,selection)
    np.testing.assert_allclose(preds["kernel_fusion"],preds["best_single"])


def test_selection_uses_past_context_and_includes_vertices():
    data=R.generate(3,"recurring");start,origin,end=400,528,560
    x=data["x"];y=data["y"].copy()
    grams=[np.exp(-R.cdist(x[start:origin,j:j+1],x[start:origin,j:j+1])) for j in range(3)]
    before=R.select_heads(grams,y[start:origin],80)
    y[origin:]=1e6  # Future labels cannot change selected weights or ridges.
    after=R.select_heads(grams,y[start:origin],80)
    assert before==after
    assert before["kernel_fusion"]["val_mse"]<=before["best_single"]["val_mse"]+1e-12
    assert before["kernel_fusion"]["val_mse"]<=before["uniform_kernel"]["val_mse"]+1e-12
    assert 0<=before["kernel_fusion"]["weight"]<=1
