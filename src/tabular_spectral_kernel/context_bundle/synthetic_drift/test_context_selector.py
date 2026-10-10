import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import numpy as np
import torch
from run_context_selector import descriptors,episode,Selector,train


def test_episode_and_descriptors_use_only_available_information():
    x,y,mu=episode(np.random.default_rng(8),'recurring')
    assert x.shape==(800,5) and y.shape==mu.shape==(800,)
    cx=x[:768].reshape(6,128,5);cy=y[:768].reshape(6,128)
    cross=np.ones((6,32,128))
    a=descriptors(cx,cy,x[768:],cross)
    changed=y.copy();changed[768:]=0
    np.testing.assert_array_equal(a,descriptors(cx,changed[:768].reshape(6,128),x[768:],cross))
    assert a.shape[:2]==(32,6) and np.isfinite(a).all()
    np.testing.assert_allclose(a[:1],descriptors(cx,cy,x[768:769],cross[:,:1]))


def test_linear_selector_learns_historical_action_costs():
    rng=np.random.default_rng(9)
    def make(n):
        x=rng.uniform(-1,1,(n,4,6,1)).astype(np.float32)
        # Best support is the largest descriptor; labels are never score inputs.
        return dict(features=x,predictions=(1-x[...,0])*.3,target=np.zeros((n,4)))
    tr,va=make(64),make(32);m,info=train(tr,va,0)
    with torch.no_grad():choice=m(torch.tensor(va['features'])).argmax(-1).numpy()
    assert np.mean(choice==va['features'][...,0].argmax(-1))>.95
    assert isinstance(m.score,torch.nn.Linear) and info['parameters']==1
