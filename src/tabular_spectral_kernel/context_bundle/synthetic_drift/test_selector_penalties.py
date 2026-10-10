import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import numpy as np
import torch
from run_selector_penalties import augment,prox,NAMES,GROUPS


def test_absolute_macro_feature_reproduces_heuristic():
    x=np.random.default_rng(4).normal(size=(3,5,6,56)).astype(np.float32)
    d=augment(dict(features=x))
    np.testing.assert_array_equal((-d['features'][...,-1]).argmax(-1),abs(x[...,9]).argmin(-1))
    assert len(NAMES)==57 and sorted(sum(GROUPS.values(),[]))==list(range(57))


def test_proximal_penalties_create_exact_sparsity():
    w=torch.ones(57)*.1;active=torch.ones(57,dtype=torch.bool);active[0]=False
    assert torch.count_nonzero(prox(w,1.,.2,'l1',active))==0
    assert torch.count_nonzero(prox(w,1.,.3,'elastic',active))==0
    assert torch.count_nonzero(prox(w,1.,.2,'group',active))==0
    z=prox(w,1.,.2,'l2',active)
    assert z[0]==0 and torch.count_nonzero(z)==56
    torch.testing.assert_close(z[1:],w[1:]/1.2)
