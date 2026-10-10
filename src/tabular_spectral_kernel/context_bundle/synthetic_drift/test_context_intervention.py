import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import numpy as np
import pytest
import torch
from context_intervention import query_only,wrong_regime,VARIANTS
from mlp_ablation import construct


@pytest.mark.parametrize('variant',VARIANTS)
def test_query_only_is_independent_of_other_queries_and_context_branches(variant):
    torch.manual_seed(77);m=construct(variant).double();q=torch.randn(1,3,8,dtype=torch.float64);q[:,:,5:]=0
    def fail(*args):raise AssertionError('Context branch executed')
    handles=[]
    for name,module in m.named_modules():
        if name in ['context','context_mix','attention','value','out'] or name.startswith('layers.'):
            handles.append(module.register_forward_pre_hook(fail))
    a=query_only(m,q)
    torch.testing.assert_close(a[:,:1],query_only(m,q[:,:1]))
    torch.testing.assert_close(a.sum(-1),torch.ones_like(a[...,0]))
    for h in handles:h.remove()


def test_wrong_regime_preserves_borrowers_and_uses_only_context_macro():
    rng=np.random.default_rng(5);x=rng.uniform(-1,1,(128,5));x[:,4]=2*(.04-.03)/.09-1
    z,y,rate=wrong_regime(x,np.random.default_rng(6))
    np.testing.assert_array_equal(z[:,:4],x[:,:4]);assert rate==.10
    assert ((y>0)&(y<1)).all()
    zz,yy,rate=wrong_regime(z,np.random.default_rng(6));assert rate==.04
    assert yy.mean()>y.mean()
    np.testing.assert_allclose(zz,x,atol=1e-14)
