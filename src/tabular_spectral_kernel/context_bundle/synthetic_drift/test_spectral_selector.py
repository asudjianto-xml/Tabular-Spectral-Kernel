import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import numpy as np
import torch
from run_spectral_selector import SpectralSelector


def model(hybrid=False):
    torch.manual_seed(41)
    return SpectralSelector(torch.zeros(5),torch.ones(5),[0,2,4],torch.arange(5,dtype=torch.float32),atoms=8,hybrid=hybrid).double()


def test_spectral_kernel_is_stationary_psd_unit_diagonal():
    m=model();x=torch.randn(17,5,dtype=torch.float64);y=torch.randn(4,5,dtype=torch.float64)
    k=m.kernel(x,x)
    assert torch.linalg.eigvalsh(k).min()>-1e-10
    torch.testing.assert_close(k.diag(),torch.ones(17,dtype=torch.float64))
    shift=torch.randn(5,dtype=torch.float64)
    torch.testing.assert_close(m.kernel(x+shift,y+shift),m.kernel(x,y))
    phase=(x[:,None,[0,2,4]]-y[None,:,[0,2,4]])@m.frequency.T
    torch.testing.assert_close(m.kernel(x,y),phase.cos().mean(-1))


def test_hybrid_starts_at_linear_baseline_and_learns_spectral_terms():
    m=model(True);x=torch.randn(2,3,6,5,dtype=torch.float64)
    torch.testing.assert_close(m(x),(x*m.base_weight).sum(-1))
    with torch.no_grad():m.coefficient.normal_()
    loss=m(x).square().mean();loss.backward()
    assert m.frequency.grad.abs().max()>0 and m.coefficient.grad.abs().max()>0
    torch.testing.assert_close(m(x[:,:1]),m(x)[:,:1])
    assert not any(isinstance(p,(torch.nn.GELU,torch.nn.ReLU,torch.nn.Tanh)) for p in m.modules())
