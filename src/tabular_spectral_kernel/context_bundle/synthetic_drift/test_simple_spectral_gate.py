import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import sys
from pathlib import Path
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import run as R
import run_transformer as T
from simple_spectral_gate import SimpleSpectralGate


def batch():
    rng=np.random.default_rng(31);n=16;qs=3
    x=rng.normal(size=(n+qs,5))
    zs=[x[:,:2],x[:,2:]]
    g=np.array([R.kernel(z[:n],z[:n],1.) for z in zs])
    q=np.array([R.kernel(z[n:],z[:n],1.) for z in zs])
    qq=np.array([R.kernel(z[n:],z[n:],1.) for z in zs])
    c=rng.normal(size=(1,n,8));query=rng.normal(size=(1,qs,8));query[:,:,5:]=0
    return dict(context=torch.tensor(c),query=torch.tensor(query),descriptor=torch.zeros(1,3,dtype=torch.float64),
        grams=torch.tensor(g[None]),cross=torch.tensor(q[None]),y=torch.tensor(c[:,:,5]),target=torch.randn(1,qs,dtype=torch.float64)),torch.tensor(qq[None])


def test_simple_pool_joint_kernel_is_psd():
    torch.manual_seed(32);model=SimpleSpectralGate().double();b,qq=batch()
    beta,a,aq,dc,dq=model.attention_blocks(b['context'],b['grams'],b['cross'])
    aq_q=torch.einsum('bj,bjst->bst',beta,qq)/torch.sqrt(dq[...,None]*dq[...,None,:])
    joint=torch.cat([torch.cat([a,aq.transpose(-1,-2)],-1),torch.cat([aq,aq_q],-1)],-2)
    torch.testing.assert_close(joint,joint.transpose(-1,-2))
    assert torch.linalg.eigvalsh(joint).min()>-1e-10
    assert sum(p.numel() for p in model.parameters())==596


def test_simple_gate_query_dependence_and_order_invariance():
    torch.manual_seed(33);model=SimpleSpectralGate().double();b,_=batch()
    a=T.gate_weights(model,b)
    torch.testing.assert_close(a.sum(-1),torch.ones(1,3,dtype=torch.float64))
    assert float(a[0,:,0].std().detach())>1e-6
    one=dict(b,query=b['query'][:,:1],cross=b['cross'][:,:,:1])
    torch.testing.assert_close(a[:,:1],T.gate_weights(model,one))
    p=torch.randperm(16);moved=dict(b,context=b['context'][:,p],grams=b['grams'][:,:,p][:,:,:,p],cross=b['cross'][:,:,:,p])
    torch.testing.assert_close(a,T.gate_weights(model,moved))


def test_simple_gate_learns_through_krr_without_trainable_geometry():
    torch.manual_seed(34);model=SimpleSpectralGate().double();b,_=batch()
    T.loss_for(model,b,.01,'krr').backward()
    for p in model.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
    assert model.context_mix.weight.grad.abs().max()>0
    assert model.value[0].weight.grad.abs().max()>0
    assert not b['grams'].requires_grad and not b['cross'].requires_grad


@pytest.mark.parametrize('normalization',['symmetric','row'])
@pytest.mark.parametrize('power',[1,2])
def test_normalization_and_sharpening(normalization,power):
    torch.manual_seed(36)
    model=SimpleSpectralGate(normalization=normalization,power=power).double()
    b,qq=batch()
    beta,a,aq,dc,dq=model.attention_blocks(b['context'],b['grams'],b['cross'])
    full=torch.cat([torch.cat([b['grams'],b['cross'].transpose(-1,-2)],-1),
                    torch.cat([b['cross'],qq],-1)],-2)
    k=torch.einsum('bj,bjst->bst',beta,full).pow(power)
    assert torch.linalg.eigvalsh(k).min()>-1e-10
    if normalization=='row':
        torch.testing.assert_close(a.sum(-1),torch.ones_like(dc))
        torch.testing.assert_close(aq.sum(-1),torch.ones_like(dq))
        # Positive kernel row normalization is softmax of log-kernel scores.
        torch.testing.assert_close(aq,torch.softmax(k[:,16:,:16].log(),-1))
        torch.testing.assert_close(dc[...,None]*a,(dc[...,None]*a).transpose(-1,-2))
    else:
        d=torch.cat([dc,dq],-1)
        normalized=k/torch.sqrt(d[...,None]*d[...,None,:])
        torch.testing.assert_close(aq,normalized[:,16:,:16])
        assert torch.linalg.eigvalsh(normalized).min()>-1e-10
    alpha=T.gate_weights(model,b)
    one=dict(b,query=b['query'][:,:1],cross=b['cross'][:,:,:1])
    torch.testing.assert_close(alpha[:,:1],T.gate_weights(model,one))
    T.loss_for(model,b,.01,'joint').backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
