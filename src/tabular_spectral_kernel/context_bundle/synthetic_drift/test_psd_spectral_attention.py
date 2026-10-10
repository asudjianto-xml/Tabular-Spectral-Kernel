import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent))
from psd_spectral_attention import PSDSpectralAttention,PSDSpectralGate


def test_full_context_query_extension_is_symmetric_psd():
    torch.manual_seed(11);torch.set_num_threads(1)
    model=PSDSpectralAttention().double()
    c=torch.randn(2,15,32,dtype=torch.float64);q=torch.randn(2,3,32,dtype=torch.float64)
    cp=torch.arange(-15,0,dtype=torch.float64).expand(2,-1);qp=torch.zeros(2,3,dtype=torch.float64)
    cc,qc,qq=model.normalized_blocks(c,cp,q,qp)
    joint=torch.cat([torch.cat([cc,qc.transpose(-1,-2)],-1),torch.cat([qc,qq],-1)],-2)
    torch.testing.assert_close(joint,joint.transpose(-1,-2),atol=1e-12,rtol=1e-12)
    assert torch.linalg.eigvalsh(joint).min()>-1e-10
    assert (joint>=0).all()
    # The query block must not depend on the presence of other query rows.
    _,one,_=model.normalized_blocks(c,cp,q[:,:1],qp[:,:1])
    torch.testing.assert_close(qc[:,:1],one,atol=1e-12,rtol=1e-12)


def test_time_changes_similarity_but_record_order_does_not():
    torch.manual_seed(12);model=PSDSpectralAttention().double()
    c=torch.randn(1,12,32,dtype=torch.float64);p=torch.arange(-12,0,dtype=torch.float64)[None]
    a=model.normalized_blocks(c,p)
    changed=model.normalized_blocks(c,p.flip(1))
    assert not torch.allclose(a,changed,atol=1e-5)
    shuffled=model.normalized_blocks(c.flip(1),p.flip(1))
    torch.testing.assert_close(a,shuffled.flip(1).flip(2))


def test_gate_is_trainable_and_simplex_valued():
    torch.manual_seed(13);model=PSDSpectralGate()
    c=torch.randn(2,16,8);c[:,:,7]=torch.arange(-4,0).repeat_interleave(4)/4
    q=torch.randn(2,3,8);q[:,:,5:]=0
    alpha=model(c,q,None)
    torch.testing.assert_close(alpha.sum(-1),torch.ones(2,3))
    alpha[:,:,0].square().mean().backward()
    for name in ["log_frequency","time_frequency","root","log_temperature"]:
        grad=getattr(model.layers[0].attention.geometry,name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
