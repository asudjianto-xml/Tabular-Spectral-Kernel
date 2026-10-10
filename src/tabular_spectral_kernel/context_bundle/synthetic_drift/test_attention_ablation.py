import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent))
import run as R
import run_transformer as T
from psd_spectral_attention import PSDSpectralGate,SpectralKernel


def test_no_rotation_retains_month_input_but_disables_temporal_phase():
    torch.manual_seed(17);torch.set_num_threads(1)
    model=PSDSpectralGate(temporal_rotation=False).double()
    for module in model.modules():
        if isinstance(module,SpectralKernel):
            assert torch.count_nonzero(module.time_frequency)==0
            assert not module.time_frequency.requires_grad
            x=torch.randn(1,7,32,dtype=torch.float64)
            a=module.embedding(x,torch.arange(7,dtype=torch.float64)[None])
            b=module.embedding(x,torch.zeros(1,7,dtype=torch.float64))
            torch.testing.assert_close(a,b,atol=0,rtol=0)
    context=torch.randn(1,16,8,dtype=torch.float64);context[:,:,7]=torch.arange(-4,0).repeat_interleave(4)/4
    query=torch.randn(1,2,8,dtype=torch.float64);query[:,:,5:]=0
    altered=context.clone();altered[:,:,7]=context[:,:,7].flip(1)
    with torch.no_grad():
        original=model(context,query,None);reversed_times=model(altered,query,None)
    assert not torch.allclose(original,reversed_times,atol=1e-9,rtol=1e-9)


def test_krr_loss_matches_decoder_and_default_joint_is_unchanged():
    torch.manual_seed(19);rng=np.random.default_rng(19);n=8;queries=3
    z=[rng.normal(size=(n+queries,3)) for _ in range(2)]
    grams=np.array([R.kernel(a[:n],a[:n],1.) for a in z]);cross=np.array([R.kernel(a[n:],a[:n],1.) for a in z])
    batch=dict(context=torch.randn(1,n,8),query=torch.randn(1,queries,8),descriptor=torch.randn(1,3),
        grams=torch.tensor(grams[None]),cross=torch.tensor(cross[None]),
        y=torch.tensor(rng.normal(size=(1,n))),target=torch.tensor(rng.normal(size=(1,queries))))
    model=T.SummaryGate();alpha=model(batch['context'],batch['query'],batch['descriptor'])
    nw,krr=T.decode(alpha,batch['grams'],batch['cross'],batch['y'],.01)
    torch.testing.assert_close(T.loss_for(model,batch,.01,'krr'),(krr-batch['target']).square().mean())
    torch.testing.assert_close(T.loss_for(model,batch,.01),.5*((nw-batch['target']).square().mean()+(krr-batch['target']).square().mean()))
