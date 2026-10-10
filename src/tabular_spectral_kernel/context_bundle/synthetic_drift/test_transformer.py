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


def test_transformer_context_permutation_and_query_independence():
    torch.manual_seed(123);torch.set_num_threads(1)
    model=T.TransformerGate().eval()
    context=torch.randn(2,128,8);query=torch.randn(2,4,8);desc=torch.randn(2,3)
    with torch.no_grad():
        original=model(context,query,desc)
        permuted=model(context[:,torch.randperm(128)],query,desc)
        one=model(context,query[:,:1],desc)
    torch.testing.assert_close(original,permuted,atol=2e-7,rtol=1e-6)
    torch.testing.assert_close(original[:,:1],one,atol=2e-7,rtol=1e-6)
    torch.testing.assert_close(original.sum(-1),torch.ones(2,4))


def test_tokens_mask_query_response_and_encode_relative_time():
    rng=np.random.default_rng(4);x,y=T.episode(rng,4)
    context,query=T.tokens(x[:128],y[:128],x[128:])
    np.testing.assert_array_equal(query[:,5:],0.)
    np.testing.assert_array_equal(context[:,5],y[:128])
    np.testing.assert_array_equal(context[:,6],1.)
    np.testing.assert_array_equal(context[:,7],np.repeat([ -1.,-.75,-.5,-.25],32))


def test_batched_query_krr_matches_independent_solve_and_has_gradients():
    rng=np.random.default_rng(8);n=10;queries=3
    zs=[rng.normal(size=(n+queries,3)) for _ in range(2)]
    grams=np.array([R.kernel(z[:n],z[:n],1.) for z in zs])
    cross=np.array([R.kernel(z[n:],z[:n],1.) for z in zs]);y=rng.normal(size=n)
    logits=torch.randn(1,queries,2,dtype=torch.float64,requires_grad=True)
    alpha=logits.softmax(-1)
    nw,krr=T.decode(alpha,torch.tensor(grams[None]),torch.tensor(cross[None]),torch.tensor(y[None]),.01)
    for i in range(queries):
        a=alpha.detach().numpy()[0,i]
        k=sum(w*g for w,g in zip(a,grams));q=sum(w*g[i:i+1] for w,g in zip(a,cross))
        np.testing.assert_allclose(krr.detach().numpy()[0,i],R.ridge_predict(k,q,y,.01)[0],atol=1e-12)
        np.testing.assert_allclose(nw.detach().numpy()[0,i],(q@y/q.sum())[0],atol=1e-12)
    (nw.square().mean()+krr.square().mean()).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().max()>1e-8
