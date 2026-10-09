import numpy as np
import torch
from tabular_spectral_kernel._paper import identifiable_spectral_california as base
from tabular_spectral_kernel._paper import spectral_stage1_extended as ext
from tabular_spectral_kernel._paper.solvers import exact_path

torch.set_num_threads(2)

def test_canonical_metric_inverse_kernel_and_gradients():
    torch.manual_seed(4);m=base.CanonicalSpectral(2,3)
    with torch.no_grad():m.offdiag.normal_(0,.03);m.diagonal.normal_(0,.05)
    R=m.normalized_root();B=R@R.T;values,Q=torch.linalg.eigh(B)
    logB=(Q*values.log())@Q.T;A=logB-logB.trace()/m.p*torch.eye(m.p,dtype=torch.float64)
    torch.testing.assert_close(A,m.generator(),atol=1e-11,rtol=1e-11)
    assert abs(float(B.trace().detach())-1)<1e-12
    x=torch.randn(24,2,dtype=torch.float64);K=m.kernel(m.embedding(x),m.embedding(x))
    assert torch.linalg.eigvalsh(K).min()>-1e-10
    K.square().mean().backward()
    for p in [m.diagonal,m.offdiag,m.log_gaps,m.log_temperature]:assert p.grad is not None and torch.isfinite(p.grad).all()

def test_ridge_matches_independent_centered_solve_and_full_nystrom():
    torch.manual_seed(8);m=base.CanonicalSpectral(2,2);x=torch.randn(20,2,dtype=torch.float64);q=torch.randn(7,2,dtype=torch.float64);y=torch.randn(20,dtype=torch.float64);ridge=.002
    pred,res=exact_path(m,x,y,q,[ridge]);approx=base.nystrom_path(m,x,y,q,torch.arange(len(x)),[ridge])
    with torch.no_grad():K=m.kernel(m.embedding(x),m.embedding(x)).numpy();Q=m.kernel(m.embedding(q),m.embedding(x)).numpy()
    H=np.eye(20)-np.ones((20,20))/20;Kc=H@K@H;Qc=Q-Q.mean(1)[:,None]-K.mean(0)[None,:]+K.mean()
    expected=Qc@np.linalg.solve(Kc+20*ridge*np.eye(20),y.numpy()-y.numpy().mean())+y.numpy().mean()
    np.testing.assert_allclose(pred.numpy()[:,0],expected,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(pred,approx,atol=2e-7,rtol=2e-7);assert max(res)<1e-10

def test_mixed_preprocessing_training_only_and_duplicate_gradients():
    import pandas as pd
    ext.DEVICE='cpu';df=pd.DataFrame({'n':[0.,1.,2.,1000.],'c':['a','b','a','unseen']})
    d=dict(df=df,nums=['n'],cats=['c'],tr=np.arange(3));prep=ext.MixedPreprocessor('standard').fit(d)
    assert prep.sc.mean_[0]==1
    x=prep.transform(df);assert np.all(x[-1,1:]==0)
    for h in prep.contrasts:np.testing.assert_allclose(h.T@h,np.eye(h.shape[1]),atol=1e-14)
    m=ext.MixedSpectral(1,prep.q);xt=ext.tensor(np.vstack([x[:3],x[:1]]));K=m.kernel(m.embedding(xt),m.embedding(xt));K.square().mean().backward()
    assert torch.isfinite(m.log_gaps.grad).all();assert K[0,-1]==1
