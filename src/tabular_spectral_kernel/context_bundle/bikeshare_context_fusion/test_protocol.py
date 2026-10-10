import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import copy
import numpy as np
import torch
import run as R


def test_solver_matches_augmented_normal_equations():
    rng=np.random.default_rng(3);f=rng.normal(size=(9,4));k=f@f.T;q=rng.normal(size=4)@f.T;y=rng.normal(size=9);lam=.003
    # Direct saddle system enforces the unpenalized intercept normal equation.
    mat=np.block([[k+9*lam*np.eye(9),np.ones((9,1))],[np.ones((1,9)),np.zeros((1,1))]])
    coef=np.linalg.solve(mat,np.r_[y,0]);ref=q@coef[:9]+coef[9]
    actual=R.solve(torch.tensor(k),torch.tensor(q),torch.tensor(y),lam)
    np.testing.assert_allclose(float(actual),ref,rtol=1e-10)


def test_every_origin_and_candidate_is_past_only():
    d=R.load()
    for day in np.unique(d['dt'].astype('datetime64[D]')):
        if day<np.datetime64('2011-07-01'):continue
        q,sp,mask=R.indices(d,day)
        assert np.all(d['dt'][sp[mask]]<np.datetime64(day))
        assert not np.intersect1d(sp[mask].ravel(),q).size
    assert not R.indices(d,np.datetime64('2011-07-01'))[2][-1]


def test_features_and_predictions_ignore_current_and_future_labels():
    d=R.load();torch.manual_seed(1)
    z=[torch.randn(len(d['y']),5,device=R.DEVICE) for _ in range(3)]
    day=np.datetime64('2012-05-01');e=R.episode(d,z,[1,1,1],day)
    changed=dict(d,y_scaled=d['y_scaled'].copy());changed['y_scaled'][d['dt']>=day]+=10000
    other=R.episode(changed,z,[1,1,1],day)
    for key in ['K','Q','y','f']:torch.testing.assert_close(e[key],other[key])
    torch.testing.assert_close(R.fixed_predictions(e,[.2,.3,.5],.001),R.fixed_predictions(other,[.2,.3,.5],.001))
    assert not torch.equal(e['target'],other['target'])


def test_spectral_rule_psd_and_query_independence():
    torch.manual_seed(4);m=R.SpectralRule(torch.zeros(8),torch.ones(8),outputs=3)
    with torch.no_grad():m.head.normal_()
    f=torch.randn(17,8);phi=m.phi(f);k=phi@phi.T
    assert torch.linalg.eigvalsh(k.double()).min()>-1e-6
    torch.testing.assert_close(k.diag(),torch.ones(17))
    torch.testing.assert_close(m(f[:1]),m(f)[:1])
    alpha=m(f).softmax(-1);assert (alpha>=0).all()
    torch.testing.assert_close(alpha.sum(-1),torch.ones(17))


def test_fusion_not_prediction_averaging():
    k1=torch.eye(2,dtype=torch.float64);k2=torch.ones((2,2),dtype=torch.float64);y=torch.tensor([1.,-1.],dtype=torch.float64)
    k=(k1+k2)/2
    fused=R.solve(k[None].expand(2,-1,-1),k,y[None].expand(2,-1),.5)
    avg=(R.solve(k1[None].expand(2,-1,-1),k1,y[None].expand(2,-1),.5)+R.solve(k2[None].expand(2,-1,-1),k2,y[None].expand(2,-1),.5))/2
    torch.testing.assert_close(fused,y/3);torch.testing.assert_close(avg,y/4)
