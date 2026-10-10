import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import numpy as np
import run as R
from run_oracle_references import MixtureCurve,oracle_predictions


def problem():
    rng=np.random.default_rng(97);z=rng.normal(size=(22,4))
    g=np.array([R.kernel(z[:16,:2],z[:16,:2],.7),R.kernel(z[:16,2:],z[:16,2:],1.2)])
    q=np.array([R.kernel(z[16:,:2],z[:16,:2],.7),R.kernel(z[16:,2:],z[:16,2:],1.2)])
    return g,q,rng.uniform(.1,.9,16)


def test_generalized_curve_matches_direct_unpenalized_intercept():
    g,q,y=problem();curve=MixtureCurve(g,q,y,.005)
    for w in [0.,.127,.5,.839,1.]:
        direct=R.ridge_predict(w*g[0]+(1-w)*g[1],w*q[0]+(1-w)*q[1],y,.005).clip(0,1)
        np.testing.assert_allclose(curve(w)[0],direct,atol=1e-11)


def test_query_oracle_attains_feasible_targets_and_dominates_candidates():
    g,q,y=problem();curve=MixtureCurve(g,q,y,.005)
    weights=np.linspace(.07,.93,6)
    target=np.array([curve(w,i)[0,0] for i,w in enumerate(weights)])
    mean=target+.02
    known=np.array([np.zeros(6),np.ones(6),weights])
    result,selected,lo,hi=oracle_predictions(curve,target,mean,known,101)
    np.testing.assert_allclose(result['oracle_query_labels'],target,atol=1e-9)
    assert np.all(lo<=target+1e-10) and np.all(hi>=target-1e-10)
    for name,w in selected.items():assert ((w>=0)&(w<=1)).all()
    expected=np.clip(mean,lo,hi)
    np.testing.assert_allclose(result['oracle_query_mean'],expected,atol=1e-9)
