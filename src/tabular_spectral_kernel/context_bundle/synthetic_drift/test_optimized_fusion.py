import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import run as R
import run_optimized_fusion as O


def fixture():
    rng=np.random.default_rng(14);x=rng.normal(size=(24,2))
    grams=[R.kernel(x[:,j:j+1],x[:,j:j+1],.8) for j in range(2)]
    return grams,np.sin(x[:,0])+rng.normal(0,.05,24)


def test_profiled_convex_objective_equals_regularized_primal():
    grams,y=fixture();ridge=.01
    for w in [0.,.23,.7,1.]:
        k=w*grams[0]+(1-w)*grams[1];coef,b=R.ridge_fit(k,y,ridge)
        primal=np.mean((y-k@coef-b)**2)+ridge*(coef@k@coef)
        np.testing.assert_allclose(O.objective(grams,y,ridge,w,"convex_mkl"),primal,atol=1e-12)
    vals=np.array([O.objective(grams,y,ridge,w,"convex_mkl") for w in np.linspace(0,1,101)])
    assert np.diff(vals,n=2).min()>-1e-12


def test_weight_optimization_beats_grid_candidates_and_refits_psd_kernel():
    grams,y=fixture()
    for method in ["validation","convex_mkl"]:
        w,info=O.optimize_weight(grams,y,.01,method,support=16)
        grid=[O.objective(grams,y,.01,v,method,support=16) for v in np.linspace(0,1,101)]
        assert 0<=w<=1 and info["objective"]<=min(grid)+1e-10
        k=w*grams[0]+(1-w)*grams[1]
        assert np.linalg.eigvalsh(k).min()>-1e-10
        if method=="validation":
            p=R.ridge_predict(k[:16,:16],k[16:,:16],y[:16],.01)
            np.testing.assert_allclose(info["objective"],np.mean((p-y[16:])**2))
