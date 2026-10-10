import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
import run_retrieval as E


def test_retrieval_rules_on_known_points():
    x=np.array([[0.,0.,-1.],[.1,0.,1.],[.8,0.,1.],[.9,0.,-1.]])
    query=np.array([0.,0.,1.]);cross=[np.array([[.1,.2,.8,.9]])]*2
    np.testing.assert_array_equal(E.retrieve("raw_nearest64",x,query,cross,k=2),[1,2])
    np.testing.assert_array_equal(E.retrieve("spectral_nearest64",x,query,cross,k=2),[2,3])
    np.testing.assert_array_equal(E.retrieve("population64",x,query,cross,k=2),[1,2])
    np.testing.assert_array_equal(E.retrieve("recent64",x,query,cross,k=2),[2,3])
    np.testing.assert_array_equal(E.retrieve("full128",x,query,cross,k=2),np.arange(4))


def test_retrieval_is_subset_and_query_dependent():
    rng=np.random.default_rng(4);x=rng.uniform(-1,1,(128,3))
    cross=[np.ones((1,128))]*2
    a=E.retrieve("raw_nearest64",x,np.array([-1.,-1.,-1.]),cross)
    b=E.retrieve("raw_nearest64",x,np.array([1.,1.,1.]),cross)
    assert len(a)==64 and len(np.unique(a))==64
    assert np.all(np.diff(a)>0) and a.min()>=0 and a.max()<128
    assert not np.array_equal(a,b)


def test_localized_generator_changes_only_designated_population():
    original=E.stream(5,"stationary");local=E.stream(5,"localized")
    np.testing.assert_array_equal(original["x"],local["x"])
    changed=(local["x"][:,2]>0)&(local["t"]>=384)
    np.testing.assert_allclose(original["y"][~changed],local["y"][~changed])
    assert not np.allclose(original["y"][changed],local["y"][changed])
