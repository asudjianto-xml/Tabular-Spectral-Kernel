import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from run_context_retrieval import fused_neighbors


def test_context_mixture_changes_neighbors_and_matches_feature_distance():
    # Two valid unit-diagonal PSD geometries with opposing neighborhoods.
    angles=np.array([0.,.1,1.,1.5])
    features=[np.column_stack([np.cos(a),np.sin(a)]) for a in [angles,angles[::-1]]]
    query=np.array([1.,0.])
    grams=[z@z.T for z in features];cross=[(z@query)[None] for z in features]
    np.testing.assert_array_equal(fused_neighbors(grams,cross,[1.,0.],2),[0,1])
    np.testing.assert_array_equal(fused_neighbors(grams,cross,[0.,1.],2),[2,3])
    alpha=np.array([.2,.8])
    z=np.concatenate([np.sqrt(w)*f for w,f in zip(alpha,features)],axis=1)
    q=np.concatenate([np.sqrt(w)*query for w in alpha])
    expected=np.sort(np.argsort(((z-q)**2).sum(1))[:2])
    np.testing.assert_array_equal(fused_neighbors(grams,cross,alpha,2),expected)
    assert np.linalg.eigvalsh(sum(w*g for w,g in zip(alpha,grams))).min()>-1e-12


def test_retrieval_invariant_to_joint_context_permutation_without_ties():
    rng=np.random.default_rng(7)
    features=[rng.normal(size=(20,4)) for _ in range(2)]
    features=[z/np.linalg.norm(z,axis=1,keepdims=True) for z in features]
    grams=[z@z.T for z in features];cross=[z[:,0][None] for z in features]
    alpha=[.8,.2];perm=rng.permutation(20)
    ix=fused_neighbors(grams,cross,alpha,6)
    shuffled=fused_neighbors([g[np.ix_(perm,perm)] for g in grams],[q[:,perm] for q in cross],alpha,6)
    np.testing.assert_array_equal(ix,np.sort(perm[shuffled]))
