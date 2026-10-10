import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import json
from pathlib import Path
import numpy as np
import torch
import synthetic_controls as S
ROOT=Path(__file__).resolve().parents[1]

def test_history_augmented_generator_preserves_original_samples():
    for sc in S.S.SCENARIOS:
        x,y,mu=S.raw_episode(np.random.default_rng(123),sc)
        xx,yy,mm=S.S.episode(np.random.default_rng(123),sc)
        np.testing.assert_array_equal(x[:,:5],xx);np.testing.assert_array_equal(y,yy);np.testing.assert_array_equal(mu,mm)


def test_unemployment_memory_is_causal():
    rates=np.linspace(.03,.12,33);changed=rates.copy();changed[20:]=.04
    _,a=S.S.U.exposure(rates);_,b=S.S.U.exposure(changed)
    np.testing.assert_array_equal(a[:20],b[:20])
    assert np.any(a[20:]!=b[20:])


def test_augmented_krr_uses_no_query_labels_and_matches_direct_solver():
    raw=S.raw_data(321,2,'recurring');p=S.rbf_path(raw,6,2.,[1e-4])[1e-4]
    x,y,mu=raw;altered=y.copy();altered[:,768:]=1-y[:,768:]
    pp=S.rbf_path((x,altered,mu),6,2.,[1e-4])[1e-4];np.testing.assert_array_equal(p,pp)
    a=x[0,640:768];b=x[0,768:]
    k=np.exp(-((a[:,None]-a[None])**2).sum(-1)/8);q=np.exp(-((b[:,None]-a[None])**2).sum(-1)/8)
    expected=S.R.ridge_predict(k,q,y[0,640:768],1e-4).clip(0,1)
    np.testing.assert_allclose(p[0],expected,atol=1e-10)


def test_oracle_ordering_on_validation_and_test():
    for folder in (ROOT/'review_experiments/bike_results').glob('seed*'):
        for part in ['original_validation','original_test','revised_validation','revised_test']:
            d=json.loads((folder/f'{part}_oracles.json').read_text())
            for kind in ['fixed','gated']:
                assert d[kind+'_query_oracle']['rmse']<=d[kind+'_day_oracle']['rmse']<=d[kind+'_recent']['rmse']


def test_saved_synthetic_oracles_are_correct_candidate_minima():
    for seed in range(3):
        for sc in S.S.SCENARIOS:
            old=np.load(ROOT/f'synthetic_drift/spectral_selector_results/seed{seed}/{sc}.npz')
            new=np.load(ROOT/f'review_experiments/synthetic_results/seed{seed}/{sc}.npz')
            p=old['candidate_predictions'];y=old['target'];mu=new['mean']
            np.testing.assert_array_equal(y,new['target'])
            for key,target in [('label_context_oracle',y),('mean_context_oracle',mu)]:
                ix=((p-target[...,None])**2).argmin(-1)
                np.testing.assert_allclose(new[key],np.take_along_axis(p,ix[...,None],-1)[...,0],atol=1e-12)
