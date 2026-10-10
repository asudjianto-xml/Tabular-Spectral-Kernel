import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
import unemployment_data as U
import run_credit_card as C


def test_macro_exposure_is_causal_and_has_recovery_memory():
    prefix=np.array([.04,.08,.10,.06])
    exp1,state=U.exposure(np.r_[prefix,.04,.04])
    exp2,_=U.exposure(np.r_[prefix,.12,.12])
    np.testing.assert_array_equal(exp1[:4],exp2[:4])
    assert exp1[4]>0 and exp1[5]<exp1[4]
    assert np.all((state>=0)&(state<=1))
    rising,_=U.exposure([.04,.04,.06])
    falling,_=U.exposure([.10,.10,.06])
    assert falling[-1]>rising[-1]  # Same current U, different macro history.


def test_higher_unemployment_reduces_repayment_at_fixed_borrowers():
    x,_=C.covariates(np.random.default_rng(1),100)
    low,_=U.conditional_mean(x,np.zeros(100))
    high,_=U.conditional_mean(x,np.ones(100))
    assert np.all(high<low)
    local,_=U.conditional_mean(x,np.ones(100),localized=True)
    stable=x[:,2]>=0
    np.testing.assert_array_equal(local[stable],low[stable])
    assert np.all(local[~stable]<low[~stable])


def test_macro_is_observed_but_memory_is_not_a_feature():
    d=U.generate(0,"localized_downturn")
    assert d["x"].shape==(1920,5) and d["raw"].shape==(1920,4)
    np.testing.assert_allclose((d["x"][:,-1]+1)*.09/2+.03,d["unemployment_rate"])
    assert np.all((d["y"]>0)&(d["y"]<1))
    np.testing.assert_allclose(d["payment_amount"]/d["statement_balance"],d["y"])
    for month in range(60):assert len(np.unique(d["unemployment_rate"][d["month"]==month]))==1


def test_borrower_covariates_and_balances_match_across_macro_scenarios():
    a=U.generate(2,"stable_economy");b=U.generate(2,"gradual_downturn")
    np.testing.assert_array_equal(a["raw"],b["raw"])
    np.testing.assert_array_equal(a["statement_balance"],b["statement_balance"])
    assert not np.array_equal(a["mean"],b["mean"])
    history=U.historical(2)
    assert history["month"].min()==-48 and history["month"].max()==-9
    assert history["x"].shape==(1200,5)


def test_five_feature_geometry_and_meta_episode(tmp_path):
    h=U.historical(1)
    model,info=C.R.train_geometry(h["x"],h["y"],np.arange(192),np.arange(192,240),1,1,tmp_path/"bank")
    assert model.d==5 and np.isfinite(info["validation_mse"])
    x,y=U.meta_episode(np.random.default_rng(10))
    assert x.shape==(129,5) and y.shape==(129,)
