import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
import run_credit_card as C


def test_repayment_data_are_bounded_and_accounting_consistent():
    d=C.generate(2,"localized_downturn")
    assert d["x"].shape==(1920,4)
    assert np.all((d["y"]>0)&(d["y"]<1))
    assert np.all((d["raw"][:,1]>=450)&(d["raw"][:,1]<=750))
    assert np.all(d["statement_balance"]>0)
    np.testing.assert_allclose(d["payment_amount"]/d["statement_balance"],d["y"])
    np.testing.assert_allclose(d["statement_balance"],d["raw"][:,0]*d["credit_limit"])


def test_localized_credit_change_preserves_other_conditional_means():
    stable=C.generate(0,"stable_economy");local=C.generate(0,"localized_downturn")
    np.testing.assert_array_equal(stable["x"],local["x"])
    affected=(local["raw"][:,2]<=2)&(local["t"]>=384)
    np.testing.assert_array_equal(stable["mean"][~affected],local["mean"][~affected])
    assert np.all(local["mean"][affected]<stable["mean"][affected])


def test_existing_spectral_trainer_supports_four_credit_features(tmp_path):
    rng=np.random.default_rng(3);x,_=C.covariates(rng,240);mu,_=C.means(x);y=C.responses(rng,mu)
    model,info=C.R.train_geometry(x,y,np.arange(192),np.arange(192,240),3,1,tmp_path/"bank")
    assert model.d==4 and model.frequencies().shape==(4,4)
    assert np.isfinite(info["validation_mse"])
