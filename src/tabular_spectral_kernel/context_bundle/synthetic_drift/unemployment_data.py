"""Unemployment-driven synthetic repayment; all coefficients are illustrative.

The fifth predictor is the unemployment rate already available at prediction
time. A lagged macro-stress state affects repayment but is not a predictor.
No evaluation response is switched by row index or a calendar breakpoint.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import csv
import numpy as np
from scipy.special import expit
import run_credit_card as C


def exposure(rate,initial=0.):
    """Causal monthly recursion. Rates are proportions, not percentage points."""
    shock=np.clip((np.asarray(rate)-.04)/.06,0.,1.)
    memory=np.empty(len(shock));state=float(initial)
    for i,s in enumerate(shock):
        state=.75*state+.25*s;memory[i]=state
    return .6*shock+.4*memory,memory


def unemployment_path(scenario,months=60):
    t=np.arange(months)
    if scenario=="recurring_downturn":
        return np.interp(t,[0,8,16,22,32,38,46,50,59],[.04,.04,.10,.10,.04,.04,.095,.095,.04])
    if scenario in ["gradual_downturn","localized_downturn"]:
        return np.interp(t,[0,8,24,32,59],[.04,.04,.10,.10,.04])
    if scenario=="stable_economy":return np.full(months,.04)
    raise ValueError(scenario)


def feature_matrix(borrower_x,rate):
    # Fixed declared physical scale; no future-data-fitted normalization.
    return np.column_stack([borrower_x,2*(np.asarray(rate)-.03)/.09-1])


def conditional_mean(borrower_x,macro_exposure,localized=False):
    u,s,b,p=(borrower_x+1).T/2
    susceptibility=np.clip(1-2*b,0,1) if localized else np.ones(len(u))
    stress=np.asarray(macro_exposure)*susceptibility
    z=.7+1.3*s+1.8*p+.7*b-1.4*u
    mean=expit(z-stress*(1.2+u+.8*(1-s)))
    return mean,stress


def historical(seed):
    rng=np.random.default_rng(41000+seed);bx,raw=C.covariates(rng,1200)
    # Forty completed historical cohorts of thirty observations, then evaluation.
    rates=np.r_[np.linspace(.04,.06,20),np.linspace(.10,.07,20)]
    macro,state=exposure(rates)
    rate=np.repeat(rates,30);expanded=np.repeat(macro,30)
    mean,stress=conditional_mean(bx,expanded)
    return dict(x=feature_matrix(bx,rate),raw=raw,y=C.responses(rng,mean),mean=mean,
        unemployment_rate=rate,macro_exposure=expanded,latent_stress=stress,
        stress_memory=np.repeat(state,30),month=np.repeat(np.arange(-48,-8),30),
        window=np.repeat([0,1],600))


def meta_episode(rng):
    bx,_=C.covariates(rng,129)
    # Eight burn-in macro months plus five support/query months. The query
    # covariate rate is known, while its payment label enters training loss only.
    rates=np.clip(rng.uniform(.035,.105)+np.cumsum(rng.normal(0,.006,13)),.03,.12)
    macro,_=exposure(rates)
    rate=np.r_[np.repeat(rates[-5:-1],32),rates[-1]]
    expanded=np.r_[np.repeat(macro[-5:-1],32),macro[-1]]
    mean,_=conditional_mean(bx,expanded,localized=bool(rng.integers(2)))
    return feature_matrix(bx,rate),C.responses(rng,mean)


def generate(seed,scenario,n=1920):
    rng=np.random.default_rng(51000+seed);bx,raw=C.covariates(rng,n)
    months=(n+31)//32;rates=unemployment_path(scenario,months)
    # Eight normal macro months separate the historical library and evaluation
    # to define a common low-stress initial state, rather than resetting state.
    historical_rates=np.r_[np.linspace(.04,.06,20),np.linspace(.10,.07,20),np.full(8,.04)]
    all_exposure,all_state=exposure(np.r_[historical_rates,rates])
    macro=all_exposure[-months:];state=all_state[-months:]
    rate=np.repeat(rates,32)[:n];expanded=np.repeat(macro,32)[:n]
    mean,stress=conditional_mean(bx,expanded,localized=scenario=="localized_downturn")
    # Separate RNG keeps monetary-accounting covariates identical across scenarios.
    monetary_rng=np.random.default_rng(52000+seed)
    limit=np.clip(monetary_rng.lognormal(np.log(5000),.55,n),500,20000)
    balance=raw[:,0]*limit;y=C.responses(rng,mean)
    return dict(x=feature_matrix(bx,rate),raw=raw,y=y,mean=mean,stress=stress,
        t=np.arange(n),month=np.arange(n)//32,unemployment_rate=rate,
        macro_exposure=expanded,stress_memory=np.repeat(state,32)[:n],
        credit_limit=limit,statement_balance=balance,payment_amount=y*balance)


def export_csv(path,d):
    with path.open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["account_month_id","month","utilization","credit_score","cash_buffer_months",
            "prior_payment_ratio","unemployment_rate","unemployment_rate_pct","credit_limit",
            "statement_balance","payment_by_due_date","payment_balance_ratio","true_conditional_mean",
            "latent_macro_exposure","latent_stress_memory","latent_borrower_stress"])
        for i in range(len(d["y"])):
            writer.writerow([i,int(d["month"][i]),*d["raw"][i],d["unemployment_rate"][i],
                100*d["unemployment_rate"][i],d["credit_limit"][i],d["statement_balance"][i],
                d["payment_amount"][i],d["y"][i],d["mean"][i],d["macro_exposure"][i],
                d["stress_memory"][i],d["stress"][i]])
