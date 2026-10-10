"""Illustrative credit-card repayment-ratio drift; not calibrated lending data.

Observable model features: utilization, credit score, deposit-derived buffer
months, prior payment ratio. No debt-burden variable. Target is payment by due
date / statement balance, capped at one (a statement-repayment fraction).
Each monthly cohort has 32 independent accounts; outcomes become available
before the next cohort. Previous four cohorts form the 128-row context.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
import run as R
import run_meta as M
import run_retrieval as E

SCENARIOS=["recurring_downturn","gradual_downturn","stable_economy","localized_downturn"]
METHODS=E.METHODS


def covariates(rng,n):
    utilization=.02+.96*rng.beta(2,2,n)
    score=450+300*rng.beta(2,3,n)
    buffer=4*rng.beta(1.5,2,n)
    previous=rng.beta(2,1.5,n)
    raw=np.column_stack([utilization,score,buffer,previous])
    x=2*(raw-np.array([0,450,0,0]))/np.array([1,300,4,1])-1
    return x,raw


def means(x):
    u,s,b,p=(x+1).T/2
    normal=expit(.7+1.3*s+1.8*p+.7*b-1.4*u)
    # Illustrative worsening of repayment, with stronger effects at high
    # utilization and low score. No claim of an estimated causal policy effect.
    downturn=expit(.7+1.3*s+1.8*p+.7*b-1.4*u-1.2-1.0*u-.8*(1-s))
    return normal,downturn


def responses(rng,mean):
    return rng.beta(60*mean,60*(1-mean))


def generate(seed,scenario,n=1920):
    rng=np.random.default_rng(51000+seed);x,raw=covariates(rng,n);a,b=means(x)
    t=np.arange(n);seg=t//384
    if scenario=="recurring_downturn":stress=(seg%2).astype(float)
    elif scenario=="gradual_downturn":
        stress=np.interp(t,[0,383,767,1151,1535,1919],[0,0,1,1,0,0])
    elif scenario=="stable_economy":stress=np.zeros(n)
    elif scenario=="localized_downturn":stress=((t>=384)&(raw[:,2]<=2)).astype(float)
    else:raise ValueError(scenario)
    mean=(1-stress)*a+stress*b;y=responses(rng,mean)
    limit=np.clip(rng.lognormal(np.log(5000),.55,n),500,20000)
    balance=raw[:,0]*limit
    return dict(x=x,raw=raw,y=y,mean=mean,stress=stress,t=t,month=t//32,
                credit_limit=limit,statement_balance=balance,payment_amount=y*balance)


def export_csv(path,d):
    with path.open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["account_month_id","month","utilization","credit_score","cash_buffer_months",
            "prior_payment_ratio","credit_limit","statement_balance","payment_by_due_date",
            "payment_balance_ratio","true_conditional_mean","latent_stress"])
        for i in range(len(d["y"])):
            writer.writerow([i,int(d["month"][i]),*d["raw"][i],d["credit_limit"][i],
                d["statement_balance"][i],d["payment_amount"][i],d["y"][i],d["mean"][i],d["stress"][i]])


def train_gate(gate,models,temps,ridge,seed,steps,episode_fn=None):
    rng=np.random.default_rng(61000+seed);opt=torch.optim.Adam(gate.parameters(),lr=.01);trace=[]
    for step in range(steps):
        if episode_fn is None:
            x,_=covariates(rng,129);a,b=means(x)
            severity=rng.uniform()
            stress=np.full(129,severity)
            if rng.uniform()<.5:stress*=((x[:,2]+1)/2<=rng.uniform(.25,.75))
            y=responses(rng,(1-stress)*a+stress*b)
        else:x,y=episode_fn(rng)
        zs=[R.embedding(m,x) for m in models]
        grams=[R.kernel(z[:128],z[:128],t) for z,t in zip(zs,temps)]
        cross=[R.kernel(z[128:],z[:128],t) for z,t in zip(zs,temps)]
        # Train with both global and query-retrieved contexts, avoiding an
        # exclusively 128-row meta-training distribution.
        method=METHODS[step%len(METHODS)]
        ix=E.retrieve(method,x[:128],x[128],cross)
        if len(ix)<2:ix=np.arange(64,128)
        ks=[k[np.ix_(ix,ix)] for k in grams];qs=[q[:,ix] for q in cross];ys=torch.tensor(y[ix])
        alpha=gate(torch.tensor(M.descriptor(ks,y[ix])))
        k=sum(w*torch.tensor(v) for w,v in zip(alpha,ks));q=sum(w*torch.tensor(v) for w,v in zip(alpha,qs))
        nw=q@ys/q.sum(1)
        factor=torch.linalg.cholesky(k+len(ix)*ridge*torch.eye(len(ix),dtype=torch.float64))
        sol=torch.cholesky_solve(torch.stack([ys,torch.ones_like(ys)],1),factor)
        intercept=sol[:,0].sum()/sol[:,1].sum();krr=q@(sol[:,0]-intercept*sol[:,1])+intercept
        loss=.5*((nw-y[128]).square().mean()+(krr-y[128]).square().mean())
        opt.zero_grad();loss.backward();opt.step()
        if step%100==0 or step==steps-1:trace.append(dict(step=step,query_loss=float(loss.detach())))
    return trace


def execute(args):
    provider=None
    if getattr(args,"unemployment",False):
        import unemployment_data as provider
    out=args.output.resolve();out.mkdir(exist_ok=False);torch.set_num_threads(1)
    R.save(out/"protocol.json",dict(seeds=args.seeds,geometry_steps=args.steps,meta_steps=args.meta_steps,
        features=["utilization","credit_score","cash_buffer_months","prior_payment_ratio"],
        response="fraction of statement balance paid by due date, capped at 1",
        observation="buffer assumes deposit/linked-account access; score is not measured debt burden",
        cohort_size=32,context_months=4,context_rows=128,retrieval_rows=64,noise="Beta with concentration 60",
        historical_windows="600 normal + 600 broad-downturn observations; all before evaluation",
        meta_tasks="independent global and buffer-localized severity tasks; query-based retrieval during pretraining",
        methods=METHODS,scenarios=SCENARIOS,profiles=["original","nw_calibrated"],
        information="Only four pre-outcome covariates and past observed ratios enter inference",
        limitations=["Illustrative response laws, not estimated from lending data",
            "Independent account-months, no longitudinal customer dependence",
            "All outcomes observed with one-cohort availability; no approval selection/censoring",
            "Population reference is supplied the cash-buffer partition",
            "No exact inactive-population preservation is imposed"] ))
    for file in [Path(__file__),Path(R.__file__),Path(M.__file__),Path(E.__file__),R.MODEL_SOURCE]:
        (out/(file.stem+"_snapshot.py")).write_bytes(file.read_bytes())
    if provider is not None:
        (out/"unemployment_data_snapshot.py").write_bytes(Path(provider.__file__).read_bytes())
        protocol=json.loads((out/"protocol.json").read_text())
        protocol.update(features=protocol["features"]+["available_unemployment_rate"],
            information="Five pre-outcome inputs; macro exposure and stress memory are audit-only latent fields",
            driver="shock=clip((U-.04)/.06,0,1); memory=.75*previous+.25*shock; exposure=.6*shock+.4*memory",
            localized_susceptibility="clip(1 - cash_buffer_months/2, 0, 1)",
            historical_windows="40 monthly cohorts of 30 observations; low then high unemployment; 8 normal macro-only gap months before evaluation",
            meta_tasks="independent unemployment histories, broad/local susceptibility and all retrieval rules",
            publication_timing="U is defined as already available at prediction; no future release or future payment labels are used",
            qualification="Synthetic rates and sensitivities; conditional drift relative to current features arises from omitted unemployment-history state")
        R.save(out/"protocol.json",protocol)
    metrics=[];started=time.perf_counter()
    for seed in args.seeds:
        folder=out/f"seed{seed}";folder.mkdir();rng=np.random.default_rng(41000+seed)
        if provider is None:
            hx,hraw=covariates(rng,1200);a,b=means(hx);hy=responses(rng,np.r_[a[:600],b[600:]])
            history=dict(x=hx,raw=hraw,y=hy,window=np.repeat([0,1],600))
        else:
            history=provider.historical(seed);hx=history["x"];hy=history["y"]
        np.savez_compressed(folder/"history.npz",**history)
        banks=[];infos=[]
        for j in range(2):
            m,info=R.train_geometry(hx,hy,np.arange(j*600,j*600+480),np.arange(j*600+480,j*600+600),
                71000+seed*10+j,args.steps,folder/["normal","downturn"][j])
            banks.append(m);infos.append(info)
        for profile in ["original","nw_calibrated"]:
            temps=[v["temperature"] for v in infos];ridges=[v["ridge"] for v in infos];calibration=[]
            if profile=="nw_calibrated":
                for j,m in enumerate(banks):
                    z=R.embedding(m,hx[j*600:(j+1)*600]);ys=hy[j*600:(j+1)*600];trials=[]
                    for scale in [.02,.05,.1,.2,.5,1.]:
                        q=R.kernel(z[480:],z[:480],temps[j]*scale)
                        pred=M.nw_predict([q],ys[:480],[1.])
                        trials.append(dict(scale=scale,mse=float(np.mean((pred-ys[480:])**2))))
                    chosen=min(trials,key=lambda d:d["mse"]);temps[j]*=chosen["scale"]
                    # Unlike the earlier sensitivity pilot, recalibrate ridge
                    # on historical labels after changing the bandwidth.
                    k=R.kernel(z[:480],z[:480],temps[j]);q=R.kernel(z[480:],z[:480],temps[j])
                    errors=[float(np.mean((R.ridge_predict(k,q,ys[:480],r)-ys[480:])**2)) for r in R.RIDGES]
                    ridges[j]=float(R.RIDGES[np.argmin(errors)])
                    calibration.append(dict(bank=j,trials=trials,selected=chosen,ridge=ridges[j],ridge_errors=errors))
            ridge=float(np.sqrt(np.prod(ridges)));torch.manual_seed(81000+seed);gate=M.ContextGate()
            trace=train_gate(gate,banks,temps,ridge,seed,args.meta_steps,
                episode_fn=None if provider is None else provider.meta_episode);gate.eval()
            torch.save(gate.state_dict(),folder/f"{profile}_gate.pt")
            R.save(folder/f"{profile}_training.json",dict(temperatures=temps,ridge=ridge,trace=trace,calibration=calibration))
            for scenario in SCENARIOS:
                d=(generate if provider is None else provider.generate)(seed,scenario);x=d["x"];y=d["y"];zs=[R.embedding(m,x) for m in banks]
                predictions={method+"_"+dec:np.full(len(y),np.nan) for method in METHODS for dec in ["nw","krr"]}
                weights={method:np.full((len(y),2),np.nan) for method in METHODS}
                indices={method:np.full((len(y),128),-1,dtype=np.int32) for method in METHODS}
                clipping={key:0 for key in predictions}
                for origin in range(128,len(y),32):
                    sl=slice(origin-128,origin);end=origin+32
                    grams=[R.kernel(z[sl],z[sl],t) for z,t in zip(zs,temps)]
                    cross=[R.kernel(z[origin:end],z[sl],t) for z,t in zip(zs,temps)]
                    for method in METHODS:
                        queries=[(origin,end)] if method in ["full128","recent64"] else [(i,i+1) for i in range(origin,end)]
                        for left,right in queries:
                            q=[g[left-origin:right-origin] for g in cross]
                            ix=E.retrieve(method,x[sl],x[left],[v[:1] for v in q])
                            if len(ix)<2:ix=np.arange(64,128)
                            alpha,nw,krr=M.forward(gate,[k[np.ix_(ix,ix)] for k in grams],[g[:,ix] for g in q],y[sl][ix],ridge)
                            for dec,pred in [("nw",nw),("krr",krr)]:
                                key=method+"_"+dec;clipping[key]+=int(np.sum((pred<0)|(pred>1)))
                                predictions[key][left:right]=np.clip(pred,0,1)
                            weights[method][left:right]=alpha;indices[method][left:right,:len(ix)]=origin-128+ix
                np.savez_compressed(folder/f"{profile}_{scenario}.npz",**d,**predictions,
                    **{key+"_alpha":v for key,v in weights.items()},**{key+"_indices":v for key,v in indices.items()})
                if profile=="original":(export_csv if provider is None else provider.export_csv)(folder/f"{scenario}.csv",d)
                valid=d["t"]>=128
                for key,pred in predictions.items():
                    method,dec=key.rsplit("_",1)
                    row=dict(seed=seed,profile=profile,scenario=scenario,method=method,decoder=dec,
                        rmse=float(np.sqrt(np.mean((pred[valid]-y[valid])**2))),
                        mean_rmse=float(np.sqrt(np.mean((pred[valid]-d["mean"][valid])**2))),
                        clipped_fraction=clipping[key]/int(valid.sum()))
                    if scenario=="localized_downturn":
                        active=(d["t"]>=384) if provider is None else (valid&(d["macro_exposure"]>.1))
                        for group,mask in [("stable",valid&(d["raw"][:,2]>2)),("changed",active&(d["raw"][:,2]<=2))]:
                            row[group+"_rmse"]=float(np.sqrt(np.mean((pred[mask]-y[mask])**2)))
                    metrics.append(row)
                print(seed,profile,scenario,{key:round(float(np.sqrt(np.mean((p[valid]-y[valid])**2))),3) for key,p in predictions.items()},flush=True)
    R.save(out/"metrics.json",metrics);summary=[]
    for profile in ["original","nw_calibrated"]:
        for scenario in SCENARIOS:
            for method in METHODS:
                for dec in ["nw","krr"]:
                    rows=[r for r in metrics if (r["profile"],r["scenario"],r["method"],r["decoder"])==(profile,scenario,method,dec)]
                    vals=[r["rmse"] for r in rows]
                    summary.append(dict(profile=profile,scenario=scenario,method=method,decoder=dec,
                        rmse_mean=float(np.mean(vals)),rmse_seed_sd=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.,
                        **({g+"_rmse":float(np.mean([r[g+"_rmse"] for r in rows])) for g in ["stable","changed"]} if scenario=="localized_downturn" else {})))
    R.save(out/"summary.json",summary)
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    for ax,(profile,dec) in zip(axs,[("original","krr"),("nw_calibrated","nw")]):
        for i,method in enumerate(METHODS):
            vals=[next(r["rmse_mean"] for r in summary if (r["profile"],r["scenario"],r["method"],r["decoder"])==(profile,sc,method,dec)) for sc in SCENARIOS]
            ax.bar(np.arange(4)+(i-2)*.15,100*np.array(vals),width=.15,label=method)
        ax.set_xticks(np.arange(4),["Recurring","Gradual","Stable","Localized"])
        ax.set_ylabel("Repayment-ratio RMSE (percentage points)");ax.set_title(profile+" / "+dec.upper());ax.grid(axis="y",alpha=.2)
    axs[0].legend(fontsize=8);fig.savefig(out/"credit_card_results.png",dpi=170);fig.savefig(out/"credit_card_results.pdf");plt.close(fig)
    R.save(out/"runtime.json",dict(seconds=time.perf_counter()-started))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"credit_card_results")
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--steps",type=int,default=400);parser.add_argument("--meta-steps",type=int,default=600)
    parser.add_argument("--unemployment",action="store_true",help="use observable unemployment and latent lagged macro stress; choose a new output directory")
    with threadpool_limits(limits=1):execute(parser.parse_args())
