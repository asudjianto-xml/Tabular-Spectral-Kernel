"""Amortized context fusion: frozen meta-encoder, paired NW/KRR readouts.

Meta-training uses independent stationary A/B mixture tasks, never evaluation
stream labels. A small encoder reads permutation-invariant leave-one-out
context-error descriptors and emits simplex weights. Its training objective
averages NW and KRR query MSE. At inference, neither weights nor ridge are
searched. KRR still needs a linear solve; NW needs only kernel aggregation.
Both use exactly the same kernels, weights, contexts, and query covariates.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run as R


class ContextGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(3,16),torch.nn.Tanh(),torch.nn.Linear(16,2)).double()
    def forward(self,descriptor):return self.net(descriptor).softmax(-1)


def descriptor(grams,y):
    """Row-order invariant; no query inputs or labels are accepted."""
    variance=max(float(np.var(y)),1e-8)
    losses=[]
    for k in grams:
        loo=k.copy();np.fill_diagonal(loo,0.)
        pred=loo@y/np.maximum(loo.sum(1),1e-300)
        losses.append(float(np.mean((pred-y)**2)))
    return np.log(np.maximum([losses[0]/variance,losses[1]/variance,variance],1e-8))


def nw_predict(grams,y,alpha):
    k=sum(w*g for w,g in zip(alpha,grams))
    return k@y/np.maximum(k.sum(1),1e-300)


def forward(gate,grams,cross,y,ridge):
    with torch.no_grad():
        weights=gate(torch.tensor(descriptor(grams,y),dtype=torch.float64)).numpy()
    nw=nw_predict(cross,y,weights)
    k=sum(w*g for w,g in zip(weights,grams));q=sum(w*g for w,g in zip(weights,cross))
    return weights,nw,R.ridge_predict(k,q,y,ridge)


def train(gate,models,temps,ridge,seed,steps):
    rng=np.random.default_rng(30000+seed)
    optimizer=torch.optim.Adam(gate.parameters(),lr=.01)
    trace=[]
    for step in range(steps):
        # Fresh support and query examples per episode. True mixture weight
        # generates labels but is not an encoder input or supervised gate target.
        weight=float(rng.uniform()) if rng.uniform()>.3 else float(rng.integers(2))
        x=rng.uniform(-1,1,(160,3));a,b=R.mechanisms(x)
        y=weight*a+(1-weight)*b+rng.normal(0,.15,len(x))
        zs=[R.embedding(m,x) for m in models]
        grams=[R.kernel(z[:128],z[:128],t) for z,t in zip(zs,temps)]
        cross=[R.kernel(z[128:],z[:128],t) for z,t in zip(zs,temps)]
        alpha=gate(torch.tensor(descriptor(grams,y[:128]),dtype=torch.float64))
        k=sum(w*torch.as_tensor(g) for w,g in zip(alpha,grams))
        q=sum(w*torch.as_tensor(g) for w,g in zip(alpha,cross))
        ys=torch.as_tensor(y[:128]);target=torch.as_tensor(y[128:])
        nw=q@ys/q.sum(1)
        chol=torch.linalg.cholesky(k+128*ridge*torch.eye(128,dtype=torch.float64))
        sol=torch.cholesky_solve(torch.stack([ys,torch.ones_like(ys)],1),chol)
        intercept=sol[:,0].sum()/sol[:,1].sum();coef=sol[:,0]-intercept*sol[:,1]
        krr=q@coef+intercept
        loss=.5*((nw-target).square().mean()+(krr-target).square().mean())
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%50==0 or step==steps-1:
            trace.append(dict(step=step,training_query_mse=float(loss.detach())))
    return trace


def execute(args):
    source=args.source.resolve();out=args.output.resolve();out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    protocol=json.loads((source/"protocol.json").read_text());seeds=protocol["seeds"]
    R.save(out/"protocol.json",dict(seeds=seeds,steps=args.steps,meta_context=128,meta_query=32,temperature_mode=args.temperature_mode,
        meta_seed_rule="30000 + seed, independent of history and evaluation",
        task_prior="stationary convex A/B response mixtures; 30% pure endpoint episodes",
        encoder="3 invariant LOO error/variance descriptors -> 16 tanh -> 2 softmax",
        objective="equal-weight NW and KRR query MSE; frozen spectral kernels",
        inference="frozen encoder; no search; fixed historical ridge; identical kernels and alpha for both decoders",
        qualification="Known two-mechanism task family; geometry was originally trained for KRR",
        historical_nw_scales=[.02,.05,.1,.2,.5,1.]))
    (out/"run_meta_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (out/"run_dependency_snapshot.py").write_bytes(Path(R.__file__).read_bytes())
    rows=[]
    for seed in seeds:
        models=[];temps=[];ridges=[];calibrations=[];folder=source/f"seed{seed}"
        history=np.load(folder/"historical_data.npz")
        for bank,name in enumerate(["bank_a","bank_b"]):
            m=R.base.CanonicalSpectral(3,4,max_frequency=3.)
            m.load_state_dict(torch.load(folder/f"{name}.pt",weights_only=True,map_location="cpu"));m.eval()
            models.append(m);temps.append(float(m.log_temperature.detach().exp()))
            ridges.append(json.loads((folder/f"{name}.json").read_text())["ridge"])
            if args.temperature_mode=="historical_nw":
                offset=600*bank;z=R.embedding(m,history["x"][offset:offset+600]);hy=history["y"][offset:offset+600]
                trials=[]
                for scale in [.02,.05,.1,.2,.5,1.]:
                    q=R.kernel(z[480:],z[:480],temps[-1]*scale)
                    pred=nw_predict([q],hy[:480],[1.])
                    trials.append(dict(scale=scale,val_mse=float(np.mean((pred-hy[480:])**2))))
                choice=min(trials,key=lambda d:d["val_mse"]);temps[-1]*=choice["scale"]
                calibrations.append(dict(bank=name,temperature=temps[-1],selected=choice,trials=trials))
        ridge=float(np.sqrt(np.prod(ridges)))  # Fixed from historical validation.
        torch.manual_seed(40000+seed);gate=ContextGate()
        trace=train(gate,models,temps,ridge,seed,args.steps);gate.eval()
        torch.save(gate.state_dict(),out/f"seed{seed}_gate.pt")
        R.save(out/f"seed{seed}_training.json",dict(trace=trace,ridge=ridge,temperatures=temps,calibrations=calibrations))
        for scenario in R.SCENARIOS:
            data=R.generate(seed,scenario);x=data["x"];y=data["y"]
            zs=[R.embedding(m,x) for m in models]
            predictions={name:np.full(len(y),np.nan) for name in ["meta_nw","meta_krr","uniform_nw"]}
            weights=[]
            for origin in range(128,len(y),32):
                sl=slice(origin-128,origin);query=slice(origin,origin+32)
                grams=[R.kernel(z[sl],z[sl],t) for z,t in zip(zs,temps)]
                cross=[R.kernel(z[query],z[sl],t) for z,t in zip(zs,temps)]
                alpha,nw,krr=forward(gate,grams,cross,y[sl],ridge)
                predictions["meta_nw"][query]=nw;predictions["meta_krr"][query]=krr
                predictions["uniform_nw"][query]=nw_predict(cross,y[sl],[.5,.5])
                weights.append(dict(origin=origin,alpha=alpha.tolist()))
            np.savez_compressed(out/f"seed{seed}_{scenario}.npz",**data,**predictions)
            R.save(out/f"seed{seed}_{scenario}_weights.json",weights)
            for name,pred in predictions.items():
                rows.append(dict(seed=seed,scenario=scenario,method=name,
                    rmse=float(np.sqrt(np.mean((pred[128:]-y[128:])**2)))))
        print("meta seed",seed,[r for r in rows if r["seed"]==seed],flush=True)
    R.save(out/"metrics.json",rows)
    summary=[]
    for scenario in R.SCENARIOS:
        for name in predictions:
            values=[r["rmse"] for r in rows if r["scenario"]==scenario and r["method"]==name]
            summary.append(dict(scenario=scenario,method=name,rmse_mean=float(np.mean(values)),
                rmse_seed_sd=float(np.std(values,ddof=1)) if len(values)>1 else 0.))
    R.save(out/"summary.json",summary);print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path(__file__).parent/"results")
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"meta_results")
    parser.add_argument("--steps",type=int,default=400)
    parser.add_argument("--temperature-mode",choices=["original","historical_nw"],default="original")
    with threadpool_limits(limits=1):execute(parser.parse_args())
