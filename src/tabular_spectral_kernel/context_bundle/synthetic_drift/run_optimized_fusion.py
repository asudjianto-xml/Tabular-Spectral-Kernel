"""Inference-time mixture optimization versus amortized fusion, fixed ridge."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import argparse
import hashlib
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits
import run as R
import run_credit_card as C


def objective(grams,y,ridge,weight,method,support=96):
    k=weight*grams[0]+(1-weight)*grams[1]
    if method=="validation":
        p=R.ridge_predict(k[:support,:support],k[support:,:support],y[:support],ridge)
        return float(np.mean((p-y[support:])**2))
    if method=="convex_mkl":
        coef,intercept=R.ridge_fit(k,y,ridge)
        # Profiled mean squared loss + ridge * RKHS norm, intercept unpenalized.
        return float(ridge*(y-intercept)@coef)
    raise ValueError(method)


def optimize_weight(grams,y,ridge,method,support=96):
    """Accept only labeled context; no query features or labels enter selection."""
    count=0
    def score(w):
        nonlocal count
        count+=1;return objective(grams,y,ridge,w,method,support)
    candidates=[(score(0.),0.),(score(1.),1.)]
    if method=="convex_mkl":
        result=minimize_scalar(score,bounds=(0.,1.),method="bounded",options={"xatol":1e-6})
        candidates.append((float(result.fun),float(result.x)))
    else:
        # Validation risk need not be convex. Include a dense grid and refine
        # each bracketed local minimum; not a formal global-optimality claim.
        grid=np.linspace(0,1,101);values=np.array([score(w) for w in grid])
        candidates.extend(zip(values,grid))
        for i in range(1,len(grid)-1):
            if values[i]<=values[i-1] and values[i]<=values[i+1]:
                r=minimize_scalar(score,bounds=(grid[i-1],grid[i+1]),method="bounded",options={"xatol":1e-6})
                candidates.append((float(r.fun),float(r.x)))
    loss,weight=min(candidates)
    return float(weight),dict(objective=float(loss),objective_evaluations=count)


def execute(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    metrics=[];selection=[];provenance={}
    def record(path):provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return path
    for f in [Path(__file__),Path(R.__file__),R.MODEL_SOURCE]:
        (out/(f.stem+"_snapshot.py")).write_bytes(f.read_bytes())
    for seed in [0,1,2]:
        src=args.source/"unemployment_results"/f"seed{seed}";banks=[]
        for name in ["normal","downturn"]:
            m=R.base.CanonicalSpectral(5,4,max_frequency=3.,device="cpu")
            m.load_state_dict(torch.load(record(src/f"{name}.pt"),weights_only=True,map_location="cpu"));m.eval();banks.append(m)
        info=json.loads(record(src/"original_training.json").read_text());ridge=info["ridge"]
        for scenario in C.SCENARIOS:
            with np.load(record(src/f"original_{scenario}.npz")) as f:data={k:f[k] for k in f.files}
            x,y=data["x"],data["y"];n=len(y);zs=[R.embedding(m,x) for m in banks]
            pred={m:np.full(n,np.nan) for m in ["validation","convex_mkl"]}
            weights={m:np.full(n,np.nan) for m in pred}
            for origin in range(128,n,32):
                sl=slice(origin-128,origin);qs=slice(origin,origin+32)
                grams=[R.kernel(z[sl],z[sl],t) for z,t in zip(zs,info["temperatures"])]
                cross=[R.kernel(z[qs],z[sl],t) for z,t in zip(zs,info["temperatures"])]
                for method in pred:
                    start=time.perf_counter();w,diagnostic=optimize_weight(grams,y[sl],ridge,method)
                    k=w*grams[0]+(1-w)*grams[1];q=w*cross[0]+(1-w)*cross[1]
                    pred[method][qs]=np.clip(R.ridge_predict(k,q,y[sl],ridge),0,1);weights[method][qs]=w
                    selection.append(dict(seed=seed,scenario=scenario,origin=origin,method=method,weight=w,
                        context_start=origin-128,context_end=origin-1,ridge=ridge,
                        selection_and_refit_seconds=time.perf_counter()-start,**diagnostic))
            with np.load(record(args.source/"latest_kernel_comparison"/f"seed{seed}_{scenario}.npz")) as f:
                for method in ["latest","transformer","rope"]:pred[method]=f[method+"_krr"]
            valid=np.arange(n)>=128
            np.savez_compressed(out/f"seed{seed}_{scenario}.npz",**pred,**{m+"_normal_weight":w for m,w in weights.items()})
            for method,p in pred.items():
                assert np.isfinite(p[valid]).all()
                row=dict(seed=seed,scenario=scenario,method=method,rmse=float(np.sqrt(np.mean((p[valid]-y[valid])**2))))
                if scenario=="localized_downturn":
                    for group,mask in [("stable",valid&(data["raw"][:,2]>2)),("changed",valid&(data["macro_exposure"]>.1)&(data["raw"][:,2]<=2))]:
                        row[group+"_rmse"]=float(np.sqrt(np.mean((p[mask]-y[mask])**2)))
                metrics.append(row)
            print(seed,scenario,{m:round(100*np.sqrt(np.mean((p[valid]-y[valid])**2)),3) for m,p in pred.items()},flush=True)
    summary=[]
    for scenario in C.SCENARIOS:
        for method in ["latest","transformer","rope","validation","convex_mkl"]:
            rr=[r for r in metrics if (r["scenario"],r["method"])==(scenario,method)]
            summary.append(dict(scenario=scenario,method=method,rmse_mean=float(np.mean([r["rmse"] for r in rr])),
                rmse_seed_sd=float(np.std([r["rmse"] for r in rr],ddof=1)),
                **({g+"_rmse":float(np.mean([r[g+"_rmse"] for r in rr])) for g in ["stable","changed"]} if scenario=="localized_downturn" else {})))
    for name,obj in [("metrics",metrics),("summary",summary),("selection",selection),("provenance",provenance)]:R.save(out/(name+".json"),obj)
    R.save(out/"protocol.json",dict(seeds=[0,1,2],decoder="KRR",profile="original",context_rows=128,
        ridge="Fixed original historical shared ridge for every method; no joint ridge search",
        validation="96 earlier context rows fit KRR; 32 newest labels select mixture; 101-point grid plus local bounded refinements",
        convex_mkl="All128 labels minimize profiled regularized KRR loss lambda*(y-b)'(K+128*lambda*I)^-1*(y-b), b optimized analytically",
        final_readout="Refit KRR on all128 labeled rows with selected mixture and unchanged ridge",
        information="Only labels available before the origin enter selection; query responses only score saved predictions",
        qualification="Optimized weights shared across each next-month cohort; Transformer weights query-conditioned; validation risk is not convex, profiled MKL objective is convex; timings exclude kernel construction"))
    fig,ax=plt.subplots(figsize=(10,4),layout="constrained")
    for j,method in enumerate(["latest","transformer","rope","validation","convex_mkl"]):
        rr=[r for r in summary if r["method"]==method]
        ax.bar(np.arange(4)+(j-2)*.15,[100*r["rmse_mean"] for r in rr],width=.15,label=method,
            yerr=[100*r["rmse_seed_sd"] for r in rr],capsize=2)
    ax.set_xticks(np.arange(4),["Recurring","Gradual","Stable","Localized"])
    ax.set_ylabel("KRR RMSE (percentage points)");ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
    ax.set_title("Inference-time mixture optimization versus frozen meta-learners")
    fig.savefig(out/"optimized_comparison.pdf");fig.savefig(out/"optimized_comparison.png",dpi=180);plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path(__file__).parent)
    p.add_argument("--output",type=Path,default=Path(__file__).parent/"optimized_fusion_results")
    with threadpool_limits(limits=1):execute(p.parse_args())
