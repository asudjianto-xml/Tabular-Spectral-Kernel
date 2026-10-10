"""Query-local retrieval ablation with frozen spectral banks and meta gates.

No fitting, gradient updates or hyperparameter search uses evaluation labels.
Both saved bandwidth profiles are reported. All methods see the same past
128 eligible labels. Full/recent controls distinguish retrieval from context
size and recency. Population retrieval is an informed-partition reference.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
import run as R
import run_meta as M

METHODS=["full128","recent64","raw_nearest64","spectral_nearest64","population64"]
SCENARIOS=["recurring","gradual","stationary","mixed","localized"]
PROFILES=["meta_results","meta_nw_calibrated_results"]


def retrieve(method,context_x,query_x,cross,k=64):
    """Label-free retrieval; indices refer only to caller's eligible context."""
    n=len(context_x);k=min(k,n)
    if method=="full128":return np.arange(n)
    if method=="recent64":return np.arange(n-k,n)
    if method=="raw_nearest64":
        # Equal known scales: all coordinates are uniform on [-1,1].
        distance=np.sum((context_x-query_x)**2,axis=1)
    elif method=="spectral_nearest64":
        distance=-np.mean(np.asarray(cross),axis=0).reshape(-1)
    elif method=="population64":
        eligible=np.flatnonzero((context_x[:,2]>0)==(query_x[2]>0))
        return eligible[-k:]
    else:raise ValueError(method)
    # Stable ties use original chronological order; preserve chronology after selection.
    return np.sort(np.argsort(distance,kind="stable")[:k])


def stream(seed,scenario):
    data=R.generate(seed,"stationary" if scenario=="localized" else scenario)
    if scenario=="localized":
        a,b=R.mechanisms(data["x"])
        changed=(data["x"][:,2]>0)&(data["t"]>=384)
        mean=np.where(changed,b,a)
        data["y"]=mean+(data["y"]-data["mean"])
        data["mean"]=mean;data["weight"]=(~changed).astype(float)
    return data


def execute(args):
    source=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    R.save(out/"protocol.json",dict(seeds=args.seeds,profiles=PROFILES,methods=METHODS,
        scenarios=SCENARIOS,eligible_context=128,retrieval_k=64,horizon=32,
        inference="frozen saved gate and banks; no optimization; same local weights for NW/KRR",
        limitations=["Gate trained on 128-row stationary tasks, evaluated also on retrieved subsets",
                     "Population reference knows the candidate x2 sign partition",
                     "NW-calibrated profile retains historical original-bandwidth KRR ridge",
                     "No exact stable-prediction preservation is imposed"],
        source=str(source)))
    for file in [Path(__file__),Path(R.__file__),Path(M.__file__)]:
        (out/(file.stem+"_snapshot.py")).write_bytes(file.read_bytes())
    rows=[];start=time.perf_counter()
    provenance={}
    for seed in args.seeds:
        banks=[];folder=source/"results"/f"seed{seed}"
        for name in ["bank_a","bank_b"]:
            model=R.base.CanonicalSpectral(3,4,max_frequency=3.)
            path=folder/f"{name}.pt"
            model.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"));model.eval()
            banks.append(model);provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for profile in PROFILES:
            gate=M.ContextGate();gatepath=source/profile/f"seed{seed}_gate.pt"
            gate.load_state_dict(torch.load(gatepath,weights_only=True,map_location="cpu"));gate.eval()
            provenance[str(gatepath)]=hashlib.sha256(gatepath.read_bytes()).hexdigest()
            info=json.loads((source/profile/f"seed{seed}_training.json").read_text())
            temps=info.get("temperatures",[float(m.log_temperature.detach().exp()) for m in banks])
            ridge=info["ridge"]
            for scenario in SCENARIOS:
                data=stream(seed,scenario);x=data["x"];y=data["y"];n=len(y)
                zs=[R.embedding(m,x) for m in banks]
                predictions={f"{method}_{decoder}":np.full(n,np.nan) for method in METHODS for decoder in ["nw","krr"]}
                weights={method:np.full((n,2),np.nan) for method in METHODS}
                indices={method:np.full((n,128),-1,dtype=np.int32) for method in METHODS}
                seconds={method:0. for method in METHODS}
                for origin in range(128,n,32):
                    sl=slice(origin-128,origin);end=min(n,origin+32)
                    grams=[R.kernel(z[sl],z[sl],t) for z,t in zip(zs,temps)]
                    cross=[R.kernel(z[origin:end],z[sl],t) for z,t in zip(zs,temps)]
                    for method in METHODS:
                        t0=time.perf_counter()
                        if method in ["full128","recent64"]:
                            ix=retrieve(method,x[sl],x[origin],[q[:1] for q in cross])
                            alpha,nw,krr=M.forward(gate,[k[np.ix_(ix,ix)] for k in grams],
                                                   [q[:,ix] for q in cross],y[sl][ix],ridge)
                            predictions[method+"_nw"][origin:end]=nw
                            predictions[method+"_krr"][origin:end]=krr
                            weights[method][origin:end]=alpha
                            indices[method][origin:end,:len(ix)]=origin-128+ix
                        else:
                            for qi,qtime in enumerate(range(origin,end)):
                                q=[g[qi:qi+1] for g in cross]
                                ix=retrieve(method,x[sl],x[qtime],q)
                                if len(ix)<2:raise RuntimeError("Insufficient eligible local context")
                                alpha,nw,krr=M.forward(gate,[k[np.ix_(ix,ix)] for k in grams],
                                    [g[:,ix] for g in q],y[sl][ix],ridge)
                                predictions[method+"_nw"][qtime]=nw[0]
                                predictions[method+"_krr"][qtime]=krr[0]
                                weights[method][qtime]=alpha
                                indices[method][qtime,:len(ix)]=origin-128+ix
                        seconds[method]+=time.perf_counter()-t0
                name=f"seed{seed}_{profile}_{scenario}"
                np.savez_compressed(out/(name+".npz"),**data,**predictions,
                    **{k+"_alpha":v for k,v in weights.items()},**{k+"_indices":v for k,v in indices.items()})
                valid=data["t"]>=128
                for method in METHODS:
                    for decoder in ["nw","krr"]:
                        pred=predictions[method+"_"+decoder]
                        row=dict(seed=seed,profile=profile,scenario=scenario,method=method,decoder=decoder,
                            rmse=float(np.sqrt(np.mean((pred[valid]-y[valid])**2))),
                            clean_rmse=float(np.sqrt(np.mean((pred[valid]-data["mean"][valid])**2))),
                            joint_decoder_seconds=seconds[method])
                        if scenario=="localized":
                            for group,mask in [("stable",valid&(x[:,2]<=0)),("changed",(data["t"]>=384)&(x[:,2]>0))]:
                                row[group+"_rmse"]=float(np.sqrt(np.mean((pred[mask]-y[mask])**2)))
                        rows.append(row)
                print(seed,profile,scenario,{r["method"]+"_"+r["decoder"]:round(r["rmse"],3) for r in rows if r["seed"]==seed and r["profile"]==profile and r["scenario"]==scenario},flush=True)
    R.save(out/"provenance.json",provenance);R.save(out/"metrics.json",rows)
    summary=[]
    for profile in PROFILES:
        for scenario in SCENARIOS:
            for method in METHODS:
                for decoder in ["nw","krr"]:
                    rr=[r for r in rows if r["profile"]==profile and r["scenario"]==scenario and r["method"]==method and r["decoder"]==decoder]
                    vals=[r["rmse"] for r in rr]
                    entry=dict(profile=profile,scenario=scenario,method=method,decoder=decoder,
                        rmse_mean=float(np.mean(vals)),rmse_seed_sd=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.)
                    if scenario=="localized":
                        entry.update({g+"_rmse":float(np.mean([r[g+"_rmse"] for r in rr])) for g in ["stable","changed"]})
                    summary.append(entry)
    R.save(out/"summary.json",summary)
    for profile in PROFILES:
        fig,axs=plt.subplots(1,2,figsize=(12,4),layout="constrained")
        for decoder,ax in zip(["nw","krr"],axs):
            for i,method in enumerate(METHODS):
                vals=[next(r["rmse_mean"] for r in summary if r["profile"]==profile and r["scenario"]==sc and r["method"]==method and r["decoder"]==decoder) for sc in SCENARIOS]
                ax.bar(np.arange(5)+(i-2)*.15,vals,width=.15,label=method)
            ax.set_xticks(np.arange(5),SCENARIOS,rotation=20);ax.set_ylabel("Mean RMSE (3 seeds)")
            ax.set_title(decoder.upper());ax.grid(axis="y",alpha=.2)
        axs[0].legend(fontsize=8);fig.suptitle(profile.replace("_"," "))
        fig.savefig(out/(profile+".png"),dpi=170);fig.savefig(out/(profile+".pdf"));plt.close(fig)
    R.save(out/"runtime.json",dict(seconds=time.perf_counter()-start,
        timing_note="Joint NW/KRR inference timings exclude kernel-block construction; query-local methods repeat gate/solve per query, full/recent controls reuse across a block."))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path(__file__).parent)
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"retrieval_results")
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    with threadpool_limits(limits=1):execute(parser.parse_args())
