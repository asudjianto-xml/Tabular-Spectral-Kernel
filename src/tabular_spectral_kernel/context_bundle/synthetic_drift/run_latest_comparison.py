"""Latest historical kernel versus context-induced fusion, matched past labels."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run as R
import run_meta as M
import run_credit_card as C


def execute(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    rows=[];provenance={}
    def read(path):
        provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    for file in [Path(__file__),Path(R.__file__),Path(M.__file__),R.MODEL_SOURCE]:
        (out/(file.stem+"_snapshot.py")).write_bytes(file.read_bytes())
    for seed in [0,1,2]:
        src=args.source/"unemployment_results"/f"seed{seed}"
        model=R.base.CanonicalSpectral(5,4,max_frequency=3.,device="cpu")
        model.load_state_dict(torch.load(read(src/"downturn.pt"),map_location="cpu",weights_only=True));model.eval()
        info=json.loads(read(src/"original_training.json").read_text())
        own=json.loads(read(src/"downturn.json").read_text())
        for scenario in C.SCENARIOS:
            with np.load(read(src/f"original_{scenario}.npz")) as f:data={k:f[k] for k in f.files}
            x,y=data["x"],data["y"];z=R.embedding(model,x);n=len(y)
            preds={key:np.full(n,np.nan) for key in ["latest_krr","latest_nw","latest_own_ridge_krr"]}
            indices=np.full((n,128),-1,dtype=int)
            for origin in range(128,n,32):
                sl=slice(origin-128,origin);qsl=slice(origin,origin+32)
                k=R.kernel(z[sl],z[sl],info["temperatures"][1]);q=R.kernel(z[qsl],z[sl],info["temperatures"][1])
                preds["latest_krr"][qsl]=np.clip(R.ridge_predict(k,q,y[sl],info["ridge"]),0,1)
                preds["latest_nw"][qsl]=np.clip(M.nw_predict([q],y[sl],[1.]),0,1)
                preds["latest_own_ridge_krr"][qsl]=np.clip(R.ridge_predict(k,q,y[sl],own["ridge"]),0,1)
                indices[qsl]=np.arange(origin-128,origin)
            for folder,method in [("transformer_results","transformer"),("transformer_rope_results","rope")]:
                with np.load(read(args.source/folder/f"seed{seed}"/(scenario+".npz"))) as f:
                    for dec in ["krr","nw"]:preds[method+"_"+dec]=f["transformer_"+dec]
            for dec in ["krr","nw"]:preds["summary_"+dec]=data["full128_"+dec]
            valid=np.arange(n)>=128;origins=(np.arange(n)//32)*32
            assert np.all(indices[valid]<origins[valid,None])
            assert np.all(indices[valid]>=origins[valid,None]-128)
            np.savez_compressed(out/f"seed{seed}_{scenario}.npz",**preds,indices=indices)
            for key,pred in preds.items():
                assert np.isfinite(pred[valid]).all()
                method,dec=key.rsplit("_",1)
                row=dict(seed=seed,scenario=scenario,method=method,decoder=dec,
                    rmse=float(np.sqrt(np.mean((pred[valid]-y[valid])**2))),
                    mae=float(np.mean(np.abs(pred[valid]-y[valid]))))
                if scenario=="localized_downturn":
                    for group,mask in [("stable",valid&(data["raw"][:,2]>2)),("changed",valid&(data["macro_exposure"]>.1)&(data["raw"][:,2]<=2))]:
                        row[group+"_rmse"]=float(np.sqrt(np.mean((pred[mask]-y[mask])**2)))
                rows.append(row)
    summary=[]
    for scenario,method,dec in sorted({(r["scenario"],r["method"],r["decoder"]) for r in rows}):
        rr=[r for r in rows if (r["scenario"],r["method"],r["decoder"])==(scenario,method,dec)]
        base=[next(r["rmse"] for r in rows if (r["scenario"],r["method"],r["decoder"],r["seed"])==(scenario,"latest",dec,s)) for s in [0,1,2]]
        delta=np.array(base)-np.array([r["rmse"] for r in rr])
        summary.append(dict(scenario=scenario,method=method,decoder=dec,
            rmse_mean=float(np.mean([r["rmse"] for r in rr])),rmse_seed_sd=float(np.std([r["rmse"] for r in rr],ddof=1)),
            improvement_pp=float(100*delta.mean()),improvement_seed_pp= (100*delta).tolist(),
            relative_rmse_reduction_pct=float(100*delta.mean()/np.mean(base)),
            **({g+"_rmse":float(np.mean([r[g+"_rmse"] for r in rr])) for g in ["stable","changed"]} if scenario=="localized_downturn" else {})))
    for name,value in [("summary",summary),("metrics",rows),("provenance",provenance)]:R.save(out/(name+".json"),value)
    R.save(out/"protocol.json",dict(seeds=[0,1,2],profile="original",context=128,horizon=32,
        latest_kernel="second/later historical window: downturn.pt; alpha=(0,1), geometry frozen",
        comparison="all methods refit readout using identical 128 past labels; same historical bandwidths and common ridge",
        sensitivity="latest_own_ridge uses its independently historical-validation-selected ridge",
        qualification="latest kernel also learns its readout from context; difference is geometry adaptation, not access to labels; no retraining of encoders; original NW bandwidth only"))
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for ax,dec in zip(axs,["krr","nw"]):
        for j,method in enumerate(["latest","transformer","rope"]):
            rr=[next(r for r in summary if (r["scenario"],r["method"],r["decoder"])==(s,method,dec)) for s in C.SCENARIOS]
            ax.bar(np.arange(4)+(j-1)*.24,[100*r["rmse_mean"] for r in rr],width=.24,label=method,
                yerr=[100*r["rmse_seed_sd"] for r in rr],capsize=2)
        ax.set_xticks(np.arange(4),["Recurring","Gradual","Stable","Localized"])
        ax.set(title=dec.upper(),ylabel="RMSE (percentage points)");ax.grid(axis="y",alpha=.2)
    axs[0].legend();fig.suptitle("Same labeled context: single latest kernel vs context-conditioned fusion")
    fig.savefig(out/"latest_comparison.png",dpi=170);fig.savefig(out/"latest_comparison.pdf");plt.close(fig)
    for r in summary:
        if r["decoder"]=="krr":print(r["scenario"],r["method"],round(100*r["rmse_mean"],4),"reduction %",round(r["relative_rmse_reduction_pct"],2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path(__file__).parent)
    p.add_argument("--output",type=Path,default=Path(__file__).parent/"latest_kernel_comparison")
    with threadpool_limits(limits=1):execute(p.parse_args())
