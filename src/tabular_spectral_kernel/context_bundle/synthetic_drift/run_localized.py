"""Localized drift extension using the saved frozen spectral banks.

python synthetic_drift/run_localized.py
Two prespecified populations x2<=0 and x2>0. Only the latter changes from
A to B at time 384. Detection sees past labels, not the regime or change time.
Freeze the old A predictor and fit gated residual KRR only in flagged groups.
The MSE-ratio monitor is a heuristic, not a calibrated sequential test.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as R


def detect(residual, groups, baseline, factor=4., minimum=32):
    flags=[];scores=[]
    for group in [0,1]:
        values=residual[groups==group]
        score=float(np.mean(values**2)) if len(values) else None
        flags.append(len(values)>=minimum and score>factor*baseline[group])
        scores.append(score)
    return flags,scores


def apply_local(base_prediction, groups, corrections):
    """Exact preservation outside groups with an explicitly supplied correction."""
    prediction=base_prediction.copy()
    for group,delta in corrections.items():prediction[groups==group]+=delta
    return prediction


def execute(args):
    source=args.source.resolve();out=args.output.resolve();out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    protocol=json.loads((source/"protocol.json").read_text())
    seeds=protocol["seeds"]
    R.save(out/"protocol.json",dict(seeds=seeds,source=str(source),context=128,support=80,horizon=32,
        population="x2 > 0 vs x2 <= 0; fixed before labels",
        monitor="group recent frozen-predictor MSE > 4 * held-out historical group MSE; min 32 rows",
        response="A everywhere before 384; B on x2>0 after 384; A on x2<=0 throughout",
        source_model_protocol=protocol["source_sha256"],geometry_updates=0,
        limitations="Known candidate partition; heuristic alarm; fully observed labels; no credit-policy simulation"))
    (out/"run_localized_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (out/"run_dependency_snapshot.py").write_bytes(Path(R.__file__).read_bytes())
    metrics=[]
    for seed in seeds:
        folder=source/f"seed{seed}";historical=np.load(folder/"historical_data.npz")
        models=[];temps=[]
        for name in ["bank_a","bank_b","pooled"]:
            model=R.base.CanonicalSpectral(3,4,max_frequency=3.)
            model.load_state_dict(torch.load(folder/f"{name}.pt",weights_only=True,map_location="cpu"));model.eval()
            models.append(model);temps.append(float(model.log_temperature.detach().exp()))
        h=historical["x"][:600];hy=historical["y"][:600]
        zh=R.embedding(models[0],h)
        info=json.loads((folder/"bank_a.json").read_text());ridge=info["ridge"]
        # Calibrate the monitor on held-out historical A labels before refitting
        # the final frozen baseline head on all 600 historical A rows.
        val_pred=R.ridge_predict(R.kernel(zh[:480],zh[:480],temps[0]),
                                R.kernel(zh[480:],zh[:480],temps[0]),hy[:480],ridge)
        hg=(h[480:,2]>0).astype(int)
        calibration=[float(np.mean((val_pred[hg==g]-hy[480:][hg==g])**2)) for g in [0,1]]
        coef,intercept=R.ridge_fit(R.kernel(zh,zh,temps[0]),hy,ridge)
        raw=R.generate(seed,"stationary");x=raw["x"];groups=(x[:,2]>0).astype(int)
        a,b=R.mechanisms(x);mean=np.where((groups==1)&(raw["t"]>=384),b,a)
        y=mean+(raw["y"]-raw["mean"])
        zs=[R.embedding(m,x) for m in models]
        base=R.kernel(zs[0],zh,temps[0])@coef+intercept
        preds={name:np.full(len(y),np.nan) for name in ["frozen_base","global_fusion","partitioned_fusion","selective_residual"]}
        trace=[]
        for origin in range(128,len(y),32):
            sl=slice(origin-128,origin);query=slice(origin,origin+32)
            grams=[R.kernel(z[sl],z[sl],temp) for z,temp in zip(zs,temps)]
            cross=[R.kernel(z[query],z[sl],temp) for z,temp in zip(zs,temps)]
            selected=R.select_heads(grams,y[sl],80)
            preds["global_fusion"][query]=R.predict_heads(grams,cross,y[sl],selected)["kernel_fusion"]
            preds["frozen_base"][query]=base[query]
            residual=y[sl]-base[sl]
            flags,scores=detect(residual,groups[sl],calibration)
            corrections={};local=[]
            for g,flag in enumerate(flags):
                mask=groups[sl]==g;qm=groups[query]==g
                local_grams=[k[np.ix_(mask,mask)] for k in grams]
                local_cross=[q[np.ix_(qm,mask)] for q in cross]
                # Matched partition-aware comparator updates both populations.
                group_selection=R.select_heads(local_grams,y[sl][mask],int(.625*mask.sum()))
                group_prediction=R.predict_heads(local_grams,local_cross,y[sl][mask],group_selection)["kernel_fusion"]
                preds["partitioned_fusion"][np.arange(origin,origin+32)[qm]]=group_prediction
                if not flag:continue
                selection=R.select_heads(local_grams,residual[mask],int(.625*mask.sum()))
                corrections[g]=R.predict_heads(local_grams,local_cross,residual[mask],selection)["kernel_fusion"]
                local.append(dict(group=g,n=int(mask.sum()),selection=selection["kernel_fusion"]))
            preds["selective_residual"][query]=apply_local(base[query],groups[query],corrections)
            trace.append(dict(origin=origin,flags=flags,context_mse=scores,corrections=local))
        valid=raw["t"]>=128;after=raw["t"]>=384;recovered=raw["t"]>=512
        for name,pred in preds.items():
            row=dict(seed=seed,method=name)
            for label,mask in [("all",valid),("stable",valid&(groups==0)),
                               ("changed_after_drift",after&(groups==1)),
                               ("changed_after_context_refresh",recovered&(groups==1))]:
                row[label+"_rmse"]=float(np.sqrt(np.mean((pred[mask]-y[mask])**2)))
            row["max_stable_prediction_change"]=float(np.max(np.abs(pred[valid&(groups==0)]-base[valid&(groups==0)])))
            metrics.append(row)
        R.save(out/f"seed{seed}_trace.json",dict(calibration=calibration,episodes=trace))
        np.savez_compressed(out/f"seed{seed}.npz",x=x,y=y,mean=mean,groups=groups,t=raw["t"],**preds)
        print("localized",seed,[r for r in metrics if r["seed"]==seed],flush=True)
    R.save(out/"metrics.json",metrics)
    summary=[]
    for method in preds:
        rows=[r for r in metrics if r["method"]==method]
        summary.append(dict(method=method,**{key:float(np.mean([r[key] for r in rows]))
            for key in rows[0] if key not in ["seed","method"]}))
    R.save(out/"summary.json",summary)
    fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True,layout="constrained")
    origins=np.arange(128,1920,32)
    for g,ax in enumerate(axs[:2]):
        for method in preds:
            curves=[]
            for seed in seeds:
                z=np.load(out/f"seed{seed}.npz")
                curves.append([np.mean((z[method][t:t+32][z["groups"][t:t+32]==g]-
                    z["y"][t:t+32][z["groups"][t:t+32]==g])**2) for t in origins])
            ax.plot(origins,np.sqrt(np.mean(curves,axis=0)),label=method.replace("_"," "))
        ax.axvline(384,color="black",ls="--");ax.axhline(.15,color="black",ls=":")
        ax.set_ylabel("Next-block RMSE");ax.set_title(["Stable population (x2 <= 0)","Changing population (x2 > 0)"][g]);ax.legend()
    for g in [0,1]:
        flags=[np.array([v["flags"][g] for v in json.loads((out/f"seed{s}_trace.json").read_text())["episodes"]]) for s in seeds]
        axs[2].plot(origins,np.mean(flags,axis=0),label=["Stable group flagged","Changing group flagged"][g])
    axs[2].set_ylabel("Fraction of seeds flagged");axs[2].set_xlabel("Stream time")
    axs[2].legend();axs[2].axvline(384,color="black",ls="--")
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(out/"localized.png",dpi=170);fig.savefig(out/"localized.pdf");plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path(__file__).parent/"results")
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"localized_results")
    with threadpool_limits(limits=1):execute(parser.parse_args())
