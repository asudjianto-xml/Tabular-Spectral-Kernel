"""Controlled context-induced retrieval ablation using saved unemployment models."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run as R
import run_meta as M
import run_credit_card as C
import run_retrieval as E


def context_weights(gate, grams, y):
    with torch.no_grad():
        return gate(torch.tensor(M.descriptor(grams, y), dtype=torch.float64)).numpy()


def fused_neighbors(grams, cross, alpha, k=64):
    # All component Laplace kernels have query self-similarity exactly one.
    # Include context diagonal explicitly: this is RKHS squared distance.
    distance = sum(w * (1 + np.diag(g) - 2*q.reshape(-1))
                   for w, g, q in zip(alpha, grams, cross))
    return np.sort(np.argsort(distance, kind="stable")[:k])


def execute(args):
    out=args.output;out.mkdir(parents=True, exist_ok=False);torch.set_num_threads(1)
    provenance={};rows=[];diagnostics=[]
    def record(path):
        provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for file in [Path(__file__),Path(R.__file__),Path(M.__file__),Path(E.__file__),R.MODEL_SOURCE]:
        (out/(file.stem+"_snapshot.py")).write_bytes(file.read_bytes())
    for seed in [0,1,2]:
        folder=args.source/f"seed{seed}";banks=[]
        for name in ["normal","downturn"]:
            path=folder/f"{name}.pt";record(path)
            model=R.base.CanonicalSpectral(5,4,max_frequency=3.,device="cpu")
            model.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"));model.eval();banks.append(model)
        for profile in ["original","nw_calibrated"]:
            path=folder/f"{profile}_gate.pt";record(path);gate=M.ContextGate()
            gate.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"));gate.eval()
            path=folder/f"{profile}_training.json";record(path);info=json.loads(path.read_text())
            for scenario in C.SCENARIOS:
                path=folder/f"{profile}_{scenario}.npz";record(path)
                with np.load(path) as saved: data={key:saved[key] for key in saved.files}
                x,y=data["x"],data["y"];n=len(y);zs=[R.embedding(m,x) for m in banks]
                preds={d:np.full(n,np.nan) for d in ["nw","krr"]}
                initial=np.full((n,2),np.nan);refined=initial.copy();indices=np.full((n,64),-1,dtype=int)
                overlaps=[];baseline_max=0.
                for origin in range(128,n,32):
                    sl=slice(origin-128,origin)
                    grams=[R.kernel(z[sl],z[sl],t) for z,t in zip(zs,info["temperatures"])]
                    cross=[R.kernel(z[origin:origin+32],z[sl],t) for z,t in zip(zs,info["temperatures"])]
                    alpha=context_weights(gate,grams,y[sl])
                    for offset in range(32):
                        i=origin+offset;q=[v[offset:offset+1] for v in cross]
                        ix=fused_neighbors(grams,q,alpha)
                        a,nw,krr=M.forward(gate,[g[np.ix_(ix,ix)] for g in grams],[v[:,ix] for v in q],y[sl][ix],info["ridge"])
                        initial[i]=alpha;refined[i]=a;indices[i]=origin-128+ix
                        for dec,p in [("nw",nw),("krr",krr)]:preds[dec][i]=np.clip(p[0],0,1)
                        old=data["spectral_nearest64_indices"][i];old=old[old>=0]
                        overlaps.append(len(np.intersect1d(indices[i],old))/64)
                        # Recompute equal-weight control on the identical blocks.
                        eq=fused_neighbors(grams,q,np.array([.5,.5]))
                        np.testing.assert_array_equal(origin-128+eq,old)
                        _,bn,bk=M.forward(gate,[g[np.ix_(eq,eq)] for g in grams],[v[:,eq] for v in q],y[sl][eq],info["ridge"])
                        for dec,p in [("nw",bn),("krr",bk)]:
                            baseline_max=max(baseline_max,abs(np.clip(p[0],0,1)-data["spectral_nearest64_"+dec][i]))
                valid=np.arange(n)>=128;origins=(np.arange(n)//32)*32
                assert np.all(indices[valid]<origins[valid,None])
                assert np.all(indices[valid]>=origins[valid,None]-128)
                np.testing.assert_allclose(initial[valid].sum(1),1)
                np.testing.assert_allclose(refined[valid].sum(1),1)
                assert baseline_max<1e-10
                np.savez_compressed(out/f"seed{seed}_{profile}_{scenario}.npz",
                    initial_alpha=initial,refined_alpha=refined,indices=indices,**preds)
                for method in E.METHODS+["context_spectral64"]:
                    for dec in ["nw","krr"]:
                        p=preds[dec] if method=="context_spectral64" else data[method+"_"+dec]
                        row=dict(seed=seed,profile=profile,scenario=scenario,method=method,decoder=dec,
                            rmse=float(np.sqrt(np.mean((p[valid]-y[valid])**2))))
                        if scenario=="localized_downturn":
                            for group,mask in [("stable",valid&(data["raw"][:,2]>2)),("changed",valid&(data["macro_exposure"]>.1)&(data["raw"][:,2]<=2))]:
                                row[group+"_rmse"]=float(np.sqrt(np.mean((p[mask]-y[mask])**2)))
                        rows.append(row)
                diagnostics.append(dict(seed=seed,profile=profile,scenario=scenario,
                    equal_weight_control_max_error=baseline_max,neighbor_overlap=float(np.mean(overlaps)),
                    initial_bank0_quantiles=np.quantile(initial[valid,0],[0,.5,1]).tolist(),
                    refined_bank0_quantiles=np.quantile(refined[valid,0],[0,.5,1]).tolist()))
                print(seed,profile,scenario,"context KRR",round(rows[-1]["rmse"]*100,3),flush=True)
    summary=[]
    for profile in ["original","nw_calibrated"]:
        for scenario in C.SCENARIOS:
            for method in E.METHODS+["context_spectral64"]:
                for dec in ["nw","krr"]:
                    rr=[r for r in rows if (r["profile"],r["scenario"],r["method"],r["decoder"])==(profile,scenario,method,dec)]
                    summary.append(dict(profile=profile,scenario=scenario,method=method,decoder=dec,
                        rmse_mean=float(np.mean([r["rmse"] for r in rr])),rmse_seed_sd=float(np.std([r["rmse"] for r in rr],ddof=1)),
                        **({g+"_rmse":float(np.mean([r[g+"_rmse"] for r in rr])) for g in ["stable","changed"]} if scenario=="localized_downturn" else {})))
    for name,value in [("metrics",rows),("summary",summary),("diagnostics",diagnostics),("provenance",provenance)]:R.save(out/(name+".json"),value)
    R.save(out/"protocol.json",dict(source=str(args.source.resolve()),seeds=[0,1,2],
        algorithm="128 past labels -> frozen gate -> fused RKHS nearest 64 -> same gate on retrieved labels -> NW and KRR",
        controls="Saved predictions, equal-weight spectral retrieval independently reproduced to 1e-10",
        limitation="Gate previously trained on full/recent/raw/equal-weight spectral/population contexts, not this two-stage adaptive selection; no retraining in this controlled ablation",
        inference="No gradients, weight search, future labels, or true macro state; KRR uses a linear solve"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path(__file__).parent/"unemployment_results")
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"context_retrieval_results")
    with threadpool_limits(limits=1):execute(parser.parse_args())
