"""Controlled temporal spectral-kernel experiment; no network or bike data.

Run: python synthetic_drift/run.py --seeds 0 1 2 --steps 400
Inputs are IID Uniform([-1,1]^3); only E[Y|X] changes. Historical windows
A and B precede all evaluation streams. Reuse CanonicalSpectral from the
existing paper implementation. Context selection is chronological held-out
KRR error over a simplex grid, not a claim of convex optimization.
Outputs include raw data, checkpoints, predictions, selection traces, metrics,
protocol/source snapshots, and PDF/PNG plots. Existing output dirs are refused.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
from tabular_spectral_kernel._paper import identifiable_spectral_california as base
MODEL_SOURCE = Path(base.__file__)
RIDGES = np.array([1e-4, 1e-3, 1e-2, .1])
WEIGHTS = np.linspace(0., 1., 11)
METHODS = ["kernel_fusion", "prediction_average", "best_single", "uniform_kernel",
           "pooled_kernel", "latest_kernel", "stale_predictor"]
SCENARIOS = ["recurring", "gradual", "stationary", "mixed"]


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def mechanisms(x):
    a = np.sin(2*np.pi*x[:, 0]) + .4*np.cos(4*np.pi*x[:, 0])
    b = np.sin(6*np.pi*x[:, 1]) + .4*np.cos(2*np.pi*x[:, 1])
    return a, b


def generate(seed, scenario, segment=384, noise=.15):
    # Common random numbers across scenarios; generator weights are never
    # passed to the model. x2 is a distractor in both historical regimes.
    rng = np.random.default_rng(10000 + seed)
    x = rng.uniform(-1, 1, (5*segment, 3))
    t = np.arange(len(x))
    if scenario == "recurring":
        weight = np.repeat([1., 0., 1., 0., 1.], segment)
    elif scenario == "gradual":
        weight = np.concatenate([np.ones(segment), np.linspace(1, 0, segment),
                                 np.zeros(segment), np.linspace(0, 1, segment),
                                 np.ones(segment)])
    elif scenario == "stationary":
        weight = np.ones(len(x))
    elif scenario == "mixed":
        weight = np.full(len(x), .5)
    else:
        raise ValueError(scenario)
    a, b = mechanisms(x)
    mean = weight*a + (1-weight)*b
    y = mean + rng.normal(0, noise, len(x))
    return dict(x=x, y=y, mean=mean, weight=weight, t=t)


def historical(seed, n=600, noise=.15):
    rng = np.random.default_rng(20000 + seed)
    x = rng.uniform(-1, 1, (2*n, 3))
    a, b = mechanisms(x)
    mean = np.concatenate([a[:n], b[n:]])
    return x, mean + rng.normal(0, noise, len(x)), mean


def ridge_fit(k, y, ridge):
    """Exact KRR with an unpenalized intercept and mean-loss ridge."""
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    factor = cho_factor(k + len(y)*ridge*np.eye(len(y)), lower=True)
    sol = cho_solve(factor, np.column_stack([y, np.ones(len(y))]))
    intercept = sol[:, 0].sum()/sol[:, 1].sum()
    return sol[:, 0] - intercept*sol[:, 1], float(intercept)


def ridge_predict(k, q, y, ridge):
    c, b = ridge_fit(k, y, ridge)
    return q@c + b


def select_heads(grams, y, support):
    """Selection accepts context Grams/labels only, never query data/labels.

    First two Grams are historical A/B; third is the pooled geometry.
    Tune each averaging member's ridge independently, then its output weight.
    All selected heads are later refitted on the full identical context.
    """
    n = len(y)
    if not 1 < support < n:
        raise ValueError("need nonempty chronological support and validation")
    target = y[support:]
    def score(k, ridge):
        pred = ridge_predict(k[:support, :support], k[support:, :support],
                             y[:support], ridge)
        return float(np.mean((pred-target)**2)), pred
    single = []
    for j, k in enumerate(grams):
        trials = [(score(k, r)[0], float(r)) for r in RIDGES]
        loss, ridge = min(trials)
        single.append(dict(kernel=j, ridge=ridge, val_mse=loss))
    fusion = []
    for w in WEIGHTS:
        k = w*grams[0]+(1-w)*grams[1]
        for ridge in RIDGES:
            loss, _ = score(k, ridge)
            fusion.append(dict(weight=float(w), ridge=float(ridge), val_mse=loss))
    members = [score(grams[j], single[j]["ridge"])[1] for j in (0, 1)]
    diff = members[0]-members[1]
    den = diff@diff
    w = float(np.clip(diff@(target-members[1])/den, 0, 1)) if den > 0 else .5
    average = dict(weight=w, ridges=[s["ridge"] for s in single[:2]],
                   val_mse=float(np.mean((w*members[0]+(1-w)*members[1]-target)**2)))
    return dict(kernel_fusion=min(fusion, key=lambda d: d["val_mse"]),
                uniform_kernel=min([d for d in fusion if d["weight"] == .5],
                                   key=lambda d: d["val_mse"]),
                best_single=min(single[:2], key=lambda d: d["val_mse"]),
                pooled_kernel=single[2], latest_kernel=single[1],
                prediction_average=average)


def predict_heads(grams, cross, y, selection):
    out = {}
    for name in ["kernel_fusion", "uniform_kernel"]:
        s = selection[name]; w = s["weight"]
        out[name] = ridge_predict(w*grams[0]+(1-w)*grams[1],
                                 w*cross[0]+(1-w)*cross[1], y, s["ridge"])
    for name in ["best_single", "pooled_kernel", "latest_kernel"]:
        s = selection[name]; j = s["kernel"]
        out[name] = ridge_predict(grams[j], cross[j], y, s["ridge"])
    s = selection["prediction_average"]
    out["prediction_average"] = sum(
        w*ridge_predict(grams[j], cross[j], y, ridge)
        for j, (w, ridge) in enumerate(zip([s["weight"], 1-s["weight"]], s["ridges"])))
    return out


def train_geometry(x, y, train, validation, seed, steps, path):
    torch.manual_seed(seed)
    model = base.CanonicalSpectral(x.shape[1], 4, max_frequency=3., device="cpu")
    # Broad, identical initialization for all banks; no true regime parameters.
    with torch.no_grad():
        omega = torch.tensor([.25, .75, 1.5, 3.], dtype=torch.float64)
        model.log_gaps.copy_(torch.diff(omega, prepend=omega.new_zeros(1)).log().repeat(x.shape[1], 1))
        model.log_ridge.fill_(np.log(.05))
    xt = torch.as_tensor(x[train], dtype=torch.float64)
    mu, sd = float(y[train].mean()), float(y[train].std())
    yt = torch.as_tensor((y[train]-mu)/sd, dtype=torch.float64)
    xv = torch.as_tensor(x[validation], dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=.015)
    gen = torch.Generator().manual_seed(seed+700)
    trace = []; best = None
    for step in range(steps+1):
        if step % 50 == 0 or step == steps:
            with torch.no_grad():
                z = model.embedding(xt); zv = model.embedding(xv)
                k = model.kernel(z,z).numpy(); q = model.kernel(zv,z).numpy()
            trials = [(float(np.mean((ridge_predict(k,q,y[train],r)-y[validation])**2)),float(r))
                      for r in RIDGES]
            loss, ridge = min(trials)
            trace.append(dict(step=step, val_mse=loss, ridge=ridge))
            if best is None or loss < best[0]:
                best = (loss, copy.deepcopy(model.state_dict()), step, ridge)
        if step == steps:
            break
        ids = torch.randperm(len(xt), generator=gen)[:128]
        z = model.embedding(xt[ids]); ys = yt[ids[:64]]
        k = model.kernel(z[:64], z[:64]); q = model.kernel(z[64:], z[:64])
        factor = torch.linalg.cholesky(k + model.log_ridge.exp()*torch.eye(64,dtype=torch.float64))
        sol = torch.cholesky_solve(torch.stack([ys,torch.ones_like(ys)],1),factor)
        b = sol[:,0].sum()/sol[:,1].sum(); c=sol[:,0]-b*sol[:,1]
        loss = (q@c+b-yt[ids[64:]]).square().mean()
        loss = loss + 1e-5*(model.diagonal.square().mean()+model.offdiag.square().mean())
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.);optimizer.step()
        with torch.no_grad():
            model.log_gaps.clamp_(-6, 2);model.log_temperature.clamp_(-4, 3)
            model.log_ridge.clamp_(np.log(1e-4), np.log(2.))
    model.load_state_dict(best[1]);model.eval()
    torch.save(model.state_dict(),path.with_suffix(".pt"))
    info=dict(selected_step=best[2],validation_mse=best[0],ridge=best[3],trace=trace,
              train_rows=train.tolist(),validation_rows=validation.tolist(),
              frequencies=model.frequencies().detach().numpy().tolist(),
              temperature=float(model.log_temperature.exp().detach()))
    save(path.with_suffix(".json"),info)
    print(path.name, "validation RMSE", round(np.sqrt(best[0]),4), "step",best[2],flush=True)
    return model, info


def embedding(model, x):
    with torch.no_grad():
        return model.embedding(torch.as_tensor(x,dtype=torch.float64)).numpy()


def kernel(z, q, temperature):
    return np.exp(-cdist(z,q)/temperature)


def run_seed(seed, args, out):
    dest=out/f"seed{seed}";dest.mkdir()
    hx,hy,hm=historical(seed)
    np.savez_compressed(dest/"historical_data.npz",x=hx,y=hy,mean=hm,window=np.repeat([0,1],600))
    models=[];infos=[]
    splits=[(np.arange(480),np.arange(480,600)),
            (np.arange(600,1080),np.arange(1080,1200)),
            (np.r_[0:480,600:1080],np.r_[480:600,1080:1200])]
    for j,(tr,va) in enumerate(splits):
        model,info=train_geometry(hx,hy,tr,va,seed*10+j,args.steps,dest/["bank_a","bank_b","pooled"][j])
        models.append(model);infos.append(info)
    temperatures=[v["temperature"] for v in infos]
    stale_x=hx[600:];stale_y=hy[600:];stale_z=embedding(models[1],stale_x)
    stale_c,stale_b=ridge_fit(kernel(stale_z,stale_z,temperatures[1]),stale_y,infos[1]["ridge"])
    results=[]
    for scenario in SCENARIOS:
        data=generate(seed,scenario);x=data["x"];y=data["y"]
        zs=[embedding(m,x) for m in models]
        predictions={name:np.full(len(y),np.nan) for name in METHODS};selections=[]
        for origin in range(args.context,len(y),args.horizon):
            end=min(origin+args.horizon,len(y)); start=origin-args.context
            grams=[kernel(z[start:origin],z[start:origin],temp) for z,temp in zip(zs,temperatures)]
            cross=[kernel(z[origin:end],z[start:origin],temp) for z,temp in zip(zs,temperatures)]
            selected=select_heads(grams,y[start:origin],args.support)
            preds=predict_heads(grams,cross,y[start:origin],selected)
            preds["stale_predictor"]=kernel(zs[1][origin:end],stale_z,temperatures[1])@stale_c+stale_b
            for name,pred in preds.items():predictions[name][origin:end]=pred
            selections.append(dict(origin=origin,context_start=start,context_end=origin-1,
                                   query_end=end-1,selection=selected))
        np.savez_compressed(dest/f"{scenario}.npz",**data,**predictions)
        save(dest/f"{scenario}_selection.json",selections)
        valid=np.arange(len(y))>=args.context
        # Same intervals for all methods; stationary/mixed have no transitions.
        transition=np.zeros(len(y),dtype=bool)
        if scenario in ("recurring","gradual"):
            for boundary in [384,768,1152,1536]:transition[boundary:boundary+args.context]=True
        for name,pred in predictions.items():
            row=dict(seed=seed,scenario=scenario,method=name,n=int(valid.sum()),
                     rmse=float(np.sqrt(np.mean((pred[valid]-y[valid])**2))),
                     mean_rmse=float(np.sqrt(np.mean((pred[valid]-data["mean"][valid])**2))),
                     mae=float(np.mean(np.abs(pred[valid]-y[valid]))))
            for label,mask in [("after_boundary",valid&transition),("outside_boundary",valid&~transition)]:
                if mask.any():row[label+"_rmse"]=float(np.sqrt(np.mean((pred[mask]-y[mask])**2)))
            results.append(row)
        print("seed",seed,scenario,{r["method"]:round(r["rmse"],3) for r in results if r["scenario"]==scenario},flush=True)
    save(dest/"metrics.json",results)
    return results


def report(out, rows, args):
    colors=dict(kernel_fusion="#0072B2",prediction_average="#D55E00",best_single="#009E73",
                uniform_kernel="#CC79A7",pooled_kernel="#6F6F6F",latest_kernel="#E69F00")
    summary=[]
    for scenario in SCENARIOS:
        for name in METHODS:
            rr=[r for r in rows if r["scenario"]==scenario and r["method"]==name]
            vals=np.array([r["rmse"] for r in rr])
            summary.append(dict(scenario=scenario,method=name,rmse_mean=float(vals.mean()),
                                rmse_seed_sd=float(vals.std(ddof=1)) if len(vals)>1 else 0.))
        fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True,layout="constrained")
        first=np.load(out/f"seed{args.seeds[0]}"/f"{scenario}.npz")
        axs[0].plot(first["t"],first["weight"],color="black",label="Response coefficient of regime A")
        axs[0].set_ylabel("True coefficient");axs[0].set_ylim(-.05,1.05);axs[0].legend(loc="upper right")
        for name in colors:
            curves=[]
            for seed in args.seeds:
                z=np.load(out/f"seed{seed}"/f"{scenario}.npz")
                curves.append([np.mean((z[name][t:t+args.horizon]-z["y"][t:t+args.horizon])**2)
                               for t in range(args.context,len(z["y"]),args.horizon)])
            t=np.arange(args.context,len(first["y"]),args.horizon)
            rmse=np.sqrt(np.mean(curves,axis=0))
            axs[1].plot(t,rmse,label=name.replace("_"," "),color=colors[name],alpha=.9)
        axs[1].axhline(.15,color="black",ls=":",label="Noise SD")
        axs[1].set_ylabel("Next-block RMSE");axs[1].legend(ncol=3,fontsize=8)
        for seed in args.seeds:
            trace=json.loads((out/f"seed{seed}"/f"{scenario}_selection.json").read_text())
            axs[2].plot([v["origin"] for v in trace],
                        [v["selection"]["kernel_fusion"]["weight"] for v in trace],
                        alpha=.6,label=f"Seed {seed}")
        axs[2].set_ylabel("Kernel A weight");axs[2].set_xlabel("Stream time (one observation per step)")
        axs[2].set_ylim(-.05,1.05);axs[2].legend(ncol=len(args.seeds))
        axs[0].set_title(scenario.capitalize()+" concept-drift experiment; frozen historical kernels")
        for ax in axs:ax.grid(alpha=.2)
        fig.savefig(out/f"{scenario}.png",dpi=170);fig.savefig(out/f"{scenario}.pdf");plt.close(fig)
    save(out/"summary.json",summary)
    lines=[r"\begin{table}[ht]",r"\centering\small",r"\begin{tabular}{lrrrr}",r"\toprule",
           r"Method & Recurring & Gradual & Stationary & Mixed \\",r"\midrule"]
    for name in METHODS:
        cells=[]
        for scenario in SCENARIOS:
            row=next(r for r in summary if r["scenario"]==scenario and r["method"]==name)
            cells.append(f"${row['rmse_mean']:.3f} \\pm {row['rmse_seed_sd']:.3f}$")
        lines.append(name.replace("_"," ")+" & "+" & ".join(cells)+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}",
              r"\caption{Synthetic pilot: mean RMSE and standard deviation across "+str(len(args.seeds))+r" seeds. Noise standard deviation is 0.15. These are seed dispersions, not confidence intervals.}",r"\end{table}"]
    (out/"results_table.tex").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    parser.add_argument("--steps",type=int,default=400)
    parser.add_argument("--context",type=int,default=128)
    parser.add_argument("--support",type=int,default=80)
    parser.add_argument("--horizon",type=int,default=32)
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"results")
    args=parser.parse_args()
    if not 2<=args.support<args.context<1920 or args.horizon<=0 or args.steps<0:
        parser.error("invalid context/support/horizon/steps")
    if len(set(args.seeds))!=len(args.seeds):parser.error("seeds must be unique")
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    source=Path(__file__).read_bytes();model_source=MODEL_SOURCE.read_bytes()
    (out/"run_snapshot.py").write_bytes(source)
    (out/"canonical_model_snapshot.py").write_bytes(model_source)
    save(out/"protocol.json",dict(**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        scenarios=SCENARIOS,methods=METHODS,ridges=RIDGES.tolist(),fusion_weights=WEIGHTS.tolist(),
        input_distribution="IID Uniform([-1,1]^3), identical across time and scenarios",
        response_a="sin(2*pi*x0)+0.4*cos(4*pi*x0)",response_b="sin(6*pi*x1)+0.4*cos(2*pi*x1)",
        noise_sd=.15,historical_window_size=600,segment_size=384,
        selection="chronological support/validation; refit full context; no query labels",
        source_sha256=hashlib.sha256(source).hexdigest(),
        canonical_model_sha256=hashlib.sha256(model_source).hexdigest(),
        versions=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__)))
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        rows=[]
        for seed in args.seeds:rows.extend(run_seed(seed,args,out))
        save(out/"metrics.json",rows);report(out,rows,args)
    save(out/"runtime.json",dict(seconds=time.perf_counter()-start))


if __name__=="__main__":main()
