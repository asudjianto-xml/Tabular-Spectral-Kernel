"""Compare completed attention experiments and audit trained PSD operators."""
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
import run_transformer as T
from psd_spectral_attention import PSDSpectralGate


def main(root):
    out=root/"transformer_psd_results";torch.set_num_threads(1)
    folders={"Transformer":"transformer_results","RoPE":"transformer_rope_results","PSD spectral":"transformer_psd_results"}
    scenarios=T.C.SCENARIOS;summary=[];training=[];audit=[];paired=[]
    common_keys=["seeds","profile","train_episodes","validation_episodes","queries_per_episode","max_epochs","batch","patience","objective","optimizer","selection"]
    reference=json.loads((root/"transformer_results/protocol.json").read_text())
    for method,folder in folders.items():
        source=root/folder;protocol=json.loads((source/"protocol.json").read_text())
        for key in common_keys:assert protocol[key]==reference[key],(folder,key)
        for filename,digest in json.loads((source/"provenance.json").read_text()).items():
            runtime.verify_artifact(filename,digest)
        records=json.loads((source/"summary.json").read_text())
        summary.extend([dict(r,method=method) for r in records if r["method"]=="transformer"])
        for seed in [0,1,2]:
            d=json.loads((source/f"seed{seed}/transformer_training.json").read_text())
            training.append(dict(method=method,seed=seed,parameters=d["parameters"],best_epoch=d["best_epoch"]+1,
                epochs_run=len(d["trace"]),best_validation_loss=d["best_validation_loss"]))
    latest=json.loads((root/"latest_kernel_comparison/summary.json").read_text())
    summary.extend([dict(r,method="Latest kernel") for r in latest if r["method"]=="latest"])
    methods=["Latest kernel",*folders]
    for seed in [0,1,2]:
        gate=PSDSpectralGate().double().eval()
        gate.load_state_dict(torch.load(out/f"seed{seed}/transformer.pt",weights_only=True,map_location="cpu"))
        for scenario in scenarios:
            with np.load(root/f"unemployment_results/seed{seed}/original_{scenario}.npz") as f:x=f["x"];y=f["y"]
            with np.load(out/f"seed{seed}/{scenario}.npz") as f:
                for key in f.files:assert np.isfinite(f[key][128:]).all()
                np.testing.assert_allclose(f["transformer_alpha"][128:].sum(1),1,atol=1e-6)
                for other,folder in folders.items():
                    if other=="PSD spectral":continue
                    with np.load(root/f"{folder}/seed{seed}/{scenario}.npz") as baseline:
                        for dec in ["krr","nw"]:
                            psd_rmse=np.sqrt(np.mean((f["transformer_"+dec][128:]-y[128:])**2))
                            other_rmse=np.sqrt(np.mean((baseline["transformer_"+dec][128:]-y[128:])**2))
                            paired.append(dict(seed=seed,scenario=scenario,decoder=dec,comparison=other,
                                psd_rmse=float(psd_rmse),other_rmse=float(other_rmse),psd_improvement_pp=float(100*(other_rmse-psd_rmse))))
            # Check trained self-attention and the joint context/query extension
            # at the initial and final evaluation origins, for every scenario.
            for origin in [128,1888]:
                c,q=T.tokens(x[origin-128:origin],y[origin-128:origin],x[origin:origin+32])
                c=torch.tensor(c[None],dtype=torch.float64);q=torch.tensor(q[None],dtype=torch.float64)
                cp=c[...,7]*4;qp=q[...,7]*4
                with torch.no_grad():
                    memory=gate.input(c);mins=[];symmetry=[]
                    for layer in gate.layers:
                        a=layer.attention.normalized_blocks(layer.norm1(memory),cp)
                        mins.append(float(torch.linalg.eigvalsh(a).min()));symmetry.append(float((a-a.transpose(-1,-2)).abs().max()))
                        memory=layer(memory,cp)
                    cc,qc,qq=gate.attention.normalized_blocks(memory,cp,gate.input(q),qp)
                    joint=torch.cat([torch.cat([cc,qc.transpose(-1,-2)],-1),torch.cat([qc,qq],-1)],-2)
                    mins.append(float(torch.linalg.eigvalsh(joint).min()));symmetry.append(float((joint-joint.transpose(-1,-2)).abs().max()))
                assert min(mins)>-1e-9 and max(symmetry)<1e-10
                audit.append(dict(seed=seed,scenario=scenario,origin=origin,min_eigenvalues=mins,max_asymmetry=max(symmetry)))
    for name,value in [("comparison_summary",summary),("comparison_training",training),("paired_comparison",paired),("trained_psd_audit",audit)]:
        (out/(name+".json")).write_text(json.dumps(value,indent=2)+"\n")
    colors={"Latest kernel":"#777777","Transformer":"#236bb5","RoPE":"#23956a","PSD spectral":"#a04aae"}
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for ax,dec in zip(axs[0],["krr","nw"]):
        for j,method in enumerate(methods):
            rr=[next(r for r in summary if (r["scenario"],r["method"],r["decoder"])==(sc,method,dec)) for sc in scenarios]
            ax.bar(np.arange(4)+(j-1.5)*.2,[100*r["rmse_mean"] for r in rr],width=.2,label=method,color=colors[method],
                yerr=[100*r["rmse_seed_sd"] for r in rr],capsize=2)
        ax.set_xticks(np.arange(4),["Recurring","Gradual","Stable","Localized"])
        ax.set(title=dec.upper(),ylabel="RMSE (percentage points)");ax.grid(axis="y",alpha=.2)
    axs[0,0].legend()
    for method,folder in folders.items():
        d=json.loads((root/folder/"seed0/transformer_training.json").read_text())
        axs[1,0].plot([r["epoch"]+1 for r in d["trace"]],[r["validation_loss"] for r in d["trace"]],label=method,color=colors[method])
        axs[1,0].scatter([d["best_epoch"]+1],[d["best_validation_loss"]],s=25,color=colors[method])
        with np.load(root/folder/"seed0/localized_downturn.npz") as f:w=f["transformer_alpha"][128:,0].reshape(-1,32)
        months=np.arange(4,60)
        axs[1,1].plot(months,w.mean(1),color=colors[method],label=method)
        axs[1,1].fill_between(months,np.quantile(w,.25,axis=1),np.quantile(w,.75,axis=1),color=colors[method],alpha=.15)
    axs[1,0].set(title="Validation loss, seed 0",xlabel="Epoch",ylabel="Joint NW/KRR MSE")
    axs[1,1].set(title="Localized drift: fusion weights, seed 0",xlabel="Month",ylabel="Normal-bank weight: mean and IQR",ylim=(-.03,1.03))
    for ax in axs[1]:ax.grid(alpha=.2)
    fig.suptitle("Same meta-training protocol and frozen regression banks; three seeds")
    fig.savefig(out/"psd_comparison.png",dpi=180);fig.savefig(out/"psd_comparison.pdf");plt.close(fig)
    lines=[r"\begin{table}[ht]",r"\centering\small",r"\begin{tabular}{llrrrr}\toprule",
           r"Readout & Encoder & Recurring & Gradual & Normal macro & Localized\\\midrule"]
    for dec in ["krr","nw"]:
        for method in methods:
            vals=[next(r["rmse_mean"] for r in summary if (r["scenario"],r["method"],r["decoder"])==(s,method,dec))*100 for s in scenarios]
            lines.append(dec.upper()+" & "+method+" & "+" & ".join(f"{v:.2f}" for v in vals)+r"\\")
        if dec=="krr":lines.append(r"\midrule")
    lines += [r"\bottomrule\end{tabular}",r"\caption{PSD spectral attention versus Transformer attention: mean RMSE in percentage points over three seeds, original bandwidth profile for both readouts.}",r"\end{table}"]
    (out/"results_table.tex").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary,indent=2));print("Trained PSD audit passed:",len(audit),"context/query sets")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,default=Path(__file__).parent)
    main(p.parse_args().source)
