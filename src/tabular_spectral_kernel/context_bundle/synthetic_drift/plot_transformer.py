"""Plot the completed Transformer ablation without selecting models."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main(folder,rope_folder=None):
    summary=json.loads((folder/"summary.json").read_text())
    scenarios=["recurring_downturn","gradual_downturn","stable_economy","localized_downturn"]
    methods=["constant","summary","transformer"]
    colors={"constant":"#777777","summary":"#d98b22","transformer":"#236bb5"}
    if rope_folder is not None:
        rope=json.loads((rope_folder/"summary.json").read_text())
        summary.extend([dict(r,method="transformer_rope") for r in rope if r["method"]=="transformer"])
        methods.append("transformer_rope");colors["transformer_rope"]="#23956a"
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for ax,decoder in zip(axs[0],["krr","nw"]):
        for j,method in enumerate(methods):
            rows=[next(r for r in summary if (r["scenario"],r["method"],r["decoder"])==(s,method,decoder)) for s in scenarios]
            width=.8/len(methods)
            ax.bar(np.arange(4)+(j-(len(methods)-1)/2)*width,[100*r["rmse_mean"] for r in rows],width=width,
                   yerr=[100*r["rmse_seed_sd"] for r in rows],capsize=2,label=method,color=colors[method])
        ax.set_xticks(np.arange(4),["Recurring","Gradual","Stable","Localized"])
        ax.set_ylabel("RMSE (percentage points)");ax.set_title(decoder.upper()+": mean ± seed SD")
        ax.grid(axis="y",alpha=.2)
    axs[0,0].legend()
    for method in methods:
        source=rope_folder if method=="transformer_rope" else folder
        key="transformer" if method=="transformer_rope" else method
        d=json.loads((source/"seed0"/(key+"_training.json")).read_text())
        axs[1,0].plot([r["epoch"]+1 for r in d["trace"]],[r["validation_loss"] for r in d["trace"]],label=method,color=colors[method])
        axs[1,0].scatter([d["best_epoch"]+1],[d["best_validation_loss"]],color=colors[method],s=25)
    axs[1,0].set(title="Independent validation loss (seed 0)",xlabel="Epoch",ylabel="Joint NW/KRR MSE")
    for method in methods:
        source=rope_folder if method=="transformer_rope" else folder
        key="transformer" if method=="transformer_rope" else method
        with np.load(source/"seed0/localized_downturn.npz") as data:
            w=data[key+"_alpha"][128:,0].reshape(-1,32)
        months=np.arange(4,4+len(w))
        axs[1,1].plot(months,w.mean(1),label=method,color=colors[method])
        if method.startswith("transformer"):
            axs[1,1].fill_between(months,np.quantile(w,.25,axis=1),np.quantile(w,.75,axis=1),alpha=.15,color=colors[method])
    axs[1,1].set(title="Localized drift: normal-bank weight (seed 0)",xlabel="Month",ylabel="Mean weight; Transformer IQR",ylim=(-.03,1.03))
    for ax in axs[1]:ax.grid(alpha=.2)
    fig.suptitle("Frozen spectral banks; matched full-context meta-training; original bandwidths")
    fig.savefig(folder/"transformer_comparison.png",dpi=180)
    fig.savefig(folder/"transformer_comparison.pdf")
    plt.close(fig)
    (folder/"combined_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=[r"\begin{table}[ht]",r"\centering\small",r"\begin{tabular}{llrrrr}\toprule",
           r"Readout & Encoder & Recurring & Gradual & Normal macro & Localized\\\midrule"]
    names={"constant":"Constant","summary":"Summary MLP","transformer":"Transformer","transformer_rope":"Transformer + RoPE"}
    for dec in ["krr","nw"]:
        for method in methods:
            vals=[next(r["rmse_mean"] for r in summary if (r["scenario"],r["method"],r["decoder"])==(s,method,dec))*100 for s in scenarios]
            lines.append(dec.upper()+" & "+names[method]+" & "+" & ".join(f"{v:.2f}" for v in vals)+r"\\")
        if dec=="krr":lines.append(r"\midrule")
    lines += [r"\bottomrule\end{tabular}",r"\caption{Matched full-context meta-training: mean RMSE in percentage points over three seeds. Both readouts use the original bandwidth profile.}",r"\end{table}"]
    (folder/"results_table.tex").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--folder",type=Path,default=Path(__file__).parent/"transformer_results")
    p.add_argument("--rope-folder",type=Path)
    args=p.parse_args();main(args.folder,args.rope_folder)
