"""Predeclared temporal-rotation x loss ablation with plain-Transformer controls."""
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
from psd_spectral_attention import PSDSpectralGate,SpectralKernel

CONFIGS=[
    ("Transformer / joint","transformer_results",False,False,"joint"),
    ("Transformer / KRR","transformer_krr_results",False,False,"krr"),
    ("PSD rotation / joint","transformer_psd_results",True,True,"joint"),
    ("PSD rotation / KRR","transformer_psd_krr_results",True,True,"krr"),
    ("PSD month only / joint","transformer_psd_norot_joint_results",True,False,"joint"),
    ("PSD month only / KRR","transformer_psd_norot_krr_results",True,False,"krr"),
]


def main(root,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    summary=[];metrics=[];training=[];sensitivity=[];audits=[];provenance={}
    ref=json.loads((root/"transformer_results/protocol.json").read_text())
    def read_json(p):
        provenance[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    for label,folder,psd,rotary,loss in CONFIGS:
        source=root/folder;protocol=read_json(source/"protocol.json")
        for key in ["seeds","profile","train_episodes","validation_episodes","queries_per_episode","max_epochs","batch","patience","optimizer"]:
            assert protocol[key]==ref[key],(label,key)
        assert ("KRR squared query error only" in protocol["objective"])==(loss=="krr")
        if psd:assert protocol.get("spectral_temporal_rotation",True)==rotary
        for filename,digest in read_json(source/"provenance.json").items():
            runtime.verify_artifact(filename,digest)
        for r in read_json(source/"summary.json"):
            if r["method"]=="transformer":summary.append(dict(r,method=label))
        for r in read_json(source/"metrics.json"):
            if r["method"]=="transformer":metrics.append(dict(r,method=label))
        for r in read_json(source/"diagnostics.json"):
            sensitivity.append(dict(method=label,seed=r["seed"],scenario=r["scenario"],
                within_cohort_query_sd=r["mean_within_cohort_query_sd"],label_shuffle_change=r["encoder_label_shuffle_mean_weight_change"],
                weights=r["weights"]["transformer"]))
        for seed in [0,1,2]:
            trace=read_json(source/f"seed{seed}/transformer_training.json")
            training.append(dict(method=label,seed=seed,parameters=trace["parameters"],best_epoch=trace["best_epoch"]+1,
                epochs_run=len(trace["trace"]),validation_loss=trace["best_validation_loss"]))
            if not psd:continue
            gate=PSDSpectralGate(temporal_rotation=rotary).double().eval()
            path=source/f"seed{seed}/transformer.pt";provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            gate.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"))
            if not rotary:
                for m in gate.modules():
                    if isinstance(m,SpectralKernel):assert torch.count_nonzero(m.time_frequency)==0 and not m.time_frequency.requires_grad
            for scenario in T.C.SCENARIOS:
                path=root/f"unemployment_results/seed{seed}/original_{scenario}.npz"
                with np.load(path) as f:x,y=f["x"],f["y"]
                for origin in [128,1888]:
                    c,q=T.tokens(x[origin-128:origin],y[origin-128:origin],x[origin:origin+32])
                    c=torch.tensor(c[None],dtype=torch.float64);q=torch.tensor(q[None],dtype=torch.float64)
                    cp=c[...,7]*4;qp=q[...,7]*4
                    with torch.no_grad():
                        z=gate.input(c);mins=[]
                        for layer in gate.layers:
                            a=layer.attention.normalized_blocks(layer.norm1(z),cp)
                            torch.testing.assert_close(a,a.transpose(-1,-2),atol=1e-10,rtol=1e-10)
                            mins.append(float(torch.linalg.eigvalsh(a).min()));z=layer(z,cp)
                        cc,qc,qq=gate.attention.normalized_blocks(z,cp,gate.input(q),qp)
                        joint=torch.cat([torch.cat([cc,qc.transpose(-1,-2)],-1),torch.cat([qc,qq],-1)],-2)
                        torch.testing.assert_close(joint,joint.transpose(-1,-2),atol=1e-10,rtol=1e-10)
                        mins.append(float(torch.linalg.eigvalsh(joint).min()))
                    assert min(mins)>-1e-9
                    audits.append(dict(method=label,seed=seed,scenario=scenario,origin=origin,min_eigenvalues=mins))
    # Explicit paired comparisons; positive means the first configuration wins.
    pairs=[(CONFIGS[1][0],CONFIGS[0][0]),(CONFIGS[3][0],CONFIGS[2][0]),(CONFIGS[5][0],CONFIGS[4][0]),
           (CONFIGS[4][0],CONFIGS[2][0]),(CONFIGS[5][0],CONFIGS[3][0]),(CONFIGS[5][0],CONFIGS[1][0])]
    paired=[]
    for better,reference in pairs:
        for scenario in T.C.SCENARIOS:
            delta=[]
            for seed in [0,1,2]:
                def rmse(method):return next(r["rmse"] for r in metrics if (r["method"],r["scenario"],r["seed"],r["decoder"])==(method,scenario,seed,"krr"))
                delta.append(100*(rmse(reference)-rmse(better)))
            paired.append(dict(candidate=better,reference=reference,scenario=scenario,improvement_seed_pp=delta,improvement_mean_pp=float(np.mean(delta))))
    for name,obj in [("summary",summary),("metrics",metrics),("training",training),("sensitivity",sensitivity),("psd_audit",audits),("paired",paired),("provenance",provenance)]:
        (out/(name+".json")).write_text(json.dumps(obj,indent=2)+"\n")
    (out/"compare_attention_ablation_snapshot.py").write_bytes(Path(__file__).read_bytes())
    labels=[c[0] for c in CONFIGS]
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout="constrained")
    for ax,dec in zip(axs,["krr","nw"]):
        for j,label in enumerate(labels):
            rr=[next(r for r in summary if (r["method"],r["decoder"],r["scenario"])==(label,dec,sc)) for sc in T.C.SCENARIOS]
            ax.bar(np.arange(4)+(j-2.5)*.13,[100*r["rmse_mean"] for r in rr],width=.13,label=label,
                yerr=[100*r["rmse_seed_sd"] for r in rr],capsize=2)
        ax.set_xticks(np.arange(4),["Recurring","Gradual","Stable","Localized"])
        ax.set(title=dec.upper(),ylabel="RMSE (percentage points)");ax.grid(axis="y",alpha=.2)
    axs[0].legend(fontsize=8)
    fig.suptitle("Temporal rotations and training loss: identical episodes, three seeds")
    fig.savefig(out/"attention_ablation.png",dpi=180);fig.savefig(out/"attention_ablation.pdf");plt.close(fig)
    lines=[r"\begin{table}[ht]",r"\centering\small",r"\begin{tabular}{lrrrr}\toprule",
        r"Encoder / training loss & Recurring & Gradual & Normal macro & Localized\\\midrule"]
    for label in labels:
        vals=[next(r["rmse_mean"] for r in summary if (r["method"],r["decoder"],r["scenario"])==(label,"krr",sc))*100 for sc in T.C.SCENARIOS]
        lines.append(label+" & "+" & ".join(f"{v:.2f}" for v in vals)+r"\\")
        print(label,[round(v,4) for v in vals])
    lines += [r"\bottomrule\end{tabular}",r"\caption{KRR RMSE in percentage points, averaged over three seeds. All encoders receive relative month. Joint loss averages NW and KRR errors; KRR-only loss changes both training and validation checkpoint selection.}",r"\end{table}"]
    (out/"results_table.tex").write_text("\n".join(lines)+"\n")
    print("PSD matrices checked:",len(audits)*3)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,default=Path(__file__).parent)
    p.add_argument("--output",type=Path,default=Path(__file__).parent/"attention_ablation_results")
    args=p.parse_args();main(args.source,args.output)
