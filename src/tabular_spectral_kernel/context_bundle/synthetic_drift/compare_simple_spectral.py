"""Accuracy, trained PSD pooling, and warmed CPU latency for the small encoder."""
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
from threadpoolctl import threadpool_limits
import run as R
import run_transformer as T
from psd_spectral_attention import PSDSpectralGate
from simple_spectral_gate import SimpleSpectralGate

CONFIGS=[
    ("Transformer / joint","transformer_results","transformer","joint"),
    ("PSD month / joint","transformer_psd_norot_joint_results","psd_month","joint"),
    ("Simple / joint","simple_spectral_joint_results","simple","joint"),
    ("Transformer / KRR","transformer_krr_results","transformer","krr"),
    ("PSD rotation / KRR","transformer_psd_krr_results","psd","krr"),
    ("Simple / KRR","simple_spectral_krr_results","simple","krr"),
]


def construct(kind):
    if kind=="simple":return SimpleSpectralGate()
    if kind.startswith("psd"):return PSDSpectralGate(temporal_rotation=kind!="psd_month")
    return T.TransformerGate()


def main(root,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    summary=[];params=[];diagnostics=[];audits=[];provenance={};timings=[]
    def read(path):
        provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    reference=read(root/"transformer_results/protocol.json")
    for label,folder,kind,loss in CONFIGS:
        source=root/folder;protocol=read(source/"protocol.json")
        for key in ["seeds","profile","train_episodes","validation_episodes","queries_per_episode","max_epochs","batch","patience","optimizer"]:
            assert protocol[key]==reference[key],(label,key)
        assert ("KRR squared query error only" in protocol["objective"])==(loss=="krr")
        for path,digest in read(source/"provenance.json").items():runtime.verify_artifact(path,digest)
        summary.extend([dict(r,method=label) for r in read(source/"summary.json") if r["method"]=="transformer"])
        diagnostics.extend([dict(r,method=label) for r in read(source/"diagnostics.json")])
        for seed in [0,1,2]:
            trace=read(source/f"seed{seed}/transformer_training.json")
            params.append(dict(method=label,seed=seed,parameters=trace["parameters"],best_epoch=trace["best_epoch"]+1,
                epochs_run=len(trace["trace"]),best_validation_loss=trace["best_validation_loss"]))
    for seed in [0,1,2]:
        source=root/f"unemployment_results/seed{seed}";banks=[]
        for name in ["normal","downturn"]:
            model=R.base.CanonicalSpectral(5,4,max_frequency=3.,device="cpu")
            model.load_state_dict(torch.load(source/f"{name}.pt",weights_only=True,map_location="cpu"));model.eval();banks.append(model)
        info=read(source/"original_training.json")
        for scenario in T.C.SCENARIOS:
            with np.load(source/f"original_{scenario}.npz") as f:x,y=f["x"],f["y"]
            for origin in [128,1888]:
                xx=x[origin-128:origin+32];g,q=T.blocks(banks,info["temperatures"],xx)
                zs=[R.embedding(m,xx) for m in banks]
                qq=[R.kernel(z[128:],z[128:],t) for z,t in zip(zs,info["temperatures"])]
                ct,qt=T.tokens(xx[:128],y[origin-128:origin],xx[128:])
                for label,folder,kind,loss in CONFIGS:
                    if kind!="simple":continue
                    model=construct(kind).double().eval()
                    path=root/folder/f"seed{seed}/transformer.pt";provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                    model.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"))
                    with torch.no_grad():
                        beta,a,aq,dc,dq=model.attention_blocks(torch.tensor(ct[None]),torch.tensor(np.asarray(g)[None]),torch.tensor(np.asarray(q)[None]))
                        aq_q=torch.einsum('bj,bjst->bst',beta,torch.tensor(np.asarray(qq)[None]))/torch.sqrt(dq[...,None]*dq[...,None,:])
                        joint=torch.cat([torch.cat([a,aq.transpose(-1,-2)],-1),torch.cat([aq,aq_q],-1)],-2)
                        asymmetry=float((joint-joint.transpose(-1,-2)).abs().max());eig=float(torch.linalg.eigvalsh(joint).min())
                    assert eig>-1e-9 and asymmetry<1e-10
                    audits.append(dict(method=label,seed=seed,scenario=scenario,origin=origin,min_eigenvalue=eig,max_asymmetry=asymmetry))
        if seed!=0:continue
        # Timings are measured after all training jobs finish. Frozen kernel
        # construction is common and excluded; the simple gate reuses blocks.
        with np.load(source/"original_localized_downturn.npz") as f:x,y=f["x"],f["y"]
        for origin in [128,896,1888]:
            xx=x[origin-128:origin+32];g,q=T.blocks(banks,info["temperatures"],xx)
            ct,qt=T.tokens(xx[:128],y[origin-128:origin],xx[128:])
            b={k:torch.tensor(np.asarray(v)[None],dtype=torch.float32 if k in ["context","query","descriptor"] else torch.float64)
               for k,v in dict(context=ct,query=qt,descriptor=T.M.descriptor(g,y[origin-128:origin]),grams=g,cross=q,y=y[origin-128:origin]).items()}
            for label,folder,kind,loss in CONFIGS:
                model=construct(kind).eval();model.load_state_dict(torch.load(root/folder/"seed0/transformer.pt",weights_only=True,map_location="cpu"))
                with torch.no_grad():
                    for _ in range(5):
                        a=T.gate_weights(model,b);T.decode(a,b['grams'],b['cross'],b['y'],info['ridge'])
                    encoder=[];full=[]
                    for _ in range(20):
                        start=time.perf_counter();a=T.gate_weights(model,b);encoder.append(time.perf_counter()-start)
                        T.decode(a,b['grams'],b['cross'],b['y'],info['ridge']);full.append(time.perf_counter()-start)
                timings.append(dict(method=label,origin=origin,encoder_ms=1000*float(np.median(encoder)),
                    encoder_and_both_readouts_ms=1000*float(np.median(full)),queries=32))
    for name,obj in [("summary",summary),("training",params),("diagnostics",diagnostics),("psd_audit",audits),("latency",timings),("provenance",provenance)]:
        (out/(name+".json")).write_text(json.dumps(obj,indent=2)+"\n")
    (out/"protocol.json").write_text(json.dumps(dict(configs=CONFIGS,
        comparison="same seeds, episodes, optimizer and validation protocol within each loss; selected prior larger-PSD references use month-only for joint and rotations for KRR",
        timing="one CPU thread; seed0 localized scenario, three origins, 5 warmups and20 timed repetitions each; median per origin; kernel construction and data loading excluded; 32 queries; both readouts share alpha",
        qualification="Exploratory three-seed comparison on previously inspected scenarios; encoder counts exclude shared frozen spectral banks; parameter reduction does not remove KRR linear solves"),indent=2)+"\n")
    (out/"compare_simple_spectral_snapshot.py").write_bytes(Path(__file__).read_bytes())
    fig,axs=plt.subplots(1,2,figsize=(13,4.5),layout="constrained")
    for ax,loss in zip(axs,["joint","krr"]):
        configs=[c for c in CONFIGS if c[3]==loss]
        for j,(label,_,_,_) in enumerate(configs):
            rr=[next(r for r in summary if (r['method'],r['decoder'],r['scenario'])==(label,'krr',sc)) for sc in T.C.SCENARIOS]
            ax.bar(np.arange(4)+(j-1)*.24,[100*r['rmse_mean'] for r in rr],width=.24,label=label,
                yerr=[100*r['rmse_seed_sd'] for r in rr],capsize=2)
        ax.set_xticks(np.arange(4),['Recurring','Gradual','Stable','Localized'])
        ax.set(title=loss.upper()+" training loss",ylabel="KRR RMSE (percentage points)");ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/"simple_comparison.png",dpi=180);fig.savefig(out/"simple_comparison.pdf");plt.close(fig)
    lines=[r"\begin{table}[ht]",r"\centering\small",r"\begin{tabular}{lrrrrr}\toprule",
        r"Encoder / loss & Parameters & Recurring & Gradual & Normal & Localized\\\midrule"]
    for label,_,_,_ in CONFIGS:
        count=next(r['parameters'] for r in params if r['method']==label)
        vals=[next(r['rmse_mean'] for r in summary if (r['method'],r['decoder'],r['scenario'])==(label,'krr',sc))*100 for sc in T.C.SCENARIOS]
        lines.append(label+f" & {count:,} & "+" & ".join(f"{v:.2f}" for v in vals)+r"\\")
        print(label,count,[round(v,4) for v in vals])
    lines += [r"\bottomrule\end{tabular}",r"\caption{One-layer frozen-kernel pooling versus larger encoders. Entries are three-seed mean KRR RMSE in percentage points; parameter counts exclude the shared frozen regression banks.}",r"\end{table}"]
    (out/"results_table.tex").write_text("\n".join(lines)+"\n")
    print('PSD audits passed:',len(audits))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path(__file__).parent)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'simple_spectral_comparison')
    args=p.parse_args()
    with threadpool_limits(limits=1):main(args.source,args.output)
