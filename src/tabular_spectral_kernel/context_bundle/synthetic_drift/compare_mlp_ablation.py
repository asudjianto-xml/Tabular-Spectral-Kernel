"""Compare nonlinear capacity and intermediate feed-forward removal."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run as R
import run_transformer as T
from mlp_ablation import construct

ROOT=Path(__file__).parent
CONFIGS=[('Simple PSD','simple_spectral_joint_results',None),
         ('PSD + large head','mlp_large_head_results','large_head'),
         ('Stacked PSD + FF','mlp_stacked_psd_results','stacked_psd'),
         ('Stacked PSD, no FF','mlp_stacked_psd_no_ff_results','stacked_psd_no_ff'),
         ('Transformer + FF','transformer_results',None),
         ('Transformer, no FF','mlp_transformer_no_ff_results','transformer_no_ff')]
PAIRS=[('PSD + large head','Simple PSD'),('Stacked PSD + FF','Simple PSD'),
       ('Stacked PSD + FF','Stacked PSD, no FF'),('Transformer + FF','Transformer, no FF')]


def main():
    out=ROOT/'mlp_ablation_comparison';out.mkdir(exist_ok=False);torch.set_num_threads(1)
    provenance={};summary=[];training=[];diagnostics=[];metrics=[];audits=[]
    def read(path):
        provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    ref=read(ROOT/'transformer_results/protocol.json')
    for label,folder,variant in CONFIGS:
        p=read(ROOT/folder/'protocol.json')
        for key in ['seeds','profile','train_episodes','validation_episodes','queries_per_episode','max_epochs','batch','patience','objective','optimizer','selection']:
            assert p[key]==ref[key],(folder,key)
        if variant:assert p['mlp_ablation']==variant
        for path,digest in read(ROOT/folder/'provenance.json').items():
            runtime.verify_artifact(path,digest)
        summary.extend(dict(r,method=label) for r in read(ROOT/folder/'summary.json') if r['method']=='transformer')
        metrics.extend(dict(r,method=label) for r in read(ROOT/folder/'metrics.json') if r['method']=='transformer')
        diagnostics.extend(dict(r,method=label) for r in read(ROOT/folder/'diagnostics.json'))
        for seed in ref['seeds']:
            r=read(ROOT/folder/f'seed{seed}/transformer_training.json')
            training.append(dict(method=label,seed=seed,parameters=r['parameters'],best_epoch=r['best_epoch']+1,
                                 epochs_run=len(r['trace']),validation_loss=r['best_validation_loss']))
    for seed in ref['seeds']:
        source=ROOT/f'unemployment_results/seed{seed}';banks=[];models=[]
        info=read(source/'original_training.json')
        for name in ['normal','downturn']:
            m=R.base.CanonicalSpectral(5,4,max_frequency=3.,device='cpu')
            m.load_state_dict(torch.load(source/f'{name}.pt',weights_only=True,map_location='cpu'));m.eval();banks.append(m)
        for label,folder,variant in CONFIGS:
            if variant not in ['large_head','stacked_psd','stacked_psd_no_ff']:continue
            m=construct(variant).double().eval();path=ROOT/folder/f'seed{seed}/transformer.pt'
            provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            m.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'));models.append((label,m))
        for sc in T.C.SCENARIOS:
            with np.load(source/f'original_{sc}.npz') as f:x,y=f['x'],f['y']
            for origin in [128,1888]:
                xx=x[origin-128:origin+32];zs=[R.embedding(m,xx) for m in banks]
                full=torch.tensor(np.array([R.kernel(z,z,t) for z,t in zip(zs,info['temperatures'])])[None])
                grams=full[:,:,:128,:128];cross=full[:,:,128:,:128]
                ct,qt=T.tokens(xx[:128],y[origin-128:origin],xx[128:]);ct=torch.tensor(ct[None]);qt=torch.tensor(qt[None])
                for label,m in models:
                    with torch.no_grad():
                        beta,a,aq,dc,dq=m.attention_blocks(ct,grams,cross)
                        k=torch.einsum('bj,bjst->bst',beta,full);d=torch.cat([dc,dq],-1)
                        normalized=k/torch.sqrt(d[...,None]*d[...,None,:])
                        eig=float(torch.linalg.eigvalsh(normalized).min());assert eig>-1e-9
                        torch.testing.assert_close(aq,normalized[:,128:,:128])
                        alpha=m(ct,qt,grams=grams,cross=cross)
                        single=m(ct,qt[:,:1],grams=grams,cross=cross[:,:,:1])
                        torch.testing.assert_close(alpha[:,:1],single)
                    audits.append(dict(method=label,seed=seed,scenario=sc,origin=origin,min_eigenvalue=eig,
                                       query_weight_sd=float(alpha[0,:,0].std())))
    paired=[]
    for left,right in PAIRS:
        for sc in T.C.SCENARIOS:
            delta=[]
            for seed in ref['seeds']:
                def get(label):return next(r['rmse'] for r in metrics if (r['method'],r['scenario'],r['seed'],r['decoder'])==(label,sc,seed,'krr'))
                delta.append(100*(get(left)-get(right)))
            paired.append(dict(left=left,right=right,scenario=sc,rmse_difference_pp_by_seed=delta,mean_difference_pp=float(np.mean(delta))))
    for name,obj in [('summary',summary),('metrics',metrics),('training',training),('diagnostics',diagnostics),('psd_audit',audits),('paired',paired),('provenance',provenance)]:R.save(out/f'{name}.json',obj)
    R.save(out/'protocol.json',dict(configs=CONFIGS,pairs=PAIRS,
        comparison='Identical episodes, three seeds, joint objective, optimizer and validation stopping. Frozen spectral banks and regression readouts unchanged.',
        architecture='Large head changes 24->16->2 to 24->128->128->2. Stacked PSD uses two width32 residual pooling/FF64 blocks and query cross-pooling, with fixed conditional PSD attention throughout. No-FF variants bypass intermediate FF and norm2, keeping shared initial weights and final nonlinear head.',
        qualification='Removal also reduces trainable capacity. Stacked-vs-simple changes depth, width, residuals and norms. Fixed-geometry stacked PSD does not reproduce learned Transformer Q/K maps. Previously inspected scenarios; exploratory and not a statistical significance claim.'))
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\begin{tabular}{lrrrrr}\toprule',r'Encoder & Parameters & Recurring & Gradual & Normal & Localized\\\midrule']
    for label,_,_ in CONFIGS:
        vals=[100*next(r['rmse_mean'] for r in summary if (r['method'],r['decoder'],r['scenario'])==(label,'krr',sc)) for sc in T.C.SCENARIOS]
        count=next(r['parameters'] for r in training if r['method']==label)
        print(label,count,[round(v,4) for v in vals])
        lines.append(label+f' & {count:,} & '+' & '.join(f'{v:.3f}' for v in vals)+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{MLP capacity and placement: three-seed mean KRR RMSE in percentage points. FF denotes the intermediate feed-forward MLPs. Counts include only trainable encoder parameters; all models retain a nonlinear final mixture head.}',r'\end{table}']
    (out/'results_table.tex').write_text('\n'.join(lines)+'\n')
    fig,axs=plt.subplots(1,4,figsize=(15,4.7),layout='constrained')
    for ax,sc,title in zip(axs,T.C.SCENARIOS,['Recurring','Gradual','Stable','Localized']):
        for j,(label,_,_) in enumerate(CONFIGS):
            r=next(r for r in summary if (r['method'],r['decoder'],r['scenario'])==(label,'krr',sc))
            ax.errorbar(j,100*r['rmse_mean'],yerr=100*r['rmse_seed_sd'],fmt='o',capsize=3)
        ax.set_xticks(range(6),[c[0] for c in CONFIGS],rotation=65,ha='right',fontsize=8);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('KRR RMSE (percentage points; seed SD)')
    fig.savefig(out/'mlp_comparison.png',dpi=180);fig.savefig(out/'mlp_comparison.pdf');plt.close(fig)
    (out/'compare_mlp_ablation_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('Trained PSD audits passed:',len(audits))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
