"""Matched normalization x Schur-power ablation; trained kernel and pooling audits."""
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
from simple_spectral_gate import SimpleSpectralGate

ROOT=Path(__file__).parent
CONFIGS=[('Symmetric, p=1','simple_spectral_joint_results','symmetric',1),
         ('Row, p=1','simple_row_p1_results','row',1),
         ('Symmetric, p=2','simple_symmetric_p2_results','symmetric',2),
         ('Row, p=2','simple_row_p2_results','row',2)]


def main():
    out=ROOT/'pool_normalization_comparison';out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    provenance={};summary=[];training=[];audits=[];diagnostics=[];metrics=[]
    def read(path):
        provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    ref=read(ROOT/CONFIGS[0][1]/'protocol.json')
    for label,folder,norm,power in CONFIGS:
        protocol=read(ROOT/folder/'protocol.json')
        for key in ['seeds','train_episodes','validation_episodes','queries_per_episode','max_epochs','batch','patience','objective','optimizer','selection','profile']:
            assert protocol[key]==ref[key],(folder,key)
        assert protocol.get('pool_normalization','symmetric')==norm
        assert protocol.get('pool_power',1)==power
        for path,digest in read(ROOT/folder/'provenance.json').items():
            runtime.verify_artifact(path,digest)
        summary.extend(dict(r,method=label) for r in read(ROOT/folder/'summary.json') if r['method']=='transformer')
        diagnostics.extend(dict(r,method=label) for r in read(ROOT/folder/'diagnostics.json'))
        metrics.extend(dict(r,method=label) for r in read(ROOT/folder/'metrics.json') if r['method']=='transformer')
        for seed in ref['seeds']:
            r=read(ROOT/folder/f'seed{seed}/transformer_training.json')
            assert r['parameters']==596
            training.append(dict(method=label,seed=seed,parameters=r['parameters'],best_epoch=r['best_epoch']+1,epochs_run=len(r['trace'])))
    summary.extend(dict(r,method='Plain Transformer') for r in read(ROOT/'transformer_results/summary.json') if r['method']=='transformer')
    for seed in ref['seeds']:
        source=ROOT/f'unemployment_results/seed{seed}';banks=[];models=[]
        info=read(source/'original_training.json')
        for name in ['normal','downturn']:
            m=R.base.CanonicalSpectral(5,4,max_frequency=3.,device='cpu')
            m.load_state_dict(torch.load(source/f'{name}.pt',weights_only=True,map_location='cpu'));m.eval();banks.append(m)
        for label,folder,norm,power in CONFIGS:
            m=SimpleSpectralGate(normalization=norm,power=power).double().eval()
            path=ROOT/folder/f'seed{seed}/transformer.pt'
            provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            m.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'));models.append(m)
        for scenario in T.C.SCENARIOS:
            with np.load(source/f'original_{scenario}.npz') as f:x,y=f['x'],f['y']
            for origin in [128,1888]:
                xx=x[origin-128:origin+32]
                full=np.array([R.kernel(R.embedding(m,xx),R.embedding(m,xx),temp) for m,temp in zip(banks,info['temperatures'])])
                full=torch.tensor(full[None]);grams=full[:,:,:128,:128];cross=full[:,:,128:,:128]
                ct,qt=T.tokens(xx[:128],y[origin-128:origin],xx[128:]);ct=torch.tensor(ct[None])
                for (label,_,norm,power),m in zip(CONFIGS,models):
                    with torch.no_grad():
                        beta,a,aq,dc,dq=m.attention_blocks(ct,grams,cross)
                        k=torch.einsum('bj,bjst->bst',beta,full).pow(power)
                        eig=float(torch.linalg.eigvalsh(k).min());assert eig>-1e-9
                        d=torch.cat([dc,dq],-1);s=k/torch.sqrt(d[...,None]*d[...,None,:])
                        seig=float(torch.linalg.eigvalsh(s).min());assert seig>-1e-9
                        row=cross.new_tensor(0.)
                        if norm=='row':
                            row=(aq.sum(-1)-1).abs().max();assert row<1e-10
                            torch.testing.assert_close(aq,torch.softmax(k[:,128:,:128].log(),-1))
                        # Concentration after making query pooling weights sum to one.
                        prob=aq/aq.sum(-1,keepdim=True)
                        entropy=-(prob*prob.clamp_min(1e-300).log()).sum(-1).mean()
                        alpha=m(ct,torch.tensor(qt[None]),grams=grams,cross=cross)
                    audits.append(dict(method=label,seed=seed,scenario=scenario,origin=origin,
                        kernel_min_eigenvalue=eig,symmetric_normalized_min_eigenvalue=seig,
                        row_sum_error=float(row),attention_asymmetry=float((a-a.transpose(-1,-2)).abs().max()),
                        normalized_pool_entropy=float(entropy),query_weight_sd=float(alpha[0,:,0].std())))
    paired=[]
    for label,_,_,_ in CONFIGS[1:]:
        for sc in T.C.SCENARIOS:
            diffs=[]
            for seed in ref['seeds']:
                def value(method):
                    return next(r['rmse'] for r in metrics if (r['method'],r['scenario'],r['seed'],r['decoder'])==(method,sc,seed,'krr'))
                diffs.append(100*(value(label)-value(CONFIGS[0][0])))
            paired.append(dict(method=label,scenario=sc,rmse_difference_pp_by_seed=diffs,mean_difference_pp=float(np.mean(diffs))))
    for name,obj in [('summary',summary),('training',training),('audit',audits),('diagnostics',diagnostics),('metrics',metrics),('paired',paired),('provenance',provenance)]:
        R.save(out/f'{name}.json',obj)
    R.save(out/'protocol.json',dict(configs=CONFIGS,loss='joint',sharpening='Schur square of fused pooling kernel; KRR/NW readout banks unchanged',
        interpretation='Row attention is generally nonsymmetric; underlying conditional kernel remains PSD. This does not isolate learned Q/K asymmetry or capacity in the larger Transformer.',
        qualification='Exploratory three-seed comparison; existing symmetric p=1 baseline reused; no evaluation-based temperature search.'))
    labels=[c[0] for c in CONFIGS]+['Plain Transformer']
    fig,axs=plt.subplots(1,4,figsize=(14,4),layout='constrained')
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\begin{tabular}{lrrrr}\toprule',r'Pooling & Recurring & Gradual & Normal & Localized\\\midrule']
    for label in labels:
        rows=[next(r for r in summary if (r['method'],r['decoder'],r['scenario'])==(label,'krr',sc)) for sc in T.C.SCENARIOS]
        vals=[r['rmse_mean']*100 for r in rows]
        print(label,[round(v,4) for v in vals])
        lines.append(label+' & '+' & '.join(f'{v:.3f}' for v in vals)+r'\\')
    for ax,sc,title in zip(axs,T.C.SCENARIOS,['Recurring','Gradual','Stable','Localized']):
        for j,label in enumerate(labels):
            r=next(r for r in summary if (r['method'],r['decoder'],r['scenario'])==(label,'krr',sc))
            ax.errorbar(j,100*r['rmse_mean'],yerr=100*r['rmse_seed_sd'],fmt='o',capsize=3)
        ax.set_xticks(range(5),labels,rotation=60,ha='right',fontsize=8);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('KRR RMSE (percentage points; seed SD)')
    fig.savefig(out/'normalization_comparison.png',dpi=180);fig.savefig(out/'normalization_comparison.pdf');plt.close(fig)
    lines += [r'\bottomrule\end{tabular}',r'\caption{Normalization and PSD-preserving sharpening: three-seed mean KRR RMSE in percentage points. All small gates have 596 parameters; the Transformer reference has 22,786. All use the joint NW/KRR training loss.}',r'\end{table}']
    (out/'results_table.tex').write_text('\n'.join(lines)+'\n')
    (out/'compare_pool_normalization_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('Trained pooling audits passed:',len(audits))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
