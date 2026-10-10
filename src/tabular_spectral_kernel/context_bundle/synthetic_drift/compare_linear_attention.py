"""Attention/kernel comparison with affine heads and no feed-forward MLPs."""
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
NEW=[('Single PSD, linear','linear_simple'),('Stacked PSD, linear','linear_stacked'),
     ('Learned PSD, linear','linear_learned_psd'),('Transformer, linear','linear_transformer')]
OLD=[('Simple PSD + MLP','simple_spectral_joint_results'),
     ('Stacked PSD, final MLP','mlp_stacked_psd_no_ff_results'),
     ('Learned PSD + MLPs','transformer_psd_norot_joint_results'),
     ('Transformer, final MLP','mlp_transformer_no_ff_results'),
     ('Transformer + all MLPs','transformer_results')]


def main():
    out=ROOT/'linear_attention_comparison';out.mkdir(exist_ok=False);torch.set_num_threads(1)
    summary=[];metrics=[];training=[];diagnostics=[];audit=[];provenance={}
    def read(path):
        provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    ref=read(ROOT/'transformer_results/protocol.json')
    configs=[(label,'mlp_'+v+'_results') for label,v in NEW]+OLD
    for label,folder in configs:
        p=read(ROOT/folder/'protocol.json')
        for k in ['seeds','profile','train_episodes','validation_episodes','queries_per_episode','max_epochs','batch','patience','objective','optimizer','selection']:assert p[k]==ref[k],(folder,k)
        for path,digest in read(ROOT/folder/'provenance.json').items():runtime.verify_artifact(path,digest)
        for filename,target in [('summary',summary),('metrics',metrics)]:
            target.extend(dict(r,method=label) for r in read(ROOT/folder/(filename+'.json')) if r['method']=='transformer')
        diagnostics.extend(dict(r,method=label) for r in read(ROOT/folder/'diagnostics.json'))
        for seed in ref['seeds']:
            r=read(ROOT/folder/f'seed{seed}/transformer_training.json')
            training.append(dict(method=label,seed=seed,parameters=r['parameters'],best_epoch=r['best_epoch']+1,epochs_run=len(r['trace'])))
    for seed in ref['seeds']:
        src=ROOT/f'unemployment_results/seed{seed}';info=read(src/'original_training.json');banks=[];models=[]
        for name in ['normal','downturn']:
            m=R.base.CanonicalSpectral(5,4,max_frequency=3.,device='cpu')
            m.load_state_dict(torch.load(src/f'{name}.pt',weights_only=True,map_location='cpu'));m.eval();banks.append(m)
        for label,v in NEW:
            path=ROOT/f'mlp_{v}_results/seed{seed}/transformer.pt';provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            m=construct(v).double().eval();m.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'));models.append((label,v,m))
        for sc in T.C.SCENARIOS:
            with np.load(src/f'original_{sc}.npz') as f:x,y=f['x'],f['y']
            for origin in [128,1888]:
                xx=x[origin-128:origin+32];zs=[R.embedding(m,xx) for m in banks]
                full=torch.tensor(np.array([R.kernel(z,z,t) for z,t in zip(zs,info['temperatures'])])[None]);g=full[:,:,:128,:128];q=full[:,:,128:,:128]
                ct,qt=T.tokens(xx[:128],y[origin-128:origin],xx[128:]);ct=torch.tensor(ct[None]);qt=torch.tensor(qt[None])
                for label,v,m in models:
                    matrices=[]
                    with torch.no_grad():
                        if getattr(m,'uses_kernel_blocks',False):
                            beta,a,aq,dc,dq=m.attention_blocks(ct,g,q)
                            k=torch.einsum('bj,bjst->bst',beta,full);d=torch.cat([dc,dq],-1)
                            matrices=[k/torch.sqrt(d[...,None]*d[...,None,:])]
                        elif v=='linear_learned_psd':
                            memory=m.input(ct);qp=m.input(qt);cp=ct[...,7]*4;posq=qt[...,7]*4
                            for layer in m.layers:
                                z=layer.norm1(memory);matrices.append(layer.attention.normalized_blocks(z,cp))
                                memory=memory+layer.attention(z,cp)
                            cc,qc,qq=m.attention.normalized_blocks(memory,cp,qp,posq)
                            matrices.append(torch.cat([torch.cat([cc,qc.transpose(-1,-2)],-1),torch.cat([qc,qq],-1)],-2))
                        b=dict(context=ct,query=qt,descriptor=None,grams=g,cross=q)
                        alpha=T.gate_weights(m,b)
                        torch.testing.assert_close(alpha[:,:1],T.gate_weights(m,dict(b,query=qt[:,:1],cross=q[:,:,:1])))
                        for layer,k in enumerate(matrices):
                            eig=float(torch.linalg.eigvalsh(k).min());sym=float((k-k.transpose(-1,-2)).abs().max())
                            assert eig>-1e-9 and sym<1e-10
                            audit.append(dict(method=label,seed=seed,scenario=sc,origin=origin,layer=layer,min_eigenvalue=eig,symmetry_error=sym))
    paired=[]
    pairs=[(NEW[0][0],OLD[0][0]),(NEW[1][0],OLD[1][0]),(NEW[2][0],OLD[2][0]),(NEW[3][0],OLD[3][0]),(NEW[2][0],NEW[3][0]),(NEW[1][0],NEW[3][0])]
    for left,right in pairs:
        for sc in T.C.SCENARIOS:
            def get(label,seed):return next(r['rmse'] for r in metrics if (r['method'],r['scenario'],r['seed'],r['decoder'])==(label,sc,seed,'krr'))
            d=[100*(get(left,s)-get(right,s)) for s in ref['seeds']]
            paired.append(dict(left=left,right=right,scenario=sc,difference_pp_by_seed=d,mean_difference_pp=float(np.mean(d))))
    for name,obj in [('summary',summary),('metrics',metrics),('training',training),('diagnostics',diagnostics),('psd_audit',audit),('paired',paired),('provenance',provenance)]:R.save(out/f'{name}.json',obj)
    R.save(out/'protocol.json',dict(configs=configs,head='Single affine map to 2 logits, then simplex softmax; no feed-forward MLPs or GELU on executed paths',
        remaining_nonlinearity='Attention/kernel computation, context mixture softmax, layer normalization where present, final softmax and KRR solve',
        qualification='Not fully linear models; architecture, kernel learning and parameter counts differ. Learned PSD removes both intermediate and final MLPs versus old reference; simple also removes value GELU. Three-seed exploratory matched-budget evaluation.'))
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\begin{tabular}{lrrrrr}\toprule',r'Encoder & Parameters & Recurring & Gradual & Normal & Localized\\\midrule']
    for label,_ in configs:
        vals=[100*next(r['rmse_mean'] for r in summary if (r['method'],r['decoder'],r['scenario'])==(label,'krr',sc)) for sc in T.C.SCENARIOS]
        count=next(r['parameters'] for r in training if r['method']==label)
        print(label,count,[round(v,4) for v in vals]);lines.append(label+f' & {count:,} & '+' & '.join(f'{v:.3f}' for v in vals)+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Attention and kernel encoders with affine mixture heads and no feed-forward MLPs (first four rows), with earlier nonlinear references. Three-seed mean KRR RMSE in percentage points.}',r'\end{table}']
    (out/'results_table.tex').write_text('\n'.join(lines)+'\n')
    fig,axs=plt.subplots(1,4,figsize=(14,4.5),layout='constrained')
    for ax,sc,title in zip(axs,T.C.SCENARIOS,['Recurring','Gradual','Stable','Localized']):
        for j,(label,_) in enumerate(NEW):
            r=next(r for r in summary if (r['method'],r['decoder'],r['scenario'])==(label,'krr',sc))
            ax.errorbar(j,100*r['rmse_mean'],yerr=100*r['rmse_seed_sd'],fmt='o',capsize=3)
        ax.set_xticks(range(4),[c[0] for c in NEW],rotation=55,ha='right',fontsize=8);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('KRR RMSE (percentage points; seed SD)')
    fig.savefig(out/'linear_attention.png',dpi=180);fig.savefig(out/'linear_attention.pdf');plt.close(fig)
    (out/'compare_linear_attention_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('PSD audits:',len(audit))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
