"""Aggregate paired frozen-model context interventions without shuffle ensembling."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import run as R
from context_intervention import ROOT,VARIANTS
import run_transformer as T

LABELS=dict(zip(VARIANTS,['Single PSD','Stacked PSD','Learned PSD','Transformer']))
CONDITIONS=['correct','shuffle_gate','labels_masked_gate','wrong_regime_gate','query_only_gate','shuffle_all','wrong_regime_all','historical_only']


def main():
    root=ROOT/'context_intervention_results';out=root/'comparison';out.mkdir(exist_ok=False)
    rows=[];provenance={}
    def read(path):
        provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    for seed in range(3):
        p=root/f'seed{seed}';raw=read(p/'metrics.json');read(p/'protocol.json')
        for path,digest in read(p/'provenance.json').items():runtime.verify_artifact(path,digest)
        for v in VARIANTS:
            for sc in T.C.SCENARIOS:
                for c in CONDITIONS:
                    records=[r for r in raw if r['variant']==v and r['scenario']==sc and (r['condition']==c or r['condition'].startswith(c+'_'))]
                    assert len(records)==(3 if c.startswith('shuffle') else 1)
                    r=dict(seed=seed,variant=v,scenario=sc,condition=c)
                    for k in ['rmse','clean_rmse','weight_abs_change','prediction_rms_change']:r[k]=float(np.mean([s[k] for s in records]))
                    rows.append(r)
    summary=[]
    for v in VARIANTS:
        for sc in T.C.SCENARIOS:
            for c in CONDITIONS:
                rr=[r for r in rows if (r['variant'],r['scenario'],r['condition'])==(v,sc,c)]
                baselines=[next(r['rmse'] for r in rows if (r['variant'],r['scenario'],r['condition'],r['seed'])==(v,sc,'correct',seed)) for seed in range(3)]
                delta=[100*(r['rmse']-base) for r,base in zip(rr,baselines)]
                summary.append(dict(variant=v,scenario=sc,condition=c,rmse_pp=float(np.mean([r['rmse'] for r in rr]))*100,
                    rmse_seed_sd_pp=float(np.std([r['rmse'] for r in rr],ddof=1))*100,
                    delta_rmse_pp=float(np.mean(delta)),delta_by_seed_pp=delta,
                    mean_absolute_normal_weight_change=float(np.mean([r['weight_abs_change'] for r in rr])),
                    prediction_rms_change_pp=float(np.mean([r['prediction_rms_change'] for r in rr]))*100))
    for name,obj in [('per_seed',rows),('summary',summary),('provenance',provenance)]:R.save(out/f'{name}.json',obj)
    R.save(out/'protocol.json',dict(aggregation='Average each shuffle replicate metric within seed, then mean across seeds; never average shuffled predictions into an ensemble.',
        gate_only='Alter encoder context; retain original recent128 KRR Gram/cross/labels.',
        joint='Alter both encoder and KRR context. Historical-only uses query-only gate and fixed pre-evaluation support.',
        caveats=['Wrong-regime and missing-context interventions are synthetic/out of distribution.',
                 'Sensitivity does not establish useful labeled-context adaptation; evaluate error direction.',
                 'No retraining, no optimal no-context baseline, no statistical significance claims.']))
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,sc in zip(axs,['recurring_downturn','localized_downturn']):
        vals=np.array([[next(r['delta_rmse_pp'] for r in summary if (r['variant'],r['scenario'],r['condition'])==(v,sc,c)) for c in CONDITIONS[1:5]] for v in VARIANTS])
        im=ax.imshow(vals,cmap='coolwarm',vmin=-max(abs(vals).max(),.001),vmax=max(abs(vals).max(),.001))
        for (i,j),val in np.ndenumerate(vals):ax.text(j,i,f'{val:+.3f}',ha='center',va='center',fontsize=10)
        ax.set_xticks(range(4),['Shuffle labels','Mask labels','Wrong regime','Query-only'],rotation=30,ha='right')
        ax.set_yticks(range(4),[LABELS[v] for v in VARIANTS]);ax.set_title(sc.replace('_',' ').title())
        fig.colorbar(im,ax=ax,label='KRR RMSE change (pp); positive = worse')
    fig.savefig(out/'encoder_interventions.png',dpi=180);fig.savefig(out/'encoder_interventions.pdf');plt.close(fig)
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\begin{tabular}{lrrrr}\toprule',r'Intervention & Single PSD & Stacked PSD & Learned PSD & Transformer\\\midrule']
    for c in CONDITIONS:
        vals=[next(r['rmse_pp'] for r in summary if (r['variant'],r['scenario'],r['condition'])==(v,'recurring_downturn',c)) for v in VARIANTS]
        name='unaltered recent context' if c=='correct' else c.replace('_',' ')
        lines.append(name+' & '+' & '.join(f'{v:.3f}' for v in vals)+r'\\');print(c,[round(v,4) for v in vals])
    lines += [r'\bottomrule\end{tabular}',r'\caption{Frozen-model context interventions on recurring drift: three-seed mean KRR RMSE in percentage points. Gate-only interventions retain the correct recent KRR support; rows labeled all change both encoder and readout support. Historical-only uses no recent labels and is a distinct baseline.}',r'\end{table}']
    (out/'results_table.tex').write_text('\n'.join(lines)+'\n')
    (out/'compare_context_intervention_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
