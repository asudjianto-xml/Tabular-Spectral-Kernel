"""Report deployable references separately from numerical hindsight/Bayes diagnostics."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import run as R
import run_transformer as T
from run_oracle_references import ROOT

LABELS={'linear_learned_psd':'Learned PSD, unaltered context','linear_transformer':'Transformer, linear head',
        'transformer_full':'Transformer, full encoder','validation':'Context validation fusion',
        'convex_mkl':'Context convex MKL','oracle_cohort_labels':'Oracle: cohort query labels',
        'oracle_query_labels':'Oracle: per-query labels','oracle_query_mean':'Oracle: per-query true mean',
        'true_conditional_mean':'True conditional mean'}


def main():
    root=ROOT/'oracle_reference_results';out=root/'comparison';out.mkdir(exist_ok=False)
    metrics=[];checks=[];hashes={}
    def read(path):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return json.loads(path.read_text())
    for seed in range(3):
        p=root/f'seed{seed}';metrics+=read(p/'metrics.json');checks+=read(p/'checks.json');read(p/'protocol.json')
        for path,digest in read(p/'provenance.json').items():runtime.verify_artifact(path,digest)
    summary=[]
    for sc in T.C.SCENARIOS:
        for method in dict.fromkeys(r['method'] for r in metrics):
            rr=[r for r in metrics if (r['method'],r['scenario'])==(method,sc)]
            summary.append(dict(method=method,scenario=sc,rmse_pp=float(np.mean([r['rmse'] for r in rr]))*100,
                rmse_seed_sd_pp=float(np.std([r['rmse'] for r in rr],ddof=1))*100,
                clean_rmse_pp=float(np.mean([r['clean_rmse'] for r in rr]))*100,
                fresh_noise_rmse_pp=float(np.mean([r['expected_fresh_noise_rmse'] for r in rr]))*100,
                expected_noise_rmse_pp=float(np.mean([r['full_state_noise_rmse'] for r in rr]))*100))
    for name,obj in [('summary',summary),('metrics',metrics),('checks',checks),('provenance',hashes)]:R.save(out/f'{name}.json',obj)
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\begin{tabular}{lrrrr}\toprule',r'Reference & Recurring & Gradual & Normal & Localized\\\midrule']
    for method,label in LABELS.items():
        vals=[next(r['rmse_pp'] for r in summary if (r['method'],r['scenario'])==(method,sc)) for sc in T.C.SCENARIOS]
        print(label,[round(v,4) for v in vals]);lines.append(label+' & '+' & '.join(f'{v:.3f}' for v in vals)+r'\\')
    vals=[next(r['expected_noise_rmse_pp'] for r in summary if (r['method'],r['scenario'])==('true_conditional_mean',sc)) for sc in T.C.SCENARIOS]
    print('Expected full-state noise RMSE',[round(v,4) for v in vals])
    lines += [r'\midrule','Expected full-state noise RMSE & '+' & '.join(f'{v:.3f}' for v in vals)+r'\\',
              r'\bottomrule\end{tabular}',r'\caption{Unaltered-context prediction is a reference, not a ceiling. Entries are three-seed mean realized RMSE in percentage points, except the final analytic-noise row. Query-label and true-mean oracles are nondeployable numerical diagnostics with fixed kernels, recent support, and ridge.}',r'\end{table}']
    (out/'results_table.tex').write_text('\n'.join(lines)+'\n')
    shown=['linear_learned_psd','validation','oracle_cohort_labels','oracle_query_labels','oracle_query_mean','true_conditional_mean']
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    sc='recurring_downturn'
    for ax,field,title in zip(axs,['rmse_pp','fresh_noise_rmse_pp'],['Realized held-out labels','Independent new noise at the same queries']):
        vals=[next(r[field] for r in summary if (r['method'],r['scenario'])==(m,sc)) for m in shown]
        ax.barh(range(len(shown)),vals,color=['#4477aa','#4477aa','#cc6677','#cc6677','#ddaa33','#228833'])
        ax.set_yticks(range(len(shown)),[LABELS[m] for m in shown],fontsize=8);ax.invert_yaxis()
        ax.set_title(title);ax.set_xlabel('RMSE (percentage points)');ax.grid(axis='x',alpha=.2)
    fig.savefig(out/'oracle_references.png',dpi=180);fig.savefig(out/'oracle_references.pdf');plt.close(fig)
    R.save(out/'protocol.json',dict(comparison='Same evaluation rows, banks, context128, fixed historical ridge; past-only optimization reused after provenance and first-origin reselection checks.',
        oracles='Numerical hindsight references, not certified global extrema or deployable predictors.',
        bayes='True conditional mean sees latent macro state; analytic noise floor is expected risk under full-state information, not an empirical lower bound.',
        table='Mean per-seed RMSE; final row is analytic noise expectation converted to RMSE per seed then averaged.'))
    (out/'compare_oracle_references_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('Direct-solve checks',len(checks),'max error',max(r['direct_solve_max_error'] for r in checks))


if __name__=='__main__':main()
