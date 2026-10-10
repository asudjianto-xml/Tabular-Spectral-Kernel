"""Report fresh-test accuracy and validation-selected sparsity."""
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
from run_selector_penalties import ROOT,PENALTIES
from run_context_selector import SCENARIOS

METHODS=['latest','macro_nearest','original','augmented_adam_l2','l2','l1','elastic','group','validation_selected']
LABELS=['Latest','Unemployment only','Original selector','Added distance / Adam L2','Matched proximal L2','L1','Elastic net','Group penalty','Validation choice']


def main():
    root=ROOT/'selector_penalty_results';out=root/'comparison';out.mkdir(exist_ok=False)
    rows=[];sparsity=[];choices=[];hashes={};paired=[]
    def read(p):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
    for seed in range(3):
        src=root/f'seed{seed}';rows+=read(src/'metrics.json');p=read(src/'protocol.json');sel=read(src/'selection.json')
        choices.append(dict(seed=seed,winner=sel['winner'],validation=sel['validation']))
        for file,digest in read(src/'provenance.json').items():runtime.verify_artifact(file,digest)
        for kind in PENALTIES:
            s=sel['selected'][kind]
            sparsity.append(dict(seed=seed,method=kind,strength=s['strength'],nonzero=s['nonzero'],active_features=s['active_features'],
                best_epoch=s['best_epoch'],epochs_run=len(s['trace']),group_norms=s['group_norms'],
                retained_groups=[g for g,v in s['group_norms'].items() if v>1e-8],
                retained_features=[k for k,v in s['coefficients'].items() if abs(v)>1e-8],
                absolute_distance_coefficient=s['coefficients']['absolute_macro_distance']))
        for sc in SCENARIOS:
            path=src/f'{sc}.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as d:
                for method in METHODS:
                    pred=np.take_along_axis(d['candidate_predictions'],d[method+'_choice'][...,None],-1)[...,0]
                    np.testing.assert_array_equal(pred,d[method])
                np.testing.assert_array_equal(d['validation_selected'],d[sel['winner']])
                for method in METHODS:
                    for baseline in ['macro_nearest','original','l2']:
                        error=((d[method]-d['target'])**2).mean();be=((d[baseline]-d['target'])**2).mean()
                        paired.append(dict(seed=seed,scenario=sc,method=method,baseline=baseline,delta_rmse_pp=float(100*(np.sqrt(error)-np.sqrt(be)))))
    summary=[]
    for sc in SCENARIOS:
        for method in METHODS:
            rr=[r for r in rows if (r['scenario'],r['method'])==(sc,method)]
            summary.append(dict(scenario=sc,method=method,rmse_pp=float(np.mean([r['rmse'] for r in rr]))*100,
                sd_pp=float(np.std([r['rmse'] for r in rr],ddof=1))*100))
    for name,obj in [('summary',summary),('sparsity',sparsity),('validation_choices',choices),('paired',paired),('provenance',hashes)]:R.save(out/f'{name}.json',obj)
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\begin{tabular}{lrrrr}\toprule',r'Selector & Recurring & Gradual & Normal & Localized\\\midrule']
    for method,label in zip(METHODS,LABELS):
        vals=[next(r['rmse_pp'] for r in summary if (r['method'],r['scenario'])==(method,sc)) for sc in SCENARIOS]
        print(label,[round(v,4) for v in vals]);lines.append(label+' & '+' & '.join(f'{v:.3f}' for v in vals)+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Penalty-selected context retrieval on a second fresh episode set. Entries are three-seed mean RMSE in percentage points. Penalty strengths and checkpoints use validation hard-selection error; the validation-choice row selects among all methods, including heuristics, before test generation.}',r'\end{table}']
    (out/'results_table.tex').write_text('\n'.join(lines)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    for method in ['macro_nearest','original','augmented_adam_l2','l2','l1','elastic','group']:
        vals=[next(r['rmse_pp'] for r in summary if (r['method'],r['scenario'])==(method,sc)) for sc in SCENARIOS]
        axs[0].plot(range(4),vals,marker='o',label=LABELS[METHODS.index(method)])
    axs[0].set_xticks(range(4),['Recurring','Gradual','Stable','Localized']);axs[0].set_ylabel('RMSE (percentage points)');axs[0].legend(fontsize=7);axs[0].grid(alpha=.2)
    for seed in range(3):
        rr=[next(r for r in sparsity if (r['seed'],r['method'])==(seed,m)) for m in PENALTIES]
        axs[1].bar(np.arange(4)+(seed-1)*.24,[r['nonzero'] for r in rr],width=.24,label=f'Seed {seed}')
    axs[1].set_xticks(range(4),['L2','L1','Elastic','Group']);axs[1].set_ylabel('Retained coefficients (absolute value > 1e-8)');axs[1].legend();axs[1].grid(axis='y',alpha=.2)
    fig.savefig(out/'penalty_comparison.png',dpi=180);fig.savefig(out/'penalty_comparison.pdf');plt.close(fig)
    R.save(out/'protocol.json',dict(scope='Feature/penalty comparison, no change to frozen banks, mixtures, ridge, six-window candidates or training action costs.',
        controls='Original56-feature saved selector; augmented57-feature old-optimizer L2; matched57-feature proximal L2; all new penalties.',
        feature_selection='Exact proximal zeros; candidate-invariant features removed from new proximal fits because they cancel from linear rankings.',
        qualification='Penalty types have different inductive biases; same finite strength grid and validation data. Fresh test observations share simulator family. No retuning after test.'))
    (out/'compare_selector_penalties_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
