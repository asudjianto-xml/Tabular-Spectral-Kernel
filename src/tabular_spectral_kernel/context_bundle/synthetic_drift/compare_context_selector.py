"""Fresh-episode evaluation of historically supervised window retrieval."""
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
from run_context_selector import ROOT,SCENARIOS

METHODS=['latest','macro_nearest','learned','oracle_labels','oracle_mean']
LABELS=['Latest window','Nearest unemployment','Learned selector','Query-label oracle','True-mean oracle']


def main():
    root=ROOT/'context_selector_results';out=root/'comparison';out.mkdir(exist_ok=False)
    metrics=[];training=[];provenance={};paired=[];diagnostics=[]
    def read(p):provenance[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
    for seed in range(3):
        src=root/f'seed{seed}';metrics+=read(src/'metrics.json')
        p=read(src/'protocol.json');t=read(src/'training.json')
        training.append(dict(seed=seed,weight=p['mixture_weight'],parameters=t['parameters'],selected_epoch=t['best_epoch'],epochs_run=len(t['trace'])))
        for path,digest in read(src/'provenance.json').items():runtime.verify_artifact(path,digest)
        for sc in SCENARIOS:
            path=src/f'{sc}.npz';provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as d:
                for name in METHODS:
                    c=d[name+'_choice'];pred=np.take_along_axis(d['candidate_predictions'],c[...,None],-1)[...,0]
                    np.testing.assert_array_equal(pred,d[name])
                    assert ((c>=0)&(c<6)).all()
                assert np.all((d['oracle_labels']-d['target'])**2<=(d['learned']-d['target'])**2+1e-12)
                assert np.all((d['oracle_mean']-d['mean'])**2<=(d['learned']-d['mean'])**2+1e-12)
                for method in ['learned','macro_nearest']:
                    delta=d[method+'_episode_mse']-d['latest_episode_mse']
                    paired.append(dict(seed=seed,scenario=sc,method=method,mean_mse_difference=float(delta.mean()),
                        rmse_difference_pp=100*(np.sqrt(d[method+'_episode_mse'].mean())-np.sqrt(d['latest_episode_mse'].mean())),
                        fraction_episodes_lower_mse=float(np.mean(delta<0))))
                c=d['learned_choice'];diagnostics.append(dict(seed=seed,scenario=sc,
                    mean_distinct_windows_per_query_cohort=float(np.mean([len(np.unique(a)) for a in c])),
                    multiwindow_cohort_fraction=float(np.mean([len(np.unique(a))>1 for a in c]))))
    summary=[]
    for sc in SCENARIOS:
        for method in METHODS:
            r=[a for a in metrics if (a['scenario'],a['method'])==(sc,method)]
            summary.append(dict(scenario=sc,method=method,rmse_pp=float(np.mean([a['rmse'] for a in r]))*100,
                rmse_seed_sd_pp=float(np.std([a['rmse'] for a in r],ddof=1))*100,
                clean_rmse_pp=float(np.mean([a['clean_rmse'] for a in r]))*100,
                older_fraction=float(np.mean([a['older_fraction'] for a in r]))))
    for name,obj in [('summary',summary),('training',training),('metrics',metrics),('paired',paired),('diagnostics',diagnostics),('provenance',provenance)]:R.save(out/f'{name}.json',obj)
    R.save(out/'protocol.json',dict(evaluation='256 fresh independent episodes per scenario per seed; three frozen bank/selector seeds. Test constructed after checkpoint frozen; no retuning.',
        comparability='Different fresh randomized episode protocol from earlier 60-month streams; do not compare absolute RMSE across protocols.',
        oracle='Exhaustive finite six-window oracle, conditional on fixed bank mixture and ridge; query labels/latent means are privileged diagnostics.',
        inference='Hard query-specific context selection; one principal-submatrix KRR per selected window/query. No global PSD claim across different selected supports.',
        limitations='Same synthetic structural law and path families as meta-training, not an unseen task family. Fixed historical mixture may select a simplex endpoint. No latency benchmark.'))
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\begin{tabular}{lrrrr}\toprule',r'Selector & Recurring & Gradual & Normal & Localized\\\midrule']
    for method,label in zip(METHODS,LABELS):
        vals=[next(r['rmse_pp'] for r in summary if (r['method'],r['scenario'])==(method,sc)) for sc in SCENARIOS]
        print(label,[round(v,4) for v in vals]);lines.append(label+' & '+' & '.join(f'{v:.3f}' for v in vals)+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Learned historical-window selection on fresh randomized episodes: three-seed mean KRR RMSE in percentage points. All methods choose one of six 128-row windows and share fixed kernels, mixture, and ridge within a seed. Oracle rows use unavailable query outcomes or latent conditional means.}',r'\end{table}']
    (out/'results_table.tex').write_text('\n'.join(lines)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for j,(method,label) in enumerate(zip(METHODS,LABELS)):
        rr=[next(r for r in summary if (r['method'],r['scenario'])==(method,sc)) for sc in SCENARIOS]
        axs[0].plot(range(4),[r['rmse_pp'] for r in rr],marker='o',label=label)
    axs[0].set_xticks(range(4),['Recurring','Gradual','Stable','Localized']);axs[0].set_ylabel('RMSE (percentage points)');axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
    for j,method in enumerate(['macro_nearest','learned']):
        rr=[next(r for r in summary if (r['method'],r['scenario'])==(method,sc)) for sc in SCENARIOS]
        axs[1].bar(np.arange(4)+(j-.5)*.35,[100*r['older_fraction'] for r in rr],width=.35,label=method)
    axs[1].set_xticks(range(4),['Recurring','Gradual','Stable','Localized']);axs[1].set_ylabel('Queries selecting an older window (%)');axs[1].legend();axs[1].grid(axis='y',alpha=.2)
    fig.savefig(out/'context_selection.png',dpi=180);fig.savefig(out/'context_selection.pdf');plt.close(fig)
    (out/'compare_context_selector_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
