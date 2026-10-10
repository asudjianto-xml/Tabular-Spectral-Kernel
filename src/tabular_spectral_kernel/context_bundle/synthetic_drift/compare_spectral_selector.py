"""Fresh-test comparison of linear and finite-spectral context selection."""
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
from run_spectral_selector import ROOT
from run_context_selector import SCENARIOS

METHODS=['latest','macro_nearest','l1','spectral_sparse','spectral_all','spectral','hybrid_sparse','hybrid_all','hybrid','validation_selected']
LABELS=['Latest window','Unemployment only','L1 linear','Spectral, sparse inputs','Spectral, all inputs','Spectral, validation choice',
        'Linear + spectral, sparse','Linear + spectral, all','Linear + spectral, chosen','Overall validation choice']


def main():
    root=ROOT/'spectral_selector_results';out=root/'comparison';out.mkdir(exist_ok=False)
    metrics=[];training=[];paired=[];audits=[];provenance={}
    def read(p):provenance[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
    for seed in range(3):
        src=root/f'seed{seed}';metrics+=read(src/'metrics.json');audits+=read(src/'psd_audit.json');sel=read(src/'selection.json');read(src/'protocol.json')
        for path,digest in read(src/'provenance.json').items():runtime.verify_artifact(path,digest)
        for name,r in sel['selected'].items():
            training.append(dict(seed=seed,method=name,config=r['config'],parameters=r['parameters'],
                descriptors=r['input_descriptors'],selected_epoch=r['best_epoch'],epochs_run=len(r['trace']),
                validation_mse=r['validation_mse'],initial_validation_mse=r['initial_validation_mse'],overall=sel['overall']))
        for sc in SCENARIOS:
            path=src/f'{sc}.npz';provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as d:
                for method in METHODS:
                    expected=np.take_along_axis(d['candidate_predictions'],d[method+'_choice'][...,None],-1)[...,0]
                    np.testing.assert_array_equal(expected,d[method])
                    for baseline in ['l1','macro_nearest']:
                        mse=np.mean((d[method]-d['target'])**2);base=np.mean((d[baseline]-d['target'])**2)
                        paired.append(dict(seed=seed,scenario=sc,method=method,baseline=baseline,delta_rmse_pp=float(100*(np.sqrt(mse)-np.sqrt(base)))))
                np.testing.assert_array_equal(d['validation_selected'],d[sel['overall']])
    summary=[]
    for sc in SCENARIOS:
        for method in METHODS:
            rows=[r for r in metrics if (r['scenario'],r['method'])==(sc,method)]
            summary.append(dict(scenario=sc,method=method,rmse_pp=float(np.mean([r['rmse'] for r in rows]))*100,
                                sd_pp=float(np.std([r['rmse'] for r in rows],ddof=1))*100))
    for name,obj in [('summary',summary),('training',training),('paired',paired),('psd_audit',audits),('provenance',provenance)]:R.save(out/f'{name}.json',obj)
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\begin{tabular}{lrrrr}\toprule',r'Selector & Recurring & Gradual & Normal & Localized\\\midrule']
    for method,label in zip(METHODS,LABELS):
        vals=[next(r['rmse_pp'] for r in summary if (r['scenario'],r['method'])==(sc,method)) for sc in SCENARIOS]
        print(label,[round(v,4) for v in vals]);lines.append(label+' & '+' & '.join(f'{v:.3f}' for v in vals)+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Linear versus learned spectral-kernel relevance scoring on a third fresh episode set: three-seed mean KRR RMSE in percentage points. All methods choose among the same six windows with unchanged prediction kernels, mixture weights, and ridge. Kernel configurations and the chosen subset use validation only.}',r'\end{table}']
    (out/'results_table.tex').write_text('\n'.join(lines)+'\n')
    fig,axs=plt.subplots(1,4,figsize=(14,4),layout='constrained')
    shown=['macro_nearest','l1','spectral','hybrid']
    for ax,sc in zip(axs,SCENARIOS):
        for j,method in enumerate(shown):
            r=next(r for r in summary if (r['scenario'],r['method'])==(sc,method))
            ax.errorbar(j,r['rmse_pp'],yerr=r['sd_pp'],fmt='o',capsize=3)
        ax.set_xticks(range(4),['Unemployment','L1 linear','Spectral','Linear + spectral'],rotation=50,ha='right',fontsize=8);ax.set_title(sc.capitalize());ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('RMSE (percentage points; seed SD)')
    fig.savefig(out/'spectral_selection.png',dpi=180);fig.savefig(out/'spectral_selection.pdf');plt.close(fig)
    R.save(out/'protocol.json',dict(comparison='Same fixed prediction stage; only context relevance scorer changes.',
        interpretation='Pure scorer lies in a learned finite spectral RKHS. Hybrid uses the sum of a frozen linear function and spectral RKHS function. No MLP.',
        selection='Sparse/all input sets and finite complexity/regularization candidates chosen by validation hard-selection MSE. Fresh tests generated after freezing.',
        limitation='Not the same parameter count/optimizer as L1; finite spectral capacity and training budget. No universal comparison of linear and kernel methods.'))
    (out/'compare_spectral_selector_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('Trained PSD audits',len(audits),'minimum',min(r['min_eigenvalue'] for r in audits))


if __name__=='__main__':main()
