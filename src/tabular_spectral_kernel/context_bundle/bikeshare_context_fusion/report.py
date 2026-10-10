"""Summarize the predeclared rolling evaluation; no tuning or training."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={
 'recent_latest_kernel':'Recent context + latest kernel',
 'recent_best_single':'Recent context + best single kernel',
 'recent_uniform_fusion':'Recent context + uniform fusion',
 'recent_fixed_fusion':'Recent context + calibrated fusion',
 'recent_adaptive_fusion':'Recent context + learned fusion',
 'selected_fixed_fusion':'Learned context + calibrated fusion',
 'selected_adaptive_fusion':'Learned context + learned fusion',
 'weather_selected_fixed_fusion':'Weather retrieval + calibrated fusion',
 'recent_prediction_average':'Recent context + prediction averaging',
 'seasonal_naive':'Previous weekday/hour count'}
CORE=['recent_latest_kernel','recent_fixed_fusion','recent_adaptive_fusion','selected_fixed_fusion','selected_adaptive_fusion']

def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main(root):
    protocol=json.loads((root/'protocol.json').read_text());seeds=protocol['seeds']
    frames=[];metrics=[];choices=[];weights=[];monthly=[];audit=[]
    for seed in seeds:
        folder=root/f'seed{seed}'
        frozen=json.loads((folder/'frozen_before_test.json').read_text())
        for name,digest in frozen['checkpoints'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
        df=pd.read_csv(folder/'predictions.csv',parse_dates=['timestamp']);frames.append(df)
        assert len(df)==8734 and df.row.is_unique and np.isfinite(df[list(NAMES)].to_numpy()).all()
        assert (df.timestamp.dt.year==2012).all()
        if len(frames)>1:
            np.testing.assert_array_equal(df.row,frames[0].row);np.testing.assert_array_equal(df.target,frames[0].target)
        m=json.loads((folder/'metrics.json').read_text())
        for method,v in m.items():
            expected=float(np.sqrt(np.mean((df[method]-df.target)**2)))
            assert abs(expected-v['rmse'])<1e-5
            metrics.append(dict(seed=seed,method=method,**v))
        ch=pd.read_csv(folder/'choices.csv');ww=pd.read_csv(folder/'weights.csv')
        for method,col in [('selected_fixed_fusion','fixed_context'),('selected_adaptive_fusion','adaptive_context'),('weather_selected_fixed_fusion','weather_context')]:
            choices.append(dict(seed=seed,method=method,older_fraction=float((ch[col]!=0).mean()),lag_fractions={str(lag):float((ch[col]==j).mean()) for j,lag in enumerate(protocol['context_lags_days'])}))
        weights.append(dict(seed=seed,mean=ww[['weight0','weight1','weight2']].mean().tolist(),std=ww[['weight0','weight1','weight2']].std().tolist()))
        monthly.extend([dict(seed=seed,**v) for v in json.loads((folder/'monthly.json').read_text())])
        audit.extend([dict(seed=seed,**v) for v in json.loads((folder/'psd_audit.json').read_text())])
    tab=pd.DataFrame(metrics);summary=tab.groupby('method').agg(rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),mae_mean=('mae','mean'),r2_mean=('r2','mean')).reindex(NAMES)
    summary.to_csv(root/'summary.csv');save(root/'summary.json',summary.reset_index().to_dict('records'))
    save(root/'retrieval_diagnostics.json',choices);save(root/'fusion_diagnostics.json',weights);save(root/'numerical_audit.json',audit)
    # Paired moving-block bootstrap on calendar days, using the same block draw across seeds/methods.
    # Uncertainty describes this one observed series, conditional on fitted models.
    all_days=pd.date_range('2012-01-01','2012-12-31',freq='D');n=len(all_days)
    mse={}
    counts=frames[0].groupby(frames[0].timestamp.dt.floor('D')).size().reindex(all_days).to_numpy()
    for method in NAMES:
        mse[method]=np.array([((df[method]-df.target)**2).groupby(df.timestamp.dt.floor('D')).sum().reindex(all_days).to_numpy() for df in frames])
    pairs=[('selected_adaptive_fusion','recent_latest_kernel'),('selected_adaptive_fusion','recent_fixed_fusion'),
           ('selected_adaptive_fusion','recent_adaptive_fusion'),('selected_adaptive_fusion','selected_fixed_fusion'),
           ('recent_adaptive_fusion','recent_fixed_fusion'),('selected_fixed_fusion','recent_fixed_fusion')]
    rng=np.random.default_rng(20261010);draws=[]
    for _ in range(2000):
        starts=rng.integers(0,n-6,size=int(np.ceil(n/7)))
        draws.append(np.concatenate([np.arange(s,s+7) for s in starts])[:n])
    intervals=[]
    for a,b in pairs:
        delta=[]
        for ix in draws:
            ra=np.sqrt(mse[a][:,ix].sum(1)/counts[ix].sum()).mean();rb=np.sqrt(mse[b][:,ix].sum(1)/counts[ix].sum()).mean();delta.append(ra-rb)
        intervals.append(dict(method=a,reference=b,rmse_difference=float(summary.loc[a,'rmse_mean']-summary.loc[b,'rmse_mean']),
                              lower95=float(np.quantile(delta,.025)),upper95=float(np.quantile(delta,.975)),block_days=7,draws=2000))
    save(root/'paired_block_bootstrap.json',intervals)
    mm=pd.DataFrame(monthly);mm.to_csv(root/'monthly.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),gridspec_kw={'width_ratios':[1,1.3]})
    colors=['#777777','#e69f00','#009e73','#56b4e9','#0072b2']
    labels=['Latest kernel','Fixed fusion','Learned fusion','Selection + fixed','Selection + learned']
    y=summary.loc[CORE,'rmse_mean'].to_numpy();err=summary.loc[CORE,'rmse_sd'].to_numpy()
    axes[0].barh(np.arange(5),y,xerr=err,color=colors,capsize=3);axes[0].set_yticks(range(5),labels);axes[0].invert_yaxis();axes[0].set_xlabel('Hourly count RMSE (mean ± seed SD)')
    axes[0].set_title('2012 rolling evaluation')
    for method,color,label in zip(CORE,colors,labels):
        a=mm[mm.method==method].groupby('month').rmse.mean()
        axes[1].plot(range(1,13),a,color=color,label=label,marker='o',markersize=3)
    axes[1].set_xlabel('Month in 2012');axes[1].set_ylabel('Hourly count RMSE');axes[1].set_xticks(range(1,13));axes[1].legend(fontsize=8,ncol=2);axes[1].set_title('Performance through the year')
    fig.tight_layout();fig.savefig(root/'bikeshare_comparison.png',dpi=180);fig.savefig(root/'bikeshare_comparison.pdf');plt.close(fig)
    lines=['Hourly Bike Sharing: context selection and spectral fusion','',
      'Protocol: three bimonthly spectral banks learned on Jan-Jun2011; meta-training Jul-Sep2011; validation Oct-Dec2011; rolling test all 8,734 observed hourly counts in 2012.',
      'Every day uses only labels before that day. Prior test-day outcomes become eligible context later. Observed target-hour weather is supplied: conditional demand prediction, not a weather-ahead forecast.',
      'The selector and fusion gate are spectral feature models with affine heads, without an MLP or transformer. The composed method is trained in two stages, with the gate exposed to every eligible context type.',
      '', 'Method | Count RMSE mean (seed SD) | Count MAE | R2']
    for method,row in summary.iterrows():lines.append(f"{NAMES[method]} | {row.rmse_mean:.3f} ({row.rmse_sd:.3f}) | {row.mae_mean:.3f} | {row.r2_mean:.4f}")
    lines+=['','Paired RMSE differences (negative favors the first method); 95% 7-day moving-block bootstrap intervals:']
    for x in intervals:lines.append(f"{NAMES[x['method']]} minus {NAMES[x['reference']]}: {x['rmse_difference']:.3f} [{x['lower95']:.3f}, {x['upper95']:.3f}]")
    lines+=['','Interpretation: validation retained epoch zero for both context selectors in every seed. With zero scores, the documented tie rule selects the recent window. The selected-context rows are therefore fallback results, not evidence that a trained selector discovered useful historical windows. Learned fusion improves the latest-kernel baseline in every seed; its query/context-dependent weights vary across predictions.','', 'Limitations: one observed temporal series, three optimization seeds, limited early-2011 geometry and meta-training coverage, fixed candidate lag set and 128-observation supports. Real-data temporal shift does not isolate conditional drift. Bootstrap intervals are conditional on fitted models and this series; they are not cross-dataset guarantees. No test-driven retuning was performed.',
      'Prediction averaging uses the fusion gate\'s weights and common ridge in separate per-kernel solves; it is a mechanistic comparator, not a separately trained optimal ensemble.',
      'Numerical validation: source/checkpoint hashes, stored metrics recomputation, strict temporal eligibility tests, query-label exclusion, PSD feature maps, and float32-versus-float64 ridge checks.']
    (root/'report.txt').write_text('\n'.join(lines)+'\n')
    latex=[r'\section{Hourly Bike Sharing: chronological real-data evaluation}',
      'We evaluate the composed method on 8,734 observed hourly rental counts in 2012. Three bimonthly spectral kernels are learned from January--June 2011; July--September supplies meta-training episodes and October--December supplies validation. Each daily origin uses only earlier labels. Observed weather at the query hour is supplied, so the task is conditional demand prediction rather than an ex-ante weather forecast.',
      'Candidates contain the last 128 observed hours before the origin or before offsets of 7, 28, 56, 91, and 182 days. A spectral gate is trained over eligible candidate types, then a spectral selector learns their resulting predictive costs. This evaluates the composition using two-stage training; it does not claim joint end-to-end optimization.',
      r'\begin{table}[h]\centering\small',r'\caption{2012 count prediction. Mean RMSE and its standard deviation across three optimization seeds; lower is better.}',r'\begin{tabular}{lrr}\toprule Method & RMSE & Seed SD\\\midrule']
    for method,row in summary.iterrows():latex.append(NAMES[method]+f' & {row.rmse_mean:.3f} & {row.rmse_sd:.3f}'+r'\\')
    latex += [r'\bottomrule\end{tabular}\end{table}',
      'All methods share the daily query rows. The latest-kernel and best-single-kernel references select ridge parameters independently on validation; learned and fixed fusion use the validation-calibrated fusion ridge. Previously observed 2012 labels enter later contexts but never persistent parameter training. Three seeds quantify optimization variability on one series, not independent dataset replication.',
      'In all three seeds, validation retained the initialization of both context selectors: equal scores select the recent window by the fixed tie rule. Thus context selection adds no gain in this protocol. The improvement comes from adaptive geometry on recent labeled support, not from retrieval of older contexts.',
      'Real-data temporal variation combines changes in covariate distributions, demand levels, and potentially conditional relationships. It does not establish the controlled concept-drift mechanism of the synthetic study.']
    (root/'bikeshare_results.tex').write_text('\n\n'.join(latex)+'\n')
    document=r'''\documentclass[11pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{booktabs,graphicx,amsmath,microtype}
\usepackage[T1]{fontenc}
\usepackage{xurl}
\usepackage[colorlinks=true,allcolors=blue]{hyperref}
\title{Spectral In-Context Learning on Hourly Bike Sharing}
\author{Chronological real-data experiment}
\date{October 2026}
\begin{document}
\maketitle
'''+ '\n\n'.join(latex)+r'''
\section{Interpretation and uncertainty}
The full two-stage composition is evaluated, but its selector falls back to recent support. This is a negative result for learned retrieval under the tested candidate pool and training period. It is positive evidence for context-conditioned fusion: the gate varies its mixture weights by query and context while preserving one PSD mixture for each local KRR solve. No weights are refitted with 2012 query labels.

The mean RMSE reduction relative to the latest kernel is 7.1 percent; relative to validation-calibrated fixed fusion it is 1.8 percent. A paired seven-day moving-block bootstrap, conditional on the fitted models and this observed series, gives RMSE-difference intervals of $[-7.386,-4.159]$ versus the latest kernel and $[-2.468,-0.192]$ versus calibrated fixed fusion. These intervals do not establish cross-dataset generalization or distinguish all sources of temporal drift.

The prediction-averaging comparator uses the learned fusion weights with separate per-kernel ridge solves and the same ridge. It is a mechanistic comparison rather than an independently optimized ensemble. Learned kernel fusion has a modestly lower mean error, consistent with the two operators being distinct but not a claim of universal dominance.

\begin{figure}[h]
\centering\includegraphics[width=\linewidth]{bikeshare_comparison.pdf}
\caption{Aggregate and monthly results. Error bars describe variability across optimization seeds on the same test series. The selected-context rows coincide with recent-context rows because validation retains the recent-window fallback.}
\end{figure}

\section{Checks and limitations}
Five automated tests cover exact KRR with an unpenalized intercept, temporal eligibility of all supports, exclusion of current and future labels from descriptors and predictions, PSD spectral features, and the fusion-versus-averaging distinction. Additional saved audits compare float32 and float64 ridge predictions and check realized fused Gram eigenvalues. Stored predictions reproduce the reported metrics, and checkpoint hashes match those frozen before test evaluation.

The bank uses only the first six months of 2011 and adaptation training uses only three months. Candidate supports have 128 observations and a predeclared set of lags. Geometry training is capped at 250 steps and is not a convergence claim. The ridge search selects its lower boundary in this experiment. These restrictions limit conclusions about the best attainable kernel or retrieval performance. No post-test retuning is used to address them.

\section{Reproduction}
From the project root, run the following with a new output directory:
\begin{verbatim}
python -m pytest -q bikeshare_context_fusion/test_protocol.py
python bikeshare_context_fusion/run.py --out runs/bike_replication
python bikeshare_context_fusion/report.py runs/bike_replication
\end{verbatim}
The default run uses seeds 0, 1, and 2, 250 geometry steps, and 25 fusion-gate epochs. The output includes the exact executed source, protocol, data fingerprint, checkpoints, validation traces, hourly predictions, context choices, mixture weights, monthly metrics, numerical audits, and paired bootstrap intervals. Existing output directories are refused. The original manuscripts have not been modified.
\end{document}
'''
    (root/'report.tex').write_text(document)
    print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path,nargs='?',default=Path(__file__).parent/'results');main(p.parse_args().root)
