"""Generate manuscript tables and diagnostics directly from saved review results."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).parent;OUT=runtime.output_path(__file__, HERE/'summary');OUT.mkdir(exist_ok=True)
def load(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def cell(mu,sd):return f'${mu:.3f}\\pm{sd:.3f}$'
scens=['recurring_downturn','gradual_downturn','stable_economy','localized_downturn']
enc=load(ROOT/'synthetic_drift/linear_attention_comparison/summary.json');opt=load(ROOT/'synthetic_drift/optimized_fusion_results/summary.json')
order=[('Latest historical kernel','latest'),('Validation-selected mixture','validation'),('Convex training-objective mixture','convex_mkl')]
rows=[]
for label,key in order:
 vals=[next(x for x in opt if x['method']==key and x['scenario']==sc) for sc in scens]
 rows.append(label+' & '+' & '.join(cell(100*v['rmse_mean'],100*v['rmse_seed_sd']) for v in vals)+r'\\')
for name in ['Single PSD, linear','Stacked PSD, linear','Learned PSD, linear','Transformer, linear','Simple PSD + MLP','Stacked PSD, final MLP','Learned PSD + MLPs','Transformer, final MLP','Transformer + all MLPs']:
 vals=[next(x for x in enc if x['method']==name and x['scenario']==sc and x['decoder']=='krr') for sc in scens]
 rows.append(name+' & '+' & '.join(cell(100*v['rmse_mean'],100*v['rmse_seed_sd']) for v in vals)+r'\\')
head=r'''\begin{table}[t]\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Fixed recent context: RMSE in percentage points, mean $\pm$ seed standard deviation (SD). Nine fusion encoders are compared with three reference rules. ``Linear'' removes intermediate feed-forward MLPs and uses an affine fusion head; spectral maps and normalization remain nonlinear.}
\begin{tabular}{lrrrr}\toprule
Geometry rule & Recurring & Gradual & Stable & Localized\\\midrule
'''
(ROOT/'context_selection_encoder_table.tex').write_text(head+'\n'.join(rows)+r'\bottomrule\end{tabular}\label{tab:fusion}\end{table}'+'\n')
syn=pd.DataFrame([r for seed in range(3) for r in load(HERE/f'synthetic_results/seed{seed}/metrics.json')]+load(HERE/'mlp32_results/metrics.json'));ss=syn.groupby(['method','scenario']).rmse_pp.agg(['mean','std']);ss.to_csv(OUT/'synthetic_summary.csv')
old=load(ROOT/'synthetic_drift/spectral_selector_results/comparison/summary.json');rows=[]
for label,key in [('Latest window','latest'),('Unemployment proximity','macro_nearest'),('Sparse linear relevance','l1'),('Spectral relevance, sparse inputs','spectral_sparse'),('Spectral relevance, all inputs','spectral_all'),('Validation-selected scorer family','validation_selected')]:
 vals=[next(x for x in old if x['method']==key and x['scenario']==sc) for sc in ['recurring','gradual','stable','localized']]
 rows.append(label+' & '+' & '.join(cell(v['rmse_pp'],v['sd_pp']) for v in vals)+r'\\')
for label,key in [('Matched-width MLP relevance','mlp32'),('Matched-grid MLP relevance','mlp_selector'),('Recent Gaussian KRR, snapshot','rbf_snapshot'),('Recent Gaussian KRR, with EWMA','rbf_ewma'),('Conditional-mean context oracle','mean_context_oracle'),('True conditional mean','conditional_mean'),('Query-label context oracle','label_context_oracle')]:
 vals=[ss.loc[key,sc] for sc in ['recurring','gradual','stable','localized']]
 rows.append(label+' & '+' & '.join(cell(v['mean'],v['std']) for v in vals)+r'\\')
head=r'''\begin{table}[t]\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Synthetic context-selection protocol: RMSE in percentage points, mean $\pm$ seed SD on identical test episodes. Retrieval rows use the fixed historical mixture. Gaussian KRR refits on recent support with independently validation-selected bandwidth and ridge. Oracle rows use unavailable information and are retrospective references.}
\begin{tabular}{lrrrr}\toprule
Method & Recurring & Gradual & Stable & Localized\\\midrule
'''
(ROOT/'context_selection_retrieval_table.tex').write_text(head+'\n'.join(rows)+r'\bottomrule\end{tabular}\label{tab:selection}\end{table}'+'\n')
frames=[];metrics=[];oracle=[];support=[]
for seed in range(3):
 folder=HERE/f'bike_results/seed{seed}';df=pd.read_csv(folder/'predictions.csv',parse_dates=['timestamp']);frames.append(df)
 for method,v in load(folder/'metrics.json').items():metrics.append(dict(seed=seed,method=method,**v))
 for part in ['original_validation','original_test','revised_validation','revised_test']:
  for method,v in load(folder/f'{part}_oracles.json').items():
   if isinstance(v,dict):oracle.append(dict(seed=seed,part=part,method=method,**v))
 for r in load(folder/'support_metrics.json'):support.append(dict(seed=seed,**r))
m=pd.DataFrame(metrics);s=m.groupby('method').agg(rmse=('rmse','mean'),sd=('rmse','std'),mae=('mae','mean'));s.to_csv(OUT/'bike_summary.csv')
os=pd.DataFrame(oracle).groupby(['part','method']).rmse.agg(['mean','std']);os.to_csv(OUT/'bike_oracles.csv')
ns=pd.DataFrame(support).groupby('n').agg(rmse=('rmse','mean'),sd=('rmse','std'),mae=('mae','mean'));ns.to_csv(OUT/'bike_support.csv')
methods=[('Recent + latest kernel','recent_latest_kernel'),('Recent + best single kernel','recent_best_single'),('Recent + uniform fusion','recent_uniform_fusion'),('Recent + calibrated fusion','recent_fixed_fusion'),('Recent + learned fusion','recent_adaptive_fusion'),('Weather retrieval + calibrated fusion','weather_selected_fixed_fusion'),('Recent + prediction averaging','recent_prediction_average'),('Seasonal naive','seasonal_naive')]
rows=[label+f" & {r.rmse:.3f} & {r.sd:.3f} & {r.mae:.3f}"+r'\\' for label,key in methods for _,r in s.loc[[key]].iterrows()]
rows.append(r'\midrule')
for n,r in ns.iterrows():rows.append(f'Calibrated fusion, {n} recent observations & {r.rmse:.3f} & {r.sd:.3f} & {r.mae:.3f}'+r'\\')
tree=load(HERE/'tree_results/metrics.json');rows.append(f"Rolling expanding-history CatBoost & {tree['rmse']:.3f} & --- & {tree['mae']:.3f}"+r'\\')
head=r'''\begin{table}[t]\centering\small
\caption{Bike Sharing, 2012 hourly count errors after the expanded validation ridge search. The first block uses 128-observation KRR support except seasonal naive. Selected-context results equal their recent counterparts and are omitted. The larger-support rows retain the calibrated kernel weights and select their own ridge. CatBoost uses all prior labels and one fixed optimization seed.}
\begin{tabular}{lrrr}\toprule Method & RMSE & Seed SD & MAE\\\midrule
'''
(ROOT/'context_selection_bike_table.tex').write_text(head+'\n'.join(rows)+r'\bottomrule\end{tabular}\label{tab:bike-results}\end{table}'+'\n')
rows=[]
for kind,label in [('fixed','Calibrated fusion'),('gated','Learned fusion')]:
 for which,desc in [('recent','recent support'),('day_oracle','day-label oracle'),('query_oracle','query-label oracle')]:
  vals=[os.loc[part,kind+'_'+which]['mean'] for part in ['revised_validation','revised_test']]
  rows.append(label+', '+desc+f' & {vals[0]:.3f} & {vals[1]:.3f}'+r'\\')
(ROOT/'context_selection_bike_oracle_table.tex').write_text(r'''\begin{table}[t]\centering\small
\caption{Retrospective best-of-six Bike Sharing context errors. Oracles minimize realized squared error using the query day or each query label. They measure retrospective error reductions using outcomes unavailable at prediction time.}
\begin{tabular}{lrr}\toprule Geometry and context choice & Validation RMSE & Test RMSE\\\midrule
'''+ '\n'.join(rows)+r'\bottomrule\end{tabular}\label{tab:bike-oracles}\end{table}'+'\n')
# Paired temporal uncertainty for both original and expanded-grid runs, including operator comparison.
days=pd.date_range('2012-01-01','2012-12-31');rng=np.random.default_rng(20261010);draws=[]
for _ in range(2000):
 starts=rng.integers(0,len(days)-6,size=53);draws.append(np.concatenate([np.arange(i,i+7) for i in starts])[:366])
ci=[]
for version,fs in [('revised',frames),('original',[pd.read_csv(ROOT/f'bikeshare_context_fusion/results/seed{i}/predictions.csv',parse_dates=['timestamp']) for i in range(3)])]:
 count=fs[0].groupby(fs[0].timestamp.dt.floor('D')).size().reindex(days).to_numpy()
 for ref in ['recent_latest_kernel','recent_fixed_fusion','recent_prediction_average']:
  a='recent_adaptive_fusion';errs={key:np.array([((f[key]-f.target)**2).groupby(f.timestamp.dt.floor('D')).sum().reindex(days).to_numpy() for f in fs]) for key in [a,ref]}
  actual=np.sqrt(errs[a].sum(1)/count.sum()).mean()-np.sqrt(errs[ref].sum(1)/count.sum()).mean()
  diff=[np.sqrt(errs[a][:,ix].sum(1)/count[ix].sum()).mean()-np.sqrt(errs[ref][:,ix].sum(1)/count[ix].sum()).mean() for ix in draws]
  ci.append(dict(version=version,method=a,reference=ref,delta=float(actual),low=float(np.quantile(diff,.025)),high=float(np.quantile(diff,.975))))
save(OUT/'paired_intervals.json',ci)
# Three distinct series, with a truncated-axis point plot and monthly curves.
core=['recent_latest_kernel','recent_fixed_fusion','recent_adaptive_fusion'];labels=['Latest kernel','Calibrated fusion','Learned fusion'];colors=['#666666','#e69f00','#0072b2']
fig,ax=plt.subplots(1,2,figsize=(11,4),gridspec_kw={'width_ratios':[1,1.5]})
for i,(key,label,color) in enumerate(zip(core,labels,colors)):
 ax[0].errorbar(s.loc[key,'rmse'],i,xerr=s.loc[key,'sd'],fmt='o',color=color,capsize=4)
 monthly=np.array([((f[key]-f.target)**2).groupby(f.timestamp.dt.month).mean().pow(.5).to_numpy() for f in frames])
 ax[1].plot(range(1,13),monthly.mean(0),'-o',color=color,label=label,markersize=3)
ax[0].set_yticks(range(3),labels);ax[0].invert_yaxis();ax[0].set_xlim(71,83);ax[0].set_xlabel('Count RMSE (mean ± seed SD)');ax[0].set_title('128-observation support')
ax[1].set_xlabel('Month in 2012');ax[1].set_ylabel('Count RMSE');ax[1].set_xticks(range(1,13));ax[1].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'bike_revision.pdf');fig.savefig(OUT/'bike_revision.png',dpi=180)
print('BIKE\n',s.round(4).to_string());print('SUPPORT\n',ns.round(4).to_string());print('ORACLES\n',os.round(4).to_string());print('PAIRED',json.dumps(ci,indent=2));print('SYNTHETIC\n',ss.round(4).to_string())
