"""Locked extensions: hourly Bike Sharing, Taiwan default and Adult income."""
import argparse,copy,hashlib,json,math,time,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,QuantileTransformer,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss,accuracy_score,mean_squared_error,mean_absolute_error,r2_score,brier_score_loss
from threadpoolctl import threadpool_limits
from . import identifiable_spectral_california as base
from .solvers import exact_path

OUT=Path.cwd()/'runs'/'spectral_stage1_extended';DATA=Path.cwd()/'data'
DEVICE='cpu'
DATASETS=['bike','taiwan','adult'];SEEDS=[0,1,2]
RIDGES=np.logspace(-7,-3,9)
STEPS=800;SUPPORT=1024;QUERY=256

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2)+'\n')

def tensor(x):return torch.as_tensor(x,dtype=torch.float64,device=DEVICE)

def load(name,seed):
    if name=='bike':
        df=pd.read_csv(DATA/'hour.csv').sort_values(['dteday','hr']).reset_index(drop=True)
        y=df.pop('cnt').to_numpy(float);dates=df['dteday'].to_numpy()
        df=df.drop(columns=['instant','dteday','casual','registered'])
        cats=['season','yr','holiday','workingday','weathersit']
        days=np.unique(dates);a=days[int(.6*len(days))];b=days[int(.8*len(days))]
        tr=np.flatnonzero(dates<a);va=np.flatnonzero((dates>=a)&(dates<b));te=np.flatnonzero(dates>=b)
        cal=np.array([],dtype=int);cls=False
        split_info=dict(train_end=str(dates[tr[-1]]),validation_start=str(a),test_start=str(b))
    elif name=='taiwan':
        z=np.load(DATA/'taiwan_credit.npz')
        names=['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']+[f'BILL_AMT{i}' for i in range(1,7)]+[f'PAY_AMT{i}' for i in range(1,7)]
        df=pd.DataFrame(z['X'],columns=names);y=z['y'].astype(float)
        cats=['SEX','EDUCATION','MARRIAGE'];cls=True
        outer,te=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
        tr,vpool=train_test_split(outer,test_size=.2,stratify=y[outer],random_state=1000+seed)
        cal,va=train_test_split(vpool,test_size=.5,stratify=y[vpool],random_state=2000+seed)
        split_info=dict(split='stratified 64/8/8/20 train/calibration/validation/test')
    else:
        cols=['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','income']
        left=pd.read_csv(DATA/'adult.data',names=cols,skipinitialspace=True,na_values='?')
        right=pd.read_csv(DATA/'adult.test',names=cols,skipinitialspace=True,na_values='?',comment='|')
        df=pd.concat([left,right],ignore_index=True)
        y=df.pop('income').str.strip().str.rstrip('.').eq('>50K').to_numpy(float)
        cats=['workclass','education','marital_status','occupation','relationship','race','sex','native_country'];cls=True
        tr,vpool=train_test_split(np.arange(len(left)),test_size=.2,stratify=y[:len(left)],random_state=1000+seed)
        cal,va=train_test_split(vpool,test_size=.5,stratify=y[vpool],random_state=2000+seed)
        te=np.arange(len(left),len(df));split_info=dict(split='Official Adult test partition; training partition split 80/10/10 into fit/calibration/validation')
    nums=[c for c in df.columns if c not in cats]
    for c in cats:df[c]=df[c].fillna('MISSING').astype(str)
    assert np.isfinite(df[nums].to_numpy(float)).all()
    for a,b in [(tr,va),(tr,cal),(tr,te),(va,cal),(va,te),(cal,te)]:assert not np.intersect1d(a,b).size
    return dict(name=name,seed=seed,df=df,y=y,tr=tr,va=va,cal=cal,te=te,cats=cats,nums=nums,classification=cls,split_info=split_info)

class MixedPreprocessor:
    def __init__(self,kind):self.kind=kind
    def fit(self,d):
        x=d['df'].iloc[d['tr']];self.nums=d['nums'];self.cats=d['cats']
        self.sc=(StandardScaler() if self.kind=='standard' else QuantileTransformer(n_quantiles=min(1000,len(x)),output_distribution='normal',random_state=123))
        self.sc.fit(x[self.nums]);self.oh=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float64).fit(x[self.cats])
        contrasts=[]
        for categories in self.oh.categories_:
            c=len(categories);h=np.zeros((c,c-1))
            for r in range(1,c):h[:r,r-1]=1/math.sqrt(r*(r+1));h[r,r-1]=-r/math.sqrt(r*(r+1))
            contrasts.append(h)
        self.contrasts=contrasts;self.q=sum(h.shape[1] for h in contrasts)
        return self
    def transform(self,df):
        x=self.sc.transform(df[self.nums]);one=self.oh.transform(df[self.cats]);cols=[];offset=0
        for h in self.contrasts:cols.append(one[:,offset:offset+len(h)]@h);offset+=len(h)
        return np.column_stack([x,*cols])

class MixedSpectral(base.CanonicalSpectral):
    def __init__(self,d,q,atoms=4,max_frequency=.5):
        torch.nn.Module.__init__(self);self.d=d;self.atoms=atoms;self.q=q;self.p=2*atoms*d+q;p=self.p
        ij=torch.triu_indices(p,p,1,device=DEVICE);self.register_buffer('ii',ij[0]);self.register_buffer('jj',ij[1])
        h=torch.zeros(p-1,p,dtype=torch.float64,device=DEVICE)
        for r in range(1,p):h[r-1,:r]=1/math.sqrt(r*(r+1));h[r-1,r]=-r/math.sqrt(r*(r+1))
        self.register_buffer('helmert',h)
        self.offdiag=torch.nn.Parameter(torch.zeros(len(ij[0]),dtype=torch.float64,device=DEVICE))
        self.diagonal=torch.nn.Parameter(torch.zeros(p-1,dtype=torch.float64,device=DEVICE))
        omega=torch.logspace(math.log10(.025),math.log10(max_frequency),atoms,dtype=torch.float64,device=DEVICE)
        self.log_gaps=torch.nn.Parameter(torch.diff(omega,prepend=omega.new_zeros(1)).log().repeat(d,1))
        self.log_temperature=torch.nn.Parameter(tensor(math.log(.5)));self.log_ridge=torch.nn.Parameter(tensor(math.log(.1)))
    def embedding(self,x,root=None):
        if root is None:root=self.normalized_root()
        phase=2*math.pi*x[:,:self.d,None]*self.frequencies()[None]
        raw=torch.cat([torch.stack([phase.cos(),phase.sin()],-1).flatten(1),x[:,self.d:]],1)
        return raw@root
    def kernel(self,a,b):
        sq=a.square().sum(1)[:,None]+b.square().sum(1)[None,:]-2*a@b.T
        ij=(sq.detach()<1e-12).nonzero()
        if len(ij):sq=sq.index_put((ij[:,0],ij[:,1]),(a[ij[:,0]]-b[ij[:,1]]).square().sum(1))
        distance=torch.where(sq>0,sq.clamp_min(1e-24).sqrt(),torch.zeros_like(sq))
        return torch.exp(-distance/self.log_temperature.exp())

def primary(d,y,p):return -roc_auc_score(y,p) if d['classification'] else float(np.sqrt(mean_squared_error(y,p)))
def metrics(d,y,p):
    if d['classification']:
        p=np.clip(p,1e-12,1-1e-12)
        return dict(auc=float(roc_auc_score(y,p)),log_loss=float(log_loss(y,p)),accuracy=float(accuracy_score(y,p>=.5)),brier=float(brier_score_loss(y,p)))
    return dict(rmse=float(np.sqrt(mean_squared_error(y,p))),mae=float(mean_absolute_error(y,p)),r2=float(r2_score(y,p)))

@torch.no_grad()
def constrain(m):
    m.log_gaps.clamp_(-9,2);m.log_temperature.clamp_(-5,3);m.log_ridge.clamp_(math.log(1e-4),0)
    eig=torch.linalg.eigvalsh(m.generator());spread=float(eig[-1]-eig[0])
    if spread>20:m.offdiag.mul_(20/spread);m.diagonal.mul_(20/spread)

def train(d,kind):
    folder=OUT/d['name']/f'seed{d["seed"]}'/kind;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'checkpoint.pt').exists():return json.loads((folder/'training.json').read_text())
    torch.manual_seed(700+d['seed']);prep=MixedPreprocessor(kind).fit(d)
    x=tensor(prep.transform(d['df'].iloc[d['tr']]));xv=tensor(prep.transform(d['df'].iloc[d['va']]))
    y=d['y'];mu=float(y[d['tr']].mean());sd=float(y[d['tr']].std());yt=tensor((y[d['tr']]-mu)/sd)
    m=MixedSpectral(len(d['nums']),prep.q)
    opt=torch.optim.Adam([dict(params=[m.diagonal,m.log_temperature,m.log_ridge],lr=.01),dict(params=[m.offdiag,m.log_gaps],lr=.002)])
    gen=torch.Generator(device=DEVICE).manual_seed(1777+d['seed']);land=torch.as_tensor(np.random.default_rng(77).permutation(len(x))[:512],device=DEVICE)
    best=None;trace=[];start=time.perf_counter()
    for step in range(STEPS+1):
        if step%200==0:
            with torch.no_grad():pred=base.nystrom_path(m,x,yt,xv,land,RIDGES).cpu().numpy()*sd+mu
            loss=[primary(d,y[d['va']],pred[:,j]) for j in range(len(RIDGES))];j=int(np.argmin(loss))
            row=dict(step=step,criterion=float(loss[j]),ridge=float(RIDGES[j]),seconds=time.perf_counter()-start);trace.append(row);save(folder/'trace.json',trace)
            print('SPECTRAL',d['name'],d['seed'],kind,row,flush=True)
            if best is None or loss[j]<best[0]:best=(loss[j],copy.deepcopy(m.state_dict()),step)
        if step==STEPS:break
        ids=torch.randperm(len(x),generator=gen,device=DEVICE)[:SUPPORT+QUERY];s,q=ids[:SUPPORT],ids[SUPPORT:]
        opt.zero_grad(set_to_none=True);z=m.embedding(x[ids]);K=m.kernel(z[:SUPPORT],z[:SUPPORT]);mean=yt[s].mean()
        L=torch.linalg.cholesky(K+m.log_ridge.exp()*torch.eye(SUPPORT,dtype=torch.float64,device=DEVICE))
        alpha=torch.cholesky_solve((yt[s]-mean)[:,None],L)[:,0];pred=m.kernel(z[SUPPORT:],z[:SUPPORT])@alpha+mean
        loss=(pred-yt[q]).square().mean()+1e-4/m.p*(m.diagonal.square().sum()+m.offdiag.square().sum())
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10.);opt.step();constrain(m)
        factor=.2+.4*(1+math.cos(math.pi*(step+1)/STEPS))
        opt.param_groups[0]['lr']=.01*factor;opt.param_groups[1]['lr']=.002*factor
    rec=dict(kind=kind,selected_step=best[2],criterion=float(best[0]),seconds=time.perf_counter()-start,p=m.p,numeric_features=len(d['nums']),category_contrasts=prep.q,train_rows=len(x))
    torch.save(dict(state_dict={k:v.cpu() for k,v in best[1].items()},mu=mu,sd=sd),folder/'checkpoint.pt');save(folder/'training.json',rec)
    return rec

@torch.no_grad()
def head(m,x,y,ridge):
    root=m.normalized_root();z=m.embedding(x,root);n=len(x);K=torch.empty((n,n),dtype=torch.float64,device=DEVICE)
    for i in range(0,n,512):K[i:i+512]=m.kernel(z[i:i+512],z)
    K.diagonal().add_(n*ridge);L=torch.linalg.cholesky(K);sol=torch.cholesky_solve(torch.stack([y,torch.ones_like(y)],1),L)
    b=sol[:,0].sum()/sol[:,1].sum();a=sol[:,0]-b*sol[:,1]
    residual=float(torch.linalg.vector_norm(K@a+b-y)/torch.linalg.vector_norm(y-y.mean()))
    return dict(root=root.cpu(),z=z.cpu(),alpha=a.cpu(),b=b.cpu(),residual=residual)

@torch.no_grad()
def head_predict(m,h,x):
    z=h['z'].to(DEVICE);a=h['alpha'].to(DEVICE);b=h['b'].to(DEVICE);q=m.embedding(x,h['root'].to(DEVICE))
    return torch.cat([m.kernel(q[i:i+512],z)@a+b for i in range(0,len(q),512)]).cpu().numpy()

def calibrate(raw,y):
    c=LogisticRegression(C=1e6,max_iter=1000).fit(np.asarray(raw).reshape(-1,1),y)
    return dict(slope=float(c.coef_[0,0]),intercept=float(c.intercept_[0]))
def apply_cal(raw,c):return expit(c['slope']*raw+c['intercept'])


def spectral_select(name,seed,kind):
    folder=OUT/name/f'seed{seed}';dest=folder/'spectral_selection.json'
    if dest.exists():return json.loads(dest.read_text())
    d=load(name,seed);kinds=['standard','quantile'] if kind is None else [kind]
    records=[train(d,k) for k in kinds];chosen=min(records,key=lambda v:v['criterion']);kind=chosen['kind']
    prep=MixedPreprocessor(kind).fit(d);m=MixedSpectral(len(d['nums']),prep.q)
    cp=torch.load(folder/kind/'checkpoint.pt',weights_only=True,map_location=DEVICE);m.load_state_dict(cp['state_dict']);m.eval()
    x=tensor(prep.transform(d['df'].iloc[d['tr']]));xv=tensor(prep.transform(d['df'].iloc[d['va']]));yt=tensor((d['y'][d['tr']]-cp['mu'])/cp['sd'])
    with torch.no_grad():
        land=torch.as_tensor(np.random.default_rng(88).permutation(len(x))[:2048],device=DEVICE)
        approx=base.nystrom_path(m,x,yt,xv,land,RIDGES).cpu().numpy()*cp['sd']+cp['mu']
    j=int(np.argmin([primary(d,d['y'][d['va']],approx[:,i]) for i in range(len(RIDGES))]));ridges=RIDGES[max(0,j-1):min(len(RIDGES),j+2)]
    exact,res=exact_path(m,x,yt,xv,ridges);pred=exact.cpu().numpy()*cp['sd']+cp['mu'];trials=[]
    for j,ridge in enumerate(ridges):
        for clip in ([False] if d['classification'] else [False,True]):
            pp=np.clip(pred[:,j],0,d['y'][d['tr']].max()) if clip else pred[:,j]
            trials.append(dict(ridge=float(ridge),clip=clip,criterion=float(primary(d,d['y'][d['va']],pp)),column=j))
    selected=min(trials,key=lambda v:v['criterion']);h=head(m,x,yt,selected['ridge']);torch.save(h,folder/'head.pt')
    calibration=None
    if d['classification']:
        xc=tensor(prep.transform(d['df'].iloc[d['cal']]));raw=head_predict(m,h,xc)*cp['sd']+cp['mu'];calibration=calibrate(raw,d['y'][d['cal']])
        np.savez_compressed(folder/'spectral_calibration.npz',indices=d['cal'],raw=raw,truth=d['y'][d['cal']])
    np.savez_compressed(folder/'spectral_validation.npz',indices=d['va'],predictions=pred,ridges=ridges,truth=d['y'][d['va']])
    eig=torch.linalg.eigvalsh(m.normalized_root()@m.normalized_root().T)
    rec=dict(dataset=name,seed=seed,kind=kind,training=chosen,selected=selected,trials=trials,calibration=calibration,max_relative_solve_residual=max(res+[h['residual']]),trace_error=abs(float(eig.sum())-1),min_metric_eigenvalue=float(eig.min()),split_info=d['split_info'],counts={a:len(d[a]) for a in ['tr','va','cal','te']})
    save(dest,rec);print('SPECTRAL SELECTED',rec,flush=True);return rec


def cat_select(name,seed,fixed=None):
    from catboost import CatBoostRegressor,CatBoostClassifier
    folder=OUT/name/f'seed{seed}';dest=folder/'catboost_selection.json'
    if dest.exists():return json.loads(dest.read_text())
    folder.mkdir(parents=True,exist_ok=True)
    d=load(name,seed);df=d['df'];y=d['y'];configs=base.tree_configs(16)['catboost'] if fixed is None else [fixed]
    C=CatBoostClassifier if d['classification'] else CatBoostRegressor
    best=None;trials=[];start=time.perf_counter()
    for i,cfg in enumerate(configs):
        model=C(**cfg,iterations=3000,loss_function='Logloss' if d['classification'] else 'RMSE',eval_metric='AUC' if d['classification'] else 'RMSE',random_seed=seed,thread_count=4,allow_writing_files=False,verbose=False)
        model.fit(df.iloc[d['tr']],y[d['tr']],cat_features=d['cats'],eval_set=(df.iloc[d['va']],y[d['va']]),early_stopping_rounds=150)
        raw=model.predict(df.iloc[d['va']],prediction_type='RawFormulaVal')
        criterion=primary(d,y[d['va']],raw);row=dict(trial=i,config=cfg,criterion=float(criterion),iterations=int(model.tree_count_));trials.append(row);save(folder/'catboost_trials.json',trials)
        if best is None or criterion<best[0]:best=(criterion,model,row)
        print('CATBOOST',name,seed,row,flush=True)
    model=best[1];model.save_model(str(folder/'catboost.cbm'));calibration=None
    if d['classification']:
        raw=model.predict(df.iloc[d['cal']],prediction_type='RawFormulaVal');calibration=calibrate(raw,y[d['cal']]);np.savez_compressed(folder/'catboost_calibration.npz',indices=d['cal'],raw=raw,truth=y[d['cal']])
    vraw=model.predict(df.iloc[d['va']],prediction_type='RawFormulaVal');np.savez_compressed(folder/'catboost_validation.npz',indices=d['va'],raw=vraw,truth=y[d['va']])
    rec=dict(dataset=name,seed=seed,selected=best[2],calibration=calibration,seconds=time.perf_counter()-start,counts={a:len(d[a]) for a in ['tr','va','cal','te']},split_info=d['split_info']);save(dest,rec);return rec


def evaluate():
    from catboost import CatBoostRegressor,CatBoostClassifier
    # No test predictions until both families have selections for every dataset.
    choices=[]
    for name in DATASETS:
        for seed in SEEDS:
            for family in ['spectral','catboost']:choices.append(json.loads((OUT/name/f'seed{seed}'/f'{family}_selection.json').read_text()))
    save(OUT/'all_selections_frozen.json',choices);results=[]
    for name in DATASETS:
        for seed in SEEDS:
            d=load(name,seed);folder=OUT/name/f'seed{seed}'
            for family in ['spectral','catboost']:
                choice=json.loads((folder/f'{family}_selection.json').read_text());start=time.perf_counter()
                if family=='spectral':
                    prep=MixedPreprocessor(choice['kind']).fit(d);m=MixedSpectral(len(d['nums']),prep.q);cp=torch.load(folder/choice['kind']/'checkpoint.pt',weights_only=True,map_location=DEVICE);m.load_state_dict(cp['state_dict']);m.eval();h=torch.load(folder/'head.pt',weights_only=True,map_location='cpu')
                    raw=head_predict(m,h,tensor(prep.transform(d['df'].iloc[d['te']])))*cp['sd']+cp['mu']
                else:
                    C=CatBoostClassifier if d['classification'] else CatBoostRegressor;model=C();model.load_model(str(folder/'catboost.cbm'));raw=model.predict(d['df'].iloc[d['te']],prediction_type='RawFormulaVal')
                if d['classification']:pred=apply_cal(raw,choice['calibration'])
                else:pred=np.clip(raw,0,d['y'][d['tr']].max()) if family=='spectral' and choice['selected']['clip'] else raw
                rec=dict(dataset=name,seed=seed,family=family,test=metrics(d,d['y'][d['te']],pred),prediction_seconds=time.perf_counter()-start,selection=choice)
                if d['classification']:rec['uncalibrated_auc']=float(roc_auc_score(d['y'][d['te']],raw))
                np.savez_compressed(folder/f'{family}_test.npz',raw=raw,prediction=pred,truth=d['y'][d['te']],test_indices=d['te'],train_indices=d['tr'],validation_indices=d['va'],calibration_indices=d['cal'])
                save(folder/f'{family}_test.json',rec);results.append(rec);print('TEST',name,seed,family,rec['test'],flush=True)
    summary={}
    for name in DATASETS:
        summary[name]={}
        for family in ['spectral','catboost']:
            rr=[r['test'] for r in results if r['dataset']==name and r['family']==family]
            summary[name][family]={k:dict(mean=float(np.mean([v[k] for v in rr])),std=float(np.std([v[k] for v in rr],ddof=1))) for k in rr[0]}
    save(OUT/'summary.json',dict(summary=summary,results=results))


def prepare():
    protocol=dict(datasets=DATASETS,seeds=SEEDS,geometry='One mixed-input bank: 4 learned frequencies per numeric input; fixed orthonormal contrasts for categorical fields; full trace-one matrix-exponential metric and learned Laplace bandwidth',
        spectral_training=dict(steps=STEPS,support=SUPPORT,query=QUERY,preprocessing_candidates=['standard','quantile'],recipe='Choose preprocessing on seed zero, freeze it for seeds one and two; per-seed checkpoint and head ridge selection',checkpoints='0,200,400,600,800 with Nystrom rank512; selected checkpoint rank2048 ridge search then neighboring three exact ridges',ridges=RIDGES.tolist(),loss='Squared error on standardized targets, also for binary labels'),
        trees='16 Sobol configurations on seed0, chosen configuration frozen for seeds1/2; 3000 iterations maximum, early-stop patience150; native categories; four CPU threads',
        classification='Primary ROC AUC; separate calibration partition fits sigmoid for both families; report log loss, Brier and 0.5-threshold accuracy. Calibration data never enter head training, early stopping or model selection.',
        splits='Bike first60%/next20%/last20% of dates; three optimizer seeds on one chronology. Taiwan stratified64/8/8/20 train/calibration/validation/test. Adult official test partition fixed; three80/10/10 fitting/calibration/validation splits of official training.',
        final='Freeze every model before test evaluation; no outer refit. Identical fitting, calibration, selection and test rows for both families.',
        bike_leakage='Exclude instant, dteday, casual, registered. Numeric calendar coordinates mnth/hr/weekday plus weather; categories season/yr/holiday/workingday/weathersit.',
        limitation='Search budgets differ by family; no replica or dictionary fusion; Adult and bike share test rows across optimizer seeds so SD is not independent-test uncertainty.')
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:save(p,protocol)
    snap=OUT/'source_snapshot';snap.mkdir(exist_ok=True);shutil.copyfile(__file__,snap/Path(__file__).name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['spectral','catboost','evaluate','inspect'],required=True);p.add_argument('--datasets',nargs='+',default=DATASETS);a=p.parse_args();prepare();torch.set_num_threads(4)
    with threadpool_limits(4):
        if a.phase=='evaluate':evaluate();return
        for name in a.datasets:
            if a.phase=='inspect':
                d=load(name,0);pr=MixedPreprocessor('standard').fit(d);print(name,len(d['y']),len(d['nums']),pr.q,{k:len(d[k]) for k in ['tr','va','cal','te']},d['split_info']);continue
            fixed=None
            for seed in SEEDS:
                if a.phase=='spectral':r=spectral_select(name,seed,fixed);fixed=r['kind']
                else:r=cat_select(name,seed,fixed);fixed=r['selected']['config']
if __name__=='__main__':main()
