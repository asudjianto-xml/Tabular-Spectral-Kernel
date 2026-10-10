"""Query-conditioned Transformer fusion on frozen unemployment spectral banks.

Matched full-context meta-training of Transformer, summary MLP, and constant
simplex weights. Evaluation stream labels never select models or settings.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import run as R
import run_meta as M
import run_credit_card as C
import unemployment_data as U


class TransformerGate(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.input=nn.Linear(8,width)
        layer=nn.TransformerEncoderLayer(width,4,64,dropout=0.,batch_first=True,
                                         activation="gelu",norm_first=True)
        self.context=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.attention=nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)
        self.norm=nn.LayerNorm(width)
        self.head=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,2))

    def forward(self, context, query, descriptor):
        memory=self.context(self.input(context))
        q=self.input(query)
        attended,_=self.attention(q,memory,memory,need_weights=False)
        return self.head(self.norm(q+attended)).softmax(-1)


class SummaryGate(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(3,16),nn.Tanh(),nn.Linear(16,2))

    def forward(self,context,query,descriptor):
        return self.net(descriptor).softmax(-1)[:,None].expand(-1,query.shape[1],-1)


class ConstantGate(nn.Module):
    def __init__(self):
        super().__init__();self.logits=nn.Parameter(torch.zeros(2))

    def forward(self,context,query,descriptor):
        return self.logits.softmax(-1).expand(len(context),query.shape[1],-1)


GATES={"constant":ConstantGate,"summary":SummaryGate,"transformer":TransformerGate}


def tokens(cx,cy,qx):
    """No query labels accepted. Time is relative cohort, not row position."""
    age=np.repeat(np.arange(-4,0)/4,32)
    context=np.column_stack([cx,cy,np.ones(len(cy)),age])
    query=np.column_stack([qx,np.zeros((len(qx),3))])
    return context,query


def episode(rng,queries):
    bx,_=C.covariates(rng,128+queries)
    rates=np.clip(rng.uniform(.035,.105)+np.cumsum(rng.normal(0,.006,13)),.03,.12)
    macro,_=U.exposure(rates)
    rate=np.r_[np.repeat(rates[-5:-1],32),np.repeat(rates[-1],queries)]
    stress=np.r_[np.repeat(macro[-5:-1],32),np.repeat(macro[-1],queries)]
    mean,_=U.conditional_mean(bx,stress,localized=bool(rng.integers(2)))
    return U.feature_matrix(bx,rate),C.responses(rng,mean)


def blocks(models,temps,x):
    zs=[R.embedding(m,x) for m in models]
    return ([R.kernel(z[:128],z[:128],t) for z,t in zip(zs,temps)],
            [R.kernel(z[128:],z[:128],t) for z,t in zip(zs,temps)])


def make_dataset(models,temps,seed,count,queries):
    rng=np.random.default_rng(seed);data={k:[] for k in ["context","query","descriptor","grams","cross","y","target"]}
    for _ in range(count):
        x,y=episode(rng,queries);g,q=blocks(models,temps,x)
        ct,qt=tokens(x[:128],y[:128],x[128:])
        for key,value in dict(context=ct,query=qt,descriptor=M.descriptor(g,y[:128]),
                              grams=g,cross=q,y=y[:128],target=y[128:]).items():data[key].append(value)
    return {k:torch.tensor(np.asarray(v),dtype=torch.float32 if k in ["context","query","descriptor"] else torch.float64) for k,v in data.items()}


def decode(alpha,grams,cross,y,ridge):
    """Same per-query simplex weights in every block of its local KRR fit."""
    alpha=alpha.double();n=y.shape[-1]
    k=torch.einsum("bqj,bjst->bqst",alpha,grams)
    q=torch.einsum("bqj,bjqs->bqs",alpha,cross)
    nw=(q*y[:,None]).sum(-1)/q.sum(-1).clamp_min(1e-300)
    chol=torch.linalg.cholesky(k+n*ridge*torch.eye(n,dtype=k.dtype,device=k.device))
    rhs=torch.stack([y,torch.ones_like(y)],-1)[:,None].expand(-1,alpha.shape[1],-1,-1)
    sol=torch.cholesky_solve(rhs,chol)
    intercept=sol[...,0].sum(-1)/sol[...,1].sum(-1)
    coef=sol[...,0]-intercept[...,None]*sol[...,1]
    krr=(q*coef).sum(-1)+intercept
    return nw,krr


def gate_weights(model,batch):
    if getattr(model,"uses_kernel_blocks",False):
        return model(batch["context"],batch["query"],batch["descriptor"],grams=batch["grams"],cross=batch["cross"])
    return model(batch["context"],batch["query"],batch["descriptor"])


def loss_for(model,batch,ridge,objective="joint"):
    alpha=gate_weights(model,batch)
    nw,krr=decode(alpha,batch["grams"],batch["cross"],batch["y"],ridge)
    if objective=="krr":return (krr-batch["target"]).square().mean()
    if objective!="joint":raise ValueError(objective)
    return .5*((nw-batch["target"]).square().mean()+(krr-batch["target"]).square().mean())


def train(kind,train_data,validation,ridge,seed,args):
    torch.manual_seed(92000+seed)
    constructor=GATES[kind]
    if getattr(args,"psd",False) and kind=="transformer":
        from psd_spectral_attention import PSDSpectralGate
        constructor=lambda:PSDSpectralGate(temporal_rotation=not getattr(args,"no_temporal_rotation",False))
    if getattr(args,"rotary",False) and kind=="transformer":
        from rotary_gate import RotaryTransformerGate
        constructor=RotaryTransformerGate
    if getattr(args,"simple",False) and kind=="transformer":
        from simple_spectral_gate import SimpleSpectralGate
        constructor=lambda:SimpleSpectralGate(normalization=getattr(args,"pool_normalization","symmetric"),power=getattr(args,"pool_power",1))
    if getattr(args,"mlp_ablation",None) and kind=="transformer":
        from mlp_ablation import construct
        constructor=lambda:construct(args.mlp_ablation)
    model=constructor()
    opt=torch.optim.Adam(model.parameters(),lr=.01 if kind=="constant" else .001)
    rng=np.random.default_rng(93000+seed);best=float("inf");best_epoch=-1;trace=[];best_state=None
    for epoch in range(args.epochs):
        model.train();loss_sum=0.;order=rng.permutation(len(train_data["y"]))
        for start in range(0,len(order),args.batch):
            ix=order[start:start+args.batch];b={k:v[ix] for k,v in train_data.items()}
            loss=loss_for(model,b,ridge,getattr(args,"loss","joint"));opt.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();loss_sum+=float(loss.detach())*len(ix)
        model.eval();vloss=0.
        with torch.no_grad():
            for start in range(0,len(validation["y"]),args.batch):
                b={k:v[start:start+args.batch] for k,v in validation.items()}
                vloss+=float(loss_for(model,b,ridge,getattr(args,"loss","joint")))*len(b["y"])
        vloss/=len(validation["y"])
        trace.append(dict(epoch=epoch,train_loss=loss_sum/len(order),validation_loss=vloss))
        if vloss<best:
            best=vloss;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
        if epoch%5==0:print(seed,kind,epoch,"train",round(trace[-1]["train_loss"],6),"validation",round(vloss,6),flush=True)
        if epoch-best_epoch>=args.patience:break
    model.load_state_dict(best_state);model.eval()
    return model,dict(best_epoch=best_epoch,best_validation_loss=best,trace=trace,
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()))


def execute(args):
    if getattr(args,"mlp_ablation",None) and any(getattr(args,k,False) for k in ['simple','psd','rotary']):
        raise ValueError('Use --mlp-ablation without other architecture flags')
    if getattr(args,"simple",False) and (getattr(args,"psd",False) or getattr(args,"rotary",False)):
        raise ValueError("Select the simple gate or a learned-attention architecture, not both")
    if getattr(args,"no_temporal_rotation",False) and not getattr(args,"psd",False):
        raise ValueError("--no-temporal-rotation applies only to --psd; the plain Transformer already uses month features without rotations")
    if getattr(args,"psd",False) and getattr(args,"rotary",False):
        raise ValueError("PSD spectral attention already includes temporal spectral rotations; select one architecture")
    out=args.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    methods=getattr(args,"methods",list(GATES))
    if "transformer" not in methods:raise ValueError("Include transformer for the encoder diagnostics")
    start=time.perf_counter();metrics=[];provenance={};diagnostics=[]
    for file in [Path(__file__),Path(R.__file__),Path(M.__file__),Path(C.__file__),Path(U.__file__),R.MODEL_SOURCE]:
        (out/(file.stem+"_snapshot.py")).write_bytes(file.read_bytes())
    if getattr(args,"rotary",False):
        file=Path(__file__).with_name("rotary_gate.py")
        (out/"rotary_gate_snapshot.py").write_bytes(file.read_bytes())
    if getattr(args,"psd",False):
        file=Path(__file__).with_name("psd_spectral_attention.py")
        (out/"psd_spectral_attention_snapshot.py").write_bytes(file.read_bytes())
    if getattr(args,"simple",False):
        file=Path(__file__).with_name("simple_spectral_gate.py")
        (out/"simple_spectral_gate_snapshot.py").write_bytes(file.read_bytes())
    if getattr(args,"mlp_ablation",None):
        for name in ['mlp_ablation.py','simple_spectral_gate.py','psd_spectral_attention.py']:
            (out/(Path(name).stem+'_snapshot.py')).write_bytes(Path(__file__).with_name(name).read_bytes())
    def record(path):provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    R.save(out/"protocol.json",dict(seeds=args.seeds,profile="original",train_episodes=args.episodes,
        validation_episodes=args.validation,queries_per_episode=args.queries,max_epochs=args.epochs,batch=args.batch,
        patience=args.patience,architecture=("MLP ablation: "+args.mlp_ablation) if getattr(args,"mlp_ablation",None) else ("596-parameter gate: context mean -> beta2; one normalized frozen-kernel query pool of 16 learned values; query+pool ->16->alpha2" if getattr(args,"simple",False) else "8 input channels -> width32, 2 context self-attention layers, 4 heads; query cross-attention; softmax2"),
        simple_spectral_pooling=getattr(args,"simple",False),
        mlp_ablation=getattr(args,"mlp_ablation",None),
        pool_normalization=getattr(args,"pool_normalization","symmetric") if getattr(args,"simple",False) else None,
        pool_power=getattr(args,"pool_power",1) if getattr(args,"simple",False) else None,
        methods=methods,rotary=getattr(args,"rotary",False),
        psd_spectral_attention=getattr(args,"psd",False),
        spectral_temporal_rotation=not getattr(args,"no_temporal_rotation",False) if getattr(args,"psd",False) else None,
        psd_details="4 spectral kernels per layer, Gaussian on learned Fourier/time embeddings, trace-normalized PSD metric, convex head fusion, symmetric reference-context normalization" if getattr(args,"psd",False) else None,
        rotary_position="relative month in self- and cross-attention; same-cohort borrowers share position; base10000" if getattr(args,"rotary",False) else None,
        channels="5 normalized features, observed response (query zero), response-available mask, relative month",
        objective="KRR squared query error only, no direct weight supervision" if getattr(args,"loss","joint")=="krr" else "equal mean NW and KRR squared query errors, no direct weight supervision",
        controls="same full128 contexts, training/validation episodes, frozen kernels, ridge, and mean query loss; independent training per seed",
        optimizer="Adam; transformer/summary lr .001, constant lr .01; gradient norm clip1; no dropout",
        selection="best independent validation KRR loss, patience stopping; evaluation streams excluded" if getattr(args,"loss","joint")=="krr" else "best independent validation joint loss, patience stopping; evaluation streams excluded",
        inference="query-dependent alpha, same alpha for local Gram and cross-vector; no inference optimization",
        limitations=["Synthetic task family shared between meta-training and evaluation","Only original bandwidth profile tested","No exact preservation constraint for stable borrowers","Each query defines a local PSD kernel, not one global Gram across differently gated queries"] ))
    for seed in args.seeds:
        src=args.source/f"seed{seed}";dest=out/f"seed{seed}";dest.mkdir();banks=[]
        for name in ["normal","downturn"]:
            path=src/f"{name}.pt";record(path);m=R.base.CanonicalSpectral(5,4,max_frequency=3.,device="cpu")
            m.load_state_dict(torch.load(path,map_location="cpu",weights_only=True));m.eval();banks.append(m)
        path=src/"original_training.json";record(path);info=json.loads(path.read_text());ridge=info["ridge"]
        train_data=make_dataset(banks,info["temperatures"],101000+seed,args.episodes,args.queries)
        validation=make_dataset(banks,info["temperatures"],102000+seed,args.validation,args.queries)
        models={}
        for kind in methods:
            model,training=train(kind,train_data,validation,ridge,seed,args);models[kind]=model
            torch.save(model.state_dict(),dest/f"{kind}.pt");R.save(dest/f"{kind}_training.json",training)
        del train_data,validation
        for scenario in C.SCENARIOS:
            path=src/f"original_{scenario}.npz";record(path)
            with np.load(path) as saved:data={k:saved[k] for k in saved.files}
            x,y=data["x"],data["y"];n=len(y)
            predictions={kind+"_"+dec:np.full(n,np.nan) for kind in methods for dec in ["nw","krr"]}
            weights={kind:np.full((n,2),np.nan) for kind in methods};query_sds=[]
            for origin in range(128,n,32):
                sl=slice(origin-128,origin);g,q=blocks(banks,info["temperatures"],x[origin-128:origin+32])
                ct,qt=tokens(x[sl],y[sl],x[origin:origin+32])
                b={k:torch.tensor(np.asarray(v)[None],dtype=torch.float32 if k in ["context","query","descriptor"] else torch.float64)
                    for k,v in dict(context=ct,query=qt,descriptor=M.descriptor(g,y[sl]),grams=g,cross=q,y=y[sl]).items()}
                with torch.no_grad():
                    for kind,model in models.items():
                        alpha=gate_weights(model,b)
                        nw,krr=decode(alpha,b["grams"],b["cross"],b["y"],ridge)
                        for dec,p in [("nw",nw),("krr",krr)]:predictions[kind+"_"+dec][origin:origin+32]=np.clip(p.numpy()[0],0,1)
                        weights[kind][origin:origin+32]=alpha.numpy()[0]
                    # Counterfactual encoder-only label shuffle; readout labels stay intact.
                    shuffled=b["context"].clone();shuffled[:,:,5]=shuffled[:,torch.arange(127,-1,-1),5]
                    altered=gate_weights(models["transformer"],dict(b,context=shuffled))
                    query_sds.append(dict(query_sd=float(alpha[0,:,0].std(unbiased=False)),
                        label_shuffle_weight_change=float((altered-alpha).abs().mean())))
                # Verify one independently solved Transformer query per origin.
                a=weights["transformer"][origin]
                check=R.ridge_predict(sum(w*v for w,v in zip(a,g)),sum(w*v[:1] for w,v in zip(a,q)),y[sl],ridge)
                np.testing.assert_allclose(np.clip(check[0],0,1),predictions["transformer_krr"][origin],atol=1e-9)
            valid=np.arange(n)>=128
            for kind,w in weights.items():
                np.testing.assert_allclose(w[valid].sum(1),1,atol=1e-6)
                assert np.isfinite(w[valid]).all() and (w[valid]>=0).all()
            np.savez_compressed(dest/(scenario+".npz"),**predictions,**{k+"_alpha":v for k,v in weights.items()})
            for kind in methods+["previous_summary","raw_retrieval"]:
                for dec in ["nw","krr"]:
                    pred=(data[("full128" if kind=="previous_summary" else "raw_nearest64")+"_"+dec]
                          if kind in ["previous_summary","raw_retrieval"] else predictions[kind+"_"+dec])
                    row=dict(seed=seed,scenario=scenario,method=kind,decoder=dec,
                        rmse=float(np.sqrt(np.mean((pred[valid]-y[valid])**2))),
                        clean_rmse=float(np.sqrt(np.mean((pred[valid]-data["mean"][valid])**2))))
                    if scenario=="localized_downturn":
                        for group,mask in [("stable",valid&(data["raw"][:,2]>2)),("changed",valid&(data["macro_exposure"]>.1)&(data["raw"][:,2]<=2))]:
                            row[group+"_rmse"]=float(np.sqrt(np.mean((pred[mask]-y[mask])**2)))
                    metrics.append(row)
            diagnostics.append(dict(seed=seed,scenario=scenario,
                weights={k:dict(normal_bank_quantiles=np.quantile(w[valid,0],[0,.25,.5,.75,1]).tolist(),sd=float(w[valid,0].std())) for k,w in weights.items()},
                mean_within_cohort_query_sd=float(np.mean([r["query_sd"] for r in query_sds])),
                encoder_label_shuffle_mean_weight_change=float(np.mean([r["label_shuffle_weight_change"] for r in query_sds]))))
            print("EVAL",seed,scenario,{k:round(100*np.sqrt(np.mean((p[valid]-y[valid])**2)),3) for k,p in predictions.items()},flush=True)
        R.save(out/"metrics.json",metrics);R.save(out/"diagnostics.json",diagnostics);R.save(out/"provenance.json",provenance)
    summary=[]
    for scenario in C.SCENARIOS:
        for kind in methods+["previous_summary","raw_retrieval"]:
            for dec in ["nw","krr"]:
                rr=[r for r in metrics if (r["scenario"],r["method"],r["decoder"])==(scenario,kind,dec)]
                summary.append(dict(scenario=scenario,method=kind,decoder=dec,
                    rmse_mean=float(np.mean([r["rmse"] for r in rr])),rmse_seed_sd=float(np.std([r["rmse"] for r in rr],ddof=1)) if len(rr)>1 else 0.,
                    **({g+"_rmse":float(np.mean([r[g+"_rmse"] for r in rr])) for g in ["stable","changed"]} if scenario=="localized_downturn" else {})))
    R.save(out/"summary.json",summary);R.save(out/"runtime.json",dict(seconds=time.perf_counter()-start))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path(__file__).parent/"unemployment_results")
    p.add_argument("--output",type=Path,default=Path(__file__).parent/"transformer_results")
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    p.add_argument("--episodes",type=int,default=512);p.add_argument("--validation",type=int,default=128)
    p.add_argument("--queries",type=int,default=4);p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--batch",type=int,default=16);p.add_argument("--patience",type=int,default=6)
    p.add_argument("--rotary",action="store_true")
    p.add_argument("--psd",action="store_true",help="PSD spectral attention with temporal Fourier features")
    p.add_argument("--simple",action="store_true",help="single PSD pooling layer reusing frozen kernels")
    p.add_argument('--mlp-ablation',choices=['large_head','transformer_no_ff','stacked_psd','stacked_psd_no_ff','linear_simple','linear_stacked','linear_transformer','linear_learned_psd'])
    p.add_argument("--pool-normalization",choices=["symmetric","row"],default="symmetric")
    p.add_argument("--pool-power",type=int,choices=[1,2],default=1,help="integer Schur power of fused pooling kernel; regression kernels unchanged")
    p.add_argument("--no-temporal-rotation",action="store_true",help="PSD ablation: freeze temporal spectral frequencies to zero; retain month input feature")
    p.add_argument("--loss",choices=["joint","krr"],default="joint")
    p.add_argument("--methods",nargs="+",choices=list(GATES),default=list(GATES))
    with threadpool_limits(limits=1):execute(p.parse_args())
