"""Exact kernel ridge path with an unpenalized intercept."""
import torch
from . import identifiable_spectral_california as base
@torch.no_grad()
def exact_path(model,x,y,xq,ridges):
    z=model.embedding(x);zq=model.embedding(xq)
    n=len(y);device=x.device
    # Allocate once rather than cat a list of full-sized kernel blocks.
    k=torch.empty((n,n),dtype=base.DTYPE,device=device)
    for start in range(0,n,512):k[start:start+512]=model.kernel(z[start:start+512],z)
    kq=base.kernel_blocks(model,zq,z,chunk=512)
    rhs=torch.stack([y,torch.ones_like(y)],dim=1)
    pred=[];residuals=[]
    for ridge in ridges:
        lam=n*ridge
        k.diagonal().add_(lam)
        l=torch.linalg.cholesky(k)
        sol=torch.cholesky_solve(rhs,l)
        intercept=sol[:,0].sum()/sol[:,1].sum()
        coef=sol[:,0]-intercept*sol[:,1]
        res=torch.linalg.vector_norm(k@coef+intercept-y)/torch.linalg.vector_norm(y-y.mean())
        pred.append(kq@coef+intercept);residuals.append(float(res))
        k.diagonal().sub_(lam)
        del l,sol,coef
    return torch.stack(pred,dim=1),residuals

