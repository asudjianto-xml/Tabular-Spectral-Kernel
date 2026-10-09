"""Single-bank adapter preserving the published checkpoint layout.

The historical class name is retained for checkpoint/optimizer compatibility;
multiple banks are not exposed by this release.
"""
import math
import numpy as np
import torch
from . import identifiable_spectral_california as base
DTYPE=base.DTYPE
class FusedCanonicalSpectral(torch.nn.Module):
    def __init__(self,d=8,atoms=8,banks=1,max_frequency=.5,device='cpu'):
        super().__init__()
        if banks != 1: raise ValueError("This release implements the paper single-bank model only")
        self.d,self.atoms,self.banks=d,atoms,banks
        self.p=2*d*atoms
        members=[]
        for temperature in np.geomspace(.25,1.,banks):
            bank=base.CanonicalSpectral(d,atoms,max_frequency,device)
            del bank.log_ridge  # A single fused ridge, no unused per-bank parameters.
            with torch.no_grad():bank.log_temperature.fill_(math.log(temperature))
            members.append(bank)
        self.members=torch.nn.ModuleList(members)
        # Fix final logit to zero, removing the additive softmax-logit gauge.
        self.weight_coordinates=torch.nn.Parameter(torch.zeros(banks-1,dtype=DTYPE,device=device))
        self.log_ridge=torch.nn.Parameter(torch.tensor(math.log(3e-4),dtype=DTYPE,device=device))

    def weights(self):
        return torch.cat([self.weight_coordinates,self.weight_coordinates.new_zeros(1)]).softmax(0)

    def normalized_root(self):
        generators=torch.stack([m.generator() for m in self.members])
        roots=torch.matrix_exp(.5*generators)
        return roots/torch.linalg.vector_norm(roots,dim=(-2,-1),keepdim=True)

    def embedding(self,x,root=None):
        if root is None:root=self.normalized_root()
        # Keep rows on axis zero so the existing fused Nystrom decoder is reusable.
        return torch.stack([m.embedding(x,root[h]) for h,m in enumerate(self.members)],dim=1)

    def kernel(self,za,zb):
        return sum(w*m.kernel(za[:,h],zb[:,h])
                   for h,(w,m) in enumerate(zip(self.weights(),self.members)))

    def regularizer(self):
        return 1e-4*sum(m.offdiag.square().sum()+m.diagonal.square().sum()
                       for m in self.members)/(self.banks*self.p)

    def clamp_parameters(self):
        with torch.no_grad():
            for m in self.members:
                m.log_gaps.clamp_(-9,2);m.log_temperature.clamp_(-5,3)
            self.log_ridge.clamp_(math.log(1e-7),math.log(.01))


def diagnostics(model):
    roots=model.normalized_root();records=[]
    for h,m in enumerate(model.members):
        a=m.generator();b=roots[h]@roots[h].T
        eig,u=torch.linalg.eigh(b)
        logb=(u*eig.log())@u.T
        inverse=logb-logb.trace()/m.p*torch.eye(m.p,dtype=DTYPE,device=b.device)
        w=b.diag().reshape(model.d,model.atoms,2).sum(-1)
        importance=w.sum(-1);amplitude=(w/importance[:,None]).sqrt()
        diag=(w.flatten()/2).sqrt().repeat_interleave(2)
        c=b/diag[:,None]/diag[None,:]
        errors=dict(trace_error=abs(float(b.trace())-1),inverse_error=float((inverse-a).abs().max()),
                    shape_trace_error=float((c.diag().reshape(-1,2).sum(-1)-2).abs().max()))
        assert errors['trace_error']<1e-12 and errors['inverse_error']<1e-6 and errors['shape_trace_error']<1e-12
        assert float(eig.min())>0 and torch.all(m.frequencies().diff(dim=1)>0)
        records.append(dict(bank=h,temperature=float(m.log_temperature.exp()),metric_min_eigenvalue=float(eig.min()),
            metric_condition=float(eig.max()/eig.min()),importance=importance.cpu().tolist(),
            amplitudes=amplitude.cpu().tolist(),frequencies=m.frequencies().cpu().tolist(),checks=errors))
    weights=model.weights().cpu().numpy()
    return dict(weights=weights.tolist(),effective_banks=float(1/(weights@weights)),banks=records,
                metric_dimension_per_bank=model.p,total_parameters=sum(p.numel() for p in model.parameters()))

