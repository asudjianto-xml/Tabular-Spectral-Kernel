"""Transformer-style context encoder with PSD spectral attention operators.

For a fixed layer and reference context C, k_C(s,t) is a symmetric PSD
kernel on arbitrary tokens. Symmetric degree normalization preserves PSD;
it is not row-softmax attention. Nonlinear residual layers are not themselves
claimed to be Mercer kernels. Time participates in the shared feature map.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import math
import torch
from torch import nn


class SpectralKernel(nn.Module):
    def __init__(self,width=32,heads=4,coordinates=8,atoms=2):
        super().__init__();self.heads=heads;self.coordinates=coordinates;self.atoms=atoms
        self.projection=nn.Linear(width,heads*coordinates)
        frequencies=torch.linspace(.5,1.5,atoms).expand(heads,coordinates,atoms).clone()
        self.log_frequency=nn.Parameter(frequencies.log())
        self.time_frequency=nn.Parameter(torch.linspace(.1,1.,coordinates*atoms).reshape(1,coordinates,atoms).expand(heads,-1,-1).clone())
        dim=2*coordinates*atoms
        self.root=nn.Parameter(torch.eye(dim).expand(heads,-1,-1).clone())
        self.log_temperature=nn.Parameter(torch.full((heads,),math.log(.25)))

    def embedding(self,x,positions):
        b,n,_=x.shape
        projected=self.projection(x).reshape(b,n,self.heads,self.coordinates).permute(0,2,1,3)
        # Paired cosine/sine features rotate with chronological position.
        phase=projected[...,None]*self.log_frequency.exp()[None,:,None]
        phase=phase+positions[:,None,:,None,None]*self.time_frequency[None,:,None]
        phi=torch.stack([phase.cos(),phase.sin()],-1).flatten(-3)/math.sqrt(self.coordinates*self.atoms)
        root=self.root/self.root.square().sum((-2,-1),keepdim=True).sqrt().clamp_min(1e-12)
        return torch.einsum("bhnd,hde->bhne",phi,root)

    def kernel(self,z,w):
        distance=(z.square().sum(-1)[...,None]+w.square().sum(-1)[...,None,:]-2*z@w.transpose(-1,-2)).clamp_min(0.)
        return torch.exp(-distance/(2*self.log_temperature.exp()[None,:,None,None].square()))


class PSDSpectralAttention(nn.Module):
    def __init__(self,width=32):
        super().__init__();self.geometry=SpectralKernel(width)
        self.head_logits=nn.Parameter(torch.zeros(self.geometry.heads))
        self.value=nn.Linear(width,width);self.out=nn.Linear(width,width)

    def normalized_blocks(self,context,context_positions,query=None,query_positions=None):
        z=self.geometry.embedding(context,context_positions);k=self.geometry.kernel(z,z)
        dc=k.sum(-1).clamp_min(1e-12)
        a=k/torch.sqrt(dc[...,None]*dc[...,None,:])
        beta=self.head_logits.softmax(0)
        self_attention=torch.einsum("h,bhnm->bnm",beta,a)
        if query is None:return self_attention
        qz=self.geometry.embedding(query,query_positions);q=self.geometry.kernel(qz,z)
        dq=q.sum(-1).clamp_min(1e-12)
        cross=q/torch.sqrt(dq[...,None]*dc[...,None,:])
        # This extends the same reference-context normalized kernel to queries.
        # Query degrees use context only, so other queries cannot change a row.
        qq=self.geometry.kernel(qz,qz)/torch.sqrt(dq[...,None]*dq[...,None,:])
        return (self_attention,torch.einsum("h,bhqn->bqn",beta,cross),
                torch.einsum("h,bhqr->bqr",beta,qq))

    def forward(self,context,context_positions,query=None,query_positions=None):
        blocks=self.normalized_blocks(context,context_positions,query,query_positions)
        weights=blocks if query is None else blocks[1]
        return self.out(weights@self.value(context))


class PSDLayer(nn.Module):
    def __init__(self,width):
        super().__init__();self.attention=PSDSpectralAttention(width)
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Linear(64,width))

    def forward(self,x,positions):
        z=self.norm1(x);x=x+self.attention(z,positions)
        return x+self.ff(self.norm2(x))


class PSDSpectralGate(nn.Module):
    """Drop-in gate for the existing full-context NW/KRR meta-training loop."""
    def __init__(self,width=32,temporal_rotation=True):
        super().__init__();self.input=nn.Linear(8,width)
        self.layers=nn.ModuleList([PSDLayer(width) for _ in range(2)])
        self.attention=PSDSpectralAttention(width);self.norm=nn.LayerNorm(width)
        self.head=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,2))
        if not temporal_rotation:
            for module in self.modules():
                if isinstance(module,SpectralKernel):
                    with torch.no_grad():module.time_frequency.zero_()
                    module.time_frequency.requires_grad_(False)

    def forward(self,context,query,descriptor):
        cp=context[...,7]*4;qp=query[...,7]*4
        memory=self.input(context)
        for layer in self.layers:memory=layer(memory,cp)
        q=self.input(query);attended=self.attention(memory,cp,q,qp)
        return self.head(self.norm(q+attended)).softmax(-1)
