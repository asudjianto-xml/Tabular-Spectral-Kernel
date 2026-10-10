"""Month-position rotary attention for context-conditioned kernel fusion."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import torch
from torch import nn
from torch.nn import functional as F


def rotate(x,positions):
    """x: batch, heads, tokens, channels; positions: batch, tokens (months)."""
    inv=10000.**(-torch.arange(0,x.shape[-1],2,device=x.device,dtype=x.dtype)/x.shape[-1])
    angle=positions[:,None,:,None]*inv
    even,odd=x[...,0::2],x[...,1::2]
    return torch.stack([even*angle.cos()-odd*angle.sin(),
                        even*angle.sin()+odd*angle.cos()],-1).flatten(-2)


class RotaryAttention(nn.Module):
    def __init__(self,width=32,heads=4):
        super().__init__();self.heads=heads
        self.q=nn.Linear(width,width);self.k=nn.Linear(width,width)
        self.v=nn.Linear(width,width);self.out=nn.Linear(width,width)

    def forward(self,query,memory,query_positions,memory_positions):
        def split(x):return x.reshape(x.shape[0],x.shape[1],self.heads,-1).transpose(1,2)
        q=rotate(split(self.q(query)),query_positions)
        k=rotate(split(self.k(memory)),memory_positions)
        v=split(self.v(memory))
        attended=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
        return self.out(attended.transpose(1,2).reshape(query.shape))


class RotaryLayer(nn.Module):
    def __init__(self,width):
        super().__init__();self.attention=RotaryAttention(width)
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Linear(64,width))

    def forward(self,x,positions):
        z=self.norm1(x);x=x+self.attention(z,z,positions,positions)
        return x+self.ff(self.norm2(x))


class RotaryTransformerGate(nn.Module):
    def __init__(self,width=32):
        super().__init__();self.input=nn.Linear(8,width)
        self.layers=nn.ModuleList([RotaryLayer(width) for _ in range(2)])
        self.attention=RotaryAttention(width);self.norm=nn.LayerNorm(width)
        self.head=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,2))

    def forward(self,context,query,descriptor):
        # The eighth channel stores month/4. Same-cohort borrowers share a time.
        cp=context[...,7]*4;qp=query[...,7]*4
        memory=self.input(context)
        for layer in self.layers:memory=layer(memory,cp)
        q=self.input(query)
        attended=self.attention(q,memory,qp,cp)
        return self.head(self.norm(q+attended)).softmax(-1)
