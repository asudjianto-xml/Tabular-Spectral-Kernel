"""One kernel-pooling layer reusing the frozen regression banks: 596 parameters."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import torch
from torch import nn


class SimpleSpectralGate(nn.Module):
    uses_kernel_blocks=True

    def __init__(self,width=16,normalization="symmetric",power=1):
        super().__init__()
        if normalization not in ("symmetric","row") or power not in (1,2):
            raise ValueError("Use symmetric/row normalization and Schur power 1 or 2")
        self.normalization=normalization
        self.power=power
        self.context_mix=nn.Linear(8,2)
        self.value=nn.Sequential(nn.Linear(8,width),nn.GELU())
        self.head=nn.Sequential(nn.Linear(width+8,width),nn.GELU(),nn.Linear(width,2))

    def attention_blocks(self,context,grams,cross):
        # Preliminary beta is context-conditioned and shared across queries.
        beta=self.context_mix(context.mean(1)).softmax(-1)
        grams=grams.to(context.dtype);cross=cross.to(context.dtype)
        k=torch.einsum("bj,bjst->bst",beta,grams)
        q=torch.einsum("bj,bjqs->bqs",beta,cross)
        # Integer Schur powers preserve PSD of the conditional fused kernel.
        k=k.pow(self.power);q=q.pow(self.power)
        dc=k.sum(-1).clamp_min(1e-12);dq=q.sum(-1).clamp_min(1e-12)
        if self.normalization=="row":
            a=k/dc[...,None];aq=q/dq[...,None]
        else:
            a=k/torch.sqrt(dc[...,None]*dc[...,None,:])
            aq=q/torch.sqrt(dq[...,None]*dc[:,None,:])
        return beta,a,aq,dc,dq

    def forward(self,context,query,descriptor=None,*,grams,cross):
        _,_,aq,_,_=self.attention_blocks(context,grams,cross)
        pooled=aq@self.value(context)
        # Final alpha varies by query; use it consistently in that query's KRR.
        return self.head(torch.cat([query,pooled],-1)).softmax(-1)
