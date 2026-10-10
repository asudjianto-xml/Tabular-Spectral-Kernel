"""MLP capacity and placement ablations with unchanged regression banks."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import torch
from torch import nn
from simple_spectral_gate import SimpleSpectralGate
from run_transformer import TransformerGate
from psd_spectral_attention import PSDSpectralGate


class LargeHeadGate(SimpleSpectralGate):
    def __init__(self):
        super().__init__()
        # Preserve initialization of beta and value; change only post-pooling head.
        self.head=nn.Sequential(nn.Linear(24,128),nn.GELU(),
                                nn.Linear(128,128),nn.GELU(),nn.Linear(128,2))


class TransformerNoFF(TransformerGate):
    def __init__(self):
        super().__init__()
        # Keep the original RNG draw sequence and all shared initial tensors.
        for layer in self.context.layers:
            for module in [layer.linear1,layer.linear2,layer.norm2]:
                module.requires_grad_(False)

    def forward(self,context,query,descriptor):
        memory=self.input(context)
        for layer in self.context.layers:
            z=layer.norm1(memory)
            memory=memory+layer.self_attn(z,z,z,need_weights=False)[0]
        q=self.input(query)
        attended=self.attention(q,memory,memory,need_weights=False)[0]
        return self.head(self.norm(q+attended)).softmax(-1)


class FrozenPSDLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1=nn.LayerNorm(32);self.value=nn.Linear(32,32);self.out=nn.Linear(32,32)
        self.norm2=nn.LayerNorm(32)
        self.ff=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,32))

    def forward(self,x,a,use_ff):
        x=x+self.out(a@self.value(self.norm1(x)))
        return x+self.ff(self.norm2(x)) if use_ff else x


class StackedFrozenPSDGate(nn.Module):
    uses_kernel_blocks=True

    def __init__(self,use_ff=True):
        super().__init__();self.use_ff=use_ff
        self.context_mix=nn.Linear(8,2);self.input=nn.Linear(8,32)
        self.layers=nn.ModuleList([FrozenPSDLayer() for _ in range(2)])
        self.value=nn.Linear(32,32);self.out=nn.Linear(32,32);self.norm=nn.LayerNorm(32)
        self.head=nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Linear(32,2))
        if not use_ff:
            for layer in self.layers:
                layer.ff.requires_grad_(False);layer.norm2.requires_grad_(False)

    def attention_blocks(self,context,grams,cross):
        beta=self.context_mix(context.mean(1)).softmax(-1)
        k=torch.einsum('bj,bjst->bst',beta,grams.to(context.dtype))
        q=torch.einsum('bj,bjqs->bqs',beta,cross.to(context.dtype))
        dc=k.sum(-1).clamp_min(1e-12);dq=q.sum(-1).clamp_min(1e-12)
        return beta,k/torch.sqrt(dc[...,None]*dc[...,None,:]),q/torch.sqrt(dq[...,None]*dc[:,None,:]),dc,dq

    def forward(self,context,query,descriptor=None,*,grams,cross):
        _,a,aq,_,_=self.attention_blocks(context,grams,cross)
        memory=self.input(context)
        for layer in self.layers:memory=layer(memory,a,self.use_ff)
        q=self.input(query)+self.out(aq@self.value(memory))
        return self.head(self.norm(q)).softmax(-1)


class LinearSimplePSD(SimpleSpectralGate):
    def __init__(self):
        super().__init__()
        self.value=self.value[0]
        self.head=nn.Linear(24,2)


class LinearTransformer(TransformerNoFF):
    def __init__(self):
        super().__init__()
        for layer in self.context.layers:
            del layer.linear1,layer.linear2,layer.norm2
        self.head=nn.Linear(32,2)


class LinearStackedPSD(StackedFrozenPSDGate):
    def __init__(self):
        super().__init__(use_ff=False)
        for layer in self.layers:
            del layer.ff,layer.norm2
        self.head=nn.Linear(32,2)


class LinearLearnedPSD(PSDSpectralGate):
    def __init__(self):
        super().__init__(temporal_rotation=False)
        for layer in self.layers:
            del layer.ff,layer.norm2
        self.head=nn.Linear(32,2)

    def forward(self,context,query,descriptor):
        cp=context[...,7]*4;qp=query[...,7]*4
        memory=self.input(context)
        for layer in self.layers:
            memory=memory+layer.attention(layer.norm1(memory),cp)
        q=self.input(query)
        return self.head(self.norm(q+self.attention(memory,cp,q,qp))).softmax(-1)


def construct(variant):
    return {'large_head':LargeHeadGate,'transformer_no_ff':TransformerNoFF,
            'stacked_psd':StackedFrozenPSDGate,
            'stacked_psd_no_ff':lambda:StackedFrozenPSDGate(False),
            'linear_simple':LinearSimplePSD,'linear_stacked':LinearStackedPSD,
            'linear_transformer':LinearTransformer,'linear_learned_psd':LinearLearnedPSD}[variant]()
