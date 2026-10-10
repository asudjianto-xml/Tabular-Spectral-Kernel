import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import pytest
import torch
import run_transformer as T
from test_simple_spectral_gate import batch
from simple_spectral_gate import SimpleSpectralGate
from mlp_ablation import construct


@pytest.mark.parametrize('variant',['large_head','transformer_no_ff','stacked_psd','stacked_psd_no_ff','linear_simple','linear_stacked','linear_transformer','linear_learned_psd'])
def test_ablation_gradients_and_query_independence(variant):
    torch.manual_seed(51);m=construct(variant).double();b,_=batch()
    a=T.gate_weights(m,b)
    torch.testing.assert_close(a.sum(-1),torch.ones_like(a[...,0]))
    one=dict(b,query=b['query'][:,:1],cross=b['cross'][:,:,:1])
    torch.testing.assert_close(a[:,:1],T.gate_weights(m,one))
    assert a[0,:,0].std()>1e-6
    T.loss_for(m,b,.01,'joint').backward()
    for p in m.parameters():
        if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all()
        else:assert p.grad is None


def test_shared_initialization_and_zero_ff_equivalence():
    torch.manual_seed(52);full=T.TransformerGate().double().eval()
    torch.manual_seed(52);removed=construct('transformer_no_ff').double().eval()
    for k,v in full.state_dict().items():torch.testing.assert_close(v,removed.state_dict()[k])
    with torch.no_grad():
        for layer in full.context.layers:
            layer.linear2.weight.zero_();layer.linear2.bias.zero_()
    b,_=batch();torch.testing.assert_close(T.gate_weights(full,b),T.gate_weights(removed,b))
    torch.manual_seed(53);small=SimpleSpectralGate()
    torch.manual_seed(53);large=construct('large_head')
    for k,v in small.state_dict().items():
        if not k.startswith('head.'):torch.testing.assert_close(v,large.state_dict()[k])


def test_stacked_shared_initialization_and_psd():
    torch.manual_seed(54);full=construct('stacked_psd').double()
    torch.manual_seed(54);removed=construct('stacked_psd_no_ff').double()
    for k,v in full.state_dict().items():torch.testing.assert_close(v,removed.state_dict()[k])
    b,qq=batch();beta,a,aq,dc,dq=full.attention_blocks(b['context'],b['grams'],b['cross'])
    kqq=torch.einsum('bj,bjst->bst',beta,qq)/torch.sqrt(dq[...,None]*dq[...,None,:])
    k=torch.cat([torch.cat([a,aq.transpose(-1,-2)],-1),torch.cat([aq,kqq],-1)],-2)
    assert torch.linalg.eigvalsh(k).min()>-1e-10
    with torch.no_grad():
        for layer in full.layers:layer.ff[-1].weight.zero_();layer.ff[-1].bias.zero_()
    torch.testing.assert_close(T.gate_weights(full,b),T.gate_weights(removed,b))


@pytest.mark.parametrize('variant',['linear_simple','linear_stacked','linear_transformer','linear_learned_psd'])
def test_linear_head_and_no_mlp_activation_executed(variant):
    m=construct(variant).double();b,_=batch()
    assert isinstance(m.head,torch.nn.Linear)
    assert not any(isinstance(s,(torch.nn.GELU,torch.nn.ReLU,torch.nn.Tanh)) for s in m.modules())
    calls=[]
    hook=m.head.register_forward_hook(lambda module,args,result:calls.append((args[0],result)))
    a=T.gate_weights(m,b);hook.remove()
    z,logits=calls[0]
    torch.testing.assert_close(logits,z@m.head.weight.T+m.head.bias)
    torch.testing.assert_close(a,logits.softmax(-1))
