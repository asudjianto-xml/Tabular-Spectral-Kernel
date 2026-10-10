import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent))
from rotary_gate import rotate,RotaryAttention,RotaryTransformerGate


def test_rotary_dot_products_depend_on_time_difference():
    torch.manual_seed(3)
    q=torch.randn(1,4,1,8,dtype=torch.float64);k=torch.randn(1,4,3,8,dtype=torch.float64)
    qp=torch.tensor([[0.]],dtype=torch.float64);kp=torch.tensor([[-3.,-2.,-1.]],dtype=torch.float64)
    scores=lambda a,b:(rotate(q,a)*rotate(k,b)).sum(-1)
    torch.testing.assert_close(scores(qp,kp),scores(qp+17,kp+17))
    assert not torch.allclose(scores(qp,kp),scores(qp,kp.flip(-1)))
    torch.testing.assert_close(rotate(k,kp).square().sum(-1),k.square().sum(-1))


def test_rotary_attention_responds_to_chronology():
    torch.manual_seed(8);attention=RotaryAttention().eval()
    q=torch.randn(1,2,32);memory=torch.randn(1,4,32)
    qp=torch.zeros(1,2);cp=torch.tensor([[-4.,-3.,-2.,-1.]])
    with torch.no_grad():
        original=attention(q,memory,qp,cp)
        reversed_times=attention(q,memory,qp,cp.flip(-1))
        # Reordering complete timestamped observations preserves their chronology.
        moved_records=attention(q,memory.flip(1),qp,cp.flip(-1))
    assert not torch.allclose(original,reversed_times,atol=1e-5)
    torch.testing.assert_close(original,moved_records,atol=1e-6,rtol=1e-5)


def test_rotary_gate_query_independence_and_simplex():
    torch.manual_seed(2);torch.set_num_threads(1);model=RotaryTransformerGate().eval()
    c=torch.randn(1,128,8);c[:,:,7]=torch.arange(-4,0).repeat_interleave(32)/4
    q=torch.randn(1,3,8);q[:,:,5:]=0;d=torch.zeros(1,3)
    with torch.no_grad():a=model(c,q,d);one=model(c,q[:,:1],d)
    torch.testing.assert_close(a[:,:1],one,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(a.sum(-1),torch.ones(1,3))
