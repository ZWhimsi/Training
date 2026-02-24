"""Test Suite for Day 31: Flash Attention Backward dV, dK
Run: pytest test_day31.py -v
"""

import pytest
import torch
import math

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day30 import flash_attention_forward
        from day31 import flash_attention_backward_dv_dk
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def reference_backward(Q, K, V, dO):
    """Reference backward using autograd."""
    Q = Q.detach().requires_grad_(True)
    K = K.detach().requires_grad_(True)
    V = V.detach().requires_grad_(True)
    
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = (Q @ K.transpose(-2, -1)) * scale
    P = torch.softmax(scores, dim=-1)
    output = P @ V
    output.backward(dO)
    
    return Q.grad, K.grad, V.grad


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day31")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_dv_dk():
    """Test dV and dK computation."""
    batch, n_heads, seq_len, head_dim = 1, 4, 64, 32
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    output, L = flash_attention_forward(Q, K, V)
    dO = torch.randn_like(output)
    
    dK, dV = flash_attention_backward_dv_dk(Q, K, V, output, dO, L)
    _, ref_dK, ref_dV = reference_backward(Q, K, V, dO)
    
    dk_err = (dK - ref_dK).abs().max().item()
    dv_err = (dV - ref_dV).abs().max().item()
    
    assert dk_err <= 1e-2, f"dK error: {dk_err:.6f}"
    assert dv_err <= 1e-2, f"dV error: {dv_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day31")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_numerical_stability():
    """Test numerical stability with large values."""
    batch, n_heads, seq_len, head_dim = 1, 2, 32, 16
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda') * 5
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda') * 5
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    output, L = flash_attention_forward(Q, K, V)
    dO = torch.randn_like(output)
    
    dK, dV = flash_attention_backward_dv_dk(Q, K, V, output, dO, L)
    
    assert not torch.isnan(dK).any(), "NaN in dK"
    assert not torch.isnan(dV).any(), "NaN in dV"
    
    _, ref_dK, ref_dV = reference_backward(Q, K, V, dO)
    
    assert torch.allclose(dK, ref_dK, atol=0.1, rtol=0.1), "dK mismatch"
    assert torch.allclose(dV, ref_dV, atol=0.1, rtol=0.1), "dV mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day31")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_various_sizes():
    """Test various tensor sizes."""
    for batch, heads, seq, dim in [(1, 2, 32, 16), (2, 4, 64, 32)]:
        Q = torch.randn(batch, heads, seq, dim, device='cuda')
        K = torch.randn(batch, heads, seq, dim, device='cuda')
        V = torch.randn(batch, heads, seq, dim, device='cuda')
        
        output, L = flash_attention_forward(Q, K, V)
        dO = torch.randn_like(output)
        
        dK, dV = flash_attention_backward_dv_dk(Q, K, V, output, dO, L)
        _, ref_dK, ref_dV = reference_backward(Q, K, V, dO)
        
        assert (dK - ref_dK).abs().max() <= 0.1, f"Failed at ({batch},{heads},{seq},{dim})"
        assert (dV - ref_dV).abs().max() <= 0.1, f"Failed at ({batch},{heads},{seq},{dim})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
