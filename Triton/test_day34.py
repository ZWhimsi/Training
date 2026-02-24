"""Test Suite for Day 34: Flash Attention Backward dQ
Run: pytest test_day34.py -v
"""

import pytest
import torch
import math

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day33 import flash_attention_v2
        from day34 import flash_attention_backward_dq
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def reference_dq(Q, K, V, dO):
    """Reference dQ using autograd."""
    Q = Q.detach().requires_grad_(True)
    K = K.detach()
    V = V.detach()
    
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = (Q @ K.transpose(-2, -1)) * scale
    P = torch.softmax(scores, dim=-1)
    output = P @ V
    output.backward(dO)
    
    return Q.grad


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day34")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_dq():
    """Test dQ computation."""
    batch, n_heads, seq_len, head_dim = 1, 4, 64, 32
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    output, L, M = flash_attention_v2(Q, K, V)
    dO = torch.randn_like(output)
    
    dQ = flash_attention_backward_dq(Q, K, V, output, dO, L, M)
    ref_dQ = reference_dq(Q, K, V, dO)
    
    max_err = (dQ - ref_dQ).abs().max().item()
    assert max_err <= 1e-2, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day34")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_numerical_stability():
    """Test numerical stability with large values."""
    batch, n_heads, seq_len, head_dim = 1, 2, 32, 16
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda') * 5
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda') * 5
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    output, L, M = flash_attention_v2(Q, K, V)
    dO = torch.randn_like(output)
    
    dQ = flash_attention_backward_dq(Q, K, V, output, dO, L, M)
    
    assert not torch.isnan(dQ).any(), "NaN in dQ"
    
    ref_dQ = reference_dq(Q, K, V, dO)
    assert torch.allclose(dQ, ref_dQ, atol=0.1, rtol=0.1), "dQ mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day34")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_various_sizes():
    """Test various tensor sizes."""
    for batch, heads, seq, dim in [(1, 2, 32, 16), (2, 4, 64, 32)]:
        Q = torch.randn(batch, heads, seq, dim, device='cuda')
        K = torch.randn(batch, heads, seq, dim, device='cuda')
        V = torch.randn(batch, heads, seq, dim, device='cuda')
        
        output, L, M = flash_attention_v2(Q, K, V)
        dO = torch.randn_like(output)
        
        dQ = flash_attention_backward_dq(Q, K, V, output, dO, L, M)
        ref_dQ = reference_dq(Q, K, V, dO)
        
        assert (dQ - ref_dQ).abs().max() <= 0.1, f"Failed at ({batch},{heads},{seq},{dim})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
