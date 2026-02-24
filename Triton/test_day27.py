"""Test Suite for Day 27: Memory-Efficient Attention
Run: pytest test_day27.py -v
"""

import pytest
import torch
import math

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day27 import single_query_attention, memory_efficient_attention
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def reference_attention(Q, K, V):
    """Reference attention implementation."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = (Q @ K.T) * scale
    weights = torch.softmax(scores, dim=-1)
    return weights @ V


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day27")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_single_query():
    """Test single query attention."""
    seq_len, head_dim = 64, 32
    Q = torch.randn(seq_len, head_dim, device='cuda')
    K = torch.randn(seq_len, head_dim, device='cuda')
    V = torch.randn(seq_len, head_dim, device='cuda')
    
    query_idx = 5
    result = single_query_attention(Q, K, V, query_idx)
    
    expected_full = reference_attention(Q, K, V)
    expected = expected_full[query_idx]
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day27")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_memory_efficient():
    """Test memory efficient attention."""
    seq_len, head_dim = 64, 32
    Q = torch.randn(seq_len, head_dim, device='cuda')
    K = torch.randn(seq_len, head_dim, device='cuda')
    V = torch.randn(seq_len, head_dim, device='cuda')
    
    result = memory_efficient_attention(Q, K, V)
    expected = reference_attention(Q, K, V)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day27")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_numerical_stability():
    """Test numerical stability with large values."""
    seq_len, head_dim = 64, 32
    Q = torch.randn(seq_len, head_dim, device='cuda') * 10
    K = torch.randn(seq_len, head_dim, device='cuda') * 10
    V = torch.randn(seq_len, head_dim, device='cuda')
    
    result = memory_efficient_attention(Q, K, V)
    
    assert not torch.isnan(result).any(), "NaN in output"
    assert not torch.isinf(result).any(), "Inf in output"
    
    expected = reference_attention(Q, K, V)
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-2, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day27")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_various_sizes():
    """Test various tensor sizes."""
    for seq_len, head_dim in [(32, 32), (64, 64), (128, 32)]:
        Q = torch.randn(seq_len, head_dim, device='cuda')
        K = torch.randn(seq_len, head_dim, device='cuda')
        V = torch.randn(seq_len, head_dim, device='cuda')
        
        result = memory_efficient_attention(Q, K, V)
        expected = reference_attention(Q, K, V)
        
        max_err = (result - expected).abs().max().item()
        assert max_err <= 1e-2, f"Failed at {seq_len}x{head_dim}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
