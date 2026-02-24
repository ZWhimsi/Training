"""Test Suite for Day 29: IO-Aware Attention
Run: pytest test_day29.py -v
"""

import pytest
import torch
import math

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day29 import io_aware_attention, attention_with_stats
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def reference_attention(Q, K, V):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    return torch.softmax((Q @ K.T) * scale, dim=-1) @ V


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day29")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_io_aware_attention():
    """Test IO-aware attention."""
    seq_len, head_dim = 128, 64
    Q = torch.randn(seq_len, head_dim, device='cuda')
    K = torch.randn(seq_len, head_dim, device='cuda')
    V = torch.randn(seq_len, head_dim, device='cuda')
    
    result = io_aware_attention(Q, K, V)
    expected = reference_attention(Q, K, V)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day29")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_with_stats():
    """Test attention with stats."""
    seq_len, head_dim = 128, 64
    Q = torch.randn(seq_len, head_dim, device='cuda')
    K = torch.randn(seq_len, head_dim, device='cuda')
    V = torch.randn(seq_len, head_dim, device='cuda')
    
    output, L, M = attention_with_stats(Q, K, V)
    expected = reference_attention(Q, K, V)
    
    max_err = (output - expected).abs().max().item()
    assert max_err <= 1e-3, f"Output error: {max_err:.6f}"
    
    assert not torch.isnan(L).any(), "NaN in L"
    assert not torch.isnan(M).any(), "NaN in M"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day29")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_numerical_stability():
    """Test numerical stability with large values."""
    seq_len, head_dim = 64, 32
    Q = torch.randn(seq_len, head_dim, device='cuda') * 10
    K = torch.randn(seq_len, head_dim, device='cuda') * 10
    V = torch.randn(seq_len, head_dim, device='cuda')
    
    result = io_aware_attention(Q, K, V)
    
    assert not torch.isnan(result).any(), "NaN in output"
    assert not torch.isinf(result).any(), "Inf in output"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day29")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_various_sizes():
    """Test various tensor sizes."""
    for seq_len, head_dim in [(64, 32), (128, 64), (256, 64)]:
        Q = torch.randn(seq_len, head_dim, device='cuda')
        K = torch.randn(seq_len, head_dim, device='cuda')
        V = torch.randn(seq_len, head_dim, device='cuda')
        
        result = io_aware_attention(Q, K, V)
        expected = reference_attention(Q, K, V)
        
        max_err = (result - expected).abs().max().item()
        assert max_err <= 1e-2, f"Failed at {seq_len}x{head_dim}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
