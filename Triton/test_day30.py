"""Test Suite for Day 30: Flash Attention Forward
Run: pytest test_day30.py -v
"""

import pytest
import torch
import math

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day30 import flash_attention, flash_attention_forward
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def reference_attention(Q, K, V, causal=False):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = (Q @ K.transpose(-2, -1)) * scale
    
    if causal:
        seq_len = Q.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    
    return torch.softmax(scores, dim=-1) @ V


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day30")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_flash_attention():
    """Test non-causal flash attention."""
    batch, n_heads, seq_len, head_dim = 2, 8, 128, 64
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    result = flash_attention(Q, K, V, causal=False)
    expected = reference_attention(Q, K, V, causal=False)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-2, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day30")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_causal():
    """Test causal flash attention."""
    batch, n_heads, seq_len, head_dim = 2, 8, 128, 64
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    result = flash_attention(Q, K, V, causal=True)
    expected = reference_attention(Q, K, V, causal=True)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-2, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day30")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_lse_stored():
    """Test that log-sum-exp is properly stored."""
    batch, n_heads, seq_len, head_dim = 2, 4, 64, 32
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    output, L = flash_attention_forward(Q, K, V)
    
    assert L.shape == (batch, n_heads, seq_len), f"L shape: {L.shape}"
    assert not torch.isnan(L).any(), "NaN in L"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day30")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_numerical_stability():
    """Test numerical stability with large values."""
    batch, n_heads, seq_len, head_dim = 1, 4, 64, 32
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda') * 10
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda') * 10
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    result = flash_attention(Q, K, V)
    
    assert not torch.isnan(result).any(), "NaN in output"
    assert not torch.isinf(result).any(), "Inf in output"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day30")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_various_sizes():
    """Test various tensor sizes."""
    for batch, heads, seq, dim in [(1, 4, 64, 32), (2, 8, 128, 64), (1, 1, 256, 64)]:
        Q = torch.randn(batch, heads, seq, dim, device='cuda')
        K = torch.randn(batch, heads, seq, dim, device='cuda')
        V = torch.randn(batch, heads, seq, dim, device='cuda')
        
        result = flash_attention(Q, K, V)
        expected = reference_attention(Q, K, V)
        
        max_err = (result - expected).abs().max().item()
        assert max_err <= 1e-2, f"Failed at ({batch},{heads},{seq},{dim})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
