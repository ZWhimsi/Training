"""Test Suite for Day 33: Flash Attention v2
Run: pytest test_day33.py -v
"""

import pytest
import torch
import math

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day33 import flash_attention_v2
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


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day33")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_basic():
    """Test basic flash attention v2."""
    batch, n_heads, seq_len, head_dim = 2, 8, 128, 64
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    output, L, M = flash_attention_v2(Q, K, V)
    expected = reference_attention(Q, K, V)
    
    max_err = (output - expected).abs().max().item()
    assert max_err <= 1e-2, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day33")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_causal():
    """Test causal flash attention v2."""
    batch, n_heads, seq_len, head_dim = 2, 8, 128, 64
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    output, _, _ = flash_attention_v2(Q, K, V, causal=True)
    expected = reference_attention(Q, K, V, causal=True)
    
    max_err = (output - expected).abs().max().item()
    assert max_err <= 1e-2, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day33")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_stats_stored():
    """Test that stats are properly stored."""
    batch, n_heads, seq_len, head_dim = 2, 4, 64, 32
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    output, L, M = flash_attention_v2(Q, K, V)
    expected = reference_attention(Q, K, V)
    
    assert L.shape == (batch, n_heads, seq_len), f"L shape: {L.shape}"
    assert M.shape == (batch, n_heads, seq_len), f"M shape: {M.shape}"
    assert not torch.isnan(L).any(), "NaN in L"
    assert not torch.isnan(M).any(), "NaN in M"
    max_err = (output - expected).abs().max().item()
    assert torch.allclose(output, expected, atol=1e-2, rtol=1e-2), f"Output mismatch: {max_err:.4f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day33")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_large_sequence():
    """Test with large sequence length."""
    batch, n_heads, seq_len, head_dim = 1, 4, 512, 64
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    output, _, _ = flash_attention_v2(Q, K, V)
    expected = reference_attention(Q, K, V)
    
    max_err = (output - expected).abs().max().item()
    assert max_err <= 0.05, f"Error: {max_err:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
