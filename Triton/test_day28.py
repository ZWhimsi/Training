"""Test Suite for Day 28: Causal Attention
Run: pytest test_day28.py -v
"""

import pytest
import torch
import math

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day28 import causal_attention, mh_causal_attention
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def reference_causal_attention(Q, K, V):
    """Reference causal attention."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = (Q @ K.transpose(-2, -1)) * scale
    
    seq_len = Q.shape[-2]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
    if Q.dim() == 3:
        mask = mask.unsqueeze(0)
    scores.masked_fill_(mask, float('-inf'))
    
    return torch.softmax(scores, dim=-1) @ V


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day28")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_causal_attention():
    """Test causal attention."""
    seq_len, head_dim = 64, 32
    Q = torch.randn(seq_len, head_dim, device='cuda')
    K = torch.randn(seq_len, head_dim, device='cuda')
    V = torch.randn(seq_len, head_dim, device='cuda')
    
    result = causal_attention(Q, K, V)
    expected = reference_causal_attention(Q, K, V)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day28")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_mh_causal_attention():
    """Test multi-head causal attention."""
    n_heads, seq_len, head_dim = 8, 64, 32
    Q = torch.randn(n_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(n_heads, seq_len, head_dim, device='cuda')
    V = torch.randn(n_heads, seq_len, head_dim, device='cuda')
    
    result = mh_causal_attention(Q, K, V)
    expected = reference_causal_attention(Q, K, V)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day28")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_causality():
    """Test that causality is properly enforced."""
    seq_len, head_dim = 32, 16
    Q = torch.randn(seq_len, head_dim, device='cuda')
    K = torch.randn(seq_len, head_dim, device='cuda')
    V = torch.randn(seq_len, head_dim, device='cuda')
    
    result = causal_attention(Q, K, V)
    
    V_modified = V.clone()
    V_modified[seq_len//2:] = torch.randn_like(V_modified[seq_len//2:])
    result_modified = causal_attention(Q, K, V_modified)
    
    first_half_err = (result[:seq_len//2] - result_modified[:seq_len//2]).abs().max().item()
    assert first_half_err <= 1e-5, "Causality violated"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day28")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_various_sizes():
    """Test various tensor sizes."""
    for seq_len, head_dim in [(32, 32), (64, 64), (100, 32)]:
        Q = torch.randn(seq_len, head_dim, device='cuda')
        K = torch.randn(seq_len, head_dim, device='cuda')
        V = torch.randn(seq_len, head_dim, device='cuda')
        
        result = causal_attention(Q, K, V)
        expected = reference_causal_attention(Q, K, V)
        
        max_err = (result - expected).abs().max().item()
        assert max_err <= 1e-2, f"Failed at {seq_len}x{head_dim}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
