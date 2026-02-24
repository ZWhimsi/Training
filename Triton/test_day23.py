"""Test Suite for Day 23: Attention Score Computation
Run: pytest test_day23.py -v
"""

import pytest
import torch
import math

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day23 import attention_scores, apply_causal_mask, mha_attention_scores
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day23")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_attention_scores():
    """Test attention score computation."""
    seq_len, head_dim = 64, 32
    Q = torch.randn(seq_len, head_dim, device='cuda')
    K = torch.randn(seq_len, head_dim, device='cuda')
    
    result = attention_scores(Q, K)
    scale = 1.0 / math.sqrt(head_dim)
    expected = (Q @ K.T) * scale
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day23")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_causal_mask():
    """Test causal mask application."""
    seq_len = 64
    scores = torch.randn(seq_len, seq_len, device='cuda')
    
    result = apply_causal_mask(scores.clone())
    
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            assert torch.isinf(result[i, j]) and result[i, j] < 0, f"scores[{i},{j}] should be -inf"
    
    for i in range(seq_len):
        for j in range(i+1):
            assert not torch.isinf(result[i, j]), f"scores[{i},{j}] should not be -inf"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day23")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_mha_scores():
    """Test multi-head attention scores."""
    n_heads, seq_len, head_dim = 8, 64, 32
    Q = torch.randn(n_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(n_heads, seq_len, head_dim, device='cuda')
    
    result = mha_attention_scores(Q, K)
    scale = 1.0 / math.sqrt(head_dim)
    expected = torch.bmm(Q, K.transpose(-2, -1)) * scale
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day23")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_scaling():
    """Test attention score scaling for various head dimensions."""
    for head_dim in [32, 64, 128]:
        seq_len = 32
        Q = torch.randn(seq_len, head_dim, device='cuda')
        K = torch.randn(seq_len, head_dim, device='cuda')
        
        result = attention_scores(Q, K)
        scale = 1.0 / math.sqrt(head_dim)
        expected = (Q @ K.T) * scale
        
        max_err = (result - expected).abs().max().item()
        assert max_err <= 1e-3, f"Failed at d={head_dim}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
