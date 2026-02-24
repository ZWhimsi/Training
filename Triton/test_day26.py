"""Test Suite for Day 26: Blocked Softmax
Run: pytest test_day26.py -v
"""

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day26 import compute_row_max, safe_blocked_softmax, blocked_attention
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day26")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_row_max():
    """Test row maximum computation."""
    x = torch.randn(64, 256, device='cuda')
    result = compute_row_max(x)
    expected = x.max(dim=-1).values
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day26")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_safe_softmax():
    """Test safe blocked softmax."""
    x = torch.randn(64, 256, device='cuda')
    result = safe_blocked_softmax(x)
    expected = torch.softmax(x, dim=-1)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"
    
    row_sums = result.sum(dim=-1)
    sum_err = (row_sums - 1.0).abs().max().item()
    assert sum_err <= 1e-4, f"Sum error: {sum_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day26")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_blocked_attention():
    """Test blocked attention."""
    seq_len, head_dim = 64, 32
    scores = torch.randn(seq_len, seq_len, device='cuda')
    V = torch.randn(seq_len, head_dim, device='cuda')
    
    result = blocked_attention(scores, V)
    expected = torch.softmax(scores, dim=-1) @ V
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day26")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_numerical_stability():
    """Test numerical stability with large values."""
    x = torch.randn(32, 128, device='cuda') * 50
    
    result = safe_blocked_softmax(x)
    
    assert not torch.isnan(result).any(), "NaN in output"
    assert not torch.isinf(result).any(), "Inf in output"
    
    expected = torch.softmax(x, dim=-1)
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
