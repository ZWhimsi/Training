"""Test Suite for Day 24: Online Softmax
Run: pytest test_day24.py -v
"""

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day24 import online_softmax, scaled_online_softmax, causal_online_softmax
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day24")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_online_softmax():
    """Test online softmax."""
    x = torch.randn(64, 256, device='cuda')
    
    result = online_softmax(x)
    expected = torch.softmax(x, dim=-1)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"
    
    row_sums = result.sum(dim=-1)
    sum_err = (row_sums - 1.0).abs().max().item()
    assert sum_err <= 1e-4, f"Row sums not 1: {sum_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day24")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_scaled_softmax():
    """Test scaled online softmax."""
    x = torch.randn(64, 256, device='cuda')
    scale = 0.125
    
    result = scaled_online_softmax(x, scale)
    expected = torch.softmax(x * scale, dim=-1)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day24")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_causal_softmax():
    """Test causal online softmax."""
    n = 32
    x = torch.randn(n, n, device='cuda')
    
    result = causal_online_softmax(x)
    
    for i in range(n):
        for j in range(i + 1, n):
            assert result[i, j].abs() <= 1e-6, f"result[{i},{j}] should be 0"
    
    for i in range(n):
        row_sum = result[i, :i+1].sum()
        assert abs(row_sum - 1.0) <= 1e-4, f"Row {i} sum: {row_sum:.4f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day24")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_numerical_stability():
    """Test numerical stability with large values."""
    x = torch.randn(32, 128, device='cuda') * 100
    
    result = online_softmax(x)
    expected = torch.softmax(x, dim=-1)
    
    assert not torch.isnan(result).any(), "NaN in output"
    assert not torch.isinf(result).any(), "Inf in output"
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day24")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_various_sizes():
    """Test various tensor sizes."""
    for rows, cols in [(16, 64), (64, 256), (128, 1024)]:
        x = torch.randn(rows, cols, device='cuda')
        
        result = online_softmax(x)
        expected = torch.softmax(x, dim=-1)
        
        max_err = (result - expected).abs().max().item()
        assert max_err <= 1e-3, f"Failed at {rows}x{cols}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
