"""Test Suite for Day 17: Batched Outer Products
Run: pytest test_day17.py -v
"""

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day17 import outer_product, batched_outer, scaled_outer
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day17")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_outer_product():
    """Test outer product."""
    a = torch.randn(64, device='cuda')
    b = torch.randn(128, device='cuda')
    
    result = outer_product(a, b)
    expected = torch.outer(a, b)
    
    assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day17")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_batched_outer():
    """Test batched outer product."""
    B, M, N = 8, 64, 128
    a = torch.randn(B, M, device='cuda')
    b = torch.randn(B, N, device='cuda')
    
    result = batched_outer(a, b)
    expected = torch.bmm(a.unsqueeze(-1), b.unsqueeze(1))
    
    assert result.shape == (B, M, N), f"Shape: {result.shape}"
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day17")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_scaled_outer():
    """Test scaled outer product."""
    a = torch.randn(64, device='cuda')
    b = torch.randn(128, device='cuda')
    scale = 0.125
    
    result = scaled_outer(a, b, scale)
    expected = scale * torch.outer(a, b)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day17")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_various_sizes():
    """Test various tensor sizes."""
    for M, N in [(32, 32), (64, 128), (100, 200)]:
        a = torch.randn(M, device='cuda')
        b = torch.randn(N, device='cuda')
        
        result = outer_product(a, b)
        expected = torch.outer(a, b)
        
        max_err = (result - expected).abs().max().item()
        assert max_err <= 1e-4, f"Failed at {M}x{N}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
