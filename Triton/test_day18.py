"""Test Suite for Day 18: Strided Memory Access
Run: pytest test_day18.py -v
"""

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day18 import strided_load, strided_2d_copy, gather, scatter_add
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day18")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_strided_load():
    """Test strided load."""
    x = torch.arange(100, device='cuda', dtype=torch.float32)
    stride = 3
    size = 30
    
    result = strided_load(x, stride, size)
    expected = x[::stride][:size]
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day18")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_strided_2d_copy():
    """Test 2D strided copy."""
    x = torch.randn(64, 128, device='cuda')
    
    result = strided_2d_copy(x)
    expected = x.clone()
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day18")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_gather():
    """Test gather operation."""
    src = torch.randn(1000, device='cuda')
    idx = torch.randint(0, 1000, (500,), device='cuda')
    
    result = gather(src, idx)
    expected = src[idx]
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day18")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_scatter_add():
    """Test scatter add operation."""
    values = torch.ones(100, device='cuda')
    indices = torch.randint(0, 50, (100,), device='cuda')
    
    result = scatter_add(values, indices, 50)
    
    expected = torch.zeros(50, device='cuda')
    for i in range(100):
        expected[indices[i]] += values[i]
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day18")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_transposed_tensor():
    """Test strided copy on transposed tensor."""
    x = torch.randn(64, 128, device='cuda')
    x_t = x.T.contiguous()
    
    result = strided_2d_copy(x_t)
    expected = x_t.clone()
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
