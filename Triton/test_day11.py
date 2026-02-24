"""Test Suite for Day 11: Element-wise Binary Operations
Run: pytest test_day11.py -v
"""

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day11 import (vector_add, vector_sub, vector_mul, vector_div,
                           scalar_add, vector_maximum)
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day11")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_vector_add():
    """Test vector addition."""
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    
    result = vector_add(x, y)
    expected = x + y
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day11")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_vector_sub():
    """Test vector subtraction."""
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    
    result = vector_sub(x, y)
    expected = x - y
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day11")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_vector_mul():
    """Test vector multiplication."""
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    
    result = vector_mul(x, y)
    expected = x * y
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day11")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_vector_div():
    """Test vector division."""
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda') + 0.1
    
    result = vector_div(x, y)
    expected = x / y
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day11")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_scalar_add():
    """Test scalar addition."""
    x = torch.randn(1024, device='cuda')
    scalar = 3.14
    
    result = scalar_add(x, scalar)
    expected = x + scalar
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day11")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_maximum():
    """Test element-wise maximum."""
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    
    result = vector_maximum(x, y)
    expected = torch.maximum(x, y)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
