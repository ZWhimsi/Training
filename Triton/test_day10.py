"""Test Suite for Day 10: Reduction Operations
Run: pytest test_day10.py -v
"""

import pytest
import torch

try:
    from day10 import vector_sum, row_wise_sum, vector_mean, vector_variance, large_sum
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day10")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vector_sum():
    """Test vector sum."""
    x = torch.arange(100, dtype=torch.float32, device='cuda')
    result = vector_sum(x)
    expected = x.sum()
    
    assert torch.allclose(result, expected.unsqueeze(0), atol=1e-4), f"Expected {expected.item()}, got {result.item()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day10")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_row_wise_sum():
    """Test row-wise sum."""
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
    result = row_wise_sum(x)
    expected = x.sum(dim=1)
    
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected.tolist()}, got {result.tolist()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day10")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vector_mean():
    """Test vector mean."""
    x = torch.randn(1000, device='cuda')
    result = vector_mean(x)
    expected = x.mean()
    
    assert torch.allclose(result, expected.unsqueeze(0), atol=1e-5), "Mean mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day10")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vector_variance():
    """Test vector variance."""
    x = torch.randn(500, device='cuda')
    result = vector_variance(x)
    expected = x.var(unbiased=False)
    
    assert torch.allclose(result, expected.unsqueeze(0), atol=1e-4), f"Expected {expected.item():.4f}, got {result.item():.4f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day10")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_sum():
    """Test large sum."""
    x = torch.randn(100000, device='cuda')
    result = large_sum(x)
    expected = x.sum()
    
    assert torch.allclose(result, expected.unsqueeze(0), rtol=1e-4), "Large sum mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
