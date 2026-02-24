"""
Test Suite for Day 3: Masking and Bounds Checking
=================================================
Run: pytest test_day03.py -v
"""

import pytest
import torch

try:
    from day03 import safe_load, threshold, clamp, positive_sum, where
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


# ============================================================================
# Reference Implementations (PyTorch)
# ============================================================================

def reference_threshold(x: torch.Tensor, thresh: float) -> torch.Tensor:
    return torch.where(x >= thresh, x, torch.zeros_like(x))

def reference_clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return torch.clamp(x, min_val, max_val)

def reference_positive_sum(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x[x > 0]).unsqueeze(0)

def reference_where(cond: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(cond, a, b)


# ============================================================================
# Test Functions
# ============================================================================

@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day03")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_threshold_basic():
    """Test threshold operation."""
    x = torch.tensor([-2, -1, 0, 1, 2, 3], dtype=torch.float32, device='cuda')
    result = threshold(x, 0.5)
    expected = reference_threshold(x, 0.5)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected.tolist()}, got {result.tolist()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day03")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_threshold_all_pass():
    """Test when all values pass threshold."""
    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device='cuda')
    result = threshold(x, 0.0)
    expected = reference_threshold(x, 0.0)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), "All values should pass"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day03")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_threshold_large():
    """Test threshold with large tensor."""
    x = torch.randn(10000, device='cuda')
    result = threshold(x, 0.0)
    expected = reference_threshold(x, 0.0)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Large tensor threshold failed"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day03")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_clamp_basic():
    """Test clamp operation."""
    x = torch.tensor([-3, -1, 0, 1, 3], dtype=torch.float32, device='cuda')
    result = clamp(x, -2.0, 2.0)
    expected = reference_clamp(x, -2.0, 2.0)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected.tolist()}, got {result.tolist()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day03")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_clamp_asymmetric():
    """Test clamp with asymmetric bounds."""
    x = torch.randn(1000, device='cuda') * 5
    result = clamp(x, -1.0, 3.0)
    expected = reference_clamp(x, -1.0, 3.0)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Asymmetric clamp failed"
    assert result.min() >= -1.0 and result.max() <= 3.0, "Values outside bounds!"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day03")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_positive_sum_basic():
    """Test positive sum."""
    x = torch.tensor([-2, -1, 1, 2, 3], dtype=torch.float32, device='cuda')
    result = positive_sum(x)
    expected = reference_positive_sum(x)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected.item()}, got {result.item()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day03")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_positive_sum_all_negative():
    """Test positive sum with all negative values."""
    x = torch.tensor([-5, -4, -3, -2, -1], dtype=torch.float32, device='cuda')
    result = positive_sum(x)
    expected = torch.tensor([0.0], device='cuda')
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), f"Expected 0, got {result.item()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day03")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_positive_sum_large():
    """Test positive sum with large tensor."""
    x = torch.randn(10000, device='cuda')
    result = positive_sum(x)
    expected = reference_positive_sum(x)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3), f"Expected ~{expected.item():.2f}, got {result.item():.2f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day03")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_where_basic():
    """Test where operation."""
    cond = torch.tensor([True, False, True, False, True], device='cuda')
    a = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device='cuda')
    b = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32, device='cuda')
    
    result = where(cond, a, b)
    expected = reference_where(cond, a, b)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected.tolist()}, got {result.tolist()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day03")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_where_large():
    """Test where with large tensors."""
    cond = torch.randint(0, 2, (10000,), device='cuda').bool()
    a = torch.randn(10000, device='cuda')
    b = torch.randn(10000, device='cuda')
    
    result = where(cond, a, b)
    expected = reference_where(cond, a, b)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Large where operation failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
