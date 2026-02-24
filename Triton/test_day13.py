"""Test Suite for Day 13: Fused Softmax
Run: pytest test_day13.py -v
"""

import pytest
import torch
import torch.nn.functional as F

try:
    from day13 import softmax_1d, softmax_2d, log_softmax_1d, softmax_temperature
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day13")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_softmax_1d():
    """Test 1D softmax."""
    x = torch.randn(128, device='cuda')
    result = softmax_1d(x)
    expected = F.softmax(x, dim=0)
    
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"
    assert torch.allclose(result.sum(), torch.tensor(1.0, device='cuda'), atol=1e-5), "Doesn't sum to 1"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day13")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_softmax_1d_stability():
    """Test 1D softmax numerical stability."""
    x = torch.tensor([1000, 1001, 1002], dtype=torch.float32, device='cuda')
    result = softmax_1d(x)
    expected = F.softmax(x, dim=0)
    
    assert not torch.any(torch.isnan(result)), "Contains NaN (not stable)"
    assert not torch.any(torch.isinf(result)), "Contains Inf (not stable)"
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day13")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_softmax_2d():
    """Test 2D row softmax."""
    x = torch.randn(32, 64, device='cuda')
    result = softmax_2d(x)
    expected = F.softmax(x, dim=1)
    
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"
    row_sums = result.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(32, device='cuda'), atol=1e-5), "Rows don't sum to 1"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day13")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_log_softmax():
    """Test log softmax."""
    x = torch.randn(64, device='cuda')
    result = log_softmax_1d(x)
    expected = F.log_softmax(x, dim=0)
    
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day13")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_softmax_temperature_high():
    """Test softmax with high temperature (more uniform)."""
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    result = softmax_temperature(x, temperature=10.0)
    expected = F.softmax(x / 10.0, dim=0)
    
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"
    assert result.max() - result.min() <= 0.2, "Not uniform enough"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day13")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_softmax_temperature_low():
    """Test softmax with low temperature (more peaked)."""
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    result = softmax_temperature(x, temperature=0.1)
    expected = F.softmax(x / 0.1, dim=0)
    
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"
    assert result.max() >= 0.95, "Not peaked enough"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
