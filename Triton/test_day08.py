"""Test Suite for Day 8: Vector Operations
Run: pytest test_day08.py -v
"""

import pytest
import torch
import torch.nn.functional as F

try:
    from day08 import gelu, silu, fused_linear_gelu, ema_update, polynomial
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day08")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gelu():
    """Test GELU activation."""
    x = torch.randn(10000, device='cuda')
    result = gelu(x)
    expected = F.gelu(x, approximate='tanh')
    
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch with torch GELU"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day08")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_silu():
    """Test SiLU activation."""
    x = torch.randn(10000, device='cuda')
    result = silu(x)
    expected = F.silu(x)
    
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day08")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fused_linear_gelu():
    """Test fused linear+GELU."""
    x = torch.randn(1000, device='cuda')
    w = torch.randn(1000, device='cuda')
    b = torch.randn(1000, device='cuda')
    result = fused_linear_gelu(x, w, b)
    expected = F.gelu(x * w + b, approximate='tanh')
    
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day08")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ema():
    """Test EMA update."""
    running = torch.randn(1000, device='cuda')
    new = torch.randn(1000, device='cuda')
    alpha = 0.1
    result = ema_update(running, new, alpha)
    expected = (1 - alpha) * running + alpha * new
    
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day08")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_polynomial():
    """Test polynomial evaluation."""
    x = torch.randn(1000, device='cuda')
    coeffs = [1.0, 2.0, 3.0, 4.0]
    result = polynomial(x, coeffs)
    expected = coeffs[0] + coeffs[1]*x + coeffs[2]*x**2 + coeffs[3]*x**3
    
    assert torch.allclose(result, expected, atol=1e-4), "Mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
