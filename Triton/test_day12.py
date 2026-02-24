"""Test Suite for Day 12: Activation Functions
Run: pytest test_day12.py -v
"""

import pytest
import torch
import torch.nn.functional as F

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day12 import sigmoid, tanh, leaky_relu, elu, softplus, mish
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day12")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_sigmoid():
    """Test sigmoid activation."""
    x = torch.randn(1024, device='cuda')
    result = sigmoid(x)
    expected = torch.sigmoid(x)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day12")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_tanh():
    """Test tanh activation."""
    x = torch.randn(1024, device='cuda')
    result = tanh(x)
    expected = torch.tanh(x)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day12")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_leaky_relu():
    """Test leaky ReLU activation."""
    x = torch.randn(1024, device='cuda')
    slope = 0.01
    result = leaky_relu(x, slope)
    expected = F.leaky_relu(x, slope)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day12")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_elu():
    """Test ELU activation."""
    x = torch.randn(1024, device='cuda')
    alpha = 1.0
    result = elu(x, alpha)
    expected = F.elu(x, alpha)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day12")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_softplus():
    """Test softplus activation."""
    x = torch.randn(1024, device='cuda')
    x = torch.clamp(x, -10, 10)
    beta = 1.0
    result = softplus(x, beta)
    expected = F.softplus(x, beta)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day12")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_mish():
    """Test mish activation."""
    x = torch.randn(1024, device='cuda')
    x = torch.clamp(x, -10, 10)
    result = mish(x)
    expected = F.mish(x)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
