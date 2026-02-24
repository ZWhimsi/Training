"""Test Suite for Day 9: Kernel Fusion
Run: pytest test_day09.py -v
"""

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day09 import (fused_add_mul, fused_bias_relu, fused_scale_shift,
                           fused_residual_add, fused_linear_bias_relu)
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day09")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_fused_add_mul():
    """Test fused add-multiply."""
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    z = torch.randn(1024, device='cuda')
    
    result = fused_add_mul(x, y, z)
    expected = (x + y) * z
    
    assert result is not None and result.sum() != 0, "Output is zero/None"
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Max error {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day09")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_fused_bias_relu():
    """Test fused bias+ReLU."""
    x = torch.randn(1024, device='cuda')
    bias = torch.randn(1024, device='cuda')
    
    result = fused_bias_relu(x, bias)
    expected = torch.relu(x + bias)
    
    assert result is not None and result.abs().sum() != 0, "Output is zero/None"
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Max error {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day09")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_fused_scale_shift():
    """Test fused scale+shift."""
    x = torch.randn(1024, device='cuda')
    scale = torch.randn(1024, device='cuda')
    shift = torch.randn(1024, device='cuda')
    
    result = fused_scale_shift(x, scale, shift)
    expected = x * scale + shift
    
    assert result is not None and result.sum() != 0, "Output is zero/None"
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Max error {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day09")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_fused_residual():
    """Test fused residual add."""
    x = torch.randn(1024, device='cuda')
    residual = torch.randn(1024, device='cuda')
    scale = 0.5
    
    result = fused_residual_add(x, residual, scale)
    expected = x * scale + residual
    
    assert result is not None and result.sum() != 0, "Output is zero/None"
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Max error {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day09")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_fused_linear_bias_relu():
    """Test fused linear+bias+ReLU."""
    x = torch.randn(1024, device='cuda')
    weight = torch.randn(1024, device='cuda')
    bias = torch.randn(1024, device='cuda')
    
    result = fused_linear_bias_relu(x, weight, bias)
    expected = torch.relu(x * weight + bias)
    
    assert result is not None and result.abs().sum() != 0, "Output is zero/None"
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Max error {max_err:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
