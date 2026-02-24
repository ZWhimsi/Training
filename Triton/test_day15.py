"""Test Suite for Day 15: Layer Normalization
Run: pytest test_day15.py -v
"""

import pytest
import torch
import torch.nn.functional as F

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day15 import layer_norm, rms_norm, layer_norm_residual
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day15")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_layer_norm_basic():
    """Test basic layer normalization."""
    batch, hidden = 32, 128
    x = torch.randn(batch, hidden, device='cuda')
    weight = torch.ones(hidden, device='cuda')
    bias = torch.zeros(hidden, device='cuda')
    
    result = layer_norm(x, weight, bias)
    expected = F.layer_norm(x, [hidden], weight, bias)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day15")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_layer_norm_affine():
    """Test layer normalization with affine parameters."""
    batch, hidden = 32, 128
    x = torch.randn(batch, hidden, device='cuda')
    weight = torch.randn(hidden, device='cuda')
    bias = torch.randn(hidden, device='cuda')
    
    result = layer_norm(x, weight, bias)
    expected = F.layer_norm(x, [hidden], weight, bias)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day15")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_rms_norm():
    """Test RMS normalization."""
    batch, hidden = 32, 128
    x = torch.randn(batch, hidden, device='cuda')
    weight = torch.ones(hidden, device='cuda')
    eps = 1e-5
    
    result = rms_norm(x, weight, eps)
    
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    expected = x / rms * weight
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day15")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_layer_norm_residual():
    """Test layer normalization with residual."""
    batch, hidden = 32, 128
    x = torch.randn(batch, hidden, device='cuda')
    residual = torch.randn(batch, hidden, device='cuda')
    weight = torch.ones(hidden, device='cuda')
    bias = torch.zeros(hidden, device='cuda')
    
    result = layer_norm_residual(x, residual, weight, bias)
    expected = F.layer_norm(x + residual, [hidden], weight, bias)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day15")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_different_sizes():
    """Test different tensor sizes."""
    for batch, hidden in [(8, 64), (64, 256), (128, 512)]:
        x = torch.randn(batch, hidden, device='cuda')
        weight = torch.ones(hidden, device='cuda')
        bias = torch.zeros(hidden, device='cuda')
        
        result = layer_norm(x, weight, bias)
        expected = F.layer_norm(x, [hidden], weight, bias)
        
        max_err = (result - expected).abs().max().item()
        assert max_err <= 1e-3, f"Failed at {batch}x{hidden}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
