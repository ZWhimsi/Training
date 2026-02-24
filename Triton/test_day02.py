"""
Test Suite for Day 2: Memory Operations
=======================================
Run: pytest test_day02.py -v
"""

import pytest
import torch

try:
    from day02 import copy, scaled_copy, strided_load, relu, add_relu
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


# ============================================================================
# Reference Implementations (PyTorch)
# ============================================================================

def reference_copy(src: torch.Tensor) -> torch.Tensor:
    return src.clone()

def reference_scaled_copy(src: torch.Tensor, scale: float) -> torch.Tensor:
    return src * scale

def reference_strided_load(src: torch.Tensor, stride: int) -> torch.Tensor:
    return src[::stride].clone()

def reference_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)

def reference_add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.relu(a + b)


# ============================================================================
# Test Functions
# ============================================================================

@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day02")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_copy_basic():
    """Test basic copy."""
    x = torch.randn(1000, device='cuda')
    result = copy(x)
    expected = reference_copy(x)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Copy values don't match"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day02")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_copy_large():
    """Test copy with large tensor."""
    x = torch.randn(100000, device='cuda')
    result = copy(x)
    expected = reference_copy(x)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Large copy failed"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day02")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_scaled_copy_basic():
    """Test scaled copy."""
    x = torch.randn(1000, device='cuda')
    scale = 2.5
    result = scaled_copy(x, scale)
    expected = reference_scaled_copy(x, scale)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Scaled values don't match"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day02")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_scaled_copy_negative():
    """Test scaled copy with negative scale."""
    x = torch.randn(1000, device='cuda')
    scale = -0.5
    result = scaled_copy(x, scale)
    expected = reference_scaled_copy(x, scale)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Negative scale failed"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day02")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_strided_load_stride2():
    """Test strided load with stride 2."""
    x = torch.arange(100, dtype=torch.float32, device='cuda')
    result = strided_load(x, stride=2)
    expected = reference_strided_load(x, stride=2)
    
    assert result is not None, "Function returned None"
    assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
    assert torch.allclose(result, expected, atol=1e-5), "Strided values don't match"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day02")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_strided_load_stride4():
    """Test strided load with stride 4."""
    x = torch.randn(1000, device='cuda')
    result = strided_load(x, stride=4)
    expected = reference_strided_load(x, stride=4)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Stride 4 values don't match"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day02")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_relu_basic():
    """Test ReLU with positive and negative values."""
    x = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32, device='cuda')
    result = relu(x)
    expected = reference_relu(x)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected.tolist()}, got {result.tolist()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day02")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_relu_large():
    """Test ReLU with large tensor."""
    x = torch.randn(10000, device='cuda')
    result = relu(x)
    expected = reference_relu(x)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Large ReLU failed"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day02")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_add_relu_basic():
    """Test fused add+ReLU."""
    a = torch.tensor([-1, 0, 1, 2], dtype=torch.float32, device='cuda')
    b = torch.tensor([0.5, -0.5, -0.5, 0.5], dtype=torch.float32, device='cuda')
    result = add_relu(a, b)
    expected = reference_add_relu(a, b)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected.tolist()}, got {result.tolist()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day02")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_add_relu_large():
    """Test fused add+ReLU with large tensors."""
    a = torch.randn(10000, device='cuda')
    b = torch.randn(10000, device='cuda')
    result = add_relu(a, b)
    expected = reference_add_relu(a, b)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Large fused operation failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
