"""
Test Suite for Day 1: Hello Triton
==================================
Run this file to check your implementations against PyTorch reference.

Usage:
    pytest test_day01.py -v
"""

import pytest
import torch

try:
    from day01 import add_one, square
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def check_cuda() -> bool:
    """Check if CUDA is available."""
    if not torch.cuda.is_available():
        return False
    return True


# ============================================================================
# Reference Implementations (PyTorch)
# ============================================================================

def reference_add_one(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: add 1 to every element."""
    return x + 1


def reference_square(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: square every element."""
    return x * x


# ============================================================================
# Test Functions
# ============================================================================

@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day01")
@pytest.mark.skipif(not check_cuda(), reason="CUDA not available")
def test_exercise_1_add_one_basic():
    """Test add_one with a small tensor."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    
    result = add_one(x)
    expected = reference_add_one(x)
    
    assert result is not None, "Function returned None (not implemented yet)"
    assert torch.allclose(result, expected, atol=1e-5), f"Mismatch: expected {expected.tolist()}, got {result.tolist()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day01")
@pytest.mark.skipif(not check_cuda(), reason="CUDA not available")
def test_exercise_1_add_one_large():
    """Test add_one with larger tensor to verify blocking."""
    x = torch.randn(10000, device='cuda')
    
    result = add_one(x)
    expected = reference_add_one(x)
    
    assert result is not None, "Function returned None"
    max_diff = (result - expected).abs().max().item()
    assert torch.allclose(result, expected, atol=1e-5), f"Mismatch on large tensor, max diff: {max_diff}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day01")
@pytest.mark.skipif(not check_cuda(), reason="CUDA not available")
def test_exercise_1_add_one_2d():
    """Test add_one with 2D tensor (flattened internally)."""
    x = torch.randn(100, 100, device='cuda')
    
    result = add_one(x)
    expected = reference_add_one(x)
    
    assert result is not None, "Function returned None"
    assert result.shape == expected.shape, f"Shape mismatch: expected {expected.shape}, got {result.shape}"
    assert torch.allclose(result, expected, atol=1e-5), "Values don't match for 2D tensor"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day01")
@pytest.mark.skipif(not check_cuda(), reason="CUDA not available")
def test_exercise_3_square_basic():
    """Test square with a small tensor."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    
    result = square(x)
    expected = reference_square(x)
    
    assert result is not None, "Function returned None (not implemented yet)"
    assert torch.allclose(result, expected, atol=1e-5), f"Mismatch: expected {expected.tolist()}, got {result.tolist()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day01")
@pytest.mark.skipif(not check_cuda(), reason="CUDA not available")
def test_exercise_3_square_negative():
    """Test square with negative numbers."""
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
    
    result = square(x)
    expected = reference_square(x)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), f"Mismatch with negatives: expected {expected.tolist()}, got {result.tolist()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day01")
@pytest.mark.skipif(not check_cuda(), reason="CUDA not available")
def test_exercise_3_square_large():
    """Test square with larger tensor."""
    x = torch.randn(5000, device='cuda')
    
    result = square(x)
    expected = reference_square(x)
    
    assert result is not None, "Function returned None"
    max_diff = (result - expected).abs().max().item()
    assert torch.allclose(result, expected, atol=1e-5), f"Mismatch on large tensor, max diff: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
