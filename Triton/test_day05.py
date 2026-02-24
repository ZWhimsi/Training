"""
Test Suite for Day 5: Block-Level Programming
=============================================
Run: pytest test_day05.py -v
"""

import pytest
import torch

try:
    from day05 import add_vectors, two_phase_sum, transpose, find_max, softmax_numerator
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


# ============================================================================
# Reference Implementations
# ============================================================================

def reference_add(a, b):
    return a + b

def reference_sum(x):
    return x.sum().unsqueeze(0)

def reference_transpose(x):
    return x.T.contiguous()

def reference_max(x):
    return x.max().unsqueeze(0)

def reference_softmax_num(x, row_maxes):
    return torch.exp(x - row_maxes.unsqueeze(1))


# ============================================================================
# Tests
# ============================================================================

@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day05")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_add_vectors_default():
    """Test add_vectors with default block size."""
    a = torch.randn(10000, device='cuda')
    b = torch.randn(10000, device='cuda')
    result = add_vectors(a, b)
    expected = reference_add(a, b)
    
    assert result is not None, "Returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Values mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day05")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_add_vectors_small_block():
    """Test add_vectors with small block size."""
    a = torch.randn(10000, device='cuda')
    b = torch.randn(10000, device='cuda')
    result = add_vectors(a, b, block_size=256)
    expected = reference_add(a, b)
    
    assert result is not None, "Returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Values mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day05")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_two_phase_sum_basic():
    """Test two_phase_sum with basic input."""
    x = torch.arange(100, dtype=torch.float32, device='cuda')
    result = two_phase_sum(x)
    expected = reference_sum(x)
    
    assert result is not None, "Returned None"
    assert torch.allclose(result, expected, atol=1e-3), f"Expected {expected.item()}, got {result.item()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day05")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_two_phase_sum_large():
    """Test two_phase_sum with large tensor."""
    x = torch.randn(100000, device='cuda')
    result = two_phase_sum(x)
    expected = reference_sum(x)
    
    assert result is not None, "Returned None"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3), "Large sum mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day05")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_transpose_square():
    """Test transpose with square matrix."""
    x = torch.randn(64, 64, device='cuda')
    result = transpose(x)
    expected = reference_transpose(x)
    
    assert result is not None, "Returned None"
    assert result.shape == expected.shape, f"Shape: expected {expected.shape}, got {result.shape}"
    assert torch.allclose(result, expected, atol=1e-5), "Values mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day05")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_transpose_rect():
    """Test transpose with rectangular matrix."""
    x = torch.randn(100, 50, device='cuda')
    result = transpose(x)
    expected = reference_transpose(x)
    
    assert result is not None, "Returned None"
    assert result.shape == expected.shape, f"Shape: expected {expected.shape}, got {result.shape}"
    assert torch.allclose(result, expected, atol=1e-5), "Values mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day05")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_find_max_basic():
    """Test find_max with basic input."""
    x = torch.tensor([-5, -2, 3, 1, 7, 2], dtype=torch.float32, device='cuda')
    result = find_max(x)
    expected = reference_max(x)
    
    assert result is not None, "Returned None"
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected.item()}, got {result.item()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day05")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_find_max_large():
    """Test find_max with large tensor."""
    x = torch.randn(50000, device='cuda')
    result = find_max(x)
    expected = reference_max(x)
    
    assert result is not None, "Returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Max mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day05")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_softmax_num_basic():
    """Test softmax_numerator with basic input."""
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
    row_maxes = x.max(dim=1).values
    result = softmax_numerator(x, row_maxes)
    expected = reference_softmax_num(x, row_maxes)
    
    assert result is not None, "Returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Softmax numerator mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day05")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_softmax_num_large():
    """Test softmax_numerator with large tensor."""
    x = torch.randn(32, 128, device='cuda')
    row_maxes = x.max(dim=1).values
    result = softmax_numerator(x, row_maxes)
    expected = reference_softmax_num(x, row_maxes)
    
    assert result is not None, "Returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Large softmax numerator mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
