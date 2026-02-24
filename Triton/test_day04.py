"""
Test Suite for Day 4: Program IDs and Grid Configuration
========================================================
Run: pytest test_day04.py -v
"""

import pytest
import torch

try:
    from day04 import row_sum, col_sum, add_matrices_2d, vector_add_cyclic, batch_scale
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


# ============================================================================
# Reference Implementations (PyTorch)
# ============================================================================

def reference_row_sum(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=1)

def reference_col_sum(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=0)

def reference_add_matrices(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

def reference_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

def reference_batch_scale(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return x * scales.unsqueeze(1)


# ============================================================================
# Test Functions
# ============================================================================

@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day04")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_row_sum_basic():
    """Test row sum."""
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
                     dtype=torch.float32, device='cuda')
    result = row_sum(x)
    expected = reference_row_sum(x)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected.tolist()}, got {result.tolist()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day04")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_row_sum_large():
    """Test row sum with larger matrix."""
    x = torch.randn(100, 256, device='cuda')
    result = row_sum(x)
    expected = reference_row_sum(x)
    
    assert result is not None, "Function returned None"
    max_diff = (result - expected).abs().max().item()
    assert torch.allclose(result, expected, atol=1e-4), f"Max diff: {max_diff}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day04")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_col_sum_basic():
    """Test column sum."""
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
                     dtype=torch.float32, device='cuda')
    result = col_sum(x)
    expected = reference_col_sum(x)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected.tolist()}, got {result.tolist()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day04")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_col_sum_large():
    """Test column sum with larger matrix."""
    x = torch.randn(256, 100, device='cuda')
    result = col_sum(x)
    expected = reference_col_sum(x)
    
    assert result is not None, "Function returned None"
    max_diff = (result - expected).abs().max().item()
    assert torch.allclose(result, expected, atol=1e-4), f"Max diff: {max_diff}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day04")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_add_matrices_2d_basic():
    """Test 2D matrix addition."""
    a = torch.randn(64, 64, device='cuda')
    b = torch.randn(64, 64, device='cuda')
    result = add_matrices_2d(a, b)
    expected = reference_add_matrices(a, b)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Matrix addition failed"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day04")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_add_matrices_2d_non_square():
    """Test 2D matrix addition with non-square matrices."""
    a = torch.randn(100, 200, device='cuda')
    b = torch.randn(100, 200, device='cuda')
    result = add_matrices_2d(a, b)
    expected = reference_add_matrices(a, b)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Non-square matrix addition failed"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day04")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vector_add_cyclic():
    """Test cyclic vector addition."""
    a = torch.randn(10000, device='cuda')
    b = torch.randn(10000, device='cuda')
    result = vector_add_cyclic(a, b)
    expected = reference_vector_add(a, b)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Vector addition failed"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day04")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batch_scale_basic():
    """Test batch scaling."""
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
    scales = torch.tensor([2.0, 0.5], device='cuda')
    result = batch_scale(x, scales)
    expected = reference_batch_scale(x, scales)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected.tolist()}, got {result.tolist()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day04")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batch_scale_large():
    """Test batch scaling with larger batches."""
    batch_size = 64
    vec_size = 128
    x = torch.randn(batch_size, vec_size, device='cuda')
    scales = torch.randn(batch_size, device='cuda')
    result = batch_scale(x, scales)
    expected = reference_batch_scale(x, scales)
    
    assert result is not None, "Function returned None"
    assert torch.allclose(result, expected, atol=1e-5), "Large batch scale failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
