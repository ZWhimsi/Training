"""Test Suite for Day 16: Tiled Matrix Multiplication
Run: pytest test_day16.py -v
"""

import pytest
import torch

try:
    from day16 import matmul, matmul_bias, batched_matmul
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day16")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matmul_square():
    """Test square matrix multiplication."""
    A = torch.randn(64, 64, device='cuda')
    B = torch.randn(64, 64, device='cuda')
    result = matmul(A, B)
    expected = torch.mm(A, B)
    
    max_diff = (result - expected).abs().max().item()
    assert torch.allclose(result, expected, atol=1e-4, rtol=1e-4), f"Max diff: {max_diff}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day16")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matmul_rect():
    """Test rectangular matrix multiplication."""
    A = torch.randn(128, 256, device='cuda')
    B = torch.randn(256, 64, device='cuda')
    result = matmul(A, B)
    expected = torch.mm(A, B)
    
    assert torch.allclose(result, expected, atol=1e-4, rtol=1e-4), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day16")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matmul_non_aligned():
    """Test non-aligned matrix multiplication."""
    A = torch.randn(100, 75, device='cuda')
    B = torch.randn(75, 50, device='cuda')
    result = matmul(A, B)
    expected = torch.mm(A, B)
    
    assert torch.allclose(result, expected, atol=1e-4, rtol=1e-4), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day16")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matmul_bias():
    """Test matrix multiplication with bias."""
    A = torch.randn(64, 128, device='cuda')
    B = torch.randn(128, 64, device='cuda')
    bias = torch.randn(64, device='cuda')
    result = matmul_bias(A, B, bias)
    expected = torch.mm(A, B) + bias
    
    assert torch.allclose(result, expected, atol=1e-4, rtol=1e-4), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day16")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batched_matmul():
    """Test batched matrix multiplication."""
    A = torch.randn(8, 32, 64, device='cuda')
    B = torch.randn(8, 64, 32, device='cuda')
    result = batched_matmul(A, B)
    expected = torch.bmm(A, B)
    
    assert torch.allclose(result, expected, atol=1e-4, rtol=1e-4), "Mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
