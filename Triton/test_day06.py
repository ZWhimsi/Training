"""
Test Suite for Day 6: Multi-dimensional Grids
=============================================
Run: pytest test_day06.py -v
"""

import pytest
import torch

try:
    from day06 import elementwise_2d, batch_matvec, batch_add_3d, sum_axis0, broadcast_add
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


# References
def ref_elementwise_2d(x, y): return x * y
def ref_batch_matvec(A, x): return torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
def ref_batch_add_3d(a, b): return a + b
def ref_sum_axis0(x): return x.sum(dim=0)
def ref_broadcast_add(x, bias): return x + bias


# Tests
@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day06")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_elementwise_2d():
    """Test elementwise 2D multiplication."""
    x = torch.randn(64, 64, device='cuda')
    y = torch.randn(64, 64, device='cuda')
    result = elementwise_2d(x, y)
    expected = ref_elementwise_2d(x, y)
    
    assert result is not None, "None"
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day06")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_elementwise_2d_rect():
    """Test elementwise 2D with rectangular matrix."""
    x = torch.randn(100, 200, device='cuda')
    y = torch.randn(100, 200, device='cuda')
    result = elementwise_2d(x, y)
    expected = ref_elementwise_2d(x, y)
    
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day06")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batch_matvec_basic():
    """Test batch matrix-vector multiplication."""
    B, M, N = 4, 8, 16
    A = torch.randn(B, M, N, device='cuda')
    x = torch.randn(B, N, device='cuda')
    result = batch_matvec(A, x)
    expected = ref_batch_matvec(A, x)
    
    assert result is not None, "None"
    assert result.shape == expected.shape, f"Shape {result.shape} vs {expected.shape}"
    assert torch.allclose(result, expected, atol=1e-4), "Values mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day06")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batch_matvec_large():
    """Test batch matvec with larger tensors."""
    B, M, N = 16, 32, 64
    A = torch.randn(B, M, N, device='cuda')
    x = torch.randn(B, N, device='cuda')
    result = batch_matvec(A, x)
    expected = ref_batch_matvec(A, x)
    
    assert torch.allclose(result, expected, atol=1e-4), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day06")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batch_add_3d():
    """Test 3D batch addition."""
    a = torch.randn(8, 32, 32, device='cuda')
    b = torch.randn(8, 32, 32, device='cuda')
    result = batch_add_3d(a, b)
    expected = ref_batch_add_3d(a, b)
    
    assert result is not None, "None"
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day06")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sum_axis0_basic():
    """Test sum along axis 0."""
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
    result = sum_axis0(x)
    expected = ref_sum_axis0(x)
    
    assert result is not None, "None"
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected.tolist()}, got {result.tolist()}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day06")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sum_axis0_large():
    """Test sum axis 0 with large tensor."""
    x = torch.randn(256, 128, device='cuda')
    result = sum_axis0(x)
    expected = ref_sum_axis0(x)
    
    assert torch.allclose(result, expected, atol=1e-4), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day06")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_broadcast_add_basic():
    """Test broadcast addition."""
    x = torch.randn(4, 8, device='cuda')
    bias = torch.randn(8, device='cuda')
    result = broadcast_add(x, bias)
    expected = ref_broadcast_add(x, bias)
    
    assert result is not None, "None"
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day06")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_broadcast_add_large():
    """Test broadcast add with large tensors."""
    x = torch.randn(128, 256, device='cuda')
    bias = torch.randn(256, device='cuda')
    result = broadcast_add(x, bias)
    expected = ref_broadcast_add(x, bias)
    
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
