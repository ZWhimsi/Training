"""Test Suite for Day 20: Efficient Transpose
Run: pytest test_day20.py -v
"""

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day20 import naive_transpose, tiled_transpose, batched_transpose
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day20")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_naive_transpose():
    """Test naive transpose."""
    x = torch.randn(128, 256, device='cuda')
    result = naive_transpose(x)
    expected = x.T
    
    assert result.shape == expected.shape, f"Shape: {result.shape} != {expected.shape}"
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day20")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_tiled_transpose():
    """Test tiled transpose."""
    x = torch.randn(256, 512, device='cuda')
    result = tiled_transpose(x)
    expected = x.T
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day20")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_batched_transpose():
    """Test batched transpose."""
    x = torch.randn(8, 64, 128, device='cuda')
    result = batched_transpose(x)
    expected = x.transpose(1, 2)
    
    assert result.shape == expected.shape, f"Shape: {result.shape}"
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day20")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_non_square():
    """Test non-square matrix transpose."""
    for shape in [(64, 128), (100, 200), (33, 77)]:
        x = torch.randn(shape, device='cuda')
        result = tiled_transpose(x)
        expected = x.T
        
        max_err = (result - expected).abs().max().item()
        assert max_err <= 1e-4, f"Failed at {shape}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day20")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_correctness():
    """Test transpose correctness with spot checks."""
    x = torch.randn(128, 128, device='cuda')
    result = tiled_transpose(x)
    
    for _ in range(10):
        i = torch.randint(0, 128, (1,)).item()
        j = torch.randint(0, 128, (1,)).item()
        assert abs(result[i, j].item() - x[j, i].item()) <= 1e-5, f"result[{i},{j}] != x[{j},{i}]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
