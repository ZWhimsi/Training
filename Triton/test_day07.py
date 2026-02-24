"""
Test Suite for Day 7: Memory Coalescing
=======================================
Run: pytest test_day07.py -v
"""

import pytest
import torch

try:
    from day07 import row_access, col_access, tiled_access, vectorized_op, benchmark_access_patterns
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def ref_double(x): return x * 2.0


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day07")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_row_access():
    """Test row access pattern."""
    x = torch.randn(64, 128, device='cuda')
    result = row_access(x)
    expected = ref_double(x)
    
    assert result is not None, "None"
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day07")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_col_access():
    """Test column access pattern."""
    x = torch.randn(64, 128, device='cuda')
    result = col_access(x)
    expected = ref_double(x)
    
    assert result is not None, "None"
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day07")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tiled_access():
    """Test tiled access pattern."""
    x = torch.randn(100, 200, device='cuda')
    result = tiled_access(x)
    expected = ref_double(x)
    
    assert result is not None, "None"
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day07")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vectorized():
    """Test vectorized operation."""
    x = torch.randn(100000, device='cuda')
    result = vectorized_op(x)
    expected = ref_double(x)
    
    assert result is not None, "None"
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
