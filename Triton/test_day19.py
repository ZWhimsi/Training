"""Test Suite for Day 19: Block Matrix Operations
Run: pytest test_day19.py -v
"""

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day19 import (extract_block, block_matrix_add, 
                           extract_block_diagonal, block_traces)
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day19")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_extract_block():
    """Test block extraction."""
    x = torch.randn(128, 128, device='cuda')
    block = extract_block(x, 1, 2, 32, 32)
    expected = x[32:64, 64:96]
    
    max_err = (block - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day19")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_block_add():
    """Test block matrix addition."""
    a = torch.randn(100, 100, device='cuda')
    b = torch.randn(100, 100, device='cuda')
    
    result = block_matrix_add(a, b)
    expected = a + b
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day19")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_block_diagonal():
    """Test block diagonal extraction."""
    x = torch.randn(128, 128, device='cuda')
    block_size = 32
    
    result = extract_block_diagonal(x, block_size)
    
    expected_shape = (4, 32, 32)
    assert result.shape == expected_shape, f"Shape: {result.shape} != {expected_shape}"
    
    expected_block = x[:32, :32]
    max_err = (result[0] - expected_block).abs().max().item()
    assert max_err <= 1e-5, f"Block 0 error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day19")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_block_traces():
    """Test block traces."""
    x = torch.eye(128, device='cuda')
    block_size = 32
    
    result = block_traces(x, block_size)
    expected = torch.tensor([32.0] * 4, device='cuda')
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day19")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_non_square():
    """Test non-square matrix block add."""
    a = torch.randn(96, 128, device='cuda')
    b = torch.randn(96, 128, device='cuda')
    
    result = block_matrix_add(a, b, block_m=32, block_n=32)
    expected = a + b
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
