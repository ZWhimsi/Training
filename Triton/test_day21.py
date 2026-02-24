"""Test Suite for Day 21: Matrix-Vector Products
Run: pytest test_day21.py -v
"""

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day21 import matvec, matvec_blocked, vecmat, batched_matvec
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day21")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_matvec():
    """Test matrix-vector multiplication."""
    M, N = 256, 128
    A = torch.randn(M, N, device='cuda')
    x = torch.randn(N, device='cuda')
    
    result = matvec(A, x)
    expected = A @ x
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day21")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_matvec_blocked():
    """Test blocked matrix-vector multiplication."""
    M, N = 256, 2048
    A = torch.randn(M, N, device='cuda')
    x = torch.randn(N, device='cuda')
    
    result = matvec_blocked(A, x)
    expected = A @ x
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day21")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_vecmat():
    """Test vector-matrix multiplication."""
    M, N = 128, 256
    v = torch.randn(M, device='cuda')
    A = torch.randn(M, N, device='cuda')
    
    result = vecmat(v, A)
    expected = v @ A
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day21")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_batched_matvec():
    """Test batched matrix-vector multiplication."""
    B, M, N = 8, 256, 128
    A = torch.randn(B, M, N, device='cuda')
    x = torch.randn(B, N, device='cuda')
    
    result = batched_matvec(A, x)
    expected = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
    
    assert result.shape == expected.shape, f"Shape: {result.shape}"
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day21")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_various_sizes():
    """Test various tensor sizes."""
    for M, N in [(64, 64), (128, 256), (100, 200)]:
        A = torch.randn(M, N, device='cuda')
        x = torch.randn(N, device='cuda')
        
        result = matvec(A, x)
        expected = A @ x
        
        max_err = (result - expected).abs().max().item()
        assert max_err <= 1e-3, f"Failed at {M}x{N}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
