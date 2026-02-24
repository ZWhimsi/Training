"""Test Suite for Day 22: Batch Matrix Multiply
Run: pytest test_day22.py -v
"""

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day22 import batched_matmul, batched_matmul_bt, scaled_batched_matmul
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day22")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_batched_matmul():
    """Test batched matrix multiplication."""
    B, M, K, N = 8, 64, 32, 64
    A = torch.randn(B, M, K, device='cuda')
    B_mat = torch.randn(B, K, N, device='cuda')
    
    result = batched_matmul(A, B_mat)
    expected = torch.bmm(A, B_mat)
    
    assert result.shape == expected.shape, f"Shape: {result.shape}"
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day22")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_batched_matmul_bt():
    """Test batched matmul with B transposed."""
    B, M, K, N = 8, 64, 32, 64
    A = torch.randn(B, M, K, device='cuda')
    B_mat = torch.randn(B, N, K, device='cuda')
    
    result = batched_matmul_bt(A, B_mat)
    expected = torch.bmm(A, B_mat.transpose(-2, -1))
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day22")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_scaled_matmul():
    """Test scaled batched matmul."""
    B, M, K, N = 8, 64, 32, 64
    A = torch.randn(B, M, K, device='cuda')
    B_mat = torch.randn(B, K, N, device='cuda')
    scale = 0.125
    
    result = scaled_batched_matmul(A, B_mat, scale)
    expected = scale * torch.bmm(A, B_mat)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day22")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_attention_pattern():
    """Test attention score pattern (Q @ K^T / sqrt(d_k))."""
    B, seq_len, d_k = 4, 128, 64
    Q = torch.randn(B, seq_len, d_k, device='cuda')
    K = torch.randn(B, seq_len, d_k, device='cuda')
    scale = 1.0 / (d_k ** 0.5)
    
    result = batched_matmul_bt(Q, K) * scale
    expected = torch.bmm(Q, K.transpose(-2, -1)) * scale
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day22")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_various_sizes():
    """Test various tensor sizes."""
    for B, M, K, N in [(4, 32, 32, 32), (8, 64, 64, 64), (16, 100, 50, 100)]:
        A = torch.randn(B, M, K, device='cuda')
        B_mat = torch.randn(B, K, N, device='cuda')
        
        result = batched_matmul(A, B_mat)
        expected = torch.bmm(A, B_mat)
        
        max_err = (result - expected).abs().max().item()
        assert max_err <= 1e-2, f"Failed at ({B},{M},{K},{N})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
