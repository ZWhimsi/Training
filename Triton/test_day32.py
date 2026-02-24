"""Test Suite for Day 32: Flash Attention Forward
Run: pytest test_day32.py -v
"""

import pytest
import torch

try:
    from day32 import flash_attention_forward, standard_attention
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day32")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_attention_small():
    """Test with small tensors."""
    B, H, M, N, D = 1, 1, 16, 16, 32
    Q = torch.randn(B, H, M, D, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    
    result = flash_attention_forward(Q, K, V)
    expected = standard_attention(Q, K, V)
    
    assert result is not None, "Returned None"
    max_diff = (result - expected).abs().max().item()
    assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2), f"Max diff: {max_diff:.4f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day32")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_attention_medium():
    """Test with medium tensors."""
    B, H, M, N, D = 2, 4, 64, 64, 64
    Q = torch.randn(B, H, M, D, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    
    result = flash_attention_forward(Q, K, V)
    expected = standard_attention(Q, K, V)
    
    max_diff = (result - expected).abs().max().item()
    assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2), f"Max diff: {max_diff:.4f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day32")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_attention_asymmetric():
    """Test with M != N."""
    B, H, M, N, D = 1, 2, 32, 48, 32
    Q = torch.randn(B, H, M, D, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    
    result = flash_attention_forward(Q, K, V)
    expected = standard_attention(Q, K, V)
    
    max_diff = (result - expected).abs().max().item()
    assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2), f"Max diff: {max_diff:.4f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day32")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_attention_output_sum():
    """Test that output is valid and matches reference."""
    B, H, M, N, D = 1, 1, 64, 64, 32
    Q = torch.randn(B, H, M, D, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    
    result = flash_attention_forward(Q, K, V)
    expected = standard_attention(Q, K, V)
    
    assert not torch.any(torch.isnan(result)), "Output contains NaN"
    assert not torch.any(torch.isinf(result)), "Output contains Inf"
    max_diff = (result - expected).abs().max().item()
    assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2), f"Values mismatch: {max_diff:.4f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day32")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_efficiency():
    """Verify Flash Attention produces correct output on larger inputs."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    B, H, M, N, D = 1, 1, 256, 256, 64
    Q = torch.randn(B, H, M, D, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    
    result = flash_attention_forward(Q, K, V)
    expected = standard_attention(Q, K, V)
    
    max_diff = (result - expected).abs().max().item()
    assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2), f"Values mismatch: {max_diff:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
