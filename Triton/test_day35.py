"""Test Suite for Day 35: Complete Flash Attention

FINAL TEST - Complete Flash Attention Implementation
Run: pytest test_day35.py -v
"""

import pytest
import torch
import math

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day35 import flash_attention, standard_attention, FlashAttentionFunction
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day35")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_forward_correctness():
    """Test forward pass correctness."""
    batch, n_heads, seq_len, head_dim = 2, 8, 128, 64
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    result = flash_attention(Q, K, V)
    expected = standard_attention(Q, K, V)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-2, f"Forward error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day35")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_causal_forward():
    """Test causal forward pass."""
    batch, n_heads, seq_len, head_dim = 2, 8, 128, 64
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    result = flash_attention(Q, K, V, causal=True)
    expected = standard_attention(Q, K, V, causal=True)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-2, f"Causal error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day35")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_backward_runs():
    """Test backward pass runs and produces gradients."""
    batch, n_heads, seq_len, head_dim = 1, 4, 64, 32
    Q1 = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    K1 = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    V1 = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    
    dO = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    output = flash_attention(Q1, K1, V1)
    output.backward(dO)
    
    assert Q1.grad is not None, "Q gradients not computed"
    assert K1.grad is not None, "K gradients not computed"
    assert V1.grad is not None, "V gradients not computed"
    assert not torch.isnan(Q1.grad).any(), "NaN in Q gradients"
    assert not torch.isnan(K1.grad).any(), "NaN in K gradients"
    assert not torch.isnan(V1.grad).any(), "NaN in V gradients"
    
    Q2 = Q1.detach().clone().requires_grad_(True)
    K2 = K1.detach().clone().requires_grad_(True)
    V2 = V1.detach().clone().requires_grad_(True)
    
    ref_out = standard_attention(Q2, K2, V2)
    ref_out.backward(dO)
    
    assert torch.allclose(Q1.grad, Q2.grad, atol=0.1, rtol=0.1), "dQ mismatch"
    assert torch.allclose(K1.grad, K2.grad, atol=0.1, rtol=0.1), "dK mismatch"
    assert torch.allclose(V1.grad, V2.grad, atol=0.1, rtol=0.1), "dV mismatch"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day35")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_backward_correctness():
    """Test backward pass correctness."""
    batch, n_heads, seq_len, head_dim = 1, 4, 32, 16
    
    Q1 = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    K1 = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    V1 = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    
    dO = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    
    out1 = flash_attention(Q1, K1, V1)
    out1.backward(dO)
    
    Q2 = Q1.detach().clone().requires_grad_(True)
    K2 = K1.detach().clone().requires_grad_(True)
    V2 = V1.detach().clone().requires_grad_(True)
    
    out2 = standard_attention(Q2, K2, V2)
    out2.backward(dO)
    
    dq_err = (Q1.grad - Q2.grad).abs().max().item()
    dk_err = (K1.grad - K2.grad).abs().max().item()
    dv_err = (V1.grad - V2.grad).abs().max().item()
    
    max_err = max(dq_err, dk_err, dv_err)
    assert max_err <= 0.1, f"Grad error: dQ={dq_err:.4f}, dK={dk_err:.4f}, dV={dv_err:.4f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day35")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_various_sizes():
    """Test various tensor sizes."""
    for batch, heads, seq, dim in [(1, 4, 64, 32), (2, 8, 128, 64), (1, 1, 256, 64)]:
        Q = torch.randn(batch, heads, seq, dim, device='cuda')
        K = torch.randn(batch, heads, seq, dim, device='cuda')
        V = torch.randn(batch, heads, seq, dim, device='cuda')
        
        result = flash_attention(Q, K, V)
        expected = standard_attention(Q, K, V)
        
        max_err = (result - expected).abs().max().item()
        assert max_err <= 0.05, f"Failed at ({batch},{heads},{seq},{dim})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
