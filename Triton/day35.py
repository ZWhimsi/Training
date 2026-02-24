"""
Day 35: Complete Flash Attention
================================
Estimated time: 1-2 hours
Prerequisites: All previous Triton days

FINAL PROJECT: Combine everything into complete Flash Attention!

Learning objectives:
- Integrate forward and backward passes
- Create a PyTorch-compatible autograd Function
- Test against standard attention
- Benchmark performance
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional, Tuple


# Import from previous days
try:
    from day33 import flash_attention_v2
    from day31 import flash_attention_backward_dv_dk
    from day34 import flash_attention_backward_dq
except ImportError:
    pass


class FlashAttentionFunction(torch.autograd.Function):
    """
    Complete Flash Attention as a PyTorch autograd Function.
    
    Forward: Efficient attention using tiled algorithm
    Backward: Recompute attention weights, compute gradients
    """
    
    @staticmethod
    def forward(ctx, Q, K, V, causal=False):
        """
        Forward pass of Flash Attention.
        
        Args:
            Q, K, V: (batch, heads, seq_len, head_dim)
            causal: Whether to apply causal mask
        
        Returns:
            output: (batch, heads, seq_len, head_dim)
        """
        # TODO: Run flash attention forward from day33
        # API hints:
        # - flash_attention_v2(Q, K, V, causal=causal) -> (output, L, M)
        # - L: log-sum-exp values for backward pass
        # - M: max values for backward pass
        
        output, L, M = None, None, None  # Exercise: call flash_attention_v2
        
        # Save for backward
        ctx.save_for_backward(Q, K, V, output, L, M)
        ctx.causal = causal
        
        return output
    
    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass of Flash Attention.
        
        Args:
            dO: Gradient of output (batch, heads, seq_len, head_dim)
        
        Returns:
            dQ, dK, dV, None (for causal flag)
        """
        Q, K, V, output, L, M = ctx.saved_tensors
        
        # TODO: Compute gradients using kernels from day31 and day34
        # API hints:
        # - flash_attention_backward_dv_dk(Q, K, V, output, dO, L) -> (dK, dV)
        # - flash_attention_backward_dq(Q, K, V, output, dO, L, M) -> dQ
        # - L, M: saved statistics from forward pass
        
        # Exercise: Call the backward kernels to compute gradients
        dK, dV = None, None  # Exercise: call flash_attention_backward_dv_dk
        dQ = None  # Exercise: call flash_attention_backward_dq
        
        # Fallback to reference implementation if kernels return None
        if dQ is None or dK is None or dV is None:
            scale = 1.0 / math.sqrt(Q.shape[-1])
            scores = (Q @ K.transpose(-2, -1)) * scale
            
            if ctx.causal:
                seq_len = Q.shape[-2]
                mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
                scores.masked_fill_(mask, float('-inf'))
            
            P = torch.softmax(scores, dim=-1)
            dV = P.transpose(-2, -1) @ dO
            dP = dO @ V.transpose(-2, -1)
            D = (P * dP).sum(dim=-1, keepdim=True)
            dS = P * (dP - D)
            dQ = dS @ K * scale
            dK = dS.transpose(-2, -1) @ Q * scale
        
        return dQ, dK, dV, None


def flash_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor,
    causal: bool = False
) -> torch.Tensor:
    """
    User-facing Flash Attention function.
    
    Supports autograd for training.
    """
    return FlashAttentionFunction.apply(Q, K, V, causal)


def standard_attention(Q, K, V, causal=False):
    """Reference standard attention for comparison."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = (Q @ K.transpose(-2, -1)) * scale
    
    if causal:
        seq_len = Q.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    
    return torch.softmax(scores, dim=-1) @ V


def benchmark(batch, n_heads, seq_len, head_dim, device='cuda'):
    """Benchmark Flash Attention vs Standard Attention."""
    import time
    
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    K = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    V = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    
    # Warmup
    _ = flash_attention(Q, K, V)
    _ = standard_attention(Q, K, V)
    torch.cuda.synchronize()
    
    # Benchmark Flash Attention
    n_runs = 100
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        _ = flash_attention(Q, K, V)
    torch.cuda.synchronize()
    flash_time = (time.time() - start) / n_runs * 1000
    
    # Benchmark Standard Attention
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        _ = standard_attention(Q, K, V)
    torch.cuda.synchronize()
    std_time = (time.time() - start) / n_runs * 1000
    
    return flash_time, std_time


if __name__ == "__main__":
    print("Day 35: Complete Flash Attention")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        # Test correctness
        batch, n_heads, seq_len, head_dim = 2, 8, 256, 64
        
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device=device, requires_grad=True)
        K = torch.randn(batch, n_heads, seq_len, head_dim, device=device, requires_grad=True)
        V = torch.randn(batch, n_heads, seq_len, head_dim, device=device, requires_grad=True)
        
        print("\n1. Testing forward pass:")
        output = flash_attention(Q, K, V)
        expected = standard_attention(Q.detach(), K.detach(), V.detach())
        print(f"   Max error: {(output - expected).abs().max().item():.6f}")
        
        print("\n2. Testing backward pass:")
        loss = output.sum()
        loss.backward()
        
        if Q.grad is not None:
            print(f"   dQ computed, max: {Q.grad.abs().max().item():.4f}")
        if K.grad is not None:
            print(f"   dK computed, max: {K.grad.abs().max().item():.4f}")
        if V.grad is not None:
            print(f"   dV computed, max: {V.grad.abs().max().item():.4f}")
        
        print("\n3. Benchmark (seq_len=512):")
        flash_ms, std_ms = benchmark(2, 8, 512, 64)
        print(f"   Flash Attention: {flash_ms:.2f} ms")
        print(f"   Standard Attention: {std_ms:.2f} ms")
        print(f"   Speedup: {std_ms/flash_ms:.2f}x")
        
        print("\n" + "=" * 50)
        print("CONGRATULATIONS! You've completed the Triton track!")
        print("You built Flash Attention from scratch!")
        print("=" * 50)
    else:
        print("CUDA not available")
    
    print("\nRun test_day35.py for final verification!")
