"""
Day 26: Multi-head Latent Attention (MLA) Basics
================================================
Estimated time: 1-2 hours
Prerequisites: Day 16 (multi-head attention), Day 22 (GQA)

Learning objectives:
- Understand the memory bottleneck in standard KV caching
- Learn the low-rank projection concept for attention
- Understand compressed KV representations
- Implement basic latent projections
- Grasp the tradeoffs between compression and expressivity

Key Concepts:
-------------
The KV Cache Problem:
    In autoregressive generation, we cache K and V for all past tokens.
    Standard MHA: KV cache size = 2 * num_layers * seq_len * num_heads * head_dim
    
    Example (7B model, 32K context):
        2 * 32 layers * 32K tokens * 32 heads * 128 dim * 2 bytes = 16.7 GB!
    
    This is a major bottleneck for long-context inference.

Multi-head Latent Attention (MLA):
    DeepSeek-V2 innovation: compress KV into a low-rank latent space.
    
    Instead of caching full K, V tensors:
    1. Project input into a compressed latent: c = x @ W_down  (d_model -> d_latent)
    2. Only cache the compressed latent c
    3. Reconstruct K, V from latent: K = c @ W_k_up, V = c @ W_v_up
    
    KV cache reduction: d_latent << num_heads * head_dim

Mathematical Formulation:
    Standard attention:
        Q = x @ W_q    (d_model -> num_heads * head_dim)
        K = x @ W_k    (d_model -> num_heads * head_dim)  
        V = x @ W_v    (d_model -> num_heads * head_dim)
    
    MLA with compression:
        c_kv = x @ W_down_kv           (d_model -> d_latent)
        K = c_kv @ W_up_k              (d_latent -> num_heads * head_dim)
        V = c_kv @ W_up_v              (d_latent -> num_heads * head_dim)
        
    Compression ratio: d_latent / (num_heads * head_dim)
    Typical: d_latent = 512, num_heads * head_dim = 4096 -> 8x compression

Why Low-Rank Works:
    - Attention patterns are often redundant across heads
    - Key/Value information is highly correlated
    - Low-rank bottleneck acts as regularization
    - Empirically: quality preserved with significant compression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ============================================================================
# Exercise 1: KV Cache Size Calculation
# ============================================================================

def calculate_kv_cache_size(
    num_layers: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    batch_size: int = 1,
    dtype_bytes: int = 2  # fp16/bf16
) -> dict:
    """
    Calculate KV cache memory requirements for standard attention.
    
    Args:
        num_layers: Number of transformer layers
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        batch_size: Batch size
        dtype_bytes: Bytes per element (2 for fp16)
    
    Returns:
        Dictionary with cache size statistics
    """
    # API hints:
    # - per_layer_kv = 2 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    # - total_bytes = num_layers * per_layer_kv
    # - total_gb = total_bytes / (1024 ** 3)
    # - kv_dim = num_heads * head_dim
    
    return None


def calculate_mla_cache_size(
    num_layers: int,
    seq_len: int,
    d_latent: int,
    batch_size: int = 1,
    dtype_bytes: int = 2
) -> dict:
    """
    Calculate KV cache memory requirements for MLA.
    
    With MLA, we only cache the compressed latent, not full K and V.
    
    Args:
        num_layers: Number of transformer layers
        seq_len: Sequence length
        d_latent: Latent dimension (compressed KV size)
        batch_size: Batch size
        dtype_bytes: Bytes per element
    
    Returns:
        Dictionary with cache size statistics
    """
    # API hints:
    # - per_layer = batch_size * seq_len * d_latent * dtype_bytes
    # - total_bytes = num_layers * per_layer
    # - total_gb = total_bytes / (1024 ** 3)
    
    return None


# ============================================================================
# Exercise 2: Low-Rank Down Projection
# ============================================================================

class DownProjection(nn.Module):
    """
    Projects input from model dimension to lower-dimensional latent space.
    
    This is the compression step: d_model -> d_latent
    """
    
    def __init__(self, d_model: int, d_latent: int):
        """
        Args:
            d_model: Input dimension
            d_latent: Compressed latent dimension (d_latent < d_model)
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_latent = d_latent
        
        # TODO: Create down projection
        # API hints:
        # - nn.Linear(d_model, d_latent, bias=False) -> linear without bias
        
        self.down_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to latent space.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Compressed latent (batch, seq_len, d_latent)
        """
        # API hints:
        # - self.down_proj(x) -> apply linear transformation
        
        return None


# ============================================================================
# Exercise 3: Low-Rank Up Projection
# ============================================================================

class UpProjection(nn.Module):
    """
    Projects latent from compressed space back to full dimension.
    
    This reconstructs K or V from the cached latent.
    """
    
    def __init__(self, d_latent: int, d_output: int):
        """
        Args:
            d_latent: Compressed latent dimension
            d_output: Output dimension (num_heads * head_dim for K/V)
        """
        super().__init__()
        
        self.d_latent = d_latent
        self.d_output = d_output
        
        # TODO: Create up projection
        # API hints:
        # - nn.Linear(d_latent, d_output, bias=False) -> linear without bias
        
        self.up_proj = None
    
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct output from latent.
        
        Args:
            c: Compressed latent (batch, seq_len, d_latent)
        
        Returns:
            Reconstructed output (batch, seq_len, d_output)
        """
        # API hints:
        # - self.up_proj(c) -> apply linear transformation
        
        return None


# ============================================================================
# Exercise 4: Combined Low-Rank KV Projection
# ============================================================================

class LowRankKVProjection(nn.Module):
    """
    Low-rank factorized KV projection.
    
    Instead of:
        K = x @ W_k  (d_model, d_kv)
        V = x @ W_v  (d_model, d_kv)
    
    We use:
        c = x @ W_down      (d_model -> d_latent)
        K = c @ W_up_k      (d_latent -> d_kv)
        V = c @ W_up_v      (d_latent -> d_kv)
    
    The latent c is what we cache, not K and V.
    """
    
    def __init__(self, d_model: int, d_latent: int, num_heads: int, head_dim: int):
        """
        Args:
            d_model: Model dimension
            d_latent: Compressed latent dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_latent = d_latent
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_kv = num_heads * head_dim
        
        # TODO: Create down projection and separate up projections for K and V
        # API hints:
        # - nn.Linear(d_model, d_latent, bias=False) -> down projection
        # - nn.Linear(d_latent, d_kv, bias=False) -> up projections for K and V
        
        self.down_proj = None
        self.up_proj_k = None
        self.up_proj_v = None
    
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress input to latent representation (for caching).
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Compressed latent (batch, seq_len, d_latent)
        """
        # API hints:
        # - self.down_proj(x) -> apply compression
        
        return None
    
    def reconstruct_kv(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct K and V from compressed latent.
        
        Args:
            c: Compressed latent (batch, seq_len, d_latent)
        
        Returns:
            K: (batch, seq_len, num_heads, head_dim)
            V: (batch, seq_len, num_heads, head_dim)
        """
        # API hints:
        # - self.up_proj_k(c) -> reconstruct K (batch, seq, d_kv)
        # - self.up_proj_v(c) -> reconstruct V (batch, seq, d_kv)
        # - tensor.view(batch, seq_len, num_heads, head_dim) -> reshape
        
        return None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward: compress then reconstruct.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            c: Compressed latent (for caching)
            K: Key tensor
            V: Value tensor
        """
        # API hints:
        # - c = self.compress(x) -> compress
        # - k, v = self.reconstruct_kv(c) -> reconstruct
        
        return None


# ============================================================================
# Exercise 5: Compression Ratio Analysis
# ============================================================================

def analyze_compression(
    d_model: int,
    d_latent: int,
    num_heads: int,
    head_dim: int
) -> dict:
    """
    Analyze the compression achieved by low-rank KV projection.
    
    Args:
        d_model: Model dimension
        d_latent: Latent dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
    
    Returns:
        Dictionary with compression statistics
    """
    # API hints:
    # - d_kv = num_heads * head_dim
    # - standard_cache_dim = 2 * d_kv (K and V)
    # - mla_cache_dim = d_latent
    # - compression_ratio = standard_cache_dim / mla_cache_dim
    # - standard_params = 2 * d_model * d_kv
    # - mla_params = d_model * d_latent + 2 * d_latent * d_kv
    
    return None


# ============================================================================
# Exercise 6: Basic MLA Attention (Simplified)
# ============================================================================

class BasicMLAAttention(nn.Module):
    """
    Simplified MLA attention to understand the core concept.
    
    This version uses low-rank KV but standard Q projection.
    """
    
    def __init__(self, d_model: int, d_latent: int, num_heads: int, 
                 head_dim: int, dropout: float = 0.0):
        """
        Args:
            d_model: Model dimension
            d_latent: Latent dimension for KV compression
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_latent = d_latent
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_kv = num_heads * head_dim
        self.scale = head_dim ** -0.5
        
        # TODO: Initialize projections
        # API hints:
        # - nn.Linear(d_model, d_kv, bias=False) -> Q projection
        # - LowRankKVProjection(d_model, d_latent, num_heads, head_dim) -> KV compression
        # - nn.Linear(d_kv, d_model, bias=False) -> output projection
        # - nn.Dropout(dropout) -> dropout layer
        
        self.W_q = None
        self.kv_proj = None
        self.W_o = None
        self.dropout = None
    
    def forward(
        self, 
        x: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional KV caching.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            kv_cache: Cached latent from previous tokens (batch, cache_len, d_latent)
            use_cache: Whether to return updated cache
        
        Returns:
            output: Attention output (batch, seq_len, d_model)
            attn_weights: Attention weights
            new_cache: Updated latent cache (if use_cache=True)
        """
        # API hints:
        # - self.W_q(x) -> query projection
        # - tensor.view(batch, seq_len, num_heads, head_dim) -> reshape
        # - tensor.transpose(1, 2) -> swap seq and heads dimensions
        # - self.kv_proj(x) -> returns (c, k, v) compressed
        # - torch.cat([kv_cache, c], dim=1) -> concat cached latent
        # - self.kv_proj.reconstruct_kv(c) -> get K, V from latent
        # - torch.matmul(q, k.transpose(-2, -1)) * self.scale -> attention scores
        # - F.softmax(scores, dim=-1) -> attention weights
        
        return None


# ============================================================================
# Exercise 7: Visualizing Compression Quality
# ============================================================================

def measure_reconstruction_error(
    kv_proj: LowRankKVProjection,
    x: torch.Tensor,
    original_W_k: torch.Tensor,
    original_W_v: torch.Tensor
) -> dict:
    """
    Measure how well low-rank projection approximates full-rank.
    
    This helps understand the quality vs compression tradeoff.
    
    Args:
        kv_proj: Low-rank KV projection module
        x: Input tensor
        original_W_k: Original full-rank K projection weights
        original_W_v: Original full-rank V projection weights
    
    Returns:
        Dictionary with reconstruction error metrics
    """
    # API hints:
    # - with torch.no_grad(): -> disable gradient computation
    # - kv_proj(x) -> returns (c, k_lr, v_lr)
    # - F.linear(x, weight) -> linear transformation with given weights
    # - torch.norm(tensor) -> compute Frobenius norm
    # - F.cosine_similarity(a.flatten(), b.flatten(), dim=0) -> cosine similarity
    # - tensor.item() -> convert scalar tensor to Python number
    
    return None


if __name__ == "__main__":
    print("Day 26: Multi-head Latent Attention (MLA) Basics")
    print("=" * 60)
    
    # Configuration
    d_model = 4096
    d_latent = 512
    num_heads = 32
    head_dim = 128
    batch_size = 2
    seq_len = 16
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  d_latent: {d_latent}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  d_kv = num_heads * head_dim: {num_heads * head_dim}")
    
    # Compare cache sizes for a typical 7B model
    print("\n" + "=" * 60)
    print("Cache Size Comparison (7B model, 32K context)")
    print("=" * 60)
    
    std_cache = calculate_kv_cache_size(32, 32768, 32, 128)
    mla_cache = calculate_mla_cache_size(32, 32768, 512)
    
    if std_cache['total_gb'] > 0:
        print(f"Standard attention: {std_cache['total_gb']:.2f} GB")
        print(f"MLA attention: {mla_cache['total_gb']:.2f} GB")
        print(f"Reduction: {std_cache['total_bytes'] / mla_cache['total_bytes']:.1f}x")
    
    # Test compression analysis
    print("\n" + "=" * 60)
    print("Compression Analysis")
    print("=" * 60)
    
    compression = analyze_compression(d_model, d_latent, num_heads, head_dim)
    if compression['compression_ratio'] > 0:
        print(f"Cache compression ratio: {compression['compression_ratio']:.1f}x")
        print(f"Parameter overhead: {compression['param_overhead']:.2f}x")
    
    # Test LowRankKVProjection
    print("\n" + "=" * 60)
    print("Testing LowRankKVProjection")
    print("=" * 60)
    
    kv_proj = LowRankKVProjection(d_model, d_latent, num_heads, head_dim)
    if kv_proj.down_proj is not None:
        x = torch.randn(batch_size, seq_len, d_model)
        c, k, v = kv_proj(x)
        print(f"Input: {x.shape}")
        print(f"Compressed latent (cached): {c.shape}")
        print(f"Reconstructed K: {k.shape}")
        print(f"Reconstructed V: {v.shape}")
    
    # Test BasicMLAAttention
    print("\n" + "=" * 60)
    print("Testing BasicMLAAttention")
    print("=" * 60)
    
    mla = BasicMLAAttention(d_model, d_latent, num_heads, head_dim)
    if mla.W_q is not None:
        x = torch.randn(batch_size, seq_len, d_model)
        output, attn, cache = mla(x, use_cache=True)
        print(f"Input: {x.shape}")
        print(f"Output: {output.shape}")
        print(f"Cache: {cache.shape if cache is not None else 'None'}")
        
        # Test with cached generation
        print("\nSimulating generation with cache:")
        new_token = torch.randn(batch_size, 1, d_model)
        output2, _, cache2 = mla(new_token, kv_cache=cache, use_cache=True)
        print(f"New token output: {output2.shape}")
        print(f"Updated cache: {cache2.shape if cache2 is not None else 'None'}")
    
    print("\nRun test_day26.py to verify your implementations!")
