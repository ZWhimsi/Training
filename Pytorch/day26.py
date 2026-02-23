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
    
    TODO: Calculate cache sizes
    HINT:
        # Per-layer KV cache: K and V, each (batch, num_heads, seq_len, head_dim)
        per_layer_kv = 2 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
        
        # Total across all layers
        total_bytes = num_layers * per_layer_kv
        
        # Convert to GB
        total_gb = total_bytes / (1024 ** 3)
        
        return {
            'per_layer_bytes': per_layer_kv,
            'total_bytes': total_bytes,
            'total_gb': total_gb,
            'kv_dim': num_heads * head_dim
        }
    """
    return {
        'per_layer_bytes': 0,
        'total_bytes': 0,
        'total_gb': 0.0,
        'kv_dim': 0
    }  # Replace


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
    
    TODO: Calculate MLA cache sizes
    HINT:
        # Per-layer: only cache compressed latent (batch, seq_len, d_latent)
        per_layer = batch_size * seq_len * d_latent * dtype_bytes
        
        total_bytes = num_layers * per_layer
        total_gb = total_bytes / (1024 ** 3)
        
        return {
            'per_layer_bytes': per_layer,
            'total_bytes': total_bytes,
            'total_gb': total_gb,
            'latent_dim': d_latent
        }
    """
    return {
        'per_layer_bytes': 0,
        'total_bytes': 0,
        'total_gb': 0.0,
        'latent_dim': 0
    }  # Replace


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
        # HINT:
        #   self.down_proj = nn.Linear(d_model, d_latent, bias=False)
        self.down_proj = None  # Replace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to latent space.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Compressed latent (batch, seq_len, d_latent)
        
        TODO: Apply down projection
        HINT:
            return self.down_proj(x)
        """
        return x  # Replace


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
        # HINT:
        #   self.up_proj = nn.Linear(d_latent, d_output, bias=False)
        self.up_proj = None  # Replace
    
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct output from latent.
        
        Args:
            c: Compressed latent (batch, seq_len, d_latent)
        
        Returns:
            Reconstructed output (batch, seq_len, d_output)
        
        TODO: Apply up projection
        HINT:
            return self.up_proj(c)
        """
        return c  # Replace


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
        # HINT:
        #   # Shared down projection (compression)
        #   self.down_proj = nn.Linear(d_model, d_latent, bias=False)
        #   
        #   # Separate up projections for K and V
        #   self.up_proj_k = nn.Linear(d_latent, self.d_kv, bias=False)
        #   self.up_proj_v = nn.Linear(d_latent, self.d_kv, bias=False)
        self.down_proj = None   # Replace
        self.up_proj_k = None   # Replace
        self.up_proj_v = None   # Replace
    
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress input to latent representation (for caching).
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Compressed latent (batch, seq_len, d_latent)
        
        TODO: Apply down projection
        HINT:
            return self.down_proj(x)
        """
        return x  # Replace
    
    def reconstruct_kv(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct K and V from compressed latent.
        
        Args:
            c: Compressed latent (batch, seq_len, d_latent)
        
        Returns:
            K: (batch, seq_len, num_heads, head_dim)
            V: (batch, seq_len, num_heads, head_dim)
        
        TODO: Reconstruct K and V from latent
        HINT:
            batch, seq_len, _ = c.shape
            
            # Reconstruct K and V
            k = self.up_proj_k(c)  # (batch, seq, d_kv)
            v = self.up_proj_v(c)  # (batch, seq, d_kv)
            
            # Reshape to (batch, seq, num_heads, head_dim)
            k = k.view(batch, seq_len, self.num_heads, self.head_dim)
            v = v.view(batch, seq_len, self.num_heads, self.head_dim)
            
            return k, v
        """
        batch, seq_len = c.shape[:2]
        return (
            torch.zeros(batch, seq_len, self.num_heads, self.head_dim),
            torch.zeros(batch, seq_len, self.num_heads, self.head_dim)
        )  # Replace
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward: compress then reconstruct.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            c: Compressed latent (for caching)
            K: Key tensor
            V: Value tensor
        
        TODO: Implement full forward pass
        HINT:
            c = self.compress(x)
            k, v = self.reconstruct_kv(c)
            return c, k, v
        """
        batch, seq_len = x.shape[:2]
        return (
            torch.zeros(batch, seq_len, self.d_latent),
            torch.zeros(batch, seq_len, self.num_heads, self.head_dim),
            torch.zeros(batch, seq_len, self.num_heads, self.head_dim)
        )  # Replace


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
    
    TODO: Compute compression metrics
    HINT:
        d_kv = num_heads * head_dim
        
        # Standard KV: store K and V
        standard_cache_dim = 2 * d_kv  # Both K and V
        
        # MLA: store only compressed latent
        mla_cache_dim = d_latent
        
        # Compression ratio
        compression_ratio = standard_cache_dim / mla_cache_dim
        
        # Parameter comparison
        # Standard: W_k (d_model, d_kv) + W_v (d_model, d_kv)
        standard_params = 2 * d_model * d_kv
        
        # MLA: W_down (d_model, d_latent) + W_up_k (d_latent, d_kv) + W_up_v (d_latent, d_kv)
        mla_params = d_model * d_latent + 2 * d_latent * d_kv
        
        return {
            'standard_cache_dim': standard_cache_dim,
            'mla_cache_dim': mla_cache_dim,
            'compression_ratio': compression_ratio,
            'standard_params': standard_params,
            'mla_params': mla_params,
            'param_overhead': mla_params / standard_params
        }
    """
    return {
        'standard_cache_dim': 0,
        'mla_cache_dim': 0,
        'compression_ratio': 0.0,
        'standard_params': 0,
        'mla_params': 0,
        'param_overhead': 0.0
    }  # Replace


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
        # HINT:
        #   # Standard Q projection
        #   self.W_q = nn.Linear(d_model, self.d_kv, bias=False)
        #   
        #   # Low-rank KV projections
        #   self.kv_proj = LowRankKVProjection(d_model, d_latent, num_heads, head_dim)
        #   
        #   # Output projection
        #   self.W_o = nn.Linear(self.d_kv, d_model, bias=False)
        #   self.dropout = nn.Dropout(dropout)
        self.W_q = None      # Replace
        self.kv_proj = None  # Replace
        self.W_o = None      # Replace
        self.dropout = None  # Replace
    
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
        
        TODO: Implement MLA attention with caching
        HINT:
            batch, seq_len, _ = x.shape
            
            # Compute Q (standard projection)
            q = self.W_q(x)  # (batch, seq, d_kv)
            q = q.view(batch, seq_len, self.num_heads, self.head_dim)
            q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
            
            # Compress KV
            c, k, v = self.kv_proj(x)
            
            # Handle caching
            if kv_cache is not None:
                # Concatenate with cached latent
                c = torch.cat([kv_cache, c], dim=1)
                # Reconstruct K, V from full cached latent
                k, v = self.kv_proj.reconstruct_kv(c)
            
            # Reshape K, V for attention
            k = k.transpose(1, 2)  # (batch, heads, cache_len + seq, head_dim)
            v = v.transpose(1, 2)
            
            # Attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output = torch.matmul(attn_weights, v)
            output = output.transpose(1, 2).reshape(batch, seq_len, self.d_kv)
            output = self.W_o(output)
            
            new_cache = c if use_cache else None
            return output, attn_weights, new_cache
        """
        batch, seq_len, _ = x.shape
        return (
            torch.zeros_like(x),
            torch.zeros(batch, self.num_heads, seq_len, seq_len),
            None
        )  # Replace


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
    
    TODO: Compute reconstruction errors
    HINT:
        with torch.no_grad():
            # Get low-rank K, V
            c, k_lr, v_lr = kv_proj(x)
            k_lr = k_lr.view(x.shape[0], x.shape[1], -1)  # Flatten heads
            v_lr = v_lr.view(x.shape[0], x.shape[1], -1)
            
            # Get full-rank K, V (what standard attention would compute)
            k_fr = F.linear(x, original_W_k)
            v_fr = F.linear(x, original_W_v)
            
            # Compute errors
            k_error = torch.norm(k_lr - k_fr) / torch.norm(k_fr)
            v_error = torch.norm(v_lr - v_fr) / torch.norm(v_fr)
            
            # Cosine similarity (measures direction preservation)
            k_cos = F.cosine_similarity(k_lr.flatten(), k_fr.flatten(), dim=0)
            v_cos = F.cosine_similarity(v_lr.flatten(), v_fr.flatten(), dim=0)
            
            return {
                'k_relative_error': k_error.item(),
                'v_relative_error': v_error.item(),
                'k_cosine_similarity': k_cos.item(),
                'v_cosine_similarity': v_cos.item()
            }
    """
    return {
        'k_relative_error': 0.0,
        'v_relative_error': 0.0,
        'k_cosine_similarity': 0.0,
        'v_cosine_similarity': 0.0
    }  # Replace


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
