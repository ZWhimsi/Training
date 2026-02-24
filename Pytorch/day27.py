"""
Day 27: MLA Key-Value Compression
=================================
Estimated time: 1-2 hours
Prerequisites: Day 26 (MLA basics)

Learning objectives:
- Implement proper KV compression matrices with RMSNorm
- Understand decoupled RoPE for compressed KV
- Build the latent projection with proper initialization
- Implement query compression (optional in MLA)
- Handle the rope_head for positional information

Key Concepts:
-------------
DeepSeek-V2 MLA Architecture:
    The full MLA has several sophisticated components:
    
    1. KV Compression:
       c_kv = RMSNorm(x @ W_down_kv)   # Compress to latent
       k_content = c_kv @ W_up_k       # Content-based key
       v = c_kv @ W_up_v               # Value
       
    2. Decoupled RoPE:
       Problem: Standard RoPE applied to compressed K breaks low-rank structure
       Solution: Use separate "rope head" that bypasses compression
       
       k_rope = x @ W_k_rope           # Small projection for positional info
       k_rope = apply_rope(k_rope)     # Apply positional encoding
       
       Final K = concat(k_content, k_rope) or k_content + positional_bias
    
    3. Query Compression (optional):
       c_q = x @ W_down_q              # Query can also be compressed
       q = c_q @ W_up_q                # Reconstruct query
       
       DeepSeek-V2 compresses Q for efficiency, but it's optional

The Decoupled RoPE Problem:
    Standard RoPE: K_rotated = RoPE(W_k @ x)
    
    With compression: K = W_up @ (W_down @ x) = W_up @ c
    If we apply RoPE to K: K_rotated = RoPE(W_up @ c)
    
    Problem: RoPE is position-dependent, so we can't cache c and apply RoPE later!
    
    Solution: Decouple positional and content information:
    - k_content = c @ W_up (cacheable, no position info)
    - k_rope = apply_rope(x @ W_rope) (position info, small dim)
    - Final attention uses both

RMSNorm in MLA:
    RMSNorm(x) = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2))
    
    Applied after down projection for stable training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ============================================================================
# Exercise 1: RMSNorm Implementation
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
    
    Simpler than LayerNorm (no mean subtraction), often works just as well.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: Dimension to normalize over
            eps: Small constant for numerical stability
        """
        super().__init__()
        
        self.eps = eps
        
        # TODO: Create learnable scale parameter
        # API hints:
        # - nn.Parameter(torch.ones(dim)) -> learnable parameter initialized to 1s
        
        self.weight = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.
        
        Args:
            x: Input tensor (..., dim)
        
        Returns:
            Normalized tensor of same shape
        """
        # API hints:
        # - x.pow(2).mean(dim=-1, keepdim=True) -> mean of squared values
        # - torch.sqrt(tensor + eps) -> square root with stability
        # - x / rms * self.weight -> normalize and scale
        
        return None


# ============================================================================
# Exercise 2: KV Down Projection with Norm
# ============================================================================

class KVDownProjection(nn.Module):
    """
    KV compression layer: projects input to latent space with normalization.
    
    c_kv = RMSNorm(x @ W_down_kv)
    """
    
    def __init__(self, d_model: int, d_latent: int, eps: float = 1e-6):
        """
        Args:
            d_model: Input dimension
            d_latent: Compressed latent dimension
            eps: RMSNorm epsilon
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_latent = d_latent
        
        # TODO: Create down projection and normalization
        # API hints:
        # - nn.Linear(d_model, d_latent, bias=False) -> down projection
        # - RMSNorm(d_latent, eps) -> normalization layer
        
        self.down_proj = None
        self.norm = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress input to normalized latent.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Compressed, normalized latent (batch, seq_len, d_latent)
        """
        # API hints:
        # - self.down_proj(x) -> apply projection
        # - self.norm(c) -> apply RMSNorm
        
        return None


# ============================================================================
# Exercise 3: Rotary Position Embedding (RoPE)
# ============================================================================

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute rotary embedding frequencies.
    
    Args:
        dim: Dimension (must be even)
        max_seq_len: Maximum sequence length
        theta: Base for frequency computation
    
    Returns:
        Complex frequencies tensor (max_seq_len, dim // 2)
    """
    # API hints:
    # - freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # - torch.arange(max_seq_len) -> position indices
    # - torch.outer(t, freqs) -> outer product (seq_len, dim//2)
    # - torch.polar(magnitude, angle) -> complex from polar form
    
    return None


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor.
    
    Args:
        x: Input tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, dim)
        freqs_cis: Precomputed frequencies (seq_len, head_dim // 2)
    
    Returns:
        Tensor with rotary embeddings applied
    """
    # API hints:
    # - x.view(*x_shape[:-1], -1, 2) -> reshape to pairs for complex
    # - torch.view_as_complex(tensor.float()) -> convert to complex
    # - freqs_cis[:seq_len] -> get frequencies for sequence length
    # - freqs.unsqueeze(0) -> add batch dimensions for broadcasting
    # - x_complex * freqs -> apply rotation via complex multiplication
    # - torch.view_as_real(complex_tensor) -> convert back to real
    # - x_out.type_as(x) -> match original dtype
    
    return None


# ============================================================================
# Exercise 4: Decoupled RoPE Key Projection
# ============================================================================

class DecoupledRoPEKey(nn.Module):
    """
    Decoupled RoPE for keys: separate content and positional projections.
    
    k_content = c_kv @ W_up_k   (from compressed latent, no RoPE)
    k_rope = x @ W_k_rope       (small projection with RoPE)
    """
    
    def __init__(self, d_model: int, d_latent: int, num_heads: int, 
                 head_dim: int, rope_dim: int, max_seq_len: int = 4096):
        """
        Args:
            d_model: Model dimension
            d_latent: Compressed latent dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head (for content)
            rope_dim: Dimension for RoPE component (typically small, e.g., 64)
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_latent = d_latent
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        
        # TODO: Create projections for content and RoPE
        # API hints:
        # - nn.Linear(d_latent, num_heads * head_dim, bias=False) -> content key projection
        # - nn.Linear(d_model, num_heads * rope_dim, bias=False) -> RoPE key projection
        # - self.register_buffer('freqs_cis', precompute_freqs_cis(rope_dim, max_seq_len))
        
        self.up_proj_k = None
        self.rope_proj = None
    
    def forward(self, x: torch.Tensor, c_kv: torch.Tensor, 
                start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute decoupled key components.
        
        Args:
            x: Original input (batch, seq_len, d_model) - for RoPE
            c_kv: Compressed latent (batch, seq_len, d_latent) - for content
            start_pos: Starting position (for generation)
        
        Returns:
            k_content: Content-based key (batch, seq_len, num_heads, head_dim)
            k_rope: Position-encoded key (batch, seq_len, num_heads, rope_dim)
        """
        # API hints:
        # - self.up_proj_k(c_kv) -> content key from latent
        # - self.rope_proj(x) -> RoPE key from input
        # - tensor.view(batch, seq_len, num_heads, dim) -> reshape
        # - self.freqs_cis[start_pos:start_pos + seq_len] -> get frequencies
        # - apply_rotary_emb(k_rope, freqs) -> apply RoPE
        
        return None


# ============================================================================
# Exercise 5: Full MLA KV Compression Module
# ============================================================================

class MLAKVCompression(nn.Module):
    """
    Complete MLA KV compression with decoupled RoPE.
    
    Produces:
    - Compressed latent c_kv (for caching)
    - Content-based K (from c_kv)
    - Position-encoded K component (from original input)
    - Value V (from c_kv)
    """
    
    def __init__(self, d_model: int, d_latent: int, num_heads: int,
                 head_dim: int, rope_dim: int = 64, max_seq_len: int = 4096):
        """
        Args:
            d_model: Model dimension
            d_latent: Compressed KV latent dimension
            num_heads: Number of attention heads
            head_dim: Content key/value dimension per head
            rope_dim: RoPE dimension per head
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_latent = d_latent
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        
        # Effective key dimension includes both content and rope
        self.total_key_dim = head_dim + rope_dim
        
        # TODO: Create all projection layers
        # API hints:
        # - KVDownProjection(d_model, d_latent) -> compression with norm
        # - nn.Linear(d_latent, num_heads * head_dim, bias=False) -> K, V up projections
        # - nn.Linear(d_model, num_heads * rope_dim, bias=False) -> RoPE projection
        # - self.register_buffer('freqs_cis', precompute_freqs_cis(rope_dim, max_seq_len))
        
        self.kv_down = None
        self.up_proj_k = None
        self.up_proj_v = None
        self.rope_proj = None
    
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress input to latent (for caching during generation).
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Compressed latent (batch, seq_len, d_latent)
        """
        # API hints:
        # - self.kv_down(x) -> compress and normalize
        
        return None
    
    def get_rope_k(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Get RoPE key component (needs original input, not cached).
        
        Args:
            x: Original input (batch, seq_len, d_model)
            start_pos: Starting position for RoPE
        
        Returns:
            k_rope: (batch, seq_len, num_heads, rope_dim)
        """
        # API hints:
        # - self.rope_proj(x) -> project to RoPE dimension
        # - tensor.view(batch, seq_len, num_heads, rope_dim) -> reshape
        # - self.freqs_cis[start_pos:start_pos + seq_len] -> get frequencies
        # - apply_rotary_emb(k_rope, freqs) -> apply rotation
        
        return None
    
    def reconstruct_kv(self, c_kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct content K and V from cached latent.
        
        Args:
            c_kv: Compressed latent (batch, seq_len, d_latent)
        
        Returns:
            k_content: (batch, seq_len, num_heads, head_dim)
            v: (batch, seq_len, num_heads, head_dim)
        """
        # API hints:
        # - self.up_proj_k(c_kv) -> reconstruct K
        # - self.up_proj_v(c_kv) -> reconstruct V
        # - tensor.view(batch, seq_len, num_heads, head_dim) -> reshape
        
        return None
    
    def forward(self, x: torch.Tensor, start_pos: int = 0
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            start_pos: Starting position for RoPE
        
        Returns:
            c_kv: Compressed latent (for caching)
            k_content: Content-based key
            k_rope: Position-encoded key component
            v: Value
        """
        # API hints:
        # - self.compress(x) -> get compressed latent
        # - self.reconstruct_kv(c_kv) -> get content K and V
        # - self.get_rope_k(x, start_pos) -> get RoPE key
        
        return None


# ============================================================================
# Exercise 6: Query Compression (Optional in MLA)
# ============================================================================

class MLAQueryCompression(nn.Module):
    """
    Optional query compression for MLA.
    
    DeepSeek-V2 also compresses Q for additional efficiency.
    Q compression uses similar low-rank structure but with different dimensions.
    """
    
    def __init__(self, d_model: int, d_q_latent: int, num_heads: int,
                 head_dim: int, rope_dim: int = 64, max_seq_len: int = 4096):
        """
        Args:
            d_model: Model dimension
            d_q_latent: Query latent dimension
            num_heads: Number of attention heads
            head_dim: Query content dimension per head
            rope_dim: RoPE dimension per head (should match KV)
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_q_latent = d_q_latent
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        
        # TODO: Create query projection layers
        # API hints:
        # - nn.Linear(d_model, d_q_latent, bias=False) -> down projection
        # - RMSNorm(d_q_latent) -> normalization
        # - nn.Linear(d_q_latent, num_heads * head_dim, bias=False) -> up projection
        # - nn.Linear(d_model, num_heads * rope_dim, bias=False) -> RoPE projection
        # - self.register_buffer('freqs_cis', precompute_freqs_cis(rope_dim, max_seq_len))
        
        self.q_down = None
        self.q_norm = None
        self.q_up = None
        self.q_rope_proj = None
    
    def forward(self, x: torch.Tensor, start_pos: int = 0
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute compressed query with RoPE.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            start_pos: Starting position for RoPE
        
        Returns:
            q_content: Content query (batch, seq_len, num_heads, head_dim)
            q_rope: RoPE query (batch, seq_len, num_heads, rope_dim)
        """
        # API hints:
        # - self.q_down(x) -> compress query
        # - self.q_norm(c_q) -> normalize
        # - self.q_up(c_q) -> expand to full dimension
        # - self.q_rope_proj(x) -> RoPE query projection
        # - apply_rotary_emb(q_rope, freqs) -> apply RoPE
        
        return None


# ============================================================================
# Exercise 7: Combining Content and RoPE for Attention
# ============================================================================

def compute_mla_attention_scores(
    q_content: torch.Tensor,
    q_rope: torch.Tensor,
    k_content: torch.Tensor,
    k_rope: torch.Tensor,
    scale: float = None
) -> torch.Tensor:
    """
    Compute attention scores with decoupled content and RoPE.
    
    The final attention score is the sum of content and position scores:
    score = q_content @ k_content.T + q_rope @ k_rope.T
    
    This allows caching k_content while applying RoPE dynamically.
    
    Args:
        q_content: Query content (batch, num_heads, seq_q, head_dim)
        q_rope: Query RoPE (batch, num_heads, seq_q, rope_dim)
        k_content: Key content (batch, num_heads, seq_k, head_dim)
        k_rope: Key RoPE (batch, num_heads, seq_k, rope_dim)
        scale: Attention scale factor
    
    Returns:
        Attention scores (batch, num_heads, seq_q, seq_k)
    """
    # API hints:
    # - total_dim = q_content.shape[-1] + q_rope.shape[-1] -> total key dim
    # - scale = total_dim ** -0.5 -> attention scale
    # - torch.matmul(q_content, k_content.transpose(-2, -1)) -> content scores
    # - torch.matmul(q_rope, k_rope.transpose(-2, -1)) -> position scores
    # - (content_scores + rope_scores) * scale -> combined scores
    
    return None


if __name__ == "__main__":
    print("Day 27: MLA Key-Value Compression")
    print("=" * 60)
    
    # Configuration
    d_model = 2048
    d_latent = 512
    num_heads = 16
    head_dim = 128
    rope_dim = 64
    batch_size = 2
    seq_len = 16
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  d_latent: {d_latent}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  rope_dim: {rope_dim}")
    
    # Test RMSNorm
    print("\n" + "=" * 60)
    print("Testing RMSNorm")
    print("=" * 60)
    
    norm = RMSNorm(d_model)
    if norm.weight is not None:
        x = torch.randn(batch_size, seq_len, d_model)
        out = norm(x)
        print(f"Input mean sq: {x.pow(2).mean().item():.4f}")
        print(f"Output mean sq: {out.pow(2).mean().item():.4f}")
    
    # Test KV Down Projection
    print("\n" + "=" * 60)
    print("Testing KVDownProjection")
    print("=" * 60)
    
    kv_down = KVDownProjection(d_model, d_latent)
    if kv_down.down_proj is not None:
        x = torch.randn(batch_size, seq_len, d_model)
        c = kv_down(x)
        print(f"Input: {x.shape}")
        print(f"Compressed: {c.shape}")
    
    # Test RoPE
    print("\n" + "=" * 60)
    print("Testing Rotary Position Embedding")
    print("=" * 60)
    
    freqs = precompute_freqs_cis(rope_dim, 128)
    if freqs.abs().sum() > 0:
        print(f"Frequencies shape: {freqs.shape}")
        x = torch.randn(batch_size, seq_len, num_heads, rope_dim)
        x_rot = apply_rotary_emb(x, freqs[:seq_len])
        print(f"Input: {x.shape}")
        print(f"Rotated: {x_rot.shape}")
    
    # Test full MLA KV Compression
    print("\n" + "=" * 60)
    print("Testing MLAKVCompression")
    print("=" * 60)
    
    mla_kv = MLAKVCompression(d_model, d_latent, num_heads, head_dim, rope_dim)
    if mla_kv.kv_down is not None:
        x = torch.randn(batch_size, seq_len, d_model)
        c_kv, k_content, k_rope, v = mla_kv(x)
        print(f"Input: {x.shape}")
        print(f"Cached latent: {c_kv.shape}")
        print(f"K content: {k_content.shape}")
        print(f"K rope: {k_rope.shape}")
        print(f"V: {v.shape}")
    
    # Test attention score computation
    print("\n" + "=" * 60)
    print("Testing MLA Attention Scores")
    print("=" * 60)
    
    q_content = torch.randn(batch_size, num_heads, seq_len, head_dim)
    q_rope = torch.randn(batch_size, num_heads, seq_len, rope_dim)
    k_content = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k_rope = torch.randn(batch_size, num_heads, seq_len, rope_dim)
    
    scores = compute_mla_attention_scores(q_content, q_rope, k_content, k_rope)
    if scores.abs().sum() > 0:
        print(f"Attention scores: {scores.shape}")
    
    print("\nRun test_day27.py to verify your implementations!")
