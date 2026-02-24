"""
Day 30: DeepSeek Block - Complete Transformer Block with MLA
============================================================
Estimated time: 1-2 hours
Prerequisites: Day 27-28 (MLA components), Day 24 (MoE), Day 17 (transformer blocks)

Learning objectives:
- Combine Multi-head Latent Attention with Feed-Forward Networks
- Implement the complete DeepSeek transformer block architecture
- Understand pre-norm vs post-norm in modern transformers
- Build residual connections with proper scaling
- Integrate optional MoE layers for DeepSeek-V2 style blocks

Key Concepts:
-------------
DeepSeek Block Architecture:
    The DeepSeek block follows the standard transformer pattern but uses
    Multi-head Latent Attention (MLA) instead of standard attention:
    
    1. Pre-LayerNorm on input
    2. Multi-head Latent Attention (compressed KV)
    3. Residual connection
    4. Pre-LayerNorm
    5. Feed-Forward Network (or MoE for some layers)
    6. Residual connection

Multi-head Latent Attention (MLA) Recap:
    - Compresses K and V into low-rank latent representations
    - c = W_c @ x  (compress to latent dim)
    - K = W_uk @ c (expand K from latent)
    - V = W_uv @ c (expand V from latent)
    - Reduces KV cache size significantly during inference

RMSNorm vs LayerNorm:
    DeepSeek uses RMSNorm (Root Mean Square Layer Normalization):
    - Simpler: no mean subtraction, just scale by RMS
    - Faster: fewer operations
    - RMSNorm(x) = x / sqrt(mean(x²) + eps) * gamma

SwiGLU Activation:
    DeepSeek uses SwiGLU in the FFN:
    - SwiGLU(x, W, V) = Swish(xW) ⊗ (xV)
    - Swish(x) = x * sigmoid(x)
    - More expressive than standard GELU

Block Variations:
    - Standard block: MLA + FFN
    - MoE block: MLA + Sparse MoE (used in some layers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DeepSeekBlockConfig:
    """Configuration for DeepSeek transformer block."""
    d_model: int = 2048           # Model dimension
    num_heads: int = 16           # Number of attention heads
    num_kv_heads: int = 4         # Number of KV heads (for GQA within MLA)
    latent_dim: int = 512         # Latent dimension for KV compression
    rope_dim: int = 64            # Dimension for RoPE (decoupled)
    d_ff: int = 5504              # FFN intermediate dimension
    dropout: float = 0.0          # Dropout rate
    use_moe: bool = False         # Whether to use MoE instead of FFN
    num_experts: int = 8          # Number of experts if using MoE
    top_k_experts: int = 2        # Top-k experts to activate
    layer_norm_eps: float = 1e-6  # LayerNorm epsilon


# ============================================================================
# Exercise 1: RMSNorm
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Simpler and faster than LayerNorm - no mean centering.
    Used in LLaMA, DeepSeek, and many modern transformers.
    
    Formula: x_norm = x / sqrt(mean(x²) + eps) * gamma
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: Dimension to normalize over
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        
        # TODO: Initialize learnable scale parameter
        # API hints:
        # - nn.Parameter(torch.ones(dim)) -> learnable scale parameter
        
        self.weight = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor (..., dim)
        
        Returns:
            Normalized tensor of same shape
        """
        # API hints:
        # - torch.mean(x ** 2, dim=-1, keepdim=True) -> mean of squared values
        # - torch.sqrt(tensor + eps) -> RMS value
        # - x / rms * self.weight -> normalize and scale
        
        return None


# ============================================================================
# Exercise 2: SwiGLU Feed-Forward Network
# ============================================================================

class SwiGLUFFN(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation.
    
    SwiGLU combines the benefits of Swish activation with gating:
    - Gate: swish(x @ W_gate)
    - Up: x @ W_up  
    - Output: (gate * up) @ W_down
    
    This is more expressive than standard FFN with GELU.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        """
        Args:
            d_model: Input/output dimension
            d_ff: Intermediate (hidden) dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # TODO: Initialize SwiGLU layers
        # API hints:
        # - nn.Linear(d_model, d_ff, bias=False) -> w_gate, w_up
        # - nn.Linear(d_ff, d_model, bias=False) -> w_down
        # - nn.Dropout(dropout) -> dropout layer
        
        self.w_gate = None
        self.w_up = None
        self.w_down = None
        self.dropout = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # API hints:
        # - F.silu(self.w_gate(x)) -> swish activation (x * sigmoid(x))
        # - self.w_up(x) -> up projection
        # - gate * up -> element-wise gating
        # - self.w_down(self.dropout(hidden)) -> output
        
        return None


# ============================================================================
# Exercise 3: Multi-head Latent Attention (MLA)
# ============================================================================

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention as used in DeepSeek-V2.
    
    Key innovation: Compress K and V into a shared low-rank latent space.
    This dramatically reduces KV cache memory during inference.
    
    Architecture:
        1. Q projection: standard or with decoupled RoPE
        2. KV compression: x -> latent (low rank)
        3. KV expansion: latent -> K, V (during attention)
        4. Attention computation with optional RoPE
        5. Output projection
    """
    
    def __init__(self, config: DeepSeekBlockConfig):
        super().__init__()
        
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.d_model // config.num_heads
        self.latent_dim = config.latent_dim
        self.rope_dim = config.rope_dim
        self.scale = self.head_dim ** -0.5
        
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        # TODO: Initialize MLA projections
        # API hints:
        # - nn.Linear(d_model, num_heads * head_dim, bias=False) -> W_q
        # - nn.Linear(d_model, latent_dim, bias=False) -> W_kv_compress
        # - nn.Linear(latent_dim, num_kv_heads * head_dim, bias=False) -> W_k_expand, W_v_expand
        # - nn.Linear(d_model, d_model, bias=False) -> W_o
        # - nn.Dropout(dropout) -> dropout
        
        self.W_q = None
        self.W_kv_compress = None
        self.W_k_expand = None
        self.W_v_expand = None
        self.W_o = None
        self.dropout = None
    
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match number of query heads."""
        if self.num_kv_groups == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x.unsqueeze(2).expand(batch, num_kv_heads, self.num_kv_groups, seq_len, head_dim)
        return x.reshape(batch, self.num_heads, seq_len, head_dim)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV caching.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            kv_cache: Optional (k_cache, v_cache) from previous steps
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, kv_len)
            new_kv_cache: Updated cache tuple
        """
        # API hints:
        # - self.W_q(x).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        # - self.W_kv_compress(x) -> latent representation
        # - self.W_k_expand(latent), self.W_v_expand(latent) -> K, V
        # - torch.cat([cache, new], dim=2) -> append to cache
        # - self._repeat_kv(k) -> repeat for GQA
        # - torch.matmul(q, k.transpose(-2, -1)) * self.scale -> attention scores
        # - scores.masked_fill(mask == 0, float('-inf')) -> apply mask
        # - F.softmax(scores, dim=-1) -> attention weights
        
        return None


# ============================================================================
# Exercise 4: DeepSeek Transformer Block
# ============================================================================

class DeepSeekBlock(nn.Module):
    """
    Complete DeepSeek transformer block.
    
    Architecture:
        x -> RMSNorm -> MLA -> residual -> RMSNorm -> FFN/MoE -> residual
    
    Uses pre-norm (normalize before each sublayer) which is more stable
    for deep networks and is standard in modern transformers.
    """
    
    def __init__(self, config: DeepSeekBlockConfig, layer_idx: int = 0):
        """
        Args:
            config: Block configuration
            layer_idx: Layer index (for logging/debugging)
        """
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        
        # TODO: Initialize block components
        # API hints:
        # - RMSNorm(d_model, eps) -> attn_norm, ffn_norm
        # - MultiHeadLatentAttention(config) -> attention
        # - SwiGLUFFN(d_model, d_ff, dropout) -> ffn
        # - nn.Dropout(dropout) -> dropout
        
        self.attn_norm = None
        self.attention = None
        self.ffn_norm = None
        self.ffn = None
        self.dropout = None
    
    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the block.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            kv_cache: Optional KV cache for inference
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, kv_len)
            new_kv_cache: Updated KV cache
        """
        # API hints:
        # - self.attn_norm(x) -> pre-norm
        # - self.attention(normed, mask, kv_cache) -> attention + cache
        # - x = x + self.dropout(attn_out) -> residual
        # - self.ffn_norm(x) -> pre-norm for FFN
        # - x = x + self.dropout(self.ffn(normed)) -> FFN with residual
        
        return None


# ============================================================================
# Exercise 5: Rotary Position Embedding (RoPE)
# ============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding for DeepSeek.
    
    RoPE encodes position by rotating the query and key vectors.
    Benefits:
    - Relative position encoding
    - Extrapolates to longer sequences
    - No learnable parameters
    """
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        """
        Args:
            dim: Dimension of embedding (must be even)
            max_seq_len: Maximum sequence length
            base: Base for computing frequencies
        """
        super().__init__()
        
        assert dim % 2 == 0, "RoPE dimension must be even"
        
        # TODO: Precompute rotation frequencies
        # API hints:
        # - inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # - self.register_buffer('inv_freq', inv_freq)
        # - t = torch.arange(max_seq_len).float()
        # - freqs = torch.outer(t, inv_freq) -> (max_seq_len, dim/2)
        # - self.register_buffer('cos_cached', torch.cos(freqs))
        # - self.register_buffer('sin_cached', torch.sin(freqs))
        
        self.dim = dim
    
    def forward(self, x: torch.Tensor, seq_len: int, 
                offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get rotation matrices for given sequence length.
        
        Args:
            x: Input tensor (for device/dtype)
            seq_len: Current sequence length
            offset: Position offset (for KV cache)
        
        Returns:
            cos, sin tensors of shape (seq_len, dim/2)
        """
        # API hints:
        # - self.cos_cached[offset:offset + seq_len] -> sliced cos
        # - self.sin_cached[offset:offset + seq_len] -> sliced sin
        
        return None


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                         cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor (batch, num_heads, seq_len, head_dim)
        k: Key tensor (batch, num_kv_heads, seq_len, head_dim)
        cos: Cosine rotation (seq_len, head_dim/2)
        sin: Sine rotation (seq_len, head_dim/2)
    
    Returns:
        Rotated q, k tensors
    """
    # API hints:
    # - rotate_half(x): x1, x2 = x[..., :half], x[..., half:]; return torch.cat([-x2, x1], dim=-1)
    # - cos.unsqueeze(0).unsqueeze(0) -> (1, 1, seq_len, dim/2)
    # - torch.cat([cos, cos], dim=-1) -> duplicate for full head_dim
    # - q_rot = q * cos + rotate_half(q) * sin -> apply rotation
    # - k_rot = k * cos + rotate_half(k) * sin -> apply rotation
    
    return None


# ============================================================================
# Exercise 6: DeepSeek Block with RoPE
# ============================================================================

class DeepSeekBlockWithRoPE(nn.Module):
    """
    DeepSeek block with integrated Rotary Position Embedding.
    
    This is the complete block as used in production DeepSeek models.
    """
    
    def __init__(self, config: DeepSeekBlockConfig, layer_idx: int = 0):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.d_model // config.num_heads
        
        # TODO: Initialize all components including RoPE
        # API hints:
        # - RMSNorm(d_model, eps) -> attn_norm, ffn_norm
        # - MultiHeadLatentAttention(config) -> attention
        # - SwiGLUFFN(d_model, d_ff, dropout) -> ffn
        # - RotaryEmbedding(head_dim) -> rope
        # - nn.Dropout(dropout) -> dropout
        
        self.attn_norm = None
        self.attention = None
        self.ffn_norm = None
        self.ffn = None
        self.rope = None
        self.dropout = None
    
    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                start_pos: int = 0,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward with RoPE integration.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            start_pos: Starting position (for RoPE with KV cache)
            kv_cache: Optional KV cache
        
        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: Updated KV cache
        """
        # API hints:
        # - Similar to DeepSeekBlock but with RoPE
        # - self.rope(x, seq_len, start_pos) -> cos, sin
        # - Pass cos/sin to attention along with x, mask, kv_cache
        
        return None


# ============================================================================
# Exercise 7: Causal Mask Creation
# ============================================================================

def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create causal attention mask.
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
    
    Returns:
        Causal mask of shape (1, 1, seq_len, seq_len)
        1 = attend, 0 = mask
    """
    # API hints:
    # - torch.tril(torch.ones(seq_len, seq_len, device=device)) -> lower triangular
    # - mask.unsqueeze(0).unsqueeze(0) -> add batch and head dims
    
    return None


def create_causal_mask_with_cache(q_len: int, kv_len: int, 
                                   device: torch.device = None) -> torch.Tensor:
    """
    Create causal mask for cached inference.
    
    During generation, q_len=1 (new token) but kv_len=total_len (all cached).
    
    Args:
        q_len: Query length (usually 1 during generation)
        kv_len: Key/Value length (total sequence so far)
        device: Device
    
    Returns:
        Mask of shape (1, 1, q_len, kv_len)
    """
    # API hints:
    # - torch.ones(q_len, kv_len, device=device) -> new tokens attend to all prev
    # - mask.unsqueeze(0).unsqueeze(0) -> add batch and head dims
    
    return None


# ============================================================================
# Verification and Testing
# ============================================================================

def count_parameters(module: nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def verify_block_dimensions(config: DeepSeekBlockConfig):
    """Verify block handles dimensions correctly."""
    block = DeepSeekBlock(config)
    
    batch, seq_len = 2, 16
    x = torch.randn(batch, seq_len, config.d_model)
    
    output, attn_weights, _ = block(x)
    
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape}"
    print(f"Block dimensions verified: {x.shape} -> {output.shape}")
    print(f"Total parameters: {count_parameters(block):,}")


if __name__ == "__main__":
    print("Day 30: DeepSeek Block - Complete Transformer Block with MLA")
    print("=" * 60)
    
    # Test configuration
    config = DeepSeekBlockConfig(
        d_model=256,
        num_heads=8,
        num_kv_heads=2,
        latent_dim=64,
        d_ff=688,
        dropout=0.0
    )
    
    print(f"\nConfiguration:")
    print(f"  d_model: {config.d_model}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  num_kv_heads: {config.num_kv_heads}")
    print(f"  latent_dim: {config.latent_dim}")
    print(f"  d_ff: {config.d_ff}")
    
    # Test RMSNorm
    print("\nTesting RMSNorm:")
    rms_norm = RMSNorm(config.d_model)
    if rms_norm.weight is not None:
        x = torch.randn(2, 16, config.d_model)
        out = rms_norm(x)
        print(f"  Input: {x.shape} -> Output: {out.shape}")
        print(f"  Output variance: {out.var(dim=-1).mean():.4f}")
    
    # Test SwiGLU FFN
    print("\nTesting SwiGLU FFN:")
    ffn = SwiGLUFFN(config.d_model, config.d_ff)
    if ffn.w_gate is not None:
        x = torch.randn(2, 16, config.d_model)
        out = ffn(x)
        print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    # Test MLA
    print("\nTesting Multi-head Latent Attention:")
    mla = MultiHeadLatentAttention(config)
    if mla.W_q is not None:
        x = torch.randn(2, 16, config.d_model)
        mask = create_causal_mask(16)
        out, attn, _ = mla(x, mask)
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")
        print(f"  Attention: {attn.shape}")
    
    # Test complete block
    print("\nTesting DeepSeek Block:")
    block = DeepSeekBlock(config)
    if block.attention is not None:
        x = torch.randn(2, 16, config.d_model)
        mask = create_causal_mask(16)
        out, attn, _ = block(x, mask)
        print(f"  Input: {x.shape} -> Output: {out.shape}")
        print(f"  Parameters: {count_parameters(block):,}")
    
    # Test RoPE
    print("\nTesting Rotary Position Embedding:")
    rope = RotaryEmbedding(config.d_model // config.num_heads)
    cos, sin = rope(torch.tensor([0]), seq_len=16)
    print(f"  cos shape: {cos.shape}")
    print(f"  sin shape: {sin.shape}")
    
    print("\nRun test_day30.py to verify your implementations!")
