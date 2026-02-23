"""
Day 31: DeepSeek Math Model Assembly
====================================
Estimated time: 1-2 hours
Prerequisites: Day 30 (DeepSeek block), Day 23 (KV cache), Day 24 (MoE)

Learning objectives:
- Assemble complete DeepSeek Math model from blocks
- Implement token and position embeddings
- Build the full transformer stack with layer caching
- Create model configuration matching DeepSeek Math architecture
- Understand weight initialization strategies
- Implement weight tying between embeddings and output

Key Concepts:
-------------
DeepSeek Math Architecture:
    DeepSeek Math is based on DeepSeek-V2's architecture optimized for
    mathematical reasoning. Key features:
    
    1. Multi-head Latent Attention (MLA) for efficient KV caching
    2. SwiGLU activation in FFN layers
    3. RMSNorm for stable training
    4. Rotary Position Embeddings (RoPE)
    5. Optional MoE layers for scaling

Model Structure:
    input_ids -> Token Embedding -> [DeepSeek Blocks] -> RMSNorm -> LM Head -> logits
    
    Token Embedding: Maps token IDs to vectors
    DeepSeek Blocks: N transformer layers with MLA + FFN
    Final RMSNorm: Normalize before output projection
    LM Head: Project to vocabulary (optionally tied to embedding)

Weight Initialization:
    - Embeddings: Normal(0, 0.02)
    - Linear layers: Normal(0, 0.02 / sqrt(2 * num_layers))
    - Output projection scaled by depth

KV Cache Management:
    Each layer has its own KV cache for efficient generation.
    Cache structure: List[Tuple[k, v]] indexed by layer.

Model Sizes (Reference):
    - DeepSeek Math 7B: 30 layers, d=4096, heads=32, kv_heads=4
    - We'll implement a smaller version for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DeepSeekMathConfig:
    """Full model configuration for DeepSeek Math."""
    
    # Model dimensions
    vocab_size: int = 32000       # Vocabulary size
    d_model: int = 2048           # Hidden dimension
    num_layers: int = 24          # Number of transformer layers
    num_heads: int = 16           # Number of attention heads
    num_kv_heads: int = 4         # Number of KV heads (for GQA in MLA)
    latent_dim: int = 512         # KV compression latent dimension
    d_ff: int = 5504              # FFN intermediate dimension
    
    # Position embedding
    max_seq_len: int = 4096       # Maximum sequence length
    rope_base: float = 10000.0    # RoPE base frequency
    rope_dim: int = 64            # RoPE dimension (head_dim by default)
    
    # Regularization
    dropout: float = 0.0          # Dropout rate
    attention_dropout: float = 0.0
    
    # MoE configuration (optional)
    use_moe: bool = False         # Whether to use MoE
    moe_layer_freq: int = 2       # Use MoE every N layers
    num_experts: int = 8          # Number of experts
    top_k_experts: int = 2        # Experts activated per token
    
    # Training
    tie_word_embeddings: bool = True  # Tie input/output embeddings
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    
    # Computed properties
    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


# ============================================================================
# Import from Day 30 (or implement simplified versions)
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(self.dropout(gate * up))


# ============================================================================
# Exercise 1: Token Embedding with Scaling
# ============================================================================

class TokenEmbedding(nn.Module):
    """
    Token embedding layer with optional scaling.
    
    DeepSeek scales embeddings by sqrt(d_model) for stable training.
    """
    
    def __init__(self, vocab_size: int, d_model: int, 
                 padding_idx: Optional[int] = None,
                 scale: bool = True):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            padding_idx: Index for padding token
            scale: Whether to scale by sqrt(d_model)
        """
        super().__init__()
        
        self.d_model = d_model
        self.scale = scale
        
        # TODO: Initialize embedding layer
        # HINT:
        #   self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.embedding = None  # Replace
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens with optional scaling.
        
        Args:
            token_ids: Token indices (batch, seq_len)
        
        Returns:
            Embeddings (batch, seq_len, d_model)
        
        TODO: Implement embedding lookup with scaling
        HINT:
            x = self.embedding(token_ids)
            if self.scale:
                x = x * math.sqrt(self.d_model)
            return x
        """
        batch, seq_len = token_ids.shape
        return torch.zeros(batch, seq_len, self.d_model)  # Replace


# ============================================================================
# Exercise 2: Rotary Position Embedding
# ============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for DeepSeek."""
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # TODO: Precompute rotation frequencies
        # HINT:
        #   inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        #   self.register_buffer('inv_freq', inv_freq)
        #   
        #   t = torch.arange(max_seq_len).float()
        #   freqs = torch.outer(t, inv_freq)
        #   self.register_buffer('cos_cached', torch.cos(freqs))
        #   self.register_buffer('sin_cached', torch.sin(freqs))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def get_cos_sin(self, seq_len: int, offset: int = 0, 
                    device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin for given sequence length."""
        t = torch.arange(offset, offset + seq_len, device=device or self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor,
                     cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    
    Args:
        q: (batch, heads, seq_len, head_dim)
        k: (batch, kv_heads, seq_len, head_dim)
        cos: (seq_len, head_dim/2)
        sin: (seq_len, head_dim/2)
    
    Returns:
        Rotated q and k tensors
    """
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    
    return q_rot, k_rot


# ============================================================================
# Exercise 3: Multi-head Latent Attention
# ============================================================================

class MultiHeadLatentAttention(nn.Module):
    """MLA with KV compression and RoPE."""
    
    def __init__(self, config: DeepSeekMathConfig):
        super().__init__()
        
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.latent_dim = config.latent_dim
        self.scale = self.head_dim ** -0.5
        self.num_kv_groups = config.num_heads // config.num_kv_heads
        
        # TODO: Initialize attention components
        # HINT:
        #   self.W_q = nn.Linear(config.d_model, config.num_heads * self.head_dim, bias=False)
        #   self.W_kv_compress = nn.Linear(config.d_model, config.latent_dim, bias=False)
        #   self.W_k_expand = nn.Linear(config.latent_dim, config.num_kv_heads * self.head_dim, bias=False)
        #   self.W_v_expand = nn.Linear(config.latent_dim, config.num_kv_heads * self.head_dim, bias=False)
        #   self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)
        #   self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.W_q = None           # Replace
        self.W_kv_compress = None # Replace
        self.W_k_expand = None    # Replace
        self.W_v_expand = None    # Replace
        self.W_o = None           # Replace
        self.attn_dropout = None  # Replace
    
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query heads."""
        if self.num_kv_groups == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x.unsqueeze(2).expand(batch, num_kv_heads, self.num_kv_groups, seq_len, head_dim)
        return x.reshape(batch, self.num_heads, seq_len, head_dim)
    
    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with RoPE and optional caching.
        
        Args:
            x: Input (batch, seq_len, d_model)
            cos, sin: RoPE rotation matrices
            mask: Attention mask
            kv_cache: Optional cached (k, v)
        
        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: Updated cache
        
        TODO: Implement MLA forward
        HINT:
            batch, seq_len, _ = x.shape
            
            # Query projection
            q = self.W_q(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # KV compression and expansion
            kv_latent = self.W_kv_compress(x)
            k = self.W_k_expand(kv_latent).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.W_v_expand(kv_latent).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
            # Apply RoPE
            q, k = apply_rotary_emb(q, k, cos, sin)
            
            # Handle cache
            if kv_cache is not None:
                k_cache, v_cache = kv_cache
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)
            new_kv_cache = (k, v)
            
            # Repeat KV for GQA
            k_expanded = self._repeat_kv(k)
            v_expanded = self._repeat_kv(v)
            
            # Attention
            scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            
            output = torch.matmul(attn, v_expanded)
            output = output.transpose(1, 2).reshape(batch, seq_len, self.d_model)
            output = self.W_o(output)
            
            return output, new_kv_cache
        """
        return x, None  # Replace


# ============================================================================
# Exercise 4: DeepSeek Block
# ============================================================================

class DeepSeekBlock(nn.Module):
    """Single transformer block for DeepSeek."""
    
    def __init__(self, config: DeepSeekMathConfig, layer_idx: int = 0):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        
        # TODO: Initialize block components
        # HINT:
        #   self.attn_norm = RMSNorm(config.d_model, config.layer_norm_eps)
        #   self.attention = MultiHeadLatentAttention(config)
        #   self.ffn_norm = RMSNorm(config.d_model, config.layer_norm_eps)
        #   self.ffn = SwiGLUFFN(config.d_model, config.d_ff, config.dropout)
        self.attn_norm = None  # Replace
        self.attention = None  # Replace
        self.ffn_norm = None   # Replace
        self.ffn = None        # Replace
    
    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through block.
        
        TODO: Implement pre-norm transformer block
        HINT:
            # Attention with pre-norm
            normed = self.attn_norm(x)
            attn_out, new_kv_cache = self.attention(normed, cos, sin, mask, kv_cache)
            x = x + attn_out
            
            # FFN with pre-norm
            normed = self.ffn_norm(x)
            x = x + self.ffn(normed)
            
            return x, new_kv_cache
        """
        return x, None  # Replace


# ============================================================================
# Exercise 5: Complete DeepSeek Math Model
# ============================================================================

class DeepSeekMathModel(nn.Module):
    """
    Complete DeepSeek Math model.
    
    This is the full transformer stack with:
    - Token embeddings
    - RoPE
    - N DeepSeek blocks
    - Final layer norm
    - LM head (optionally tied to embeddings)
    """
    
    def __init__(self, config: DeepSeekMathConfig):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        
        # TODO: Initialize all model components
        # HINT:
        #   # Token embedding
        #   self.token_embedding = TokenEmbedding(
        #       config.vocab_size, config.d_model, scale=True
        #   )
        #   
        #   # Rotary embedding
        #   self.rope = RotaryEmbedding(
        #       config.head_dim, config.max_seq_len, config.rope_base
        #   )
        #   
        #   # Transformer blocks
        #   self.layers = nn.ModuleList([
        #       DeepSeekBlock(config, layer_idx=i)
        #       for i in range(config.num_layers)
        #   ])
        #   
        #   # Final normalization
        #   self.final_norm = RMSNorm(config.d_model, config.layer_norm_eps)
        #   
        #   # LM head (tied to embedding if configured)
        #   if config.tie_word_embeddings:
        #       self.lm_head = None  # Will use embedding weights
        #   else:
        #       self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        #   
        #   # Initialize weights
        #   self.apply(self._init_weights)
        self.token_embedding = None  # Replace
        self.rope = None             # Replace
        self.layers = None           # Replace
        self.final_norm = None       # Replace
        self.lm_head = None          # Replace
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights using DeepSeek's strategy."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def get_output_embeddings(self) -> nn.Module:
        """Get output projection layer."""
        if self.config.tie_word_embeddings and self.token_embedding is not None:
            return self.token_embedding.embedding
        return self.lm_head
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                start_pos: int = 0
                ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Optional attention mask
            kv_caches: Optional list of (k, v) caches per layer
            start_pos: Starting position for RoPE (for generation)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            new_kv_caches: Updated caches for each layer
        
        TODO: Implement full forward pass
        HINT:
            batch, seq_len = input_ids.shape
            
            # Get embeddings
            x = self.token_embedding(input_ids)
            
            # Get RoPE rotations
            cos, sin = self.rope.get_cos_sin(seq_len, offset=start_pos, device=x.device)
            
            # Create causal mask if not provided
            if attention_mask is None:
                total_len = start_pos + seq_len
                mask = torch.tril(torch.ones(total_len, total_len, device=x.device))
                mask = mask[start_pos:total_len, :total_len]
                mask = mask.unsqueeze(0).unsqueeze(0)
            else:
                mask = attention_mask
            
            # Initialize caches list
            if kv_caches is None:
                kv_caches = [None] * self.config.num_layers
            new_kv_caches = []
            
            # Forward through layers
            for i, layer in enumerate(self.layers):
                x, new_cache = layer(x, cos, sin, mask, kv_caches[i])
                new_kv_caches.append(new_cache)
            
            # Final norm
            x = self.final_norm(x)
            
            # Project to vocabulary
            if self.config.tie_word_embeddings:
                logits = F.linear(x, self.token_embedding.embedding.weight)
            else:
                logits = self.lm_head(x)
            
            return logits, new_kv_caches
        """
        batch, seq_len = input_ids.shape
        return torch.zeros(batch, seq_len, self.config.vocab_size), []  # Replace


# ============================================================================
# Exercise 6: Model Creation Helpers
# ============================================================================

def create_deepseek_math_7b_config() -> DeepSeekMathConfig:
    """
    Create configuration matching DeepSeek Math 7B.
    
    Note: This is approximate - actual DeepSeek Math may differ slightly.
    
    TODO: Return appropriate config
    HINT:
        return DeepSeekMathConfig(
            vocab_size=102400,    # Large vocab for math tokens
            d_model=4096,
            num_layers=30,
            num_heads=32,
            num_kv_heads=4,       # 8x KV compression
            latent_dim=512,       # Latent dimension for MLA
            d_ff=11008,           # ~2.7x d_model for SwiGLU
            max_seq_len=4096,
            tie_word_embeddings=True
        )
    """
    return DeepSeekMathConfig()  # Replace


def create_deepseek_math_small_config() -> DeepSeekMathConfig:
    """
    Create a smaller config for testing/development.
    
    TODO: Return smaller config
    HINT:
        return DeepSeekMathConfig(
            vocab_size=32000,
            d_model=512,
            num_layers=6,
            num_heads=8,
            num_kv_heads=2,
            latent_dim=128,
            d_ff=1376,
            max_seq_len=512
        )
    """
    return DeepSeekMathConfig(
        vocab_size=32000,
        d_model=512,
        num_layers=6,
        num_heads=8,
        num_kv_heads=2,
        latent_dim=128,
        d_ff=1376,
        max_seq_len=512
    )


# ============================================================================
# Exercise 7: Weight Initialization Analysis
# ============================================================================

def analyze_model_stats(model: DeepSeekMathModel) -> Dict[str, float]:
    """
    Analyze model statistics.
    
    Args:
        model: DeepSeek Math model
    
    Returns:
        Dictionary with model statistics
    
    TODO: Compute various statistics
    HINT:
        stats = {}
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        stats['total_parameters'] = total_params
        stats['trainable_parameters'] = trainable_params
        
        # Compute memory estimate (fp16)
        stats['memory_mb_fp16'] = total_params * 2 / (1024 * 1024)
        stats['memory_gb_fp16'] = stats['memory_mb_fp16'] / 1024
        
        # Weight statistics
        all_weights = torch.cat([p.flatten() for p in model.parameters()])
        stats['weight_mean'] = all_weights.mean().item()
        stats['weight_std'] = all_weights.std().item()
        stats['weight_min'] = all_weights.min().item()
        stats['weight_max'] = all_weights.max().item()
        
        return stats
    """
    return {
        'total_parameters': 0,
        'trainable_parameters': 0,
        'memory_mb_fp16': 0.0,
        'memory_gb_fp16': 0.0,
        'weight_mean': 0.0,
        'weight_std': 0.0
    }  # Replace


def count_parameters_by_component(model: DeepSeekMathModel) -> Dict[str, int]:
    """
    Count parameters broken down by component.
    
    TODO: Count parameters for each major component
    HINT:
        counts = {}
        
        if model.token_embedding is not None:
            counts['embedding'] = sum(p.numel() for p in model.token_embedding.parameters())
        
        if model.layers is not None:
            attention_params = 0
            ffn_params = 0
            norm_params = 0
            
            for layer in model.layers:
                if layer.attention is not None:
                    attention_params += sum(p.numel() for p in layer.attention.parameters())
                if layer.ffn is not None:
                    ffn_params += sum(p.numel() for p in layer.ffn.parameters())
                if layer.attn_norm is not None:
                    norm_params += sum(p.numel() for p in layer.attn_norm.parameters())
                if layer.ffn_norm is not None:
                    norm_params += sum(p.numel() for p in layer.ffn_norm.parameters())
            
            counts['attention'] = attention_params
            counts['ffn'] = ffn_params
            counts['norm'] = norm_params
        
        if model.lm_head is not None:
            counts['lm_head'] = sum(p.numel() for p in model.lm_head.parameters())
        
        return counts
    """
    return {}  # Replace


# ============================================================================
# Verification
# ============================================================================

def verify_model_output(model: DeepSeekMathModel, config: DeepSeekMathConfig):
    """Verify model produces correct output shapes."""
    batch, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    
    logits, caches = model(input_ids)
    
    expected_logits = (batch, seq_len, config.vocab_size)
    assert logits.shape == expected_logits, f"Logits shape {logits.shape} != {expected_logits}"
    
    assert len(caches) == config.num_layers, f"Cache count {len(caches)} != {config.num_layers}"
    
    print(f"Model verification passed!")
    print(f"  Input: {input_ids.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Caches: {len(caches)} layers")


if __name__ == "__main__":
    print("Day 31: DeepSeek Math Model Assembly")
    print("=" * 60)
    
    # Create small test config
    config = create_deepseek_math_small_config()
    
    print(f"\nTest Configuration:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  num_kv_heads: {config.num_kv_heads}")
    print(f"  latent_dim: {config.latent_dim}")
    
    # Test token embedding
    print("\nTesting Token Embedding:")
    embed = TokenEmbedding(config.vocab_size, config.d_model)
    if embed.embedding is not None:
        ids = torch.randint(0, config.vocab_size, (2, 16))
        emb = embed(ids)
        print(f"  Input: {ids.shape} -> Output: {emb.shape}")
    
    # Test RoPE
    print("\nTesting Rotary Embedding:")
    rope = RotaryEmbedding(config.head_dim, config.max_seq_len)
    cos, sin = rope.get_cos_sin(16)
    print(f"  cos: {cos.shape}, sin: {sin.shape}")
    
    # Test full model
    print("\nTesting DeepSeek Math Model:")
    model = DeepSeekMathModel(config)
    if model.layers is not None:
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        logits, caches = model(input_ids)
        print(f"  Input: {input_ids.shape}")
        print(f"  Logits: {logits.shape}")
        print(f"  Caches: {len(caches)} layers")
        
        stats = analyze_model_stats(model)
        print(f"\nModel Statistics:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v:,}")
    
    print("\nRun test_day31.py to verify your implementations!")
