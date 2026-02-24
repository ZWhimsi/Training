"""
Day 29: DeepSeek Architecture Components
========================================
Estimated time: 1-2 hours
Prerequisites: Day 28 (MLA full implementation), Day 24 (MoE basics)

Learning objectives:
- Implement SwiGLU activation function
- Build the DeepSeek-style FFN (SwiGLU FFN)
- Understand auxiliary-loss-free load balancing
- Implement shared expert routing
- Build a complete DeepSeek-style transformer block
- Combine MLA with MoE for the full architecture

Key Concepts:
-------------
SwiGLU Activation:
    DeepSeek uses SwiGLU (Swish-Gated Linear Unit) instead of standard GELU.
    
    Standard FFN:
        FFN(x) = GELU(x @ W1) @ W2
    
    SwiGLU FFN:
        FFN(x) = (Swish(x @ W_gate) * (x @ W_up)) @ W_down
        
        where Swish(x) = x * sigmoid(x)
    
    The gating mechanism allows the network to learn which dimensions to activate.
    Empirically leads to better performance, especially in larger models.

DeepSeek-V2 MoE Innovations:
    1. Shared Experts:
       - Some experts are always activated for all tokens (shared)
       - Remaining experts are routed (sparse)
       - Output = shared_output + routed_output
    
    2. Auxiliary-Loss-Free Load Balancing:
       - Instead of auxiliary loss, use device-limited routing
       - Each device only routes to a subset of experts
       - Natural load balancing without extra loss term
    
    3. Expert Parallelism:
       - Experts distributed across devices
       - Efficient all-to-all communication pattern

DeepSeek-V2 Architecture Summary:
    - MLA for attention (compressed KV cache)
    - MoE with shared experts for FFN
    - SwiGLU activation in all FFN components
    - RMSNorm for all normalization
    - Pre-norm transformer structure

Parameter Count:
    DeepSeek-V2 236B has:
    - 21B activated parameters per token
    - Thanks to MoE sparsity, only ~21B params used per forward pass
    - Full parameter count much larger for model capacity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


# Try to import from previous days
try:
    from day27 import RMSNorm
except ImportError:
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        
        def forward(self, x):
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x / rms * self.weight


# ============================================================================
# Exercise 1: Swish Activation
# ============================================================================

def swish(x: torch.Tensor) -> torch.Tensor:
    """
    Swish activation function: x * sigmoid(x)
    
    Also known as SiLU (Sigmoid Linear Unit).
    
    Args:
        x: Input tensor
    
    Returns:
        Swish-activated tensor
    """
    # API hints:
    # - torch.sigmoid(x) -> sigmoid function
    # - x * torch.sigmoid(x) -> Swish activation
    
    return None


class Swish(nn.Module):
    """Swish activation as a module."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x)


# ============================================================================
# Exercise 2: SwiGLU Feed-Forward Network
# ============================================================================

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    FFN(x) = (Swish(x @ W_gate) * (x @ W_up)) @ W_down
    
    The gating mechanism allows selective activation of dimensions.
    """
    
    def __init__(self, d_model: int, d_ffn: int = None, dropout: float = 0.0,
                 multiple_of: int = 256):
        """
        Args:
            d_model: Model dimension
            d_ffn: FFN hidden dimension (default: 8/3 * d_model, rounded)
            dropout: Dropout rate
            multiple_of: Round d_ffn to multiple of this (for efficiency)
        """
        super().__init__()
        
        # Compute hidden dimension
        # SwiGLU has 3 projections vs 2 in standard FFN
        # To match param count, use 2/3 of what you'd use for standard FFN
        if d_ffn is None:
            d_ffn = int(8 / 3 * d_model)
            d_ffn = multiple_of * ((d_ffn + multiple_of - 1) // multiple_of)
        
        self.d_model = d_model
        self.d_ffn = d_ffn
        
        # TODO: Create projections
        # API hints:
        # - nn.Linear(d_model, d_ffn, bias=False) -> W_gate, W_up
        # - nn.Linear(d_ffn, d_model, bias=False) -> W_down
        # - nn.Dropout(dropout) -> dropout layer
        
        self.W_gate = None
        self.W_up = None
        self.W_down = None
        self.dropout = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwiGLU FFN.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # API hints:
        # - swish(self.W_gate(x)) -> gated path
        # - self.W_up(x) -> up path
        # - gate * up -> element-wise multiplication
        # - self.W_down(self.dropout(hidden)) -> output
        
        return None


# ============================================================================
# Exercise 3: Expert with SwiGLU
# ============================================================================

class SwiGLUExpert(nn.Module):
    """
    Single expert using SwiGLU activation.
    
    Each expert is a small SwiGLU FFN that specializes in certain inputs.
    """
    
    def __init__(self, d_model: int, d_expert: int = None, dropout: float = 0.0):
        """
        Args:
            d_model: Model dimension
            d_expert: Expert hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        if d_expert is None:
            d_expert = d_model * 4
        
        self.d_model = d_model
        self.d_expert = d_expert
        
        # TODO: Create SwiGLU expert
        # API hints:
        # - SwiGLUFFN(d_model, d_expert, dropout) -> SwiGLU feed-forward
        
        self.ffn = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, d_model) or (num_tokens, d_model)
        
        Returns:
            Output of same shape
        """
        # API hints:
        # - self.ffn(x) -> forward through SwiGLU FFN
        
        return None


# ============================================================================
# Exercise 4: Router with Top-K Selection
# ============================================================================

class DeepSeekRouter(nn.Module):
    """
    Router for DeepSeek-style MoE.
    
    Features:
    - Top-k expert selection
    - Optional noise for exploration during training
    - Normalized gating weights
    """
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2,
                 noise_std: float = 0.1):
        """
        Args:
            d_model: Model dimension
            num_experts: Number of routed experts
            top_k: Number of experts to select per token
            noise_std: Noise standard deviation for training
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        
        # TODO: Create router
        # API hints:
        # - nn.Linear(d_model, num_experts, bias=False) -> router gate
        
        self.gate = None
    
    def forward(self, x: torch.Tensor, training: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            x: Input (batch, seq_len, d_model)
            training: Whether in training mode (add noise)
        
        Returns:
            gates: Top-k gate weights (batch, seq_len, top_k)
            indices: Expert indices (batch, seq_len, top_k)
            router_logits: Full logits (batch, seq_len, num_experts)
        """
        # API hints:
        # - self.gate(x) -> router logits (batch, seq, num_experts)
        # - torch.randn_like(tensor) * noise_std -> add noise
        # - F.softmax(logits, dim=-1) -> softmax over experts
        # - torch.topk(probs, top_k, dim=-1) -> top-k selection
        # - top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9) -> renormalize
        
        return None


# ============================================================================
# Exercise 5: MoE with Shared Experts
# ============================================================================

class DeepSeekMoE(nn.Module):
    """
    DeepSeek-style Mixture of Experts with shared experts.
    
    - Shared experts: Always activated for every token
    - Routed experts: Sparsely activated based on routing
    - Output = shared_output + routed_output
    """
    
    def __init__(self, d_model: int, num_shared_experts: int = 2,
                 num_routed_experts: int = 64, top_k: int = 6,
                 d_expert: int = None, dropout: float = 0.0):
        """
        Args:
            d_model: Model dimension
            num_shared_experts: Number of always-active experts
            num_routed_experts: Number of sparsely routed experts
            top_k: Experts selected per token (from routed)
            d_expert: Expert hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        
        # TODO: Create shared and routed experts
        # API hints:
        # - nn.ModuleList([SwiGLUExpert(...) for _ in range(n)]) -> expert lists
        # - DeepSeekRouter(d_model, num_routed_experts, top_k) -> router
        
        self.shared_experts = None
        self.routed_experts = None
        self.router = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through MoE.
        
        Args:
            x: Input (batch, seq_len, d_model)
        
        Returns:
            output: MoE output (batch, seq_len, d_model)
            aux_info: Dictionary with routing info for analysis
        """
        # API hints:
        # - Loop over self.shared_experts, sum outputs, average
        # - self.router(x, self.training) -> gates, indices, router_logits
        # - (indices == i).any(dim=-1) -> mask for expert i
        # - torch.where(condition, x, y) -> conditional selection
        # - expert_gates.unsqueeze(-1) * expert_out -> weighted output
        # - Return aux_info with router_logits, selected_experts, gate_weights
        
        return None


# ============================================================================
# Exercise 6: Complete DeepSeek Block
# ============================================================================

@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek architecture."""
    d_model: int = 2048
    num_heads: int = 16
    head_dim: int = 128
    
    # MLA config
    d_kv_latent: int = 512
    d_q_latent: int = 384
    rope_dim: int = 64
    max_seq_len: int = 4096
    
    # MoE config
    num_shared_experts: int = 2
    num_routed_experts: int = 64
    top_k: int = 6
    d_expert: int = None
    
    # Other
    dropout: float = 0.0


class DeepSeekBlock(nn.Module):
    """
    Complete DeepSeek transformer block.
    
    Combines:
    - MLA (Multi-head Latent Attention)
    - MoE with shared experts and SwiGLU
    - RMSNorm
    - Pre-norm structure
    """
    
    def __init__(self, config: DeepSeekConfig, use_moe: bool = True):
        """
        Args:
            config: DeepSeek configuration
            use_moe: Whether to use MoE (False for dense baseline)
        """
        super().__init__()
        
        self.config = config
        self.use_moe = use_moe
        
        # TODO: Create block components
        # API hints:
        # - RMSNorm(config.d_model) -> normalization layers
        # - nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        # - DeepSeekMoE(...) for MoE, SwiGLUFFN(...) for dense
        
        self.attn_norm = None
        self.attn = None
        self.ffn_norm = None
        self.ffn = None
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through DeepSeek block.
        
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Attention mask
        
        Returns:
            output: Block output
            aux_info: MoE routing info (if using MoE)
        """
        # API hints:
        # - self.attn_norm(x) -> pre-norm
        # - self.attn(normed, normed, normed, attn_mask=mask) -> attention
        # - x = x + attn_out -> residual connection
        # - self.ffn(normed) -> returns (output, aux_info) for MoE
        
        return None


# ============================================================================
# Exercise 7: DeepSeek Model
# ============================================================================

class DeepSeekModel(nn.Module):
    """
    Complete DeepSeek model with multiple blocks.
    """
    
    def __init__(self, config: DeepSeekConfig, num_layers: int = 4,
                 moe_layers: List[int] = None):
        """
        Args:
            config: DeepSeek configuration
            num_layers: Number of transformer layers
            moe_layers: Which layers use MoE (default: all except first)
        """
        super().__init__()
        
        self.config = config
        self.num_layers = num_layers
        
        # Default: MoE in all layers except the first (for stability)
        if moe_layers is None:
            moe_layers = list(range(1, num_layers))
        
        # TODO: Create layers
        # API hints:
        # - nn.ModuleList([DeepSeekBlock(config, use_moe=(i in moe_layers)) for i in range(num_layers)])
        # - RMSNorm(config.d_model) -> final normalization
        
        self.layers = None
        self.final_norm = None
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[dict]]:
        """
        Forward pass through all layers.
        
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Attention mask
        
        Returns:
            output: Final hidden states
            all_aux_info: List of aux info from each layer
        """
        # API hints:
        # - for layer in self.layers: x, aux = layer(x, mask)
        # - all_aux_info.append(aux_info) -> collect routing info
        # - self.final_norm(x) -> final normalization
        
        return None


# ============================================================================
# Exercise 8: Analyze Expert Usage
# ============================================================================

def analyze_expert_usage(aux_info_list: List[dict], num_routed_experts: int) -> dict:
    """
    Analyze expert usage patterns across layers.
    
    Args:
        aux_info_list: List of aux info dicts from each layer
        num_routed_experts: Number of routed experts
    
    Returns:
        Analysis dictionary
    """
    # API hints:
    # - torch.zeros(num_routed_experts) -> initialize expert counts
    # - aux['selected_experts'] -> indices tensor (batch, seq, top_k)
    # - (indices == i).sum().item() -> count tokens routed to expert i
    # - expert_counts / total_tokens -> utilization ratio
    # - Return dict with expert_counts, expert_utilization, load_imbalance
    
    return None


def compute_activated_params(config: DeepSeekConfig, num_layers: int,
                             moe_layers: List[int] = None) -> dict:
    """
    Compute number of activated parameters per forward pass.
    
    Args:
        config: DeepSeek configuration
        num_layers: Number of layers
        moe_layers: Which layers use MoE
    
    Returns:
        Parameter count dictionary
    """
    # API hints:
    # - d_ffn = int(8 / 3 * config.d_model) -> SwiGLU hidden dim
    # - d_expert = config.d_expert or config.d_model * 4
    # - attn_params = sum of Q, K, V, O projection params
    # - dense_ffn_params = 3 * d_model * d_ffn (W_gate, W_up, W_down)
    # - expert_params = 3 * d_model * d_expert per expert
    # - moe_ffn_params = shared_params + routed_params (top_k experts)
    
    return None


if __name__ == "__main__":
    print("Day 29: DeepSeek Architecture Components")
    print("=" * 60)
    
    # Test Swish
    print("\n" + "=" * 60)
    print("Testing Swish Activation")
    print("=" * 60)
    
    x = torch.randn(2, 8, 256)
    y = swish(x)
    if not torch.equal(x, y):
        print(f"Swish applied, output range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Test SwiGLU FFN
    print("\n" + "=" * 60)
    print("Testing SwiGLU FFN")
    print("=" * 60)
    
    d_model = 256
    ffn = SwiGLUFFN(d_model)
    if ffn.W_gate is not None:
        x = torch.randn(2, 16, d_model)
        out = ffn(x)
        print(f"Input: {x.shape}")
        print(f"Output: {out.shape}")
        print(f"FFN hidden dim: {ffn.d_ffn}")
    
    # Test DeepSeek Router
    print("\n" + "=" * 60)
    print("Testing DeepSeek Router")
    print("=" * 60)
    
    num_experts = 8
    top_k = 2
    router = DeepSeekRouter(d_model, num_experts, top_k)
    if router.gate is not None:
        x = torch.randn(2, 16, d_model)
        gates, indices, logits = router(x)
        print(f"Gates: {gates.shape}")
        print(f"Selected experts: {indices.shape}")
        print(f"Top-k experts per token: {indices[0, 0].tolist()}")
    
    # Test DeepSeek MoE
    print("\n" + "=" * 60)
    print("Testing DeepSeek MoE")
    print("=" * 60)
    
    moe = DeepSeekMoE(d_model, num_shared_experts=2, num_routed_experts=8, top_k=2)
    if moe.router is not None:
        x = torch.randn(2, 16, d_model)
        out, aux = moe(x)
        print(f"Input: {x.shape}")
        print(f"Output: {out.shape}")
        print(f"Shared experts: {moe.num_shared_experts}")
        print(f"Routed experts: {moe.num_routed_experts}")
    
    # Test DeepSeek Block
    print("\n" + "=" * 60)
    print("Testing DeepSeek Block")
    print("=" * 60)
    
    config = DeepSeekConfig(
        d_model=256,
        num_heads=4,
        head_dim=64,
        d_kv_latent=64,
        d_q_latent=48,
        num_shared_experts=1,
        num_routed_experts=4,
        top_k=2
    )
    
    block = DeepSeekBlock(config, use_moe=True)
    if block.ffn is not None:
        x = torch.randn(2, 16, config.d_model)
        out, aux = block(x)
        print(f"Block output: {out.shape}")
    
    # Test full model
    print("\n" + "=" * 60)
    print("Testing DeepSeek Model")
    print("=" * 60)
    
    model = DeepSeekModel(config, num_layers=4, moe_layers=[1, 2, 3])
    if model.layers is not None:
        x = torch.randn(2, 16, config.d_model)
        out, aux_list = model(x)
        print(f"Model output: {out.shape}")
        print(f"Layers with MoE: {[i for i, l in enumerate(model.layers) if l.use_moe]}")
    
    # Compute activated params
    print("\n" + "=" * 60)
    print("Parameter Analysis")
    print("=" * 60)
    
    params = compute_activated_params(config, num_layers=4, moe_layers=[1, 2, 3])
    if params['total_activated_params'] > 0:
        print(f"Activated params per forward: {params['total_activated_params']:,}")
        print(f"Dense FFN layers: {params['num_dense_layers']}")
        print(f"MoE layers: {params['num_moe_layers']}")
    
    print("\nRun test_day29.py to verify your implementations!")
