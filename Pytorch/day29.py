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
    
    TODO: Implement Swish
    HINT:
        return x * torch.sigmoid(x)
    """
    return x  # Replace


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
        # HINT:
        #   self.W_gate = nn.Linear(d_model, d_ffn, bias=False)
        #   self.W_up = nn.Linear(d_model, d_ffn, bias=False)
        #   self.W_down = nn.Linear(d_ffn, d_model, bias=False)
        #   self.dropout = nn.Dropout(dropout)
        self.W_gate = None  # Replace
        self.W_up = None    # Replace
        self.W_down = None  # Replace
        self.dropout = None # Replace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwiGLU FFN.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        
        TODO: Implement SwiGLU FFN
        HINT:
            # Gate path: x -> W_gate -> Swish
            gate = swish(self.W_gate(x))
            
            # Up path: x -> W_up
            up = self.W_up(x)
            
            # Element-wise multiplication (gating)
            hidden = gate * up
            
            # Down projection
            hidden = self.dropout(hidden)
            output = self.W_down(hidden)
            
            return output
        """
        return x  # Replace


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
        # HINT:
        #   self.ffn = SwiGLUFFN(d_model, d_expert, dropout)
        self.ffn = None  # Replace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, d_model) or (num_tokens, d_model)
        
        Returns:
            Output of same shape
        
        TODO: Forward through SwiGLU
        HINT:
            return self.ffn(x)
        """
        return x  # Replace


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
        # HINT:
        #   self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.gate = None  # Replace
    
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
        
        TODO: Implement routing
        HINT:
            # Compute logits
            router_logits = self.gate(x)  # (batch, seq, num_experts)
            
            # Add noise during training for exploration
            if training and self.noise_std > 0:
                noise = torch.randn_like(router_logits) * self.noise_std
                router_logits = router_logits + noise
            
            # Softmax over experts
            router_probs = F.softmax(router_logits, dim=-1)
            
            # Top-k selection
            top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
            
            # Renormalize top-k probabilities
            top_k_gates = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
            
            return top_k_gates, top_k_indices, router_logits
        """
        batch, seq_len, _ = x.shape
        return (
            torch.zeros(batch, seq_len, self.top_k),
            torch.zeros(batch, seq_len, self.top_k, dtype=torch.long),
            torch.zeros(batch, seq_len, self.num_experts)
        )  # Replace


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
        # HINT:
        #   # Shared experts (always active)
        #   self.shared_experts = nn.ModuleList([
        #       SwiGLUExpert(d_model, d_expert, dropout)
        #       for _ in range(num_shared_experts)
        #   ])
        #   
        #   # Routed experts (sparsely active)
        #   self.routed_experts = nn.ModuleList([
        #       SwiGLUExpert(d_model, d_expert, dropout)
        #       for _ in range(num_routed_experts)
        #   ])
        #   
        #   # Router for routed experts
        #   self.router = DeepSeekRouter(d_model, num_routed_experts, top_k)
        self.shared_experts = None   # Replace
        self.routed_experts = None   # Replace
        self.router = None           # Replace
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through MoE.
        
        Args:
            x: Input (batch, seq_len, d_model)
        
        Returns:
            output: MoE output (batch, seq_len, d_model)
            aux_info: Dictionary with routing info for analysis
        
        TODO: Implement MoE with shared and routed experts
        HINT:
            batch, seq_len, d_model = x.shape
            
            # Shared experts (always active)
            shared_output = torch.zeros_like(x)
            for expert in self.shared_experts:
                shared_output = shared_output + expert(x)
            shared_output = shared_output / len(self.shared_experts)
            
            # Routed experts (sparse)
            gates, indices, router_logits = self.router(x, self.training)
            
            routed_output = torch.zeros_like(x)
            
            for i, expert in enumerate(self.routed_experts):
                # Find tokens routed to this expert
                expert_mask = (indices == i).any(dim=-1)  # (batch, seq)
                
                if expert_mask.any():
                    # Get gate weights for this expert
                    expert_gates = torch.where(
                        indices == i,
                        gates,
                        torch.zeros_like(gates)
                    ).sum(dim=-1)  # (batch, seq)
                    
                    # Apply expert to all tokens (for simplicity) and weight
                    expert_out = expert(x)
                    routed_output = routed_output + expert_gates.unsqueeze(-1) * expert_out
            
            # Combine shared and routed
            output = shared_output + routed_output
            
            aux_info = {
                'router_logits': router_logits,
                'selected_experts': indices,
                'gate_weights': gates
            }
            
            return output, aux_info
        """
        return x, {}  # Replace


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
        # HINT:
        #   # Attention (simplified - use standard attention for this exercise)
        #   self.attn_norm = RMSNorm(config.d_model)
        #   self.attn = nn.MultiheadAttention(
        #       config.d_model, config.num_heads,
        #       dropout=config.dropout, batch_first=True
        #   )
        #   
        #   # FFN (MoE or dense SwiGLU)
        #   self.ffn_norm = RMSNorm(config.d_model)
        #   if use_moe:
        #       self.ffn = DeepSeekMoE(
        #           config.d_model, config.num_shared_experts,
        #           config.num_routed_experts, config.top_k,
        #           config.d_expert, config.dropout
        #       )
        #   else:
        #       self.ffn = SwiGLUFFN(config.d_model, dropout=config.dropout)
        self.attn_norm = None  # Replace
        self.attn = None       # Replace
        self.ffn_norm = None   # Replace
        self.ffn = None        # Replace
    
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
        
        TODO: Implement block forward
        HINT:
            # Pre-norm attention
            normed = self.attn_norm(x)
            attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
            x = x + attn_out
            
            # Pre-norm FFN
            normed = self.ffn_norm(x)
            if self.use_moe:
                ffn_out, aux_info = self.ffn(normed)
            else:
                ffn_out = self.ffn(normed)
                aux_info = {}
            x = x + ffn_out
            
            return x, aux_info
        """
        return x, {}  # Replace


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
        # HINT:
        #   self.layers = nn.ModuleList([
        #       DeepSeekBlock(config, use_moe=(i in moe_layers))
        #       for i in range(num_layers)
        #   ])
        #   self.final_norm = RMSNorm(config.d_model)
        self.layers = None      # Replace
        self.final_norm = None  # Replace
    
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
        
        TODO: Implement forward
        HINT:
            all_aux_info = []
            
            for layer in self.layers:
                x, aux_info = layer(x, mask)
                all_aux_info.append(aux_info)
            
            x = self.final_norm(x)
            
            return x, all_aux_info
        """
        return x, []  # Replace


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
    
    TODO: Analyze routing patterns
    HINT:
        total_tokens = 0
        expert_counts = torch.zeros(num_routed_experts)
        
        for aux in aux_info_list:
            if 'selected_experts' in aux:
                indices = aux['selected_experts']  # (batch, seq, top_k)
                for i in range(num_routed_experts):
                    expert_counts[i] += (indices == i).sum().item()
                total_tokens += indices.numel()
        
        if total_tokens > 0:
            expert_utilization = expert_counts / total_tokens
            
            return {
                'expert_counts': expert_counts.tolist(),
                'expert_utilization': expert_utilization.tolist(),
                'max_utilization': expert_utilization.max().item(),
                'min_utilization': expert_utilization.min().item(),
                'load_imbalance': expert_utilization.max() / (expert_utilization.min() + 1e-9)
            }
        
        return {}
    """
    return {}  # Replace


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
    
    TODO: Compute activated parameters
    HINT:
        if moe_layers is None:
            moe_layers = list(range(1, num_layers))
        
        d_ffn = int(8 / 3 * config.d_model)
        d_expert = config.d_expert or config.d_model * 4
        
        # Attention params (same for all layers with MLA)
        attn_params = (
            config.d_model * config.d_q_latent +  # q down
            config.d_q_latent * config.num_heads * config.head_dim +  # q up
            config.d_model * config.d_kv_latent +  # kv down
            config.d_kv_latent * config.num_heads * config.head_dim * 2 +  # k/v up
            config.num_heads * config.head_dim * config.d_model  # output
        )
        
        # Dense SwiGLU FFN params
        dense_ffn_params = 3 * config.d_model * d_ffn  # W_gate, W_up, W_down
        
        # MoE FFN params (activated per token)
        expert_params = 3 * config.d_model * d_expert
        shared_params = config.num_shared_experts * expert_params
        routed_params = config.top_k * expert_params
        moe_ffn_params = shared_params + routed_params
        
        # Total
        total_attn = num_layers * attn_params
        dense_layers = num_layers - len(moe_layers)
        total_ffn = dense_layers * dense_ffn_params + len(moe_layers) * moe_ffn_params
        
        return {
            'attention_params_per_layer': attn_params,
            'dense_ffn_params_per_layer': dense_ffn_params,
            'moe_activated_params_per_layer': moe_ffn_params,
            'total_activated_params': total_attn + total_ffn,
            'num_moe_layers': len(moe_layers),
            'num_dense_layers': dense_layers
        }
    """
    return {
        'attention_params_per_layer': 0,
        'dense_ffn_params_per_layer': 0,
        'moe_activated_params_per_layer': 0,
        'total_activated_params': 0,
        'num_moe_layers': 0,
        'num_dense_layers': 0
    }  # Replace


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
