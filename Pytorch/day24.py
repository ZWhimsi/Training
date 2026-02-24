"""
Day 24: Mixture of Experts (MoE) Basics
=======================================
Estimated time: 1-2 hours
Prerequisites: Day 17 (transformer blocks), basic understanding of routing

Learning objectives:
- Understand the MoE architecture and its benefits
- Implement expert layers (specialized FFN networks)
- Build a router/gating network with top-k selection
- Handle load balancing across experts
- Implement a complete MoE layer

Key Concepts:
-------------
Mixture of Experts:
    Instead of one large FFN, use multiple smaller "expert" FFNs.
    A router network decides which experts process each token.
    Only K experts (typically 1-2) are activated per token.

Benefits:
    - Increased model capacity without proportional compute increase
    - Can learn specialized experts for different input types
    - Enables scaling to very large models efficiently

Architecture:
    1. Router: Takes input, outputs expert probabilities
    2. Top-K Selection: Choose K experts per token
    3. Expert Processing: Each expert is a small FFN
    4. Weighted Combination: Combine expert outputs using router weights

Load Balancing:
    Without balancing, some experts may be overloaded while others unused.
    Auxiliary loss encourages even distribution of tokens across experts.

    balance_loss = num_experts * sum(fraction_i * probability_i)
    where:
        fraction_i = fraction of tokens routed to expert i
        probability_i = mean router probability for expert i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


# ============================================================================
# Exercise 1: Expert Network
# ============================================================================

class Expert(nn.Module):
    """
    Single expert network - a standard FFN.
    
    Each expert is a small feed-forward network that specializes
    in processing certain types of inputs.
    """
    
    def __init__(self, d_model: int, d_expert: int = None, dropout: float = 0.0):
        """
        Args:
            d_model: Input/output dimension
            d_expert: Hidden dimension (defaults to 4 * d_model like standard FFN)
            dropout: Dropout rate
        """
        super().__init__()
        
        d_expert = d_expert or d_model * 4
        
        # TODO: Create expert FFN layers
        # API hints:
        # - nn.Linear(d_model, d_expert) -> expansion layer
        # - nn.Linear(d_expert, d_model) -> projection layer
        # - nn.Dropout(dropout) -> dropout layer
        self.w1 = None
        self.w2 = None
        self.dropout = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model) or (num_tokens, d_model)
        
        Returns:
            Output tensor of same shape
        """
        # TODO: Implement expert forward pass
        # API hints:
        # - self.w1(x) -> expansion
        # - F.gelu(x) -> GELU activation
        # - self.dropout(x) -> dropout
        # - self.w2(x) -> projection
        return x


# ============================================================================
# Exercise 2: Router Network
# ============================================================================

class Router(nn.Module):
    """
    Router network that assigns tokens to experts.
    
    Takes input tokens and outputs probabilities/scores for each expert.
    """
    
    def __init__(self, d_model: int, num_experts: int):
        """
        Args:
            d_model: Input dimension
            num_experts: Number of experts to route to
        """
        super().__init__()
        
        self.num_experts = num_experts
        
        # TODO: Create routing layer
        # API hints:
        # - nn.Linear(d_model, num_experts, bias=False) -> routing gate
        self.gate = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute routing scores for each expert.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Router logits (batch, seq_len, num_experts)
        """
        # TODO: Compute routing scores
        # API hints:
        # - self.gate(x) -> project to num_experts logits
        batch, seq_len, _ = x.shape
        return torch.zeros(batch, seq_len, self.num_experts)


# ============================================================================
# Exercise 3: Top-K Gating
# ============================================================================

def top_k_gating(router_logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform top-k gating to select experts.
    
    Args:
        router_logits: (batch, seq_len, num_experts) - raw routing scores
        k: Number of experts to select per token
    
    Returns:
        gates: (batch, seq_len, k) - softmax weights for selected experts
        indices: (batch, seq_len, k) - indices of selected experts
        router_probs: (batch, seq_len, num_experts) - full softmax probabilities
    """
    # TODO: Implement top-k selection
    # API hints:
    # - F.softmax(router_logits, dim=-1) -> get full softmax probabilities
    # - torch.topk(router_probs, k, dim=-1) -> select top-k, returns (values, indices)
    # - top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9) -> renormalize
    batch, seq_len, num_experts = router_logits.shape
    return (
        torch.zeros(batch, seq_len, k),
        torch.zeros(batch, seq_len, k, dtype=torch.long),
        torch.zeros(batch, seq_len, num_experts)
    )


# ============================================================================
# Exercise 4: Load Balancing Loss
# ============================================================================

def compute_load_balancing_loss(router_probs: torch.Tensor, 
                                 expert_indices: torch.Tensor,
                                 num_experts: int) -> torch.Tensor:
    """
    Compute auxiliary loss for load balancing across experts.
    
    This loss encourages the router to distribute tokens evenly.
    
    Args:
        router_probs: (batch, seq_len, num_experts) - softmax probabilities
        expert_indices: (batch, seq_len, k) - selected expert indices
        num_experts: Total number of experts
    
    Returns:
        Scalar load balancing loss
    
    The loss is: num_experts * sum_i(f_i * P_i)
    where:
        f_i = fraction of tokens assigned to expert i
        P_i = mean probability assigned to expert i
    """
    # TODO: Implement load balancing loss
    # API hints:
    # - F.one_hot(expert_indices, num_experts) -> one-hot encoding of selections
    # - expert_mask.sum(dim=2) -> sum over k selections
    # - tokens_per_expert = expert_mask.sum(dim=[0, 1]).float() -> count per expert
    # - f = tokens_per_expert / total_tokens -> fraction per expert
    # - P = router_probs.mean(dim=[0, 1]) -> mean probability per expert
    # - loss = num_experts * (f * P).sum() -> load balancing loss
    return torch.tensor(0.0)


# ============================================================================
# Exercise 5: Sparse MoE Layer
# ============================================================================

class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts layer.
    
    Each token is routed to top-k experts based on router scores.
    Only the selected experts are computed (sparse activation).
    """
    
    def __init__(self, d_model: int, num_experts: int, d_expert: int = None,
                 top_k: int = 2, dropout: float = 0.0):
        """
        Args:
            d_model: Model dimension
            num_experts: Number of expert networks
            d_expert: Expert hidden dimension
            top_k: Number of experts per token
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        # TODO: Create router and experts
        # API hints:
        # - Router(d_model, num_experts) -> routing network
        # - nn.ModuleList([Expert(d_model, d_expert, dropout) for _ in range(num_experts)]) -> experts
        self.router = None
        self.experts = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: Load balancing loss (scalar)
        """
        # TODO: Implement MoE forward pass
        # API hints:
        # - self.router(x) -> get routing logits
        # - top_k_gating(router_logits, self.top_k) -> get gates, indices, router_probs
        # - compute_load_balancing_loss(router_probs, indices, self.num_experts) -> aux loss
        # - (indices == i).any(dim=-1) -> mask for tokens routed to expert i
        # - expert(x) -> process through expert
        # - torch.where(indices == i, gates, zeros).sum(dim=-1) -> gate weights for expert
        # - output + expert_gates.unsqueeze(-1) * expert_output -> weighted combination
        return x, torch.tensor(0.0)


# ============================================================================
# Exercise 6: Efficient Batched MoE (Advanced)
# ============================================================================

class BatchedMoE(nn.Module):
    """
    More efficient MoE implementation using batched operations.
    
    Instead of looping over experts, we batch tokens by their assigned expert
    and process them together.
    """
    
    def __init__(self, d_model: int, num_experts: int, d_expert: int = None,
                 top_k: int = 2, dropout: float = 0.0):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        d_expert = d_expert or d_model * 4
        
        # Router
        self.router = Router(d_model, num_experts)
        
        # Shared expert weights for batched computation
        # TODO: Create batched expert parameters
        # API hints:
        # - nn.Parameter(torch.randn(num_experts, d_model, d_expert) * 0.02) -> w1 weights
        # - nn.Parameter(torch.randn(num_experts, d_expert, d_model) * 0.02) -> w2 weights
        self.expert_w1 = None
        self.expert_w2 = None
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched MoE forward pass.
        """
        # TODO: Implement batched expert computation
        # API hints:
        # - self.router(x) -> routing logits
        # - top_k_gating(router_logits, self.top_k) -> gates, indices, router_probs
        # - x.view(-1, d_model) -> flatten batch and seq
        # - self.expert_w1[expert_idx], self.expert_w2[expert_idx] -> get expert weights
        # - torch.bmm(x_flat.unsqueeze(1), w1).squeeze(1) -> batched matmul
        # - F.gelu(hidden) -> activation
        # - gate_weight.unsqueeze(-1) * expert_out -> weighted output
        # - output.view(batch, seq_len, d_model) -> reshape back
        return x, torch.tensor(0.0)


# ============================================================================
# Exercise 7: MoE Transformer Block
# ============================================================================

class MoETransformerBlock(nn.Module):
    """
    Transformer block with MoE replacing the standard FFN.
    """
    
    def __init__(self, d_model: int, num_heads: int, num_experts: int,
                 top_k: int = 2, d_expert: int = None, dropout: float = 0.0):
        super().__init__()
        
        # TODO: Create attention, MoE, and layer norms
        # API hints:
        # - nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True) -> attention
        # - SparseMoE(d_model, num_experts, d_expert, top_k, dropout) -> MoE layer
        # - nn.LayerNorm(d_model) -> layer norm
        # - nn.Dropout(dropout) -> dropout layer
        self.attention = None
        self.moe = None
        self.norm1 = None
        self.norm2 = None
        self.dropout = None
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning output and auxiliary loss.
        """
        # TODO: Implement pre-norm transformer block with MoE
        # API hints:
        # - self.norm1(x) -> pre-norm before attention
        # - self.attention(normed, normed, normed, attn_mask=mask) -> attention (q, k, v, mask)
        # - x + self.dropout(attn_out) -> residual connection
        # - self.norm2(x) -> pre-norm before MoE
        # - self.moe(normed) -> returns (output, aux_loss)
        return x, torch.tensor(0.0)


# ============================================================================
# Visualization: Expert Specialization
# ============================================================================

def analyze_expert_usage(router_probs: torch.Tensor, 
                         expert_indices: torch.Tensor,
                         num_experts: int) -> dict:
    """
    Analyze how tokens are distributed across experts.
    
    Args:
        router_probs: (batch, seq_len, num_experts)
        expert_indices: (batch, seq_len, k)
        num_experts: Number of experts
    
    Returns:
        Dictionary with usage statistics
    """
    batch, seq_len, k = expert_indices.shape
    total_assignments = batch * seq_len * k
    
    # Count assignments per expert
    counts = torch.zeros(num_experts)
    for i in range(num_experts):
        counts[i] = (expert_indices == i).sum().float()
    
    # Compute statistics
    mean_assignments = counts.mean().item()
    std_assignments = counts.std().item()
    max_assignments = counts.max().item()
    min_assignments = counts.min().item()
    
    # Load imbalance ratio
    imbalance = max_assignments / (min_assignments + 1e-9)
    
    return {
        'assignments_per_expert': counts.tolist(),
        'mean_assignments': mean_assignments,
        'std_assignments': std_assignments,
        'imbalance_ratio': imbalance,
        'total_assignments': total_assignments
    }


if __name__ == "__main__":
    print("Day 24: Mixture of Experts (MoE) Basics")
    print("=" * 50)
    
    # Configuration
    d_model = 256
    num_experts = 8
    top_k = 2
    batch_size = 2
    seq_len = 16
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  num_experts: {num_experts}")
    print(f"  top_k: {top_k}")
    
    # Test expert network
    print("\nTesting Expert Network:")
    expert = Expert(d_model)
    if expert.w1 is not None:
        x = torch.randn(batch_size, seq_len, d_model)
        out = expert(x)
        print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    # Test router
    print("\nTesting Router:")
    router = Router(d_model, num_experts)
    if router.gate is not None:
        x = torch.randn(batch_size, seq_len, d_model)
        logits = router(x)
        print(f"  Router logits: {logits.shape}")
    
    # Test top-k gating
    print("\nTesting Top-K Gating:")
    router_logits = torch.randn(batch_size, seq_len, num_experts)
    gates, indices, probs = top_k_gating(router_logits, top_k)
    print(f"  Gates: {gates.shape}")
    print(f"  Indices: {indices.shape}")
    print(f"  Probs: {probs.shape}")
    
    # Test load balancing loss
    print("\nTesting Load Balancing Loss:")
    aux_loss = compute_load_balancing_loss(probs, indices, num_experts)
    print(f"  Auxiliary loss: {aux_loss.item():.4f}")
    
    # Test SparseMoE
    print("\nTesting SparseMoE:")
    moe = SparseMoE(d_model, num_experts, top_k=top_k)
    if moe.router is not None:
        x = torch.randn(batch_size, seq_len, d_model)
        output, aux_loss = moe(x)
        print(f"  Input: {x.shape} -> Output: {output.shape}")
        print(f"  Auxiliary loss: {aux_loss.item():.4f}")
    
    # Test MoE Transformer Block
    print("\nTesting MoE Transformer Block:")
    block = MoETransformerBlock(d_model, num_heads=8, num_experts=num_experts, top_k=top_k)
    if block.moe is not None:
        x = torch.randn(batch_size, seq_len, d_model)
        output, aux_loss = block(x)
        print(f"  Block output: {output.shape}")
    
    print("\nRun test_day24.py to verify your implementations!")
