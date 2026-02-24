"""
Day 12: Dropout and Regularization
==================================
Estimated time: 1-2 hours
Prerequisites: Day 11 (batch normalization)

Learning objectives:
- Understand dropout as a regularization technique
- Implement dropout manually
- Apply weight decay (L2 regularization)
- Implement L1 regularization
- Combine multiple regularization techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================================
# CONCEPT: Dropout
# ============================================================================
"""
Dropout randomly zeros elements during training to prevent overfitting:

During training:
    1. Generate a binary mask (keep_prob = 1 - dropout_prob)
    2. Multiply input by mask
    3. Scale by 1/keep_prob to maintain expected values

During evaluation:
    - No dropout applied (identity function)

Key insight: Dropout forces the network to learn redundant representations,
making it more robust and reducing co-adaptation between neurons.
"""


# ============================================================================
# Exercise 1: Manual Dropout Implementation
# ============================================================================

def manual_dropout(x: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
    """
    Implement dropout manually.
    
    Args:
        x: Input tensor
        p: Dropout probability (probability of zeroing an element)
        training: Whether in training mode
    
    Returns:
        Output tensor with dropout applied (training) or unchanged (eval)
    """
    # API hints:
    # - torch.rand_like(x) -> random tensor same shape as x, values in [0, 1)
    # - (tensor > p).float() -> binary mask as float tensor
    # - Scaling: multiply by 1/(1-p) to maintain expected value
    # - If not training or p==0, return x unchanged
    return None


# ============================================================================
# Exercise 2: Dropout Layer
# ============================================================================

class ManualDropout(nn.Module):
    """
    Manual implementation of Dropout layer.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        # API hints:
        # - Validate p is in range [0, 1], raise ValueError if not
        # - Store self.p for use in forward
        self.p = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout based on training mode (self.training)."""
        # API hints:
        # - self.training -> bool indicating if module is in train mode
        # - manual_dropout(x, self.p, self.training) -> apply dropout
        return None


# ============================================================================
# Exercise 3: Dropout2D for CNNs
# ============================================================================

def dropout2d(x: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
    """
    Implement 2D dropout (drops entire channels).
    
    Unlike regular dropout which drops individual elements,
    Dropout2D drops entire feature map channels.
    
    Args:
        x: Input tensor of shape (N, C, H, W)
        p: Dropout probability
        training: Whether in training mode
    
    Returns:
        Output with some channels zeroed
    """
    # API hints:
    # - x.shape -> (N, C, H, W)
    # - torch.rand(N, C, 1, 1, device=x.device) -> random per-channel mask
    # - Mask shape (N, C, 1, 1) broadcasts across spatial dimensions
    # - Scale by 1/(1-p) to maintain expected value
    return None


# ============================================================================
# Exercise 4: Weight Decay (L2 Regularization)
# ============================================================================

def compute_l2_regularization(model: nn.Module, weight_decay: float) -> torch.Tensor:
    """
    Compute L2 regularization penalty.
    
    L2 penalty = (weight_decay / 2) * sum(w^2) for all weights
    
    Args:
        model: Neural network module
        weight_decay: L2 regularization strength
    
    Returns:
        L2 penalty term (scalar tensor)
    """
    # API hints:
    # - model.parameters() -> iterate over all parameters
    # - (param ** 2).sum() -> sum of squared values
    # - torch.tensor(0.0) -> initialize accumulator
    # - Formula: (weight_decay / 2) * sum(w^2)
    return None


def train_step_with_l2(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                       loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                       weight_decay: float) -> float:
    """
    Training step with manual L2 regularization added to loss.
    """
    # API hints:
    # - optimizer.zero_grad() -> clear gradients
    # - model(x) -> forward pass
    # - loss_fn(pred, y) -> compute base loss
    # - compute_l2_regularization(model, weight_decay) -> L2 penalty
    # - total_loss = loss + l2_penalty
    # - total_loss.backward() -> compute gradients
    # - optimizer.step() -> update parameters
    # - tensor.item() -> get Python scalar
    return None


# ============================================================================
# Exercise 5: L1 Regularization
# ============================================================================

def compute_l1_regularization(model: nn.Module, l1_lambda: float) -> torch.Tensor:
    """
    Compute L1 regularization penalty.
    
    L1 penalty = l1_lambda * sum(|w|) for all weights
    L1 regularization encourages sparsity.
    
    Args:
        model: Neural network module
        l1_lambda: L1 regularization strength
    
    Returns:
        L1 penalty term (scalar tensor)
    """
    # API hints:
    # - model.parameters() -> iterate over parameters
    # - param.abs().sum() -> sum of absolute values
    # - torch.tensor(0.0) -> initialize accumulator
    # - Formula: l1_lambda * sum(|w|)
    return None


# ============================================================================
# Exercise 6: Elastic Net (Combined L1 + L2)
# ============================================================================

def compute_elastic_net_penalty(model: nn.Module, l1_lambda: float, 
                                 l2_lambda: float) -> torch.Tensor:
    """
    Compute Elastic Net penalty (combination of L1 and L2).
    
    Elastic Net = l1_lambda * |w| + (l2_lambda/2) * w^2
    Combines sparsity-inducing L1 with weight shrinkage of L2.
    """
    # API hints:
    # - compute_l1_regularization(model, l1_lambda) -> L1 term
    # - compute_l2_regularization(model, l2_lambda) -> L2 term
    # - Return sum of both penalties
    return None


# ============================================================================
# Exercise 7: Network with Dropout
# ============================================================================

class RegularizedMLP(nn.Module):
    """
    MLP with dropout regularization.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout_p: float = 0.5):
        super().__init__()
        # API hints:
        # - nn.Linear(in_features, out_features) -> fully connected layer
        # - nn.Dropout(p=dropout_p) -> dropout layer
        # - nn.ReLU() -> activation function
        # - Pattern: fc -> relu -> dropout (after each hidden layer)
        self.fc1 = None
        self.dropout1 = None
        self.fc2 = None
        self.dropout2 = None
        self.fc3 = None
        self.relu = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: fc1 -> relu -> dropout -> fc2 -> relu -> dropout -> fc3."""
        # API hints:
        # - Apply relu after each hidden fc layer
        # - Apply dropout after relu
        # - No activation/dropout after final fc layer
        return None


# ============================================================================
# Exercise 8: Comparing Regularization Effects
# ============================================================================

def compare_dropout_behavior(model: nn.Module, x: torch.Tensor, 
                             num_runs: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compare model behavior in train vs eval mode with dropout.
    
    Args:
        model: Model with dropout layers
        x: Input tensor
        num_runs: Number of forward passes to average
    
    Returns:
        Tuple of (train_outputs_variance, eval_output)
        - train_outputs_variance: Variance of outputs across runs (train mode)
        - eval_output: Single output in eval mode
    """
    # API hints:
    # - model.train() -> set to training mode (dropout active)
    # - model.eval() -> set to eval mode (dropout inactive)
    # - Run forward pass num_runs times in train mode
    # - torch.stack(outputs) -> stack list of tensors
    # - tensor.var(dim=0) -> variance along first dimension
    # - torch.no_grad() -> disable gradient computation for eval
    return None


if __name__ == "__main__":
    print("Day 12: Dropout and Regularization")
    print("=" * 50)
    
    # Test manual dropout
    x = torch.randn(10, 20)
    out = manual_dropout(x, p=0.5, training=True)
    print(f"\nManual dropout: input {x.shape} -> output {out.shape}")
    
    # Count zeros (approximately 50% should be zero)
    zero_fraction = (out == 0).float().mean().item()
    print(f"Fraction of zeros: {zero_fraction:.2%}")
    
    # Test dropout layer
    dropout = ManualDropout(p=0.3)
    dropout.train()
    out_train = dropout(x)
    dropout.eval()
    out_eval = dropout(x)
    print(f"\nDropout layer (p=0.3):")
    print(f"  Train mode zeros: {(out_train == 0).float().mean().item():.2%}")
    print(f"  Eval mode zeros: {(out_eval == 0).float().mean().item():.2%}")
    
    # Test regularization
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5)
    )
    l2 = compute_l2_regularization(model, weight_decay=0.01)
    l1 = compute_l1_regularization(model, l1_lambda=0.01)
    print(f"\nRegularization penalties:")
    print(f"  L2: {l2.item():.4f}")
    print(f"  L1: {l1.item():.4f}")
    
    print("\nRun test_day12.py to verify all implementations!")
