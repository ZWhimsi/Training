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
    
    TODO: Implement dropout
    HINT:
        if not training or p == 0:
            return x
        
        # Create binary mask: 1 with probability (1-p), 0 with probability p
        mask = (torch.rand_like(x) > p).float()
        
        # Scale to maintain expected value
        return x * mask / (1 - p)
    """
    return x  # Replace


# ============================================================================
# Exercise 2: Dropout Layer
# ============================================================================

class ManualDropout(nn.Module):
    """
    Manual implementation of Dropout layer.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        """
        TODO: Store dropout probability
        HINT:
            if p < 0 or p > 1:
                raise ValueError("Dropout probability must be between 0 and 1")
            self.p = p
        """
        self.p = 0.5  # Replace with validation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Apply dropout based on training mode
        HINT: return manual_dropout(x, self.p, self.training)
        """
        return x  # Replace


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
    
    TODO: Implement channel-wise dropout
    HINT:
        if not training or p == 0:
            return x
        
        # Create mask of shape (N, C, 1, 1) - same channel mask for all spatial locations
        N, C, H, W = x.shape
        mask = (torch.rand(N, C, 1, 1, device=x.device) > p).float()
        
        return x * mask / (1 - p)
    """
    return x  # Replace


# ============================================================================
# Exercise 4: Weight Decay (L2 Regularization)
# ============================================================================

def compute_l2_regularization(model: nn.Module, weight_decay: float) -> torch.Tensor:
    """
    Compute L2 regularization penalty.
    
    L2 penalty = (weight_decay / 2) * sum(w^2) for all weights
    
    Note: In practice, PyTorch's optimizer handles this automatically
    with the weight_decay parameter. This is for understanding.
    
    Args:
        model: Neural network module
        weight_decay: L2 regularization strength
    
    Returns:
        L2 penalty term (scalar tensor)
    
    TODO: Compute L2 penalty
    HINT:
        l2_penalty = torch.tensor(0.0)
        for param in model.parameters():
            l2_penalty = l2_penalty + (param ** 2).sum()
        return (weight_decay / 2) * l2_penalty
    """
    return torch.tensor(0.0)  # Replace


def train_step_with_l2(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                       loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                       weight_decay: float) -> float:
    """
    Training step with manual L2 regularization.
    
    TODO: Perform training step with L2 penalty added to loss
    HINT:
        optimizer.zero_grad()
        
        pred = model(x)
        loss = loss_fn(pred, y)
        
        # Add L2 penalty
        l2_penalty = compute_l2_regularization(model, weight_decay)
        total_loss = loss + l2_penalty
        
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
    """
    return 0.0  # Replace


# ============================================================================
# Exercise 5: L1 Regularization
# ============================================================================

def compute_l1_regularization(model: nn.Module, l1_lambda: float) -> torch.Tensor:
    """
    Compute L1 regularization penalty.
    
    L1 penalty = l1_lambda * sum(|w|) for all weights
    
    L1 regularization encourages sparsity (many weights become exactly zero).
    
    Args:
        model: Neural network module
        l1_lambda: L1 regularization strength
    
    Returns:
        L1 penalty term (scalar tensor)
    
    TODO: Compute L1 penalty
    HINT:
        l1_penalty = torch.tensor(0.0)
        for param in model.parameters():
            l1_penalty = l1_penalty + param.abs().sum()
        return l1_lambda * l1_penalty
    """
    return torch.tensor(0.0)  # Replace


# ============================================================================
# Exercise 6: Elastic Net (Combined L1 + L2)
# ============================================================================

def compute_elastic_net_penalty(model: nn.Module, l1_lambda: float, 
                                 l2_lambda: float) -> torch.Tensor:
    """
    Compute Elastic Net penalty (combination of L1 and L2).
    
    Elastic Net = l1_lambda * |w| + (l2_lambda/2) * w^2
    
    This combines the sparsity-inducing property of L1 with
    the weight shrinkage of L2.
    
    TODO: Compute combined penalty
    HINT:
        l1 = compute_l1_regularization(model, l1_lambda)
        l2 = compute_l2_regularization(model, l2_lambda)
        return l1 + l2
    """
    return torch.tensor(0.0)  # Replace


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
        """
        TODO: Create MLP with dropout after each hidden layer
        HINT:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.dropout1 = nn.Dropout(p=dropout_p)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.dropout2 = nn.Dropout(p=dropout_p)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
        """
        self.fc1 = None      # Replace
        self.dropout1 = None # Replace
        self.fc2 = None      # Replace
        self.dropout2 = None # Replace
        self.fc3 = None      # Replace
        self.relu = None     # Replace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Forward pass with dropout
        HINT:
            x = self.relu(self.fc1(x))
            x = self.dropout1(x)
            x = self.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            return x
        """
        return x  # Replace


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
    
    TODO: Run model multiple times in train mode and once in eval mode
    HINT:
        model.train()
        train_outputs = []
        for _ in range(num_runs):
            out = model(x)
            train_outputs.append(out)
        train_outputs = torch.stack(train_outputs)
        train_var = train_outputs.var(dim=0)
        
        model.eval()
        with torch.no_grad():
            eval_output = model(x)
        
        return train_var, eval_output
    """
    return torch.zeros_like(x), x  # Replace


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
