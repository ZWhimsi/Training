"""
Day 6: Loss Functions
=====================
Estimated time: 1-2 hours
Prerequisites: Day 5 (activation functions)

Learning objectives:
- Understand common loss functions
- Implement MSE, Cross-Entropy, BCE losses
- Learn about reduction modes
- Understand numerically stable implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Exercise 1: Mean Squared Error Loss
# ============================================================================

def mse_loss_manual(pred: torch.Tensor, target: torch.Tensor, 
                    reduction: str = 'mean') -> torch.Tensor:
    """
    Implement MSE loss: L = (pred - target)²
    
    Args:
        pred: Predictions
        target: Ground truth
        reduction: 'none', 'mean', or 'sum'
    """
    # API hints:
    # - (pred - target) ** 2 -> element-wise squared error
    # - tensor.mean() -> mean of all elements
    # - tensor.sum() -> sum of all elements
    # - Use if/elif to handle reduction modes
    
    return None


# ============================================================================
# Exercise 2: Binary Cross-Entropy Loss
# ============================================================================

def bce_loss_manual(pred: torch.Tensor, target: torch.Tensor,
                    reduction: str = 'mean') -> torch.Tensor:
    """
    Implement BCE loss: L = -[y * log(p) + (1-y) * log(1-p)]
    
    Args:
        pred: Probabilities (0 to 1)
        target: Binary targets (0 or 1)
        reduction: 'none', 'mean', or 'sum'
    """
    # API hints:
    # - torch.clamp(pred, min, max) -> clamp for numerical stability
    # - torch.log(x) -> element-wise natural log
    # - Use small epsilon (1e-7) to avoid log(0)
    
    return None


def bce_with_logits_manual(logits: torch.Tensor, target: torch.Tensor,
                           reduction: str = 'mean') -> torch.Tensor:
    """
    BCE with logits (more numerically stable).
    
    L = max(logits, 0) - logits * target + log(1 + exp(-|logits|))
    
    This is more stable than sigmoid + BCE.
    """
    # API hints:
    # - F.relu(logits) -> max(logits, 0)
    # - torch.abs(logits) -> absolute value
    # - torch.exp(x) -> element-wise exponential
    # - torch.log(x) -> element-wise natural log
    
    return None


# ============================================================================
# Exercise 3: Cross-Entropy Loss
# ============================================================================

def cross_entropy_manual(logits: torch.Tensor, target: torch.Tensor,
                         reduction: str = 'mean') -> torch.Tensor:
    """
    Cross-entropy loss for multi-class classification.
    
    L = -log(softmax(logits)[target_class])
    
    Uses log-softmax for numerical stability.
    
    Args:
        logits: (batch, num_classes) unnormalized scores
        target: (batch,) class indices
    """
    # API hints:
    # - F.log_softmax(logits, dim=-1) -> log of softmax (stable)
    # - tensor.gather(dim, index) -> gather values at indices
    # - tensor.unsqueeze(-1) -> add dimension at end
    # - tensor.squeeze(-1) -> remove dimension at end
    
    return None


def cross_entropy_smooth(logits: torch.Tensor, target: torch.Tensor,
                         smoothing: float = 0.1,
                         reduction: str = 'mean') -> torch.Tensor:
    """
    Cross-entropy with label smoothing.
    
    Instead of hard labels [0, 0, 1, 0], use smoothed distribution.
    """
    # API hints:
    # - F.log_softmax(logits, dim=-1) -> log of softmax
    # - torch.full_like(tensor, value) -> tensor filled with value
    # - tensor.scatter_(dim, index, value) -> scatter values at indices (in-place)
    # - (targets * log_probs).sum(dim=-1) -> weighted sum
    
    n_classes = logits.shape[-1]
    log_probs = F.log_softmax(logits, dim=-1)
    
    return None


# ============================================================================
# Exercise 4: Huber Loss
# ============================================================================

def huber_loss_manual(pred: torch.Tensor, target: torch.Tensor,
                      delta: float = 1.0, reduction: str = 'mean') -> torch.Tensor:
    """
    Huber loss (smooth L1): less sensitive to outliers than MSE.
    
    L = 0.5 * (pred - target)²  if |pred - target| <= delta
    L = delta * (|pred - target| - 0.5 * delta)  otherwise
    """
    # API hints:
    # - torch.abs(x) -> element-wise absolute value
    # - torch.where(condition, x, y) -> select x where True, y where False
    # - diff ** 2 -> element-wise square
    
    return None


# ============================================================================
# Exercise 5: Custom Loss Module
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p) = -alpha * (1 - p)^gamma * log(p)
    
    Reduces loss for well-classified examples, focuses on hard ones.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # API hints:
        # - F.cross_entropy(logits, target, reduction='none') -> per-sample CE loss
        # - torch.exp(-ce_loss) -> probability of correct class
        # - (1 - p) ** gamma -> focal weight
        # - loss.mean() -> mean reduction
        
        return None


if __name__ == "__main__":
    print("Day 6: Loss Functions")
    print("=" * 50)
    
    # Test data
    batch_size, n_classes = 8, 10
    logits = torch.randn(batch_size, n_classes)
    target = torch.randint(0, n_classes, (batch_size,))
    pred = torch.sigmoid(torch.randn(batch_size, n_classes))
    target_binary = torch.randint(0, 2, (batch_size, n_classes)).float()
    
    print("\nTesting loss functions:")
    
    # Test MSE
    mse_result = mse_loss_manual(pred, target_binary)
    if mse_result is not None:
        mse_expected = F.mse_loss(pred, target_binary)
        print(f"  MSE: error = {(mse_result - mse_expected).abs().item():.6f}")
    else:
        print("  MSE: NOT IMPLEMENTED")
    
    # Test Cross-Entropy
    ce_result = cross_entropy_manual(logits, target)
    if ce_result is not None:
        ce_expected = F.cross_entropy(logits, target)
        print(f"  Cross-Entropy: error = {(ce_result - ce_expected).abs().item():.6f}")
    else:
        print("  Cross-Entropy: NOT IMPLEMENTED")
    
    print("\nRun test_day06.py to verify all implementations!")
