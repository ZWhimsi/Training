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
    
    TODO: Compute squared error and apply reduction
    HINT: 
        loss = (pred - target) ** 2
        if reduction == 'mean': return loss.mean()
        elif reduction == 'sum': return loss.sum()
        else: return loss
    """
    return None  # Replace


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
    
    TODO: Implement BCE formula
    HINT: 
        eps = 1e-7  # Numerical stability
        pred = torch.clamp(pred, eps, 1 - eps)
        loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    """
    return None  # Replace


def bce_with_logits_manual(logits: torch.Tensor, target: torch.Tensor,
                           reduction: str = 'mean') -> torch.Tensor:
    """
    BCE with logits (more numerically stable).
    
    L = max(logits, 0) - logits * target + log(1 + exp(-|logits|))
    
    This is more stable than sigmoid + BCE.
    """
    # TODO: Implement stable BCE with logits
    # HINT: 
    # pos_part = F.relu(logits)
    # neg_part = logits * target
    # log_part = torch.log(1 + torch.exp(-torch.abs(logits)))
    # loss = pos_part - neg_part + log_part
    return None  # Replace


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
    
    TODO: Implement cross-entropy
    HINT:
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    """
    return None  # Replace


def cross_entropy_smooth(logits: torch.Tensor, target: torch.Tensor,
                         smoothing: float = 0.1,
                         reduction: str = 'mean') -> torch.Tensor:
    """
    Cross-entropy with label smoothing.
    
    Instead of hard labels [0, 0, 1, 0], use:
    [(smoothing/n_classes), (smoothing/n_classes), (1-smoothing) + (smoothing/n_classes), ...]
    """
    n_classes = logits.shape[-1]
    log_probs = F.log_softmax(logits, dim=-1)
    
    # TODO: Create smoothed targets and compute loss
    # HINT:
    # One-hot encoding with smoothing
    # smooth_targets = torch.full_like(log_probs, smoothing / n_classes)
    # smooth_targets.scatter_(-1, target.unsqueeze(-1), 1 - smoothing + smoothing / n_classes)
    # loss = -(smooth_targets * log_probs).sum(dim=-1)
    return None  # Replace


# ============================================================================
# Exercise 4: Huber Loss
# ============================================================================

def huber_loss_manual(pred: torch.Tensor, target: torch.Tensor,
                      delta: float = 1.0, reduction: str = 'mean') -> torch.Tensor:
    """
    Huber loss (smooth L1): less sensitive to outliers than MSE.
    
    L = 0.5 * (pred - target)²  if |pred - target| <= delta
    L = delta * (|pred - target| - 0.5 * delta)  otherwise
    
    TODO: Implement Huber loss
    HINT:
        diff = torch.abs(pred - target)
        loss = torch.where(diff <= delta, 
                          0.5 * diff ** 2,
                          delta * (diff - 0.5 * delta))
    """
    return None  # Replace


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
        """
        TODO: Implement Focal Loss
        
        HINT:
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        p = torch.exp(-ce_loss)  # Probability of correct class
        focal_weight = self.alpha * (1 - p) ** self.gamma
        loss = focal_weight * ce_loss
        return loss.mean()
        """
        return None  # Replace


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
