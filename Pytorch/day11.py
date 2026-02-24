"""
Day 11: Batch Normalization
===========================
Estimated time: 1-2 hours
Prerequisites: Day 10 (model save/load)

Learning objectives:
- Understand batch normalization mathematics
- Implement batch norm forward pass manually
- Handle running statistics for train vs eval modes
- Apply batch norm in neural networks
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


# ============================================================================
# CONCEPT: Batch Normalization
# ============================================================================
"""
Batch Normalization normalizes inputs across the batch dimension:

    y = (x - mean) / sqrt(var + eps) * gamma + beta

Where:
- mean, var: computed from the batch (training) or running stats (eval)
- gamma, beta: learnable scale and shift parameters
- eps: small constant for numerical stability

Key insight: During training, we compute batch statistics AND update running
statistics. During evaluation, we use the running statistics.
"""


# ============================================================================
# Exercise 1: Manual Batch Normalization Forward Pass
# ============================================================================

def batch_norm_forward(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                       eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute batch normalization forward pass (training mode).
    
    Args:
        x: Input tensor of shape (N, C) or (N, C, H, W)
        gamma: Scale parameter of shape (C,)
        beta: Shift parameter of shape (C,)
        eps: Small constant for numerical stability
    
    Returns:
        Tuple of (normalized_output, batch_mean, batch_var)
    """
    # API hints:
    # - x.dim() -> number of dimensions (2 for FC, 4 for CNN)
    # - x.mean(dim=...) -> compute mean over specified dimensions
    # - x.var(dim=..., unbiased=False) -> compute variance (use unbiased=False for batch norm)
    # - For 2D: mean/var over dim=0 (batch)
    # - For 4D: mean/var over dim=(0, 2, 3) (batch, height, width)
    # - tensor.view(1, -1, 1, 1) -> reshape for broadcasting
    # - torch.sqrt(var + eps) -> add eps before sqrt for stability
    # - Formula: out = gamma * (x - mean) / sqrt(var + eps) + beta
    return None


# ============================================================================
# Exercise 2: Batch Norm with Running Statistics
# ============================================================================

class ManualBatchNorm1d(nn.Module):
    """
    Manual implementation of 1D Batch Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics (not parameters, but persistent state)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with train/eval mode handling.
        Training: use batch stats and update running stats.
        Eval: use running stats.
        """
        # API hints:
        # - self.training -> bool indicating train mode
        # - x.mean(dim=0), x.var(dim=0, unbiased=False) -> batch statistics
        # - self.running_mean, self.running_var -> running statistics (buffers)
        # - Update running stats: running = (1 - momentum) * running + momentum * batch
        # - Normalize: x_norm = (x - mean) / sqrt(var + eps)
        # - Scale and shift: output = gamma * x_norm + beta
        return None


# ============================================================================
# Exercise 3: Batch Norm 2D (for CNNs)
# ============================================================================

class ManualBatchNorm2d(nn.Module):
    """
    Manual implementation of 2D Batch Normalization for CNNs.
    
    Key difference from 1D: Statistics are computed over (N, H, W) dimensions,
    keeping the channel dimension C separate.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 4D input (N, C, H, W).
        Statistics computed over N, H, W dimensions (per-channel normalization).
        """
        # API hints:
        # - x.mean(dim=(0, 2, 3)), x.var(dim=(0, 2, 3), unbiased=False) -> per-channel stats
        # - self.training -> check if in training mode
        # - Update running stats with exponential moving average
        # - Reshape params for broadcasting: tensor.view(1, -1, 1, 1) -> (1, C, 1, 1)
        # - Normalize: (x - mean) / sqrt(var + eps)
        # - Scale and shift: gamma * x_norm + beta
        return None


# ============================================================================
# Exercise 4: Train vs Eval Mode Behavior
# ============================================================================

def demonstrate_train_eval_difference(bn_layer: nn.Module, 
                                       x_train: torch.Tensor,
                                       x_eval: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Show the difference between train and eval mode outputs.
    
    Args:
        bn_layer: A batch norm layer
        x_train: Input for training mode
        x_eval: Input for eval mode
    
    Returns:
        Tuple of (train_output, eval_output)
    """
    # API hints:
    # - bn_layer.train() -> set module to training mode
    # - bn_layer.eval() -> set module to evaluation mode
    # - bn_layer(x) -> forward pass
    return None


# ============================================================================
# Exercise 5: CNN with Batch Normalization
# ============================================================================

class ConvBNReLU(nn.Module):
    """
    Convolutional block: Conv -> BatchNorm -> ReLU
    
    This is a common pattern in modern CNNs.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        # API hints:
        # - nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        #   Note: bias=False because BatchNorm has its own bias (beta)
        # - nn.BatchNorm2d(num_features) -> batch norm for 4D input
        # - nn.ReLU(inplace=True) -> ReLU activation
        self.conv = None
        self.bn = None
        self.relu = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv -> bn -> relu sequentially."""
        # API hints:
        # - Chain: self.relu(self.bn(self.conv(x)))
        return None


# ============================================================================
# Exercise 6: MLP with Batch Normalization
# ============================================================================

class MLPWithBatchNorm(nn.Module):
    """
    Multi-layer perceptron with batch normalization.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # API hints:
        # - nn.Linear(in_features, out_features) -> fully connected layer
        # - nn.BatchNorm1d(num_features) -> batch norm for 2D input (N, C)
        # - nn.ReLU() -> ReLU activation
        # - Pattern: Linear -> BatchNorm -> ReLU (no BN on final layer)
        self.fc1 = None
        self.bn1 = None
        self.fc2 = None
        self.bn2 = None
        self.fc3 = None
        self.relu = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: fc1 -> bn1 -> relu -> fc2 -> bn2 -> relu -> fc3."""
        # API hints:
        # - Chain layers: relu(bn(fc(x)))
        # - No activation/batchnorm after final fc layer
        return None


if __name__ == "__main__":
    print("Day 11: Batch Normalization")
    print("=" * 50)
    
    # Test manual batch norm forward
    x = torch.randn(32, 64)  # (batch, features)
    gamma = torch.ones(64)
    beta = torch.zeros(64)
    
    out, mean, var = batch_norm_forward(x, gamma, beta)
    print(f"\nManual BN forward: input {x.shape} -> output {out.shape}")
    
    # Test ManualBatchNorm1d
    bn1d = ManualBatchNorm1d(64)
    out = bn1d(x)
    print(f"ManualBatchNorm1d: {out.shape}")
    
    # Test ManualBatchNorm2d
    x_4d = torch.randn(8, 32, 16, 16)  # (N, C, H, W)
    bn2d = ManualBatchNorm2d(32)
    out_4d = bn2d(x_4d)
    print(f"ManualBatchNorm2d: {out_4d.shape}")
    
    # Test ConvBNReLU
    conv_bn = ConvBNReLU(3, 64)
    x_img = torch.randn(4, 3, 32, 32)
    if conv_bn.conv is not None:
        out_conv = conv_bn(x_img)
        print(f"ConvBNReLU: {x_img.shape} -> {out_conv.shape}")
    
    print("\nRun test_day11.py to verify all implementations!")
