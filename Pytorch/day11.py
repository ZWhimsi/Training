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
    
    TODO: Implement batch normalization
    HINT:
        # For 2D input (N, C), compute mean/var over batch dimension
        # For 4D input (N, C, H, W), compute mean/var over N, H, W dimensions
        
        if x.dim() == 2:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
        else:  # 4D
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
        
        # Normalize
        x_norm = (x - mean.view(1, -1, *([1]*(x.dim()-2)))) / torch.sqrt(var.view(1, -1, *([1]*(x.dim()-2))) + eps)
        
        # Scale and shift
        out = gamma.view(1, -1, *([1]*(x.dim()-2))) * x_norm + beta.view(1, -1, *([1]*(x.dim()-2)))
        
        return out, mean, var
    """
    return x, torch.zeros(x.shape[1]), torch.ones(x.shape[1])  # Replace


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
        
        TODO: Implement forward pass
        HINT:
            if self.training:
                # Compute batch statistics
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)
                
                # Update running statistics
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            else:
                mean = self.running_mean
                var = self.running_var
            
            # Normalize
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            
            # Scale and shift
            return self.gamma * x_norm + self.beta
        """
        return x  # Replace


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
        
        TODO: Implement 2D batch norm forward
        HINT:
            if self.training:
                # Compute mean/var over N, H, W (keep C separate)
                mean = x.mean(dim=(0, 2, 3))
                var = x.var(dim=(0, 2, 3), unbiased=False)
                
                # Update running stats
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            else:
                mean = self.running_mean
                var = self.running_var
            
            # Reshape for broadcasting: (C,) -> (1, C, 1, 1)
            mean = mean.view(1, -1, 1, 1)
            var = var.view(1, -1, 1, 1)
            gamma = self.gamma.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
            
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            return gamma * x_norm + beta
        """
        return x  # Replace


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
    
    TODO: Run the layer in both modes
    HINT:
        bn_layer.train()
        out_train = bn_layer(x_train)
        
        bn_layer.eval()
        out_eval = bn_layer(x_eval)
        
        return out_train, out_eval
    """
    return x_train, x_eval  # Replace


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
        """
        TODO: Create the Conv -> BN -> ReLU block
        HINT:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        
        Note: bias=False in conv because BN has its own bias (beta parameter)
        """
        self.conv = None  # Replace
        self.bn = None    # Replace
        self.relu = None  # Replace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Apply conv -> bn -> relu
        HINT: return self.relu(self.bn(self.conv(x)))
        """
        return x  # Replace


# ============================================================================
# Exercise 6: MLP with Batch Normalization
# ============================================================================

class MLPWithBatchNorm(nn.Module):
    """
    Multi-layer perceptron with batch normalization.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        """
        TODO: Create MLP with batch norm after each linear layer (except last)
        HINT:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
        """
        self.fc1 = None  # Replace
        self.bn1 = None  # Replace
        self.fc2 = None  # Replace
        self.bn2 = None  # Replace
        self.fc3 = None  # Replace
        self.relu = None # Replace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Forward pass with BN after activations
        HINT:
            x = self.relu(self.bn1(self.fc1(x)))
            x = self.relu(self.bn2(self.fc2(x)))
            x = self.fc3(x)
            return x
        """
        return x  # Replace


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
