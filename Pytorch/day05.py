"""
Day 5: Activation Functions
===========================
Estimated time: 1-2 hours
Prerequisites: Day 4 (neural network building blocks)

Learning objectives:
- Understand common activation functions
- Implement ReLU, GELU, Sigmoid, Softmax
- Learn when to use each activation
- Compare activation behaviors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Exercise 1: ReLU Activation
# ============================================================================

def relu_manual(x: torch.Tensor) -> torch.Tensor:
    """
    Implement ReLU manually: f(x) = max(0, x)
    
    TODO: Return x where x > 0, else 0
    HINT: torch.clamp(x, min=0) or torch.maximum(x, torch.zeros_like(x))
    """
    return None  # Replace


def leaky_relu_manual(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """
    Implement Leaky ReLU: f(x) = x if x > 0 else negative_slope * x
    
    TODO: Use torch.where for conditional
    HINT: torch.where(x > 0, x, negative_slope * x)
    """
    return None  # Replace


# ============================================================================
# Exercise 2: GELU Activation
# ============================================================================

def gelu_manual(x: torch.Tensor) -> torch.Tensor:
    """
    Implement GELU (Gaussian Error Linear Unit).
    
    GELU(x) = x * Φ(x) where Φ is the CDF of standard normal.
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    
    TODO: Implement the approximation
    HINT: Use torch.tanh and math.sqrt(2.0 / math.pi)
    """
    # Constants
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    
    # TODO: Implement GELU approximation
    return None  # Replace


# ============================================================================
# Exercise 3: Sigmoid Activation
# ============================================================================

def sigmoid_manual(x: torch.Tensor) -> torch.Tensor:
    """
    Implement sigmoid: σ(x) = 1 / (1 + exp(-x))
    
    TODO: Implement sigmoid
    HINT: 1.0 / (1.0 + torch.exp(-x))
    """
    return None  # Replace


def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Hard sigmoid approximation (faster but less accurate).
    
    f(x) = clamp((x + 3) / 6, 0, 1)
    """
    # TODO: Implement hard sigmoid
    return None  # Replace


# ============================================================================
# Exercise 4: Softmax Activation
# ============================================================================

def softmax_manual(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Implement softmax: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    
    Numerically stable version subtracts max first.
    
    TODO: Implement stable softmax
    HINT: 
        1. x_max = x.max(dim=dim, keepdim=True).values
        2. x_exp = torch.exp(x - x_max)
        3. return x_exp / x_exp.sum(dim=dim, keepdim=True)
    """
    return None  # Replace


def log_softmax_manual(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Implement log_softmax: log(softmax(x))
    
    More stable than log(softmax(x)) computed separately.
    log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    """
    # TODO: Implement log_softmax
    return None  # Replace


# ============================================================================
# Exercise 5: SiLU/Swish Activation
# ============================================================================

def silu_manual(x: torch.Tensor) -> torch.Tensor:
    """
    Implement SiLU (Sigmoid Linear Unit), also known as Swish.
    
    SiLU(x) = x * sigmoid(x)
    
    Used in EfficientNet, LLaMA, and many modern architectures.
    """
    # TODO: Implement SiLU
    return None  # Replace


# ============================================================================
# Exercise 6: Create Custom Activation Module
# ============================================================================

class Mish(nn.Module):
    """
    Mish activation: f(x) = x * tanh(softplus(x))
    
    softplus(x) = ln(1 + exp(x))
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement Mish
        # HINT: softplus = torch.log(1 + torch.exp(x))
        #       return x * torch.tanh(softplus)
        return None  # Replace


if __name__ == "__main__":
    print("Day 5: Activation Functions")
    print("=" * 50)
    
    # Test with sample input
    x = torch.randn(4, 8)
    
    print("\nTesting activations:")
    activations = [
        ("relu_manual", relu_manual, F.relu),
        ("gelu_manual", gelu_manual, F.gelu),
        ("sigmoid_manual", sigmoid_manual, torch.sigmoid),
        ("softmax_manual", lambda t: softmax_manual(t, dim=-1), lambda t: F.softmax(t, dim=-1)),
        ("silu_manual", silu_manual, F.silu),
    ]
    
    for name, our_fn, torch_fn in activations:
        result = our_fn(x)
        if result is not None:
            expected = torch_fn(x)
            err = (result - expected).abs().max().item()
            print(f"  {name}: max error = {err:.6f}")
        else:
            print(f"  {name}: NOT IMPLEMENTED")
    
    print("\nRun test_day05.py to verify all implementations!")
