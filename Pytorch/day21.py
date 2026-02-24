"""
Day 21: Weight Initialization Strategies
========================================
Estimated time: 1-2 hours
Prerequisites: Basic understanding of neural networks and gradients

Learning objectives:
- Understand why proper weight initialization matters
- Implement Xavier/Glorot initialization
- Implement Kaiming/He initialization
- Implement orthogonal initialization
- Learn when to use each initialization method
- Apply initialization to different layer types

Key Concepts:
-------------
Why initialization matters:
- Too small: Gradients vanish, slow learning
- Too large: Gradients explode, unstable training
- Goal: Keep variance stable across layers

1. Xavier/Glorot Initialization (2010):
   - For linear/tanh/sigmoid activations
   - Uniform: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
   - Normal: N(0, sqrt(2/(fan_in+fan_out)))
   
2. Kaiming/He Initialization (2015):
   - For ReLU and variants
   - Accounts for ReLU's "killing" of negative values
   - Normal: N(0, sqrt(2/fan_in))
   - Uniform: U(-sqrt(6/fan_in), sqrt(6/fan_in))
   
3. Orthogonal Initialization:
   - Preserves gradient norms in RNNs
   - W @ W^T = I (orthogonal matrix)
   
4. Transformer-specific:
   - Embeddings: N(0, sqrt(1/d_model)) or N(0, 0.02)
   - Output projections: often scaled by 1/sqrt(2*num_layers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Exercise 1: Xavier/Glorot Initialization
# ============================================================================

def xavier_uniform_(tensor, gain=1.0):
    """
    Xavier uniform initialization.
    
    Fill tensor with values from U(-a, a) where:
    a = gain * sqrt(6 / (fan_in + fan_out))
    
    Args:
        tensor: Tensor to initialize (modified in-place)
        gain: Scaling factor (1.0 for linear, sqrt(2) for ReLU)
    
    Returns:
        tensor: The initialized tensor
    """
    if tensor.dim() < 2:
        raise ValueError("Xavier init requires at least 2D tensor")
    
    fan_in = tensor.shape[1]
    fan_out = tensor.shape[0]
    
    # TODO: Compute bound and fill tensor with uniform values
    # API hints:
    # - math.sqrt(6.0 / (fan_in + fan_out)) -> base bound
    # - a = gain * bound -> scaled bound
    # - tensor.uniform_(-a, a) -> fill in-place with uniform values
    return tensor


def xavier_normal_(tensor, gain=1.0):
    """
    Xavier normal initialization.
    
    Fill tensor with values from N(0, std) where:
    std = gain * sqrt(2 / (fan_in + fan_out))
    
    Args:
        tensor: Tensor to initialize
        gain: Scaling factor
    
    Returns:
        tensor: The initialized tensor
    """
    if tensor.dim() < 2:
        raise ValueError("Xavier init requires at least 2D tensor")
    
    fan_in = tensor.shape[1]
    fan_out = tensor.shape[0]
    
    # TODO: Compute std and fill tensor with normal values
    # API hints:
    # - math.sqrt(2.0 / (fan_in + fan_out)) -> base std
    # - std = gain * base_std -> scaled std
    # - tensor.normal_(0, std) -> fill in-place with normal values
    return tensor


# ============================================================================
# Exercise 2: Kaiming/He Initialization
# ============================================================================

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu'):
    """
    Kaiming uniform initialization.
    
    For ReLU: Fill with U(-bound, bound) where:
    bound = sqrt(6 / fan_mode) for gain=sqrt(2)
    
    Args:
        tensor: Tensor to initialize
        a: Negative slope for leaky_relu (0 for relu)
        mode: 'fan_in' (default) or 'fan_out'
        nonlinearity: 'relu' or 'leaky_relu'
    
    Returns:
        tensor: The initialized tensor
    """
    if tensor.dim() < 2:
        raise ValueError("Kaiming init requires at least 2D tensor")
    
    fan_in = tensor.shape[1]
    fan_out = tensor.shape[0]
    
    # TODO: Select fan, compute gain, std, bound, and fill tensor
    # API hints:
    # - fan = fan_in if mode == 'fan_in' else fan_out
    # - gain = math.sqrt(2.0) for relu, math.sqrt(2.0 / (1 + a**2)) for leaky_relu
    # - std = gain / math.sqrt(fan)
    # - bound = math.sqrt(3.0) * std
    # - tensor.uniform_(-bound, bound) -> fill in-place
    return tensor


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='relu'):
    """
    Kaiming normal initialization.
    
    For ReLU: Fill with N(0, std) where:
    std = sqrt(2 / fan_mode)
    
    Args:
        tensor: Tensor to initialize
        a: Negative slope for leaky_relu
        mode: 'fan_in' or 'fan_out'
        nonlinearity: 'relu' or 'leaky_relu'
    
    Returns:
        tensor: The initialized tensor
    """
    if tensor.dim() < 2:
        raise ValueError("Kaiming init requires at least 2D tensor")
    
    fan_in = tensor.shape[1]
    fan_out = tensor.shape[0]
    
    # TODO: Select fan, compute gain and std, fill tensor
    # API hints:
    # - fan = fan_in if mode == 'fan_in' else fan_out
    # - gain = math.sqrt(2.0) for relu, math.sqrt(2.0 / (1 + a**2)) for leaky_relu
    # - std = gain / math.sqrt(fan)
    # - tensor.normal_(0, std) -> fill in-place with normal values
    return tensor


# ============================================================================
# Exercise 3: Orthogonal Initialization
# ============================================================================

def orthogonal_(tensor, gain=1.0):
    """
    Orthogonal initialization.
    
    Creates an orthogonal matrix using QR decomposition.
    Useful for RNNs to prevent exploding/vanishing gradients.
    
    Args:
        tensor: Tensor to initialize (must be 2D or reshaped to 2D)
        gain: Scaling factor
    
    Returns:
        tensor: The initialized tensor
    """
    if tensor.dim() < 2:
        raise ValueError("Orthogonal init requires at least 2D tensor")
    
    rows = tensor.shape[0]
    cols = tensor.numel() // rows
    
    # TODO: Create orthogonal matrix using QR decomposition
    # API hints:
    # - torch.randn(rows, cols) -> random matrix
    # - torch.linalg.qr(flat) -> QR decomposition, returns (Q, R)
    # - torch.diag(r, 0) -> diagonal of R
    # - d.sign() -> sign of diagonal elements
    # - q * ph -> ensure consistent sign
    # - q * gain -> apply gain
    # - tensor.view(rows, cols).copy_(q) -> copy to tensor
    return tensor


# ============================================================================
# Exercise 4: Specialized Initializations
# ============================================================================

def zeros_(tensor):
    """Initialize tensor with zeros."""
    return tensor.zero_()


def ones_(tensor):
    """Initialize tensor with ones."""
    return tensor.fill_(1.0)


def constant_(tensor, value):
    """Initialize tensor with constant value."""
    return tensor.fill_(value)


def normal_(tensor, mean=0.0, std=1.0):
    """Initialize tensor with normal distribution."""
    return tensor.normal_(mean, std)


def truncated_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """
    Truncated normal initialization.
    
    Values are drawn from N(mean, std) but resampled if outside [a, b].
    Used in some BERT implementations.
    
    Args:
        tensor: Tensor to initialize
        mean: Mean of normal distribution
        std: Standard deviation
        a, b: Truncation bounds (in terms of std)
    
    Returns:
        tensor: The initialized tensor
    """
    # TODO: Fill with normal values and clamp to truncation bounds
    # API hints:
    # - tensor.normal_(mean, std) -> fill with normal values
    # - tensor.clamp_(mean + a * std, mean + b * std) -> clamp to bounds
    return tensor


# ============================================================================
# Exercise 5: Transformer-Specific Initialization
# ============================================================================

def init_transformer_weights(module, d_model, num_layers=None):
    """
    Initialize weights for a Transformer module.
    
    Common practices:
    - Embeddings: N(0, sqrt(1/d_model)) or N(0, 0.02)
    - Linear layers: Xavier or Kaiming depending on activation
    - Output projections: Scale by 1/sqrt(2*num_layers) for residual
    - Biases: Zero
    - LayerNorm: gamma=1, beta=0
    
    Args:
        module: nn.Module to initialize
        d_model: Model dimension
        num_layers: Number of layers (for residual scaling)
    """
    # TODO: Initialize based on module type
    # API hints:
    # - isinstance(module, nn.Linear) -> check if linear layer
    # - xavier_normal_(module.weight) -> Xavier init for linear
    # - zeros_(module.bias) -> zero init for biases
    # - isinstance(module, nn.Embedding) -> check if embedding
    # - normal_(module.weight, mean=0.0, std=math.sqrt(1.0/d_model)) -> embedding init
    # - isinstance(module, nn.LayerNorm) -> check if layer norm
    # - ones_(module.weight), zeros_(module.bias) -> layer norm init
    pass


def init_gpt_weights(module, n_layer, n_embd):
    """
    GPT-2 style initialization (as described in the paper).
    
    - Embeddings and linear: N(0, 0.02)
    - Residual projections: N(0, 0.02/sqrt(2*n_layer))
    """
    # TODO: Initialize based on module type
    # API hints:
    # - isinstance(module, nn.Linear) -> check if linear layer
    # - module.weight.data.normal_(mean=0.0, std=0.02) -> normal init
    # - module.bias.data.zero_() -> zero init for biases
    # - isinstance(module, nn.Embedding) -> check if embedding
    # - For residual projections: std = 0.02 / math.sqrt(2 * n_layer)
    pass


# ============================================================================
# Exercise 6: Applying Initialization to a Model
# ============================================================================

class SimpleTransformerBlock(nn.Module):
    """A simple transformer block for demonstrating initialization."""
    
    def __init__(self, d_model, num_heads, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        
        self.attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.ffn_1 = nn.Linear(d_model, d_ff)
        self.ffn_2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Simplified - just demonstrates structure
        return x


def apply_custom_init(model, init_fn):
    """
    Apply custom initialization to all parameters in a model.
    
    Args:
        model: nn.Module to initialize
        init_fn: Function that takes (name, param) and initializes param
    """
    for name, param in model.named_parameters():
        init_fn(name, param)


def smart_init(name, param):
    """
    Smart initialization based on parameter name.
    
    Demonstrates how to apply different init strategies
    based on the layer type/name.
    """
    if param.dim() < 2:
        # Skip 1D params (biases, norms)
        return
    
    if 'embedding' in name.lower():
        # Embeddings: normal with small std
        param.data.normal_(0, 0.02)
    elif 'norm' in name.lower():
        # Layer norm: ones for weight
        param.data.fill_(1.0)
    elif 'qkv' in name.lower() or 'query' in name.lower():
        # Attention projections: Xavier
        nn.init.xavier_uniform_(param.data)
    elif 'ffn' in name.lower() or 'mlp' in name.lower():
        # FFN with ReLU/GELU: Kaiming
        nn.init.kaiming_uniform_(param.data, a=math.sqrt(5))
    else:
        # Default: Xavier
        nn.init.xavier_uniform_(param.data)


# ============================================================================
# Exercise 7: Verifying Initialization Statistics
# ============================================================================

def check_init_statistics(tensor, name="tensor"):
    """
    Print statistics of initialized tensor.
    
    Good initialization should have:
    - Mean close to 0 (for most cases)
    - Std appropriate for the layer size
    """
    mean = tensor.mean().item()
    std = tensor.std().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    
    print(f"{name}:")
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std: {std:.6f}")
    print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
    
    return {'mean': mean, 'std': std, 'min': min_val, 'max': max_val}


def verify_variance_preservation(layers, input_tensor):
    """
    Test if initialization preserves variance through layers.
    
    Good init should keep variance roughly constant.
    """
    x = input_tensor
    print(f"Input variance: {x.var().item():.6f}")
    
    for i, layer in enumerate(layers):
        x = layer(x)
        if hasattr(layer, 'weight'):
            var = x.var().item()
            print(f"After layer {i} variance: {var:.6f}")
    
    return x


# ============================================================================
# Comparing Different Initializations
# ============================================================================

def compare_initializations():
    """Compare different initialization strategies."""
    print("="*60)
    print("Comparing Weight Initialization Strategies")
    print("="*60)
    
    in_features, out_features = 512, 512
    
    # Create tensors for each method
    w_xavier_u = torch.empty(out_features, in_features)
    w_xavier_n = torch.empty(out_features, in_features)
    w_kaiming_u = torch.empty(out_features, in_features)
    w_kaiming_n = torch.empty(out_features, in_features)
    w_orthogonal = torch.empty(out_features, in_features)
    
    # Apply initializations
    xavier_uniform_(w_xavier_u)
    xavier_normal_(w_xavier_n)
    kaiming_uniform_(w_kaiming_u)
    kaiming_normal_(w_kaiming_n)
    orthogonal_(w_orthogonal)
    
    print("\n1. Xavier Uniform:")
    check_init_statistics(w_xavier_u, "  Xavier Uniform")
    
    print("\n2. Xavier Normal:")
    check_init_statistics(w_xavier_n, "  Xavier Normal")
    
    print("\n3. Kaiming Uniform (ReLU):")
    check_init_statistics(w_kaiming_u, "  Kaiming Uniform")
    
    print("\n4. Kaiming Normal (ReLU):")
    check_init_statistics(w_kaiming_n, "  Kaiming Normal")
    
    print("\n5. Orthogonal:")
    check_init_statistics(w_orthogonal, "  Orthogonal")
    
    # Check orthogonality
    if w_orthogonal is not None:
        orth_check = torch.mm(w_orthogonal, w_orthogonal.t())
        identity = torch.eye(out_features)
        orth_error = (orth_check - identity).abs().mean().item()
        print(f"  Orthogonality error: {orth_error:.6f}")


def when_to_use_what():
    """Guidelines for choosing initialization."""
    print("\n" + "="*60)
    print("When to Use Which Initialization")
    print("="*60)
    
    print("""
Xavier/Glorot:
  - Use with: tanh, sigmoid, softmax, linear activations
  - Why: Preserves variance with symmetric activations
  
Kaiming/He:
  - Use with: ReLU, LeakyReLU, PReLU, ELU
  - Why: Accounts for ReLU killing half the values
  
Orthogonal:
  - Use with: RNN/LSTM hidden-to-hidden weights
  - Why: Preserves gradient norms, prevents vanishing/exploding
  
Small Normal (N(0, 0.02)):
  - Use with: Transformer embeddings, GPT-style models
  - Why: Empirically works well, as found by OpenAI
  
Zeros:
  - Use with: Biases (usually)
  - Why: Starting neutral, let learning find optimal bias
  
Ones:
  - Use with: LayerNorm/BatchNorm scale parameters
  - Why: Start with identity transformation
""")


if __name__ == "__main__":
    print("Day 21: Weight Initialization Strategies")
    print("=" * 50)
    
    # Compare initializations
    compare_initializations()
    
    # Guidelines
    when_to_use_what()
    
    # Demo: Initialize a model
    print("\n" + "="*50)
    print("Demo: Initializing a Transformer Block")
    print("="*50)
    
    block = SimpleTransformerBlock(d_model=256, num_heads=4)
    
    print("\nBefore custom init:")
    for name, param in block.named_parameters():
        if param.dim() >= 2:
            print(f"  {name}: std={param.std().item():.4f}")
    
    # Apply smart initialization
    apply_custom_init(block, smart_init)
    
    print("\nAfter custom init:")
    for name, param in block.named_parameters():
        if param.dim() >= 2:
            print(f"  {name}: std={param.std().item():.4f}")
    
    print("\nRun test_day21.py to verify your implementations!")
