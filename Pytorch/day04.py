"""
Day 4: Neural Network Building Blocks
=====================================
Estimated time: 1-2 hours
Prerequisites: Day 3 (autograd)

Learning objectives:
- Understand nn.Module as the building block
- Create simple neural network layers
- Use nn.Linear for fully connected layers
- Chain layers together
"""

import torch
import torch.nn as nn


# ============================================================================
# Exercise 1: Your First nn.Module
# ============================================================================

class SimpleLayer(nn.Module):
    """
    A simple layer that computes y = x * weight + bias.
    
    This is essentially what nn.Linear does!
    """
    
    def __init__(self, in_features, out_features):
        super().__init__()
        
        # TODO: Create weight parameter [out_features, in_features]
        # HINT: nn.Parameter(torch.randn(out_features, in_features))
        self.weight = None  # Replace
        
        # TODO: Create bias parameter [out_features]
        self.bias = None  # Replace
    
    def forward(self, x):
        """
        Forward pass: y = x @ weight.T + bias
        
        Args:
            x: Input tensor [batch, in_features]
        
        Returns:
            Output tensor [batch, out_features]
        """
        # TODO: Compute linear transformation
        # HINT: return x @ self.weight.T + self.bias
        return None  # Replace


# ============================================================================
# Exercise 2: Using nn.Linear
# ============================================================================

class TwoLayerNet(nn.Module):
    """
    A simple two-layer network.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        # TODO: Create first linear layer (input -> hidden)
        self.fc1 = None  # Replace: nn.Linear(input_size, hidden_size)
        
        # TODO: Create second linear layer (hidden -> output)
        self.fc2 = None  # Replace: nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass with ReLU activation between layers.
        """
        # TODO: First layer + ReLU
        x = None  # Replace: torch.relu(self.fc1(x))
        
        # TODO: Second layer (no activation - raw output)
        x = None  # Replace: self.fc2(x)
        
        return x


# ============================================================================
# Exercise 3: Counting Parameters
# ============================================================================

def count_parameters(model):
    """
    Count total number of trainable parameters in a model.
    
    Args:
        model: An nn.Module
    
    Returns:
        Total number of parameters
    """
    # TODO: Sum up all parameter sizes
    # HINT: sum(p.numel() for p in model.parameters())
    return None  # Replace


# ============================================================================
# Exercise 4: Accessing Parameters
# ============================================================================

def get_parameter_info(model):
    """
    Get information about model parameters.
    
    Returns:
        List of dicts with 'name', 'shape', 'requires_grad'
    """
    info = []
    
    # TODO: Iterate over named_parameters
    # HINT: for name, param in model.named_parameters():
    for name, param in model.named_parameters():
        info.append({
            'name': name,
            'shape': None,  # Replace: tuple(param.shape)
            'requires_grad': None  # Replace: param.requires_grad
        })
    
    return info


# ============================================================================
# Exercise 5: nn.Sequential
# ============================================================================

def create_sequential_net(input_size, hidden_size, output_size):
    """
    Create a network using nn.Sequential.
    
    Returns:
        nn.Sequential model
    """
    # TODO: Create sequential model with:
    # - Linear(input_size, hidden_size)
    # - ReLU
    # - Linear(hidden_size, output_size)
    
    model = None  # Replace with nn.Sequential(...)
    
    return model


# ============================================================================
# Exercise 6: Forward Pass Test
# ============================================================================

def test_forward_pass():
    """
    Test that the network can do a forward pass.
    
    Returns:
        dict with input shape, output shape, and output
    """
    model = TwoLayerNet(10, 20, 5)
    
    if model.fc1 is None:
        return {'error': 'Model not implemented'}
    
    # Create random input
    x = torch.randn(4, 10)  # Batch of 4, 10 features
    
    # TODO: Forward pass
    output = None  # Replace: model(x)
    
    return {
        'input_shape': tuple(x.shape),
        'output_shape': tuple(output.shape) if output is not None else None,
        'output_sample': output[0].tolist() if output is not None else None
    }


if __name__ == "__main__":
    print("Day 4: Neural Network Building Blocks")
    print("=" * 50)
    
    print("\nTesting TwoLayerNet:")
    result = test_forward_pass()
    print(f"  Input shape: {result.get('input_shape')}")
    print(f"  Output shape: {result.get('output_shape')}")
    
    print("\nRun test_day04.py to verify!")
