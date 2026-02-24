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
        
        # API hints:
        # - nn.Parameter(tensor) -> wraps tensor as learnable parameter
        # - torch.randn(size) -> random tensor with normal distribution
        # - Weight shape: [out_features, in_features]
        # - Bias shape: [out_features]
        
        # TODO: Create weight parameter [out_features, in_features]
        self.weight = None
        
        # TODO: Create bias parameter [out_features]
        self.bias = None
    
    def forward(self, x):
        """
        Forward pass: y = x @ weight.T + bias
        
        Args:
            x: Input tensor [batch, in_features]
        
        Returns:
            Output tensor [batch, out_features]
        """
        # API hints:
        # - x @ self.weight.T -> matrix multiplication with transposed weight
        # - Add bias after matmul
        
        # TODO: Compute linear transformation
        return None


# ============================================================================
# Exercise 2: Using nn.Linear
# ============================================================================

class TwoLayerNet(nn.Module):
    """
    A simple two-layer network.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        # API hints:
        # - nn.Linear(in_features, out_features) -> fully connected layer
        
        # TODO: Create first linear layer (input -> hidden)
        self.fc1 = None
        
        # TODO: Create second linear layer (hidden -> output)
        self.fc2 = None
    
    def forward(self, x):
        """
        Forward pass with ReLU activation between layers.
        """
        # API hints:
        # - torch.relu(tensor) or F.relu(tensor) -> ReLU activation
        # - self.fc1(x) -> pass x through first layer
        # - self.fc2(x) -> pass x through second layer
        
        # TODO: First layer + ReLU
        x = None
        
        # TODO: Second layer (no activation - raw output)
        x = None
        
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
    # API hints:
    # - model.parameters() -> iterator over all parameters
    # - param.numel() -> number of elements in parameter tensor
    # - sum(generator) -> sum up values
    
    # TODO: Sum up all parameter sizes
    return None


# ============================================================================
# Exercise 4: Accessing Parameters
# ============================================================================

def get_parameter_info(model):
    """
    Get information about model parameters.
    
    Returns:
        List of dicts with 'name', 'shape', 'requires_grad'
    """
    # API hints:
    # - model.named_parameters() -> iterator of (name, param) tuples
    # - tuple(param.shape) -> parameter shape as tuple
    # - param.requires_grad -> boolean
    
    info = []
    
    for name, param in model.named_parameters():
        info.append({
            'name': name,
            'shape': None,
            'requires_grad': None
        })
    
    return info


# ============================================================================
# Exercise 5: nn.Sequential
# ============================================================================

def create_sequential_net(input_size, hidden_size, output_size):
    """
    Create a network using nn.Sequential.
    
    Returns:
        nn.Sequential model with Linear -> ReLU -> Linear
    """
    # API hints:
    # - nn.Sequential(layer1, layer2, ...) -> sequential container
    # - nn.Linear(in_features, out_features) -> linear layer
    # - nn.ReLU() -> ReLU activation module
    
    # TODO: Create sequential model
    model = None
    
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
    # API hints:
    # - model(x) -> forward pass through model
    # - tuple(tensor.shape) -> shape as tuple
    # - tensor[0].tolist() -> first sample as Python list
    
    model = TwoLayerNet(10, 20, 5)
    
    if model.fc1 is None:
        return {'error': 'Model not implemented'}
    
    # Create random input
    x = torch.randn(4, 10)  # Batch of 4, 10 features
    
    # TODO: Forward pass
    output = None
    
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
