"""
Day 1: Tensor Basics
====================
Estimated time: 1-2 hours
Prerequisites: None (first day!)

Learning objectives:
- Create tensors from various sources
- Understand tensor properties (shape, dtype, device)
- Perform basic tensor indexing and slicing
- Move tensors between CPU and GPU

Hints:
- torch.tensor() creates a tensor from data
- torch.zeros(), torch.ones(), torch.randn() for initialized tensors
- .shape, .dtype, .device to inspect properties
- .to('cuda') or .cuda() to move to GPU
"""

import torch


# ============================================================================
# Exercise 1: Creating Tensors
# ============================================================================

def create_tensors():
    """
    Create various tensors and return them as a dictionary.
    
    Returns:
        dict with keys: 'from_list', 'zeros', 'ones', 'random', 'range'
    """
    # TODO: Create a tensor from a Python list [1, 2, 3, 4, 5]
    from_list = None  # Replace with torch.tensor([1, 2, 3, 4, 5])
    
    # TODO: Create a 3x4 tensor of zeros
    zeros = None  # Replace with torch.zeros(3, 4)
    
    # TODO: Create a 2x3 tensor of ones with dtype float32
    ones = None  # Replace with torch.ones(2, 3, dtype=torch.float32)
    
    # TODO: Create a 5x5 tensor of random normal values
    random = None  # Replace with torch.randn(5, 5)
    
    # TODO: Create a tensor with values 0, 2, 4, 6, 8 (use torch.arange)
    range_tensor = None  # Replace with torch.arange(0, 10, 2)
    
    return {
        'from_list': from_list,
        'zeros': zeros,
        'ones': ones,
        'random': random,
        'range': range_tensor
    }


# ============================================================================
# Exercise 2: Tensor Properties
# ============================================================================

def get_tensor_properties(t: torch.Tensor) -> dict:
    """
    Return properties of the given tensor.
    
    Returns:
        dict with keys: 'shape', 'dtype', 'device', 'ndim', 'numel'
    """
    # TODO: Get the tensor's shape (as a tuple)
    shape = None  # Replace with tuple(t.shape)
    
    # TODO: Get the data type
    dtype = None  # Replace with t.dtype
    
    # TODO: Get the device (cpu or cuda)
    device = None  # Replace with str(t.device)
    
    # TODO: Get number of dimensions
    ndim = None  # Replace with t.ndim
    
    # TODO: Get total number of elements
    numel = None  # Replace with t.numel()
    
    return {
        'shape': shape,
        'dtype': dtype,
        'device': device,
        'ndim': ndim,
        'numel': numel
    }


# ============================================================================
# Exercise 3: Indexing and Slicing
# ============================================================================

def tensor_indexing(t: torch.Tensor) -> dict:
    """
    Given a 2D tensor t, extract various parts.
    
    Args:
        t: A 2D tensor of shape (M, N)
    
    Returns:
        dict with keys: 'first_row', 'last_col', 'top_left_2x2', 'every_other_row'
    """
    # TODO: Get the first row
    first_row = None  # Replace with t[0]
    
    # TODO: Get the last column
    last_col = None  # Replace with t[:, -1]
    
    # TODO: Get the top-left 2x2 submatrix
    top_left_2x2 = None  # Replace with t[:2, :2]
    
    # TODO: Get every other row (0, 2, 4, ...)
    every_other_row = None  # Replace with t[::2]
    
    return {
        'first_row': first_row,
        'last_col': last_col,
        'top_left_2x2': top_left_2x2,
        'every_other_row': every_other_row
    }


# ============================================================================
# Exercise 4: Reshaping
# ============================================================================

def reshape_tensors(t: torch.Tensor) -> dict:
    """
    Reshape a tensor in various ways.
    
    Args:
        t: A tensor with 12 elements
    
    Returns:
        dict with keys: 'as_3x4', 'as_2x6', 'as_flat', 'with_batch_dim'
    """
    # TODO: Reshape to 3x4
    as_3x4 = None  # Replace with t.reshape(3, 4)
    
    # TODO: Reshape to 2x6
    as_2x6 = None  # Replace with t.reshape(2, 6)
    
    # TODO: Flatten to 1D
    as_flat = None  # Replace with t.flatten()
    
    # TODO: Add a batch dimension at the front (1, 12)
    with_batch_dim = None  # Replace with t.unsqueeze(0)
    
    return {
        'as_3x4': as_3x4,
        'as_2x6': as_2x6,
        'as_flat': as_flat,
        'with_batch_dim': with_batch_dim
    }


# ============================================================================
# Exercise 5: Device Management
# ============================================================================

def device_operations():
    """
    Demonstrate moving tensors between devices.
    
    Returns:
        dict with keys: 'cpu_tensor', 'gpu_tensor' (or None if no GPU), 'back_to_cpu'
    """
    # TODO: Create a tensor on CPU
    cpu_tensor = None  # Replace with torch.randn(3, 3)
    
    # TODO: Move to GPU if available, else keep on CPU
    if torch.cuda.is_available():
        gpu_tensor = None  # Replace with cpu_tensor.to('cuda')
    else:
        gpu_tensor = None
    
    # TODO: Move back to CPU
    if gpu_tensor is not None:
        back_to_cpu = None  # Replace with gpu_tensor.to('cpu')
    else:
        back_to_cpu = cpu_tensor
    
    return {
        'cpu_tensor': cpu_tensor,
        'gpu_tensor': gpu_tensor,
        'back_to_cpu': back_to_cpu
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Day 1: Tensor Basics")
    print("=" * 50)
    print("Run test_day01.py to verify your implementations!")
    
    # Quick demo
    print("\nTensor creation demo:")
    x = torch.tensor([1, 2, 3])
    print(f"  x = {x}")
    print(f"  shape: {x.shape}, dtype: {x.dtype}")
