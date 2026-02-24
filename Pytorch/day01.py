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
    # API hints:
    # - torch.tensor(data) -> creates tensor from Python list/array
    # - torch.zeros(*size) -> tensor filled with zeros
    # - torch.ones(*size, dtype=dtype) -> tensor filled with ones
    # - torch.randn(*size) -> tensor with random normal values
    # - torch.arange(start, end, step) -> 1D tensor with range of values
    
    # TODO: Create a tensor from a Python list [1, 2, 3, 4, 5]
    from_list = None
    
    # TODO: Create a 3x4 tensor of zeros
    zeros = None
    
    # TODO: Create a 2x3 tensor of ones with dtype float32
    ones = None
    
    # TODO: Create a 5x5 tensor of random normal values
    random = None
    
    # TODO: Create a tensor with values 0, 2, 4, 6, 8
    range_tensor = None
    
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
    # API hints:
    # - tensor.shape -> torch.Size object (convert with tuple())
    # - tensor.dtype -> data type (e.g., torch.float32)
    # - tensor.device -> device location (convert with str())
    # - tensor.ndim -> number of dimensions (int)
    # - tensor.numel() -> total number of elements (int)
    
    # TODO: Get the tensor's shape (as a tuple)
    shape = None
    
    # TODO: Get the data type
    dtype = None
    
    # TODO: Get the device (cpu or cuda)
    device = None
    
    # TODO: Get number of dimensions
    ndim = None
    
    # TODO: Get total number of elements
    numel = None
    
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
    # API hints:
    # - tensor[i] -> select row i
    # - tensor[:, j] -> select column j
    # - tensor[start:end, start:end] -> slice rows and columns
    # - tensor[::step] -> select every step-th row
    # - negative indices (-1) select from end
    
    # TODO: Get the first row
    first_row = None
    
    # TODO: Get the last column
    last_col = None
    
    # TODO: Get the top-left 2x2 submatrix
    top_left_2x2 = None
    
    # TODO: Get every other row (0, 2, 4, ...)
    every_other_row = None
    
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
    # API hints:
    # - tensor.reshape(*shape) -> returns tensor with new shape
    # - tensor.flatten() -> returns 1D tensor
    # - tensor.unsqueeze(dim) -> adds dimension at position dim
    
    # TODO: Reshape to 3x4
    as_3x4 = None
    
    # TODO: Reshape to 2x6
    as_2x6 = None
    
    # TODO: Flatten to 1D
    as_flat = None
    
    # TODO: Add a batch dimension at the front (1, 12)
    with_batch_dim = None
    
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
    # API hints:
    # - torch.randn(*size) -> random tensor on CPU by default
    # - torch.cuda.is_available() -> check if GPU is available
    # - tensor.to(device) -> move tensor to device ('cuda' or 'cpu')
    # - tensor.cuda() -> shorthand for tensor.to('cuda')
    # - tensor.cpu() -> shorthand for tensor.to('cpu')
    
    # TODO: Create a tensor on CPU
    cpu_tensor = None
    
    # TODO: Move to GPU if available, else keep on CPU
    if torch.cuda.is_available():
        gpu_tensor = None
    else:
        gpu_tensor = None
    
    # TODO: Move back to CPU
    if gpu_tensor is not None:
        back_to_cpu = None
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
