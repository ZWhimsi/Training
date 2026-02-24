"""
Day 2: Tensor Operations
========================
Estimated time: 1-2 hours
Prerequisites: Day 1 (tensor basics)

Learning objectives:
- Perform arithmetic operations on tensors
- Understand broadcasting rules
- Use reduction operations (sum, mean, max)
- Apply mathematical functions
"""

import torch


# ============================================================================
# Exercise 1: Basic Arithmetic
# ============================================================================

def arithmetic_ops(a: torch.Tensor, b: torch.Tensor) -> dict:
    """
    Perform basic arithmetic operations.
    
    Returns:
        dict with keys: 'add', 'sub', 'mul', 'div', 'pow'
    """
    # API hints:
    # - a + b or torch.add(a, b) -> element-wise addition
    # - a - b or torch.sub(a, b) -> element-wise subtraction
    # - a * b or torch.mul(a, b) -> element-wise multiplication
    # - a / b or torch.div(a, b) -> element-wise division
    # - a ** b or torch.pow(a, b) -> element-wise power
    
    # TODO: Addition
    add = None
    
    # TODO: Subtraction
    sub = None
    
    # TODO: Element-wise multiplication
    mul = None
    
    # TODO: Element-wise division
    div = None
    
    # TODO: Element-wise power (a^b)
    pow_result = None
    
    return {
        'add': add,
        'sub': sub,
        'mul': mul,
        'div': div,
        'pow': pow_result
    }


# ============================================================================
# Exercise 2: Matrix Operations
# ============================================================================

def matrix_ops(A: torch.Tensor, B: torch.Tensor) -> dict:
    """
    Perform matrix operations.
    
    Args:
        A: Shape [M, K]
        B: Shape [K, N]
    
    Returns:
        dict with keys: 'matmul', 'transpose', 'inner_product'
    """
    # API hints:
    # - A @ B or torch.mm(A, B) -> matrix multiplication
    # - A.T or A.transpose(dim0, dim1) -> transpose
    # - tensor.flatten() -> 1D tensor
    # - torch.dot(a, b) -> dot product of 1D tensors
    
    # TODO: Matrix multiplication
    matmul = None
    
    # TODO: Transpose of A
    transpose = None
    
    # TODO: Inner product (flatten A and compute dot product with itself)
    inner_product = None
    
    return {
        'matmul': matmul,
        'transpose': transpose,
        'inner_product': inner_product
    }


# ============================================================================
# Exercise 3: Broadcasting
# ============================================================================

def broadcasting_examples() -> dict:
    """
    Demonstrate broadcasting rules.
    
    Returns:
        dict with different broadcast examples
    """
    # API hints:
    # - tensor + scalar -> broadcasts scalar to all elements
    # - matrix + row_vec -> broadcasts row across all rows
    # - matrix + col_vec -> broadcasts column across all columns
    # - Broadcasting: smaller tensor is expanded to match larger tensor's shape
    
    # TODO: Add a scalar to a tensor
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    scalar_add = None
    
    # TODO: Add a row vector to a matrix (broadcasts over rows)
    matrix = torch.ones(3, 4)
    row_vec = torch.tensor([1, 2, 3, 4])
    row_broadcast = None
    
    # TODO: Add a column vector to a matrix (broadcasts over columns)
    col_vec = torch.tensor([[1], [2], [3]])  # Shape [3, 1]
    col_broadcast = None
    
    # TODO: Outer product via broadcasting [3, 1] * [1, 4] = [3, 4]
    a = torch.tensor([[1], [2], [3]], dtype=torch.float32)  # [3, 1]
    b = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)    # [1, 4]
    outer = None
    
    return {
        'scalar_add': scalar_add,
        'row_broadcast': row_broadcast,
        'col_broadcast': col_broadcast,
        'outer_product': outer
    }


# ============================================================================
# Exercise 4: Reduction Operations
# ============================================================================

def reductions(t: torch.Tensor) -> dict:
    """
    Perform reduction operations on a 2D tensor.
    
    Returns:
        dict with various reductions
    """
    # API hints:
    # - tensor.sum() -> sum of all elements
    # - tensor.mean() -> mean of all elements
    # - tensor.sum(dim=0) -> sum along dimension 0 (rows)
    # - tensor.sum(dim=1) -> sum along dimension 1 (columns)
    # - tensor.max() -> maximum value
    # - tensor.argmax() -> index of maximum value (flattened)
    
    # TODO: Sum of all elements
    total_sum = None
    
    # TODO: Mean of all elements
    total_mean = None
    
    # TODO: Sum along rows (result shape: [num_cols])
    row_sum = None
    
    # TODO: Sum along columns (result shape: [num_rows])
    col_sum = None
    
    # TODO: Maximum value and its index
    max_val = None
    argmax = None
    
    return {
        'total_sum': total_sum,
        'total_mean': total_mean,
        'row_sum': row_sum,
        'col_sum': col_sum,
        'max_val': max_val,
        'argmax': argmax
    }


# ============================================================================
# Exercise 5: Mathematical Functions
# ============================================================================

def math_functions(t: torch.Tensor) -> dict:
    """
    Apply mathematical functions.
    
    Returns:
        dict with various math function results
    """
    # API hints:
    # - torch.exp(t) -> element-wise exponential
    # - torch.log(t) -> element-wise natural logarithm
    # - torch.abs(t) -> element-wise absolute value
    # - torch.sqrt(t) -> element-wise square root
    # - torch.sin(t) -> element-wise sine
    # - torch.clamp(t, min, max) -> clamp values to range
    
    # TODO: Exponential
    exp = None
    
    # TODO: Natural logarithm (of abs + small epsilon to avoid log(0))
    log = None
    
    # TODO: Square root (of abs)
    sqrt = None
    
    # TODO: Sine
    sin = None
    
    # TODO: Clamp to range [-1, 1]
    clamped = None
    
    return {
        'exp': exp,
        'log': log,
        'sqrt': sqrt,
        'sin': sin,
        'clamped': clamped
    }


if __name__ == "__main__":
    print("Day 2: Tensor Operations")
    print("Run test_day02.py to verify!")
