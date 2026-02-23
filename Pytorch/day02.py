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
    # TODO: Addition
    add = None  # Replace: a + b
    
    # TODO: Subtraction
    sub = None  # Replace: a - b
    
    # TODO: Element-wise multiplication
    mul = None  # Replace: a * b
    
    # TODO: Element-wise division
    div = None  # Replace: a / b
    
    # TODO: Element-wise power (a^b)
    pow_result = None  # Replace: a ** b or torch.pow(a, b)
    
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
    # TODO: Matrix multiplication
    matmul = None  # Replace: A @ B or torch.mm(A, B)
    
    # TODO: Transpose of A
    transpose = None  # Replace: A.T or A.transpose(0, 1)
    
    # TODO: Inner product (flatten and dot)
    # Compute A.flatten() dot A.flatten()
    inner_product = None  # Replace: torch.dot(A.flatten(), A.flatten())
    
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
    # TODO: Add a scalar to a tensor
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    scalar_add = None  # Replace: tensor + 5
    
    # TODO: Add a row vector to a matrix (broadcasts over rows)
    matrix = torch.ones(3, 4)
    row_vec = torch.tensor([1, 2, 3, 4])
    row_broadcast = None  # Replace: matrix + row_vec
    
    # TODO: Add a column vector to a matrix (broadcasts over columns)
    col_vec = torch.tensor([[1], [2], [3]])  # Shape [3, 1]
    col_broadcast = None  # Replace: matrix + col_vec
    
    # TODO: Outer product via broadcasting
    # [3, 1] * [1, 4] = [3, 4]
    a = torch.tensor([[1], [2], [3]], dtype=torch.float32)  # [3, 1]
    b = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)    # [1, 4]
    outer = None  # Replace: a * b
    
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
    # TODO: Sum of all elements
    total_sum = None  # Replace: t.sum()
    
    # TODO: Mean of all elements
    total_mean = None  # Replace: t.mean()
    
    # TODO: Sum along rows (result shape: [num_cols])
    row_sum = None  # Replace: t.sum(dim=0)
    
    # TODO: Sum along columns (result shape: [num_rows])
    col_sum = None  # Replace: t.sum(dim=1)
    
    # TODO: Maximum value and its index
    max_val = None  # Replace: t.max()
    argmax = None   # Replace: t.argmax()
    
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
    # TODO: Exponential
    exp = None  # Replace: torch.exp(t)
    
    # TODO: Natural logarithm (of abs + small epsilon to avoid log(0))
    log = None  # Replace: torch.log(torch.abs(t) + 1e-8)
    
    # TODO: Square root (of abs)
    sqrt = None  # Replace: torch.sqrt(torch.abs(t))
    
    # TODO: Sine
    sin = None  # Replace: torch.sin(t)
    
    # TODO: Clamp to range [-1, 1]
    clamped = None  # Replace: torch.clamp(t, -1, 1)
    
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
