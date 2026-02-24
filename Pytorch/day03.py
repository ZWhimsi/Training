"""
Day 3: Autograd - Automatic Differentiation
==========================================
Estimated time: 1-2 hours
Prerequisites: Day 2 (tensor operations)

Learning objectives:
- Understand requires_grad and computational graphs
- Use .backward() to compute gradients
- Access gradients via .grad attribute
- Control gradient computation with torch.no_grad()

This is the magic that makes neural network training work!
"""

import torch


# ============================================================================
# Exercise 1: Basic Gradient Computation
# ============================================================================

def compute_gradient_simple():
    """
    Compute gradient of f(x) = x^2 at x=3.
    
    Returns:
        dict with 'x', 'y', 'gradient' (gradient should be 6.0)
    """
    # API hints:
    # - torch.tensor(value, requires_grad=True) -> tensor that tracks gradients
    # - tensor ** 2 -> element-wise square
    # - tensor.backward() -> compute gradients via backpropagation
    # - tensor.grad -> access computed gradient
    
    # TODO: Create tensor with requires_grad=True
    x = None
    
    # TODO: Compute y = x^2
    y = None
    
    # TODO: Compute gradients
    pass
    
    # TODO: Get gradient
    gradient = None
    
    return {
        'x': x,
        'y': y,
        'gradient': gradient
    }


# ============================================================================
# Exercise 2: Chain Rule in Action
# ============================================================================

def compute_chain_rule():
    """
    Compute gradient of f(x) = (2x + 1)^2 at x=2.
    
    Expected gradient: f'(2) = 4 * (2*2 + 1) = 20.0
    
    Returns:
        dict with 'x', 'y', 'gradient' (gradient should be 20.0)
    """
    # API hints:
    # - torch.tensor(value, requires_grad=True) -> tensor that tracks gradients
    # - Arithmetic operations build computational graph automatically
    # - tensor.backward() -> compute gradients via backpropagation
    # - tensor.grad -> access computed gradient
    
    # TODO: Create x with requires_grad
    x = None
    
    # TODO: Compute y = (2x + 1)^2
    y = None
    
    # TODO: Backward pass
    pass
    
    gradient = None
    
    return {
        'x': x,
        'y': y,
        'gradient': gradient
    }


# ============================================================================
# Exercise 3: Multiple Variables
# ============================================================================

def compute_multi_variable():
    """
    Compute gradients of f(x, y) = x^2 * y + y^3 at (x=2, y=3).
    
    Expected: df/dx = 12, df/dy = 31
    
    Returns:
        dict with 'x', 'y', 'z', 'grad_x', 'grad_y'
    """
    # API hints:
    # - torch.tensor(value, requires_grad=True) -> tensor that tracks gradients
    # - Multiple tensors can have requires_grad=True
    # - tensor.backward() -> computes gradients for all tensors with requires_grad
    # - tensor.grad -> access computed gradient
    
    # TODO: Create x and y with requires_grad
    x = None
    y = None
    
    # TODO: Compute z = x^2 * y + y^3
    z = None
    
    # TODO: Backward pass
    pass
    
    return {
        'x': x,
        'y': y,
        'z': z,
        'grad_x': None,
        'grad_y': None
    }


# ============================================================================
# Exercise 4: Gradient Accumulation
# ============================================================================

def gradient_accumulation():
    """
    Show that gradients accumulate across backward() calls.
    
    Returns:
        dict with gradients after each backward call
        Expected: first=4.0, second=8.0 (accumulated), after_zero=4.0 (reset)
    """
    # API hints:
    # - tensor.grad.item() -> get gradient as Python scalar
    # - tensor.grad.zero_() -> zero the gradient in-place
    # - Gradients accumulate by default (add up across backward calls)
    
    x = torch.tensor(2.0, requires_grad=True)
    
    # First forward/backward
    y1 = x ** 2
    y1.backward()
    grad_after_first = x.grad.item()
    
    # TODO: Second forward/backward (without zeroing grad)
    y2 = x ** 2
    y2.backward()
    grad_after_second = None
    
    # TODO: Zero the gradient
    pass
    
    # Third forward/backward
    y3 = x ** 2
    y3.backward()
    grad_after_zero = None
    
    return {
        'grad_after_first': grad_after_first,
        'grad_after_second': grad_after_second,
        'grad_after_zero': grad_after_zero
    }


# ============================================================================
# Exercise 5: Detaching and No Grad
# ============================================================================

def control_gradient_flow():
    """
    Demonstrate detach() and torch.no_grad().
    
    Returns:
        dict with various results
    """
    # API hints:
    # - tensor.detach() -> returns tensor detached from computation graph
    # - tensor.requires_grad -> boolean indicating if gradients are tracked
    # - torch.no_grad() -> context manager that disables gradient tracking
    
    x = torch.tensor(3.0, requires_grad=True)
    
    # Normal computation
    y = x ** 2
    
    # TODO: Detach y from the graph
    y_detached = None
    
    # TODO: Check if y_detached requires grad
    detached_requires_grad = None
    
    # TODO: Use torch.no_grad() context
    with torch.no_grad():
        z = x ** 2
        no_grad_requires = z.requires_grad  # Should be False
    
    # TODO: After no_grad context
    w = x ** 2
    after_no_grad_requires = None
    
    return {
        'y_detached': y_detached,
        'detached_requires_grad': detached_requires_grad,
        'no_grad_requires': no_grad_requires,
        'after_no_grad_requires': after_no_grad_requires
    }


# ============================================================================
# Exercise 6: Gradient with Tensors
# ============================================================================

def tensor_gradients():
    """
    Compute gradients with multi-dimensional tensors.
    
    Returns:
        dict with tensor and its gradient (gradient should be 2 * x)
    """
    # API hints:
    # - torch.tensor(data, requires_grad=True) -> 2D tensor with gradient tracking
    # - (tensor ** 2).sum() -> sum of squared elements (scalar output)
    # - tensor.backward() -> compute gradients
    # - tensor.grad -> gradient tensor (same shape as input)
    
    # TODO: Create a 2x2 tensor with requires_grad
    x = None
    
    # TODO: Compute sum of squares
    y = None
    
    # TODO: Backward
    pass
    
    return {
        'x': x,
        'y': y,
        'gradient': None
    }


if __name__ == "__main__":
    print("Day 3: Autograd - Automatic Differentiation")
    print("=" * 50)
    
    # Quick demo
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()
    print(f"\nDemo: d/dx(x^2) at x=3 = {x.grad}")
    
    print("\nRun test_day03.py to verify your implementations!")
