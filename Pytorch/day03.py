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
        dict with 'x', 'y', 'gradient'
    """
    # TODO: Create tensor with requires_grad=True
    x = None  # Replace: torch.tensor(3.0, requires_grad=True)
    
    # TODO: Compute y = x^2
    y = None  # Replace: x ** 2
    
    # TODO: Compute gradients
    # HINT: y.backward()
    pass  # Replace
    
    # TODO: Get gradient
    gradient = None  # Replace: x.grad
    
    return {
        'x': x,
        'y': y,
        'gradient': gradient  # Should be 6.0 (derivative of x^2 is 2x)
    }


# ============================================================================
# Exercise 2: Chain Rule in Action
# ============================================================================

def compute_chain_rule():
    """
    Compute gradient of f(x) = (2x + 1)^2 at x=2.
    
    f(x) = (2x + 1)^2
    f'(x) = 2 * (2x + 1) * 2 = 4 * (2x + 1)
    f'(2) = 4 * (2*2 + 1) = 4 * 5 = 20
    
    Returns:
        dict with 'x', 'y', 'gradient'
    """
    # TODO: Create x with requires_grad
    x = None  # Replace
    
    # TODO: Compute y = (2x + 1)^2
    y = None  # Replace: (2 * x + 1) ** 2
    
    # TODO: Backward pass
    pass  # Replace
    
    gradient = None  # Replace with x.grad
    
    return {
        'x': x,
        'y': y,
        'gradient': gradient  # Should be 20.0
    }


# ============================================================================
# Exercise 3: Multiple Variables
# ============================================================================

def compute_multi_variable():
    """
    Compute gradients of f(x, y) = x^2 * y + y^3 at (x=2, y=3).
    
    df/dx = 2xy = 2*2*3 = 12
    df/dy = x^2 + 3y^2 = 4 + 27 = 31
    
    Returns:
        dict with 'x', 'y', 'z', 'grad_x', 'grad_y'
    """
    # TODO: Create x and y with requires_grad
    x = None  # Replace
    y = None  # Replace
    
    # TODO: Compute z = x^2 * y + y^3
    z = None  # Replace: x ** 2 * y + y ** 3
    
    # TODO: Backward pass
    pass  # Replace
    
    return {
        'x': x,
        'y': y,
        'z': z,
        'grad_x': None,  # Replace with x.grad (should be 12)
        'grad_y': None   # Replace with y.grad (should be 31)
    }


# ============================================================================
# Exercise 4: Gradient Accumulation
# ============================================================================

def gradient_accumulation():
    """
    Show that gradients accumulate across backward() calls.
    
    Returns:
        dict with gradients after each backward call
    """
    x = torch.tensor(2.0, requires_grad=True)
    
    # First forward/backward
    y1 = x ** 2
    y1.backward()
    grad_after_first = x.grad.item()
    
    # TODO: Second forward/backward (without zeroing grad)
    y2 = x ** 2
    y2.backward()
    grad_after_second = None  # Replace with x.grad.item()
    
    # TODO: Zero the gradient
    # HINT: x.grad.zero_()
    pass  # Replace
    
    # Third forward/backward
    y3 = x ** 2
    y3.backward()
    grad_after_zero = None  # Replace with x.grad.item()
    
    return {
        'grad_after_first': grad_after_first,   # 4.0
        'grad_after_second': grad_after_second, # 8.0 (accumulated!)
        'grad_after_zero': grad_after_zero      # 4.0 (reset)
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
    x = torch.tensor(3.0, requires_grad=True)
    
    # Normal computation
    y = x ** 2
    
    # TODO: Detach y from the graph
    y_detached = None  # Replace: y.detach()
    
    # TODO: Check if y_detached requires grad
    detached_requires_grad = None  # Replace: y_detached.requires_grad
    
    # TODO: Use torch.no_grad() context
    with torch.no_grad():
        z = x ** 2
        no_grad_requires = z.requires_grad  # Should be False
    
    # TODO: After no_grad context
    w = x ** 2
    after_no_grad_requires = None  # Replace: w.requires_grad (should be True)
    
    return {
        'y_detached': y_detached,
        'detached_requires_grad': detached_requires_grad,  # False
        'no_grad_requires': no_grad_requires,              # False
        'after_no_grad_requires': after_no_grad_requires   # True
    }


# ============================================================================
# Exercise 6: Gradient with Tensors
# ============================================================================

def tensor_gradients():
    """
    Compute gradients with multi-dimensional tensors.
    
    Returns:
        dict with tensor and its gradient
    """
    # TODO: Create a 2x2 tensor with requires_grad
    x = None  # Replace: torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    
    # TODO: Compute sum of squares
    y = None  # Replace: (x ** 2).sum()
    
    # TODO: Backward
    pass  # Replace
    
    # Gradient should be 2 * x
    return {
        'x': x,
        'y': y,
        'gradient': None  # Replace with x.grad
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
