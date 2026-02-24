"""
Day 3: Partial Derivatives and Gradients
========================================
Estimated time: 1-2 hours
Prerequisites: Day 2 (chain rule)

Learning objectives:
- Understand partial derivatives for multivariate functions
- Compute gradients (vectors of partial derivatives)
- Apply partial derivatives numerically
- Understand how gradients point toward steepest ascent
"""

import numpy as np


# ============================================================================
# CONCEPT: Partial Derivatives
# ============================================================================
# For f(x, y), we can take derivatives with respect to each variable:
# - ∂f/∂x: derivative treating y as constant
# - ∂f/∂y: derivative treating x as constant
#
# The gradient is the vector of all partial derivatives: ∇f = [∂f/∂x, ∂f/∂y]
# ============================================================================


# ============================================================================
# Exercise 1: Partial Derivative with respect to x
# ============================================================================

def partial_x_numerical(f, x, y, h=1e-5):
    """
    Compute ∂f/∂x numerically using central difference.
    
    Args:
        f: Function of two variables f(x, y)
        x, y: Point at which to compute partial
        h: Step size
    
    Returns:
        Approximate ∂f/∂x at (x, y)
    """
    # API hints:
    # - Central difference for partial: (f(x+h, y) - f(x-h, y)) / (2*h)
    # - Keep y constant while varying x
    return None


# ============================================================================
# Exercise 2: Partial Derivative with respect to y
# ============================================================================

def partial_y_numerical(f, x, y, h=1e-5):
    """
    Compute ∂f/∂y numerically using central difference.
    """
    # API hints:
    # - Central difference for partial: (f(x, y+h) - f(x, y-h)) / (2*h)
    # - Keep x constant while varying y
    return None


# ============================================================================
# Exercise 3: Gradient (Vector of Partials)
# ============================================================================

def gradient_numerical(f, x, y, h=1e-5):
    """
    Compute the gradient ∇f = [∂f/∂x, ∂f/∂y] numerically.
    
    Returns:
        numpy array [∂f/∂x, ∂f/∂y]
    """
    # API hints:
    # - Use partial_x_numerical() and partial_y_numerical()
    # - Return as np.array([df_dx, df_dy])
    df_dx = partial_x_numerical(f, x, y, h)
    df_dy = partial_y_numerical(f, x, y, h)
    
    return None


# ============================================================================
# Exercise 4: Analytical Gradient of f(x,y) = x² + y²
# ============================================================================

def gradient_x2_plus_y2(x, y):
    """
    Compute gradient of f(x,y) = x² + y².
    
    Returns:
        numpy array [∂f/∂x, ∂f/∂y]
    """
    # API hints:
    # - ∂f/∂x: derivative of x² + y² w.r.t. x (treat y as constant)
    # - ∂f/∂y: derivative of x² + y² w.r.t. y (treat x as constant)
    # - Return as np.array([df_dx, df_dy])
    return None


# ============================================================================
# Exercise 5: Analytical Gradient of f(x,y) = x*y
# ============================================================================

def gradient_xy(x, y):
    """
    Compute gradient of f(x,y) = x * y.
    """
    # API hints:
    # - ∂(x*y)/∂x: derivative treating y as constant
    # - ∂(x*y)/∂y: derivative treating x as constant
    # - Return as np.array([df_dx, df_dy])
    return None


# ============================================================================
# Exercise 6: Analytical Gradient of f(x,y) = sin(x) * cos(y)
# ============================================================================

def gradient_sin_cos(x, y):
    """
    Compute gradient of f(x,y) = sin(x) * cos(y).
    """
    # API hints:
    # - ∂f/∂x: use product rule, d/dx(sin(x)) = cos(x)
    # - ∂f/∂y: use product rule, d/dy(cos(y)) = -sin(y)
    # - Use np.sin(), np.cos()
    # - Return as np.array([df_dx, df_dy])
    df_dx = None
    df_dy = None
    
    return np.array([df_dx, df_dy]) if df_dx is not None else None


# ============================================================================
# Exercise 7: Verify Numerical vs Analytical
# ============================================================================

def verify_gradient():
    """
    Verify numerical gradient matches analytical for f(x,y) = x² + y².
    
    Returns:
        dict with numerical, analytical gradients and whether they match
    """
    f = lambda x, y: x**2 + y**2
    x, y = 3.0, 4.0
    
    # TODO: Compute numerical gradient
    numerical = gradient_numerical(f, x, y)
    
    # TODO: Compute analytical gradient
    analytical = gradient_x2_plus_y2(x, y)
    
    # Check if they match
    match = None
    if numerical is not None and analytical is not None:
        match = np.allclose(numerical, analytical, atol=1e-4)
    
    return {
        'numerical': numerical,
        'analytical': analytical,
        'match': match
    }


# ============================================================================
# Exercise 8: Gradient Descent Step
# ============================================================================

def gradient_descent_step(f, x, y, learning_rate=0.1):
    """
    Perform one step of gradient descent.
    
    Returns:
        tuple (x_new, y_new)
    """
    grad = gradient_numerical(f, x, y)
    
    if grad is None:
        return x, y
    
    # API hints:
    # - Gradient descent: move opposite to gradient direction
    # - x_new = x - learning_rate * grad[0]
    # - y_new = y - learning_rate * grad[1]
    x_new = None
    y_new = None
    
    return x_new, y_new


if __name__ == "__main__":
    print("Day 3: Partial Derivatives and Gradients")
    print("=" * 50)
    
    print("\nVerifying gradient of f(x,y) = x² + y² at (3, 4):")
    result = verify_gradient()
    print(f"  Numerical:  {result['numerical']}")
    print(f"  Analytical: {result['analytical']}")
    print(f"  Match: {result['match']}")
    
    print("\nRun test_day03.py to verify your implementations!")
