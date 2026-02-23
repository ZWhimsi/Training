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
    # TODO: Implement partial with respect to x (y held constant)
    # HINT: (f(x+h, y) - f(x-h, y)) / (2*h)
    return None  # Replace


# ============================================================================
# Exercise 2: Partial Derivative with respect to y
# ============================================================================

def partial_y_numerical(f, x, y, h=1e-5):
    """
    Compute ∂f/∂y numerically using central difference.
    """
    # TODO: Implement partial with respect to y (x held constant)
    # HINT: (f(x, y+h) - f(x, y-h)) / (2*h)
    return None  # Replace


# ============================================================================
# Exercise 3: Gradient (Vector of Partials)
# ============================================================================

def gradient_numerical(f, x, y, h=1e-5):
    """
    Compute the gradient ∇f = [∂f/∂x, ∂f/∂y] numerically.
    
    Returns:
        numpy array [∂f/∂x, ∂f/∂y]
    """
    # TODO: Compute both partials and return as array
    df_dx = partial_x_numerical(f, x, y, h)
    df_dy = partial_y_numerical(f, x, y, h)
    
    return None  # Replace: np.array([df_dx, df_dy])


# ============================================================================
# Exercise 4: Analytical Gradient of f(x,y) = x² + y²
# ============================================================================

def gradient_x2_plus_y2(x, y):
    """
    Compute gradient of f(x,y) = x² + y².
    
    ∂f/∂x = 2x
    ∂f/∂y = 2y
    
    Returns:
        numpy array [2x, 2y]
    """
    # TODO: Return the analytical gradient
    return None  # Replace: np.array([2*x, 2*y])


# ============================================================================
# Exercise 5: Analytical Gradient of f(x,y) = x*y
# ============================================================================

def gradient_xy(x, y):
    """
    Compute gradient of f(x,y) = x * y.
    
    ∂f/∂x = y
    ∂f/∂y = x
    """
    # TODO: Return the analytical gradient
    return None  # Replace: np.array([y, x])


# ============================================================================
# Exercise 6: Analytical Gradient of f(x,y) = sin(x) * cos(y)
# ============================================================================

def gradient_sin_cos(x, y):
    """
    Compute gradient of f(x,y) = sin(x) * cos(y).
    
    ∂f/∂x = cos(x) * cos(y)
    ∂f/∂y = sin(x) * (-sin(y)) = -sin(x) * sin(y)
    """
    # TODO: Return the analytical gradient
    df_dx = None  # Replace: np.cos(x) * np.cos(y)
    df_dy = None  # Replace: -np.sin(x) * np.sin(y)
    
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
    
    x_new = x - learning_rate * ∂f/∂x
    y_new = y - learning_rate * ∂f/∂y
    
    Returns:
        tuple (x_new, y_new)
    """
    grad = gradient_numerical(f, x, y)
    
    if grad is None:
        return x, y
    
    # TODO: Update x and y in the direction of negative gradient
    x_new = None  # Replace: x - learning_rate * grad[0]
    y_new = None  # Replace: y - learning_rate * grad[1]
    
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
