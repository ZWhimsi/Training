"""
Day 1: Introduction to Derivatives
==================================
Estimated time: 1-2 hours
Prerequisites: None (first day!)

Learning objectives:
- Review single-variable derivatives
- Implement numerical differentiation
- Understand the limit definition
- Compare numerical vs analytical derivatives

Hints:
- Derivative definition: f'(x) = lim(h→0) [f(x+h) - f(x)] / h
- Use small h for numerical approximation (e.g., 1e-5)
- Central difference is more accurate: [f(x+h) - f(x-h)] / (2h)
"""

import numpy as np


# ============================================================================
# CONCEPT: What is a Derivative?
# ============================================================================
# The derivative measures how a function changes as its input changes.
# Geometrically: the slope of the tangent line at a point.
# 
# Examples:
# - f(x) = x²  →  f'(x) = 2x
# - f(x) = sin(x)  →  f'(x) = cos(x)
# - f(x) = eˣ  →  f'(x) = eˣ
# ============================================================================


# ============================================================================
# Exercise 1: Numerical Derivative (Forward Difference)
# ============================================================================

def forward_difference(f, x: float, h: float = 1e-5) -> float:
    """
    Compute derivative using forward difference.
    
    f'(x) ≈ [f(x + h) - f(x)] / h
    
    Args:
        f: A function of one variable
        x: Point at which to compute derivative
        h: Small step size
    
    Returns:
        Approximate derivative at x
    """
    # TODO: Implement forward difference
    # HINT: return (f(x + h) - f(x)) / h
    return None  # Replace


# ============================================================================
# Exercise 2: Central Difference (More Accurate)
# ============================================================================

def central_difference(f, x: float, h: float = 1e-5) -> float:
    """
    Compute derivative using central difference.
    
    f'(x) ≈ [f(x + h) - f(x - h)] / (2h)
    
    This is more accurate than forward difference (O(h²) vs O(h) error).
    """
    # TODO: Implement central difference
    # HINT: return (f(x + h) - f(x - h)) / (2 * h)
    return None  # Replace


# ============================================================================
# Exercise 3: Analytical Derivatives
# ============================================================================

def derivative_x_squared(x: float) -> float:
    """
    Analytical derivative of f(x) = x²
    
    f'(x) = 2x
    """
    # TODO: Return the analytical derivative
    return None  # Replace with 2 * x


def derivative_x_cubed(x: float) -> float:
    """
    Analytical derivative of f(x) = x³
    
    f'(x) = 3x²
    """
    # TODO: Return the analytical derivative
    return None  # Replace with 3 * x**2


def derivative_sin(x: float) -> float:
    """
    Analytical derivative of f(x) = sin(x)
    
    f'(x) = cos(x)
    """
    # TODO: Return the analytical derivative
    return None  # Replace with np.cos(x)


def derivative_exp(x: float) -> float:
    """
    Analytical derivative of f(x) = eˣ
    
    f'(x) = eˣ
    """
    # TODO: Return the analytical derivative
    return None  # Replace with np.exp(x)


# ============================================================================
# Exercise 4: Comparing Numerical and Analytical
# ============================================================================

def compare_derivatives(f, df_analytical, x: float) -> dict:
    """
    Compare numerical and analytical derivatives.
    
    Args:
        f: The original function
        df_analytical: The analytical derivative function
        x: Point at which to compare
    
    Returns:
        dict with 'numerical', 'analytical', 'error'
    """
    # TODO: Compute numerical derivative using central difference
    numerical = None  # Replace with central_difference(f, x)
    
    # TODO: Compute analytical derivative
    analytical = None  # Replace with df_analytical(x)
    
    # TODO: Compute absolute error
    error = None  # Replace with abs(numerical - analytical)
    
    return {
        'numerical': numerical,
        'analytical': analytical,
        'error': error
    }


# ============================================================================
# Exercise 5: Second Derivative
# ============================================================================

def second_derivative(f, x: float, h: float = 1e-4) -> float:
    """
    Compute second derivative numerically.
    
    f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
    """
    # TODO: Implement second derivative
    # HINT: return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
    return None  # Replace


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Day 1: Introduction to Derivatives")
    print("=" * 50)
    
    # Demo: derivative of x² at x=3 should be 6
    f = lambda x: x ** 2
    x = 3.0
    
    print(f"\nf(x) = x² at x = {x}")
    print(f"  Numerical (forward): {forward_difference(f, x)}")
    print(f"  Numerical (central): {central_difference(f, x)}")
    print(f"  Analytical: {derivative_x_squared(x)}")
    
    print("\nRun test_day01.py to verify your implementations!")
