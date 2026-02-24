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
    
    Args:
        f: A function of one variable
        x: Point at which to compute derivative
        h: Small step size
    
    Returns:
        Approximate derivative at x
    """
    # API hints:
    # - Forward difference formula: (f(x+h) - f(x)) / h
    # - Evaluate f at x and at x+h
    # - Return the difference quotient
    return None


# ============================================================================
# Exercise 2: Central Difference (More Accurate)
# ============================================================================

def central_difference(f, x: float, h: float = 1e-5) -> float:
    """
    Compute derivative using central difference.
    
    This is more accurate than forward difference (O(h²) vs O(h) error).
    """
    # API hints:
    # - Central difference formula: (f(x+h) - f(x-h)) / (2*h)
    # - Evaluate f at x+h and x-h
    # - Divide difference by 2*h
    return None


# ============================================================================
# Exercise 3: Analytical Derivatives
# ============================================================================

def derivative_x_squared(x: float) -> float:
    """
    Analytical derivative of f(x) = x².
    """
    # API hints:
    # - Power rule: d/dx(x^n) = n * x^(n-1)
    # - Apply to x^2
    return None


def derivative_x_cubed(x: float) -> float:
    """
    Analytical derivative of f(x) = x³.
    """
    # API hints:
    # - Power rule: d/dx(x^n) = n * x^(n-1)
    # - Apply to x^3
    return None


def derivative_sin(x: float) -> float:
    """
    Analytical derivative of f(x) = sin(x).
    """
    # API hints:
    # - d/dx(sin(x)) = cos(x)
    # - Use np.cos()
    return None


def derivative_exp(x: float) -> float:
    """
    Analytical derivative of f(x) = eˣ.
    """
    # API hints:
    # - d/dx(e^x) = e^x
    # - Use np.exp()
    return None


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
    # API hints:
    # - Use central_difference(f, x) for numerical derivative
    # - Call df_analytical(x) for analytical derivative
    # - Compute error with abs(numerical - analytical)
    numerical = None
    analytical = None
    error = None
    
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
    """
    # API hints:
    # - Second derivative formula: (f(x+h) - 2*f(x) + f(x-h)) / h²
    # - Evaluate f at three points: x-h, x, x+h
    # - Divide by h squared
    return None


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
