"""
Day 5: Forward Mode vs Reverse Mode AD
======================================
Estimated time: 1-2 hours
Prerequisites: Day 4 (computational graphs)

Learning objectives:
- Understand forward mode automatic differentiation
- Understand reverse mode (backpropagation)
- Compare complexity of both modes
- Know when to use each mode
"""

import numpy as np


# ============================================================================
# CONCEPT: Forward Mode vs Reverse Mode
# ============================================================================
#
# Forward mode: Propagates derivatives FROM inputs TO outputs
#   - Computes ∂output/∂x_i for ONE input x_i at a time
#   - Good when: few inputs, many outputs
#   - Complexity: O(n) passes for n inputs
#
# Reverse mode: Propagates derivatives FROM outputs TO inputs
#   - Computes ∂y/∂x_i for ALL inputs at once
#   - Good when: many inputs, few outputs (like neural networks!)
#   - Complexity: O(m) passes for m outputs (usually m=1 for loss)
#
# Neural networks have millions of parameters (inputs) but one loss (output)
# → Reverse mode is essential for deep learning!
# ============================================================================


class DualNumber:
    """
    Dual number for forward mode AD.
    
    A dual number is: a + b*ε where ε² = 0
    Used to track both value and derivative simultaneously.
    """
    
    def __init__(self, value, derivative=0.0):
        self.value = value
        self.derivative = derivative
    
    def __repr__(self):
        return f"Dual({self.value}, d={self.derivative})"
    
    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(
                self.value + other.value,
                self.derivative + other.derivative
            )
        return DualNumber(self.value + other, self.derivative)
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if isinstance(other, DualNumber):
            # (a + b*ε) * (c + d*ε) = ac + (ad + bc)*ε (since ε² = 0)
            return DualNumber(
                self.value * other.value,
                self.value * other.derivative + self.derivative * other.value
            )
        return DualNumber(self.value * other, self.derivative * other)
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, n):
        # d/dx(x^n) = n * x^(n-1)
        return DualNumber(
            self.value ** n,
            n * (self.value ** (n-1)) * self.derivative
        )


# ============================================================================
# Exercise 1: Forward Mode with Dual Numbers
# ============================================================================

def forward_mode_x_squared(x):
    """
    Compute f(x) = x² and its derivative using forward mode.
    
    To find df/dx: set x = Dual(x_value, 1)
    The output's derivative will be df/dx.
    
    Returns:
        (value, derivative)
    """
    # TODO: Create dual number with derivative 1 (we want df/dx)
    x_dual = DualNumber(x, 1.0)
    
    # TODO: Compute x²
    result = x_dual ** 2
    
    # TODO: Return value and derivative
    return None, None  # Replace: (result.value, result.derivative)


# ============================================================================
# Exercise 2: Forward Mode for Polynomial
# ============================================================================

def forward_mode_polynomial(x):
    """
    Compute f(x) = x³ + 2x² + 3x + 4 and its derivative.
    
    Returns:
        (value, derivative)
    """
    # TODO: Create dual number
    x_dual = None  # Replace: DualNumber(x, 1.0)
    
    # TODO: Compute polynomial
    result = None  # Replace: x_dual**3 + 2*x_dual**2 + 3*x_dual + 4
    
    return None, None  # Replace: (result.value, result.derivative)


# ============================================================================
# Exercise 3: Forward Mode for Two Variables
# ============================================================================

def forward_mode_two_vars(x, y):
    """
    Compute f(x,y) = x*y + x² and ∂f/∂x using forward mode.
    
    To get ∂f/∂x: set dx=1, dy=0
    
    Returns:
        (value, df_dx)
    """
    # TODO: Create dual numbers (dx=1, dy=0 for ∂f/∂x)
    x_dual = None  # Replace: DualNumber(x, 1.0)  # dx = 1
    y_dual = None  # Replace: DualNumber(y, 0.0)  # dy = 0
    
    # TODO: Compute f = x*y + x²
    result = None  # Replace: x_dual * y_dual + x_dual ** 2
    
    return None, None  # Replace: (result.value, result.derivative)


# ============================================================================
# Exercise 4: Reverse Mode Manually
# ============================================================================

def reverse_mode_manual(x, y):
    """
    Compute gradients of f(x,y) = (x + y) * x using reverse mode.
    
    Forward pass: track values
    Backward pass: propagate gradients from output to inputs
    
    Returns:
        dict with 'value', 'df_dx', 'df_dy'
    """
    # Forward pass
    a = x + y           # a = x + y
    f = a * x           # f = a * x = (x+y) * x = x² + xy
    
    # Backward pass
    # Start with df/df = 1
    df_df = 1.0
    
    # f = a * x
    # df/da = x, df/dx (through this op) = a
    # TODO: Compute gradient flowing to a and direct gradient to x
    df_da = None  # Replace: x * df_df
    df_dx_from_f = None  # Replace: a * df_df
    
    # a = x + y
    # da/dx = 1, da/dy = 1
    # TODO: Compute gradients flowing through a
    df_dx_from_a = None  # Replace: 1 * df_da
    df_dy = None  # Replace: 1 * df_da
    
    # TODO: Total df/dx (sum of all paths)
    df_dx = None  # Replace: df_dx_from_f + df_dx_from_a
    
    return {
        'value': f,
        'df_dx': df_dx,  # Should be 2x + y
        'df_dy': df_dy   # Should be x
    }


# ============================================================================
# Exercise 5: Complexity Comparison
# ============================================================================

def complexity_analysis():
    """
    Compare forward vs reverse mode complexity.
    
    For f: R^n → R^m
    - Forward mode: O(n) passes (one per input)
    - Reverse mode: O(m) passes (one per output)
    
    Returns:
        dict with analysis for different scenarios
    """
    scenarios = {
        # (n_inputs, n_outputs): which mode is better
        'neural_net_loss': {
            'n_inputs': 1000000,  # Parameters
            'n_outputs': 1,       # Scalar loss
            'better_mode': None,  # Replace: 'reverse' (O(1) vs O(1M))
            'forward_passes': None,  # Replace: 1000000
            'reverse_passes': None,  # Replace: 1
        },
        'jacobian_narrow': {
            'n_inputs': 3,
            'n_outputs': 1000,
            'better_mode': None,  # Replace: 'forward' (O(3) vs O(1000))
            'forward_passes': None,  # Replace: 3
            'reverse_passes': None,  # Replace: 1000
        },
        'square_jacobian': {
            'n_inputs': 100,
            'n_outputs': 100,
            'better_mode': None,  # Replace: 'either' (O(100) both)
            'forward_passes': None,  # Replace: 100
            'reverse_passes': None,  # Replace: 100
        },
    }
    
    return scenarios


if __name__ == "__main__":
    print("Day 5: Forward Mode vs Reverse Mode AD")
    print("=" * 50)
    
    print("\nForward mode for f(x) = x² at x=3:")
    val, deriv = forward_mode_x_squared(3.0)
    print(f"  Value: {val}, Derivative: {deriv}")
    
    print("\nReverse mode for f(x,y) = (x+y)*x at x=2, y=3:")
    result = reverse_mode_manual(2.0, 3.0)
    print(f"  Value: {result['value']}, df/dx: {result['df_dx']}, df/dy: {result['df_dy']}")
    
    print("\nRun test_day05.py to verify your implementations!")
