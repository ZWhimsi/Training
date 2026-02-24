"""
Day 2: The Chain Rule
====================
Estimated time: 1-2 hours
Prerequisites: Day 1 (derivatives)

Learning objectives:
- Understand the chain rule for composite functions
- Apply chain rule to nested functions
- Implement chain rule numerically
- Build intuition for backpropagation

The chain rule is THE fundamental concept for backpropagation!
"""

import numpy as np


# ============================================================================
# CONCEPT: The Chain Rule
# ============================================================================
# If y = f(g(x)), then dy/dx = dy/dg * dg/dx
#
# In other words: the derivative of a composition is the product of derivatives.
#
# Example: y = sin(x²)
#   Let u = x², y = sin(u)
#   dy/dx = dy/du * du/dx = cos(u) * 2x = cos(x²) * 2x
# ============================================================================


# ============================================================================
# Exercise 1: Simple Chain Rule
# ============================================================================

def chain_rule_two_functions(df_dg, dg_dx):
    """
    Apply chain rule for y = f(g(x)).
    
    Args:
        df_dg: Derivative of outer function w.r.t. inner
        dg_dx: Derivative of inner function w.r.t. x
    
    Returns:
        The composed derivative dy/dx
    """
    # API hints:
    # - Chain rule: dy/dx = df/dg * dg/dx
    # - Multiply the two derivative values
    return None


# ============================================================================
# Exercise 2: Chain Rule with Three Functions
# ============================================================================

def chain_rule_three_functions(df_dg, dg_dh, dh_dx):
    """
    Apply chain rule for y = f(g(h(x))).
    """
    # API hints:
    # - Extended chain rule: dy/dx = df/dg * dg/dh * dh/dx
    # - Multiply all three derivative values
    return None


# ============================================================================
# Exercise 3: Numerical Chain Rule
# ============================================================================

def numerical_chain_rule(f, g, x, h=1e-5):
    """
    Compute d/dx[f(g(x))] numerically.
    
    Compare this with the analytical chain rule!
    """
    # API hints:
    # - Create composite function: composite(val) = f(g(val))
    # - Apply central difference: (composite(x+h) - composite(x-h)) / (2*h)
    def composite(val):
        return f(g(val))
    
    return None


# ============================================================================
# Exercise 4: Analytical vs Numerical
# ============================================================================

def verify_chain_rule():
    """
    Verify chain rule: d/dx[sin(x²)] at x=1.
    
    Returns:
        dict with 'analytical' and 'numerical' derivatives
    """
    x = 1.0
    
    # API hints:
    # - For sin(x²): let u = x², y = sin(u)
    # - dy/dx = cos(u) * du/dx = cos(x²) * 2x
    # - Use np.cos(), np.sin() for trig functions
    # - Use numerical_chain_rule(f, g, x) for numerical result
    analytical = None
    
    f = np.sin
    g = lambda t: t ** 2
    numerical = numerical_chain_rule(f, g, x)
    
    return {
        'analytical': analytical,
        'numerical': numerical,
        'match': None
    }


# ============================================================================
# Exercise 5: Chain Rule for exp(sin(x))
# ============================================================================

def derivative_exp_sin(x):
    """
    Compute d/dx[exp(sin(x))].
    """
    # API hints:
    # - Let u = sin(x), y = exp(u)
    # - Chain rule: dy/dx = dy/du * du/dx
    # - dy/du = exp(u), du/dx = cos(x)
    # - Use np.exp(), np.sin(), np.cos()
    return None


# ============================================================================
# Exercise 6: Chain Rule for (x² + 1)³
# ============================================================================

def derivative_polynomial_power(x):
    """
    Compute d/dx[(x² + 1)³].
    """
    # API hints:
    # - Let u = x² + 1, y = u³
    # - Chain rule: dy/dx = dy/du * du/dx
    # - dy/du = 3*u², du/dx = 2x
    # - Simplify the expression
    return None


# ============================================================================
# Exercise 7: Why Chain Rule Matters for Neural Networks
# ============================================================================

def neural_network_gradient_intuition():
    """
    In a neural network: output = activation(weight * input + bias)
    
    To learn, we need d(loss)/d(weight).
    This is the chain rule in action!
    
    Returns:
        Example gradient computation
    """
    # Simple example: output = relu(w*x), loss = (output - target)²
    w = 2.0
    x = 3.0
    target = 5.0
    
    # Forward pass
    pre_activation = w * x
    output = max(0, pre_activation)  # relu
    loss = (output - target) ** 2
    
    # API hints:
    # - dloss_doutput: derivative of squared error = 2 * (output - target)
    # - doutput_dpre: relu derivative = 1 if pre_activation > 0 else 0
    # - dpre_dw: derivative of w*x w.r.t. w = x
    # - Chain rule: dloss_dw = dloss_doutput * doutput_dpre * dpre_dw
    dloss_doutput = None
    doutput_dpre = None
    dpre_dw = None
    dloss_dw = None
    
    return {
        'loss': loss,
        'dloss_dw': dloss_dw,
    }


if __name__ == "__main__":
    print("Day 2: The Chain Rule")
    print("=" * 50)
    
    print("\nVerifying chain rule for sin(x²) at x=1:")
    result = verify_chain_rule()
    print(f"  Analytical: {result['analytical']}")
    print(f"  Numerical:  {result['numerical']}")
    
    print("\nRun test_day02.py to verify your implementations!")
