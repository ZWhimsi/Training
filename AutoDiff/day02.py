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
    Apply chain rule: dy/dx = df/dg * dg/dx
    
    Args:
        df_dg: Derivative of outer function w.r.t. inner
        dg_dx: Derivative of inner function w.r.t. x
    
    Returns:
        The composed derivative dy/dx
    """
    # TODO: Multiply the derivatives
    # HINT: return df_dg * dg_dx
    return None  # Replace


# ============================================================================
# Exercise 2: Chain Rule with Three Functions
# ============================================================================

def chain_rule_three_functions(df_dg, dg_dh, dh_dx):
    """
    Apply chain rule for y = f(g(h(x))):
    dy/dx = df/dg * dg/dh * dh/dx
    """
    # TODO: Multiply all three derivatives
    return None  # Replace


# ============================================================================
# Exercise 3: Numerical Chain Rule
# ============================================================================

def numerical_chain_rule(f, g, x, h=1e-5):
    """
    Compute d/dx[f(g(x))] numerically.
    
    Compare this with the analytical chain rule!
    """
    # TODO: Compute using central difference on the composition
    # HINT: Let composite(x) = f(g(x))
    # HINT: return (composite(x+h) - composite(x-h)) / (2*h)
    def composite(val):
        return f(g(val))
    
    return None  # Replace


# ============================================================================
# Exercise 4: Analytical vs Numerical
# ============================================================================

def verify_chain_rule():
    """
    Verify chain rule: d/dx[sin(x²)] at x=1
    
    Let u = x², y = sin(u)
    dy/dx = cos(u) * 2x = cos(x²) * 2x
    At x=1: cos(1) * 2 ≈ 1.0806
    
    Returns:
        dict with 'analytical' and 'numerical' derivatives
    """
    x = 1.0
    
    # TODO: Compute analytical derivative
    # dy/dx = cos(x²) * 2x
    analytical = None  # Replace: np.cos(x**2) * 2 * x
    
    # TODO: Compute numerical derivative
    f = np.sin
    g = lambda t: t ** 2
    numerical = numerical_chain_rule(f, g, x)
    
    return {
        'analytical': analytical,
        'numerical': numerical,
        'match': None  # Replace: abs(analytical - numerical) < 1e-6
    }


# ============================================================================
# Exercise 5: Chain Rule for exp(sin(x))
# ============================================================================

def derivative_exp_sin(x):
    """
    Compute d/dx[exp(sin(x))].
    
    Let u = sin(x), y = exp(u)
    dy/dx = exp(u) * cos(x) = exp(sin(x)) * cos(x)
    """
    # TODO: Implement analytical derivative
    return None  # Replace: np.exp(np.sin(x)) * np.cos(x)


# ============================================================================
# Exercise 6: Chain Rule for (x² + 1)³
# ============================================================================

def derivative_polynomial_power(x):
    """
    Compute d/dx[(x² + 1)³].
    
    Let u = x² + 1, y = u³
    dy/dx = 3u² * 2x = 6x(x² + 1)²
    """
    # TODO: Implement analytical derivative
    return None  # Replace: 6 * x * (x**2 + 1)**2


# ============================================================================
# Exercise 7: Why Chain Rule Matters for Neural Networks
# ============================================================================

def neural_network_gradient_intuition():
    """
    In a neural network: output = activation(weight * input + bias)
    
    To learn, we need d(loss)/d(weight).
    
    If loss = L(output) and output = f(w*x + b):
    d(loss)/d(weight) = dL/d(output) * d(output)/d(w)
                      = dL/d(output) * f'(w*x + b) * x
    
    This is the chain rule in action!
    
    Returns:
        Example gradient computation
    """
    # Simple example: output = relu(w*x), loss = (output - target)²
    w = 2.0
    x = 3.0
    target = 5.0
    
    # Forward pass
    pre_activation = w * x  # 6.0
    output = max(0, pre_activation)  # relu: 6.0
    loss = (output - target) ** 2  # (6-5)² = 1
    
    # TODO: Backward pass using chain rule
    # d(loss)/d(output) = 2 * (output - target)
    dloss_doutput = None  # Replace: 2 * (output - target)
    
    # d(output)/d(pre_act) = 1 if pre_act > 0 else 0 (relu derivative)
    doutput_dpre = None  # Replace: 1.0 if pre_activation > 0 else 0.0
    
    # d(pre_act)/d(w) = x
    dpre_dw = None  # Replace: x
    
    # Chain rule: d(loss)/d(w) = d(loss)/d(output) * d(output)/d(pre) * d(pre)/d(w)
    dloss_dw = None  # Replace: dloss_doutput * doutput_dpre * dpre_dw
    
    return {
        'loss': loss,
        'dloss_dw': dloss_dw,  # Should be 2 * 1 * 1 * 3 = 6
    }


if __name__ == "__main__":
    print("Day 2: The Chain Rule")
    print("=" * 50)
    
    print("\nVerifying chain rule for sin(x²) at x=1:")
    result = verify_chain_rule()
    print(f"  Analytical: {result['analytical']}")
    print(f"  Numerical:  {result['numerical']}")
    
    print("\nRun test_day02.py to verify your implementations!")
