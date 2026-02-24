"""
Day 10: Numerical Gradient Checking
====================================
Estimated time: 1.5-2 hours
Prerequisites: Days 8-9 (backward passes for operations and activations)

Learning objectives:
- Understand finite difference approximation of derivatives
- Implement forward difference gradient checking
- Implement central difference gradient checking
- Compare analytical vs numerical gradients
- Debug autodiff implementations using gradient checking

Mathematical background:
- Forward difference: f'(x) ≈ [f(x + h) - f(x)] / h
- Central difference: f'(x) ≈ [f(x + h) - f(x - h)] / (2h)
- Central difference is O(h²) accurate, forward is O(h)

Why gradient checking?
- Verify correctness of backward pass implementations
- Debug complex computational graphs
- Catch subtle bugs in gradient computations
"""

import math


class Value:
    """Complete Value class for gradient checking."""
    
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        
        for v in reversed(topo):
            v._backward()
    
    # All operations with backward passes
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)
    
    def __rtruediv__(self, other):
        return Value(other) / self
    
    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')
        
        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out


# ============================================================================
# Exercise 1: Forward Difference
# ============================================================================

def forward_difference(f, x, h=1e-5):
    """
    Compute numerical gradient using forward difference.
    
    This is O(h) accurate.
    
    Args:
        f: Function that takes a float and returns a float
        x: Point at which to compute gradient
        h: Step size (default 1e-5)
    
    Returns:
        Approximate gradient at x
    """
    # API hints:
    # - Forward difference: (f(x + h) - f(x)) / h
    # - Evaluate f at x and x+h
    return None


# ============================================================================
# Exercise 2: Central Difference
# ============================================================================

def central_difference(f, x, h=1e-5):
    """
    Compute numerical gradient using central difference.
    
    This is O(h²) accurate - more accurate than forward difference!
    
    Args:
        f: Function that takes a float and returns a float
        x: Point at which to compute gradient
        h: Step size (default 1e-5)
    
    Returns:
        Approximate gradient at x
    """
    # API hints:
    # - Central difference: (f(x + h) - f(x - h)) / (2 * h)
    # - Evaluate f at x+h and x-h
    return None


# ============================================================================
# Exercise 3: Compare Accuracy
# ============================================================================

def compare_difference_methods(f, x, true_grad, h_values=None):
    """
    Compare forward and central difference accuracy.
    
    Args:
        f: Function to differentiate
        x: Point to evaluate
        true_grad: Exact analytical gradient
        h_values: List of step sizes to try
    
    Returns:
        dict with errors for each method and step size
    """
    if h_values is None:
        h_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    
    results = []
    
    for h in h_values:
        # API hints:
        # - Use forward_difference(f, x, h) and central_difference(f, x, h)
        # - Compute errors: abs(computed - true_grad)
        fwd = forward_difference(f, x, h) if forward_difference(f, x, h) else 0
        ctr = central_difference(f, x, h) if central_difference(f, x, h) else 0
        
        fwd_error = None
        ctr_error = None
        
        results.append({
            'h': h,
            'forward': fwd,
            'central': ctr,
            'forward_error': fwd_error,
            'central_error': ctr_error
        })
    
    return results


# ============================================================================
# Exercise 4: Gradient Check for Single Variable
# ============================================================================

def gradient_check_single(expression_fn, x_val, h=1e-5, tol=1e-4):
    """
    Check if analytical gradient matches numerical gradient.
    
    Args:
        expression_fn: Function that takes a Value and returns a Value
        x_val: Float value at which to check gradient
        h: Step size for numerical gradient
        tol: Tolerance for comparison
    
    Returns:
        dict with analytical gradient, numerical gradient, and whether they match
    """
    # TODO: Compute analytical gradient via backprop
    x = Value(x_val)
    y = expression_fn(x)
    y.backward()
    analytical = x.grad
    
    # API hints:
    # - Create wrapper function: f(val) = expression_fn(Value(val)).data
    # - Use central_difference(f, x_val, h) for numerical gradient
    # - Check match: abs(analytical - numerical) < tol
    def f(val):
        return expression_fn(Value(val)).data
    
    numerical = None
    matches = None
    
    return {
        'analytical': analytical,
        'numerical': numerical,
        'matches': matches,
        'difference': abs(analytical - numerical) if numerical else None
    }


# ============================================================================
# Exercise 5: Gradient Check for Multiple Variables
# ============================================================================

def gradient_check_multi(expression_fn, values_dict, h=1e-5, tol=1e-4):
    """
    Check gradients for an expression with multiple variables.
    
    Args:
        expression_fn: Function that takes a dict of Values and returns a Value
        values_dict: Dict mapping names to float values
        h: Step size
        tol: Tolerance
    
    Returns:
        dict mapping variable names to gradient check results
    """
    results = {}
    
    for var_name, var_val in values_dict.items():
        # TODO: Compute analytical gradient
        values = {k: Value(v) for k, v in values_dict.items()}
        y = expression_fn(values)
        y.backward()
        analytical = values[var_name].grad
        
        # API hints:
        # - Create wrapper that perturbs one variable at a time
        # - Use central_difference(f, var_val, h) for numerical gradient
        # - Check match: abs(analytical - numerical) < tol
        def f(val):
            perturbed = values_dict.copy()
            perturbed[var_name] = val
            values_temp = {k: Value(v) for k, v in perturbed.items()}
            return expression_fn(values_temp).data
        
        numerical = None
        matches = None
        
        results[var_name] = {
            'analytical': analytical,
            'numerical': numerical,
            'matches': matches
        }
    
    return results


# ============================================================================
# Exercise 6: Relative Error Check
# ============================================================================

def relative_error(analytical, numerical):
    """
    Compute relative error between analytical and numerical gradients.
    
    Using max with 1e-8 prevents division by zero.
    
    Args:
        analytical: Analytical gradient
        numerical: Numerical gradient
    
    Returns:
        Relative error (0 = perfect match)
    """
    # API hints:
    # - Formula: |a - n| / max(|a|, |n|, 1e-8)
    # - Use abs() for absolute values
    # - Use max() to get denominator, include 1e-8 to prevent div by zero
    return None


def gradient_check_relative(expression_fn, x_val, h=1e-5, tol=1e-5):
    """
    Gradient check using relative error (better for varying magnitudes).
    
    Args:
        expression_fn: Function that takes a Value and returns a Value
        x_val: Value at which to check
        h: Step size
        tol: Relative error tolerance
    
    Returns:
        dict with gradients and relative error
    """
    # Analytical
    x = Value(x_val)
    y = expression_fn(x)
    y.backward()
    analytical = x.grad
    
    # Numerical
    def f(val):
        return expression_fn(Value(val)).data
    
    numerical = central_difference(f, x_val, h) if central_difference(f, x_val, h) else 0
    
    # TODO: Compute relative error
    rel_error = relative_error(analytical, numerical) if relative_error(analytical, numerical) else float('inf')
    
    return {
        'analytical': analytical,
        'numerical': numerical,
        'relative_error': rel_error,
        'passes': rel_error < tol if rel_error else False
    }


# ============================================================================
# Exercise 7: Comprehensive Gradient Checker
# ============================================================================

def comprehensive_gradient_check(expression_fn, x_val, name=""):
    """
    Run comprehensive gradient checks with multiple step sizes.
    
    Args:
        expression_fn: Function to check
        x_val: Input value
        name: Name of the function being tested
    
    Returns:
        dict with all check results
    """
    h_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    
    results = {
        'name': name,
        'x': x_val,
        'checks': []
    }
    
    for h in h_values:
        check = gradient_check_single(expression_fn, x_val, h=h)
        check['h'] = h
        results['checks'].append(check)
    
    # Overall pass if majority of step sizes pass
    passing = sum(1 for c in results['checks'] if c.get('matches', False))
    results['overall_pass'] = passing >= len(h_values) // 2
    
    return results


# ============================================================================
# Test Functions for Gradient Checking
# ============================================================================

def test_functions():
    """
    Collection of test functions and their analytical derivatives.
    
    Returns:
        List of (name, function, derivative_function)
    """
    return [
        ('x^2', lambda x: x ** 2, lambda x: 2 * x),
        ('x^3', lambda x: x ** 3, lambda x: 3 * x ** 2),
        ('sin(x)', lambda x: Value(math.sin(x.data)) if isinstance(x, Value) else math.sin(x), 
         lambda x: math.cos(x)),
        ('sigmoid(x)', lambda x: x.sigmoid() if isinstance(x, Value) else 1/(1+math.exp(-x)), 
         lambda x: (1/(1+math.exp(-x))) * (1 - 1/(1+math.exp(-x)))),
        ('tanh(x)', lambda x: x.tanh() if isinstance(x, Value) else math.tanh(x),
         lambda x: 1 - math.tanh(x)**2),
    ]


if __name__ == "__main__":
    print("Day 10: Numerical Gradient Checking")
    print("=" * 60)
    
    # Test central vs forward difference
    print("\nExercise 1-2: Forward vs Central Difference")
    print("-" * 60)
    
    def f_squared(x):
        return x ** 2
    
    x_test = 3.0
    true_grad = 6.0  # d/dx(x^2) = 2x = 6 at x=3
    
    fwd = forward_difference(f_squared, x_test)
    ctr = central_difference(f_squared, x_test)
    
    print(f"f(x) = x² at x = {x_test}")
    print(f"True gradient: {true_grad}")
    print(f"Forward difference: {fwd}")
    print(f"Central difference: {ctr}")
    
    # Test with Value class
    print("\nExercise 4: Gradient Check Single Variable")
    print("-" * 60)
    
    def simple_expr(x):
        return x ** 2 + 2 * x + 1
    
    result = gradient_check_single(simple_expr, 2.0)
    print(f"f(x) = x² + 2x + 1 at x = 2")
    print(f"Analytical: {result['analytical']}")
    print(f"Numerical: {result['numerical']}")
    print(f"Match: {result['matches']}")
    
    # Test multi-variable
    print("\nExercise 5: Gradient Check Multiple Variables")
    print("-" * 60)
    
    def multi_expr(values):
        x, y = values['x'], values['y']
        return x * y + x ** 2
    
    multi_result = gradient_check_multi(multi_expr, {'x': 2.0, 'y': 3.0})
    print("f(x, y) = x*y + x² at x=2, y=3")
    for var, data in multi_result.items():
        print(f"  d f/d{var}: analytical={data['analytical']}, numerical={data['numerical']}")
    
    # Comprehensive check on activations
    print("\nExercise 7: Comprehensive Checks")
    print("-" * 60)
    
    for name, func, _ in test_functions()[:2]:
        result = comprehensive_gradient_check(func, 2.0, name)
        status = "PASS" if result['overall_pass'] else "FAIL"
        print(f"  {name}: {status}")
    
    print("\nRun test_day10.py to verify your implementation!")
