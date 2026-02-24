"""Test Suite for Day 10: Numerical Gradient Checking"""

import math
import pytest

from day10 import (
    Value,
    forward_difference,
    central_difference,
    compare_difference_methods,
    gradient_check_single,
    gradient_check_multi,
    relative_error,
    gradient_check_relative,
    comprehensive_gradient_check
)


def test_forward_difference_quadratic():
    """Test forward difference on x^2."""
    def f(x):
        return x ** 2
    
    grad = forward_difference(f, 3.0)
    assert grad is not None, "Returned None"
    
    # True gradient of x^2 at x=3 is 6
    assert abs(grad - 6.0) <= 0.01, f"Got {grad}, expected ~6.0"


def test_forward_difference_cubic():
    """Test forward difference on x^3."""
    def f(x):
        return x ** 3
    
    grad = forward_difference(f, 2.0)
    assert grad is not None, "Returned None"
    
    # True gradient of x^3 at x=2 is 12
    assert abs(grad - 12.0) <= 0.01, f"Got {grad}, expected ~12.0"


def test_central_difference_quadratic():
    """Test central difference on x^2."""
    def f(x):
        return x ** 2
    
    grad = central_difference(f, 3.0)
    assert grad is not None, "Returned None"
    
    # Central diff should be very accurate for polynomial
    assert abs(grad - 6.0) <= 1e-8, f"Got {grad}, expected 6.0"


def test_central_difference_trig():
    """Test central difference on sin(x)."""
    def f(x):
        return math.sin(x)
    
    x = math.pi / 4
    grad = central_difference(f, x)
    assert grad is not None, "Returned None"
    
    # d/dx sin(x) = cos(x)
    expected = math.cos(x)
    assert abs(grad - expected) <= 1e-6, f"Got {grad}, expected {expected}"


def test_central_more_accurate():
    """Verify central difference is more accurate than forward."""
    def f(x):
        return x ** 2
    
    x, h = 3.0, 1e-4
    true_grad = 6.0
    
    fwd = forward_difference(f, x, h)
    ctr = central_difference(f, x, h)
    
    assert fwd is not None and ctr is not None, "One method returned None"
    
    fwd_error = abs(fwd - true_grad)
    ctr_error = abs(ctr - true_grad)
    
    assert ctr_error < fwd_error, f"Central error {ctr_error} >= forward error {fwd_error}"


def test_gradient_check_single_basic():
    """Test gradient check on simple expression."""
    def expr(x):
        return x ** 2
    
    result = gradient_check_single(expr, 3.0)
    
    assert result['analytical'] is not None and result['numerical'] is not None, "Returned None values"
    assert result.get('matches', False), f"Gradients don't match: {result}"


def test_gradient_check_single_complex():
    """Test gradient check on complex expression."""
    def expr(x):
        return (x ** 2 + x).sigmoid()
    
    result = gradient_check_single(expr, 1.0)
    
    assert result['analytical'] is not None and result['numerical'] is not None, "Returned None values"
    assert result['difference'] is not None and result['difference'] <= 1e-4, f"Difference too large: {result['difference']}"


def test_gradient_check_multi_basic():
    """Test multi-variable gradient check."""
    def expr(vals):
        return vals['x'] * vals['y']
    
    result = gradient_check_multi(expr, {'x': 2.0, 'y': 3.0})
    
    # d(x*y)/dx = y = 3, d(x*y)/dy = x = 2
    assert result['x']['numerical'] is not None and result['y']['numerical'] is not None, "Numerical gradients are None"
    assert result['x'].get('matches') or result['y'].get('matches'), f"Gradients don't match: {result}"


def test_gradient_check_multi_complex():
    """Test multi-variable gradient check on complex expression."""
    def expr(vals):
        x, y = vals['x'], vals['y']
        return (x ** 2 + y ** 2).tanh()
    
    result = gradient_check_multi(expr, {'x': 0.5, 'y': 0.5})
    
    x_ok = result['x'].get('matches', False)
    y_ok = result['y'].get('matches', False)
    
    assert x_ok and y_ok, f"x match: {x_ok}, y match: {y_ok}"


def test_relative_error_basic():
    """Test relative error calculation."""
    err = relative_error(1.0, 1.0)
    assert err is not None, "Returned None"
    assert abs(err) <= 1e-10, f"Same values should have 0 error, got {err}"
    
    err2 = relative_error(1.0, 1.1)
    expected = 0.1 / 1.1  # |1.0 - 1.1| / max(1.0, 1.1)
    assert abs(err2 - expected) <= 1e-10, f"Expected {expected}, got {err2}"


def test_relative_error_small_values():
    """Test relative error with small values (avoid div by zero)."""
    err = relative_error(1e-10, 1e-10)
    assert err is not None, "Returned None"
    assert math.isfinite(err), f"Got non-finite value: {err}"


def test_gradient_check_relative_basic():
    """Test relative gradient check."""
    def expr(x):
        return x ** 3
    
    result = gradient_check_relative(expr, 2.0)
    
    assert result['relative_error'] is not None, "Relative error is None"
    assert result.get('passes', False), f"Should pass: rel_error = {result['relative_error']}"


def test_comprehensive_check_passes():
    """Test comprehensive gradient checker on simple function."""
    def expr(x):
        return x ** 2 + 2 * x
    
    result = comprehensive_gradient_check(expr, 3.0, "x²+2x")
    
    assert result.get('overall_pass', False), "Should pass overall"
    
    checks = result.get('checks', [])
    assert len(checks) >= 3, "Not enough checks performed"


def test_comprehensive_check_relu():
    """Test comprehensive gradient checker on ReLU."""
    def expr(x):
        return x.relu()
    
    # Test at positive value (gradient should be 1)
    result = comprehensive_gradient_check(expr, 2.0, "relu")
    
    assert result.get('overall_pass', False), "ReLU check should pass at x=2"


def test_comprehensive_check_sigmoid():
    """Test comprehensive gradient checker on sigmoid."""
    def expr(x):
        return x.sigmoid()
    
    result = comprehensive_gradient_check(expr, 0.0, "sigmoid")
    
    assert result.get('overall_pass', False), "Sigmoid check should pass"
    
    # Check specific gradient at 0
    analytical = result['checks'][0]['analytical']
    expected = 0.25  # sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5
    assert abs(analytical - expected) <= 1e-4, f"sigmoid'(0) = {analytical}, expected {expected}"


def test_gradient_check_chain_rule():
    """Test gradient check with chain rule."""
    def expr(x):
        return (x ** 2).sigmoid()
    
    result = gradient_check_single(expr, 1.0)
    
    assert result.get('matches', False), f"Chain rule check failed: {result}"


def test_step_size_effect():
    """Test that step size affects accuracy appropriately."""
    def f(x):
        return x ** 2
    
    x, true_grad = 3.0, 6.0
    
    # Larger h should give worse results
    h_large = central_difference(f, x, h=1e-2)
    h_small = central_difference(f, x, h=1e-6)
    
    assert h_large is not None and h_small is not None, "Returned None"
    
    err_large = abs(h_large - true_grad)
    err_small = abs(h_small - true_grad)
    
    # Small h should generally be more accurate (up to numerical precision)
    assert err_large >= err_small * 10, "Larger h should have larger error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
