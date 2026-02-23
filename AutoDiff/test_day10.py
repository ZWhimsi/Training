"""Test Suite for Day 10: Numerical Gradient Checking"""

import math
import sys
from typing import Tuple

try:
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
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_forward_difference_quadratic() -> Tuple[bool, str]:
    """Test forward difference on x^2."""
    try:
        def f(x):
            return x ** 2
        
        grad = forward_difference(f, 3.0)
        if grad is None:
            return False, "Returned None"
        
        # True gradient of x^2 at x=3 is 6
        if abs(grad - 6.0) > 0.01:  # Forward diff has larger error
            return False, f"Got {grad}, expected ~6.0"
        return True, f"d/dx(x²)|_3 ≈ {grad:.4f}"
    except Exception as e:
        return False, str(e)


def test_forward_difference_cubic() -> Tuple[bool, str]:
    """Test forward difference on x^3."""
    try:
        def f(x):
            return x ** 3
        
        grad = forward_difference(f, 2.0)
        if grad is None:
            return False, "Returned None"
        
        # True gradient of x^3 at x=2 is 12
        if abs(grad - 12.0) > 0.01:
            return False, f"Got {grad}, expected ~12.0"
        return True, f"d/dx(x³)|_2 ≈ {grad:.4f}"
    except Exception as e:
        return False, str(e)


def test_central_difference_quadratic() -> Tuple[bool, str]:
    """Test central difference on x^2."""
    try:
        def f(x):
            return x ** 2
        
        grad = central_difference(f, 3.0)
        if grad is None:
            return False, "Returned None"
        
        # Central diff should be very accurate for polynomial
        if abs(grad - 6.0) > 1e-8:
            return False, f"Got {grad}, expected 6.0"
        return True, f"d/dx(x²)|_3 = {grad:.6f}"
    except Exception as e:
        return False, str(e)


def test_central_difference_trig() -> Tuple[bool, str]:
    """Test central difference on sin(x)."""
    try:
        def f(x):
            return math.sin(x)
        
        x = math.pi / 4
        grad = central_difference(f, x)
        if grad is None:
            return False, "Returned None"
        
        # d/dx sin(x) = cos(x)
        expected = math.cos(x)
        if abs(grad - expected) > 1e-6:
            return False, f"Got {grad}, expected {expected}"
        return True, "d/dx sin(π/4) = cos(π/4)"
    except Exception as e:
        return False, str(e)


def test_central_more_accurate() -> Tuple[bool, str]:
    """Verify central difference is more accurate than forward."""
    try:
        def f(x):
            return x ** 2
        
        x, h = 3.0, 1e-4
        true_grad = 6.0
        
        fwd = forward_difference(f, x, h)
        ctr = central_difference(f, x, h)
        
        if fwd is None or ctr is None:
            return False, "One method returned None"
        
        fwd_error = abs(fwd - true_grad)
        ctr_error = abs(ctr - true_grad)
        
        if ctr_error >= fwd_error:
            return False, f"Central error {ctr_error} >= forward error {fwd_error}"
        return True, f"Central: {ctr_error:.2e} < Forward: {fwd_error:.2e}"
    except Exception as e:
        return False, str(e)


def test_gradient_check_single_basic() -> Tuple[bool, str]:
    """Test gradient check on simple expression."""
    try:
        def expr(x):
            return x ** 2
        
        result = gradient_check_single(expr, 3.0)
        
        if result['analytical'] is None or result['numerical'] is None:
            return False, "Returned None values"
        if not result.get('matches', False):
            return False, f"Gradients don't match: {result}"
        return True, "x² gradient verified"
    except Exception as e:
        return False, str(e)


def test_gradient_check_single_complex() -> Tuple[bool, str]:
    """Test gradient check on complex expression."""
    try:
        def expr(x):
            return (x ** 2 + x).sigmoid()
        
        result = gradient_check_single(expr, 1.0)
        
        if result['analytical'] is None or result['numerical'] is None:
            return False, "Returned None values"
        if result['difference'] is None or result['difference'] > 1e-4:
            return False, f"Difference too large: {result['difference']}"
        return True, "sigmoid(x²+x) gradient verified"
    except Exception as e:
        return False, str(e)


def test_gradient_check_multi_basic() -> Tuple[bool, str]:
    """Test multi-variable gradient check."""
    try:
        def expr(vals):
            return vals['x'] * vals['y']
        
        result = gradient_check_multi(expr, {'x': 2.0, 'y': 3.0})
        
        # d(x*y)/dx = y = 3, d(x*y)/dy = x = 2
        if result['x']['numerical'] is None or result['y']['numerical'] is None:
            return False, "Numerical gradients are None"
        
        if not result['x'].get('matches') or not result['y'].get('matches'):
            return False, f"Gradients don't match: {result}"
        return True, "x*y gradients verified"
    except Exception as e:
        return False, str(e)


def test_gradient_check_multi_complex() -> Tuple[bool, str]:
    """Test multi-variable gradient check on complex expression."""
    try:
        def expr(vals):
            x, y = vals['x'], vals['y']
            return (x ** 2 + y ** 2).tanh()
        
        result = gradient_check_multi(expr, {'x': 0.5, 'y': 0.5})
        
        x_ok = result['x'].get('matches', False)
        y_ok = result['y'].get('matches', False)
        
        if not (x_ok and y_ok):
            return False, f"x match: {x_ok}, y match: {y_ok}"
        return True, "tanh(x²+y²) gradients verified"
    except Exception as e:
        return False, str(e)


def test_relative_error_basic() -> Tuple[bool, str]:
    """Test relative error calculation."""
    try:
        err = relative_error(1.0, 1.0)
        if err is None:
            return False, "Returned None"
        if abs(err) > 1e-10:
            return False, f"Same values should have 0 error, got {err}"
        
        err2 = relative_error(1.0, 1.1)
        expected = 0.1 / 1.1  # |1.0 - 1.1| / max(1.0, 1.1)
        if abs(err2 - expected) > 1e-10:
            return False, f"Expected {expected}, got {err2}"
        
        return True, "Relative error correct"
    except Exception as e:
        return False, str(e)


def test_relative_error_small_values() -> Tuple[bool, str]:
    """Test relative error with small values (avoid div by zero)."""
    try:
        err = relative_error(1e-10, 1e-10)
        if err is None:
            return False, "Returned None"
        if not math.isfinite(err):
            return False, f"Got non-finite value: {err}"
        return True, "Handles small values"
    except Exception as e:
        return False, str(e)


def test_gradient_check_relative_basic() -> Tuple[bool, str]:
    """Test relative gradient check."""
    try:
        def expr(x):
            return x ** 3
        
        result = gradient_check_relative(expr, 2.0)
        
        if result['relative_error'] is None:
            return False, "Relative error is None"
        if not result.get('passes', False):
            return False, f"Should pass: rel_error = {result['relative_error']}"
        return True, "x³ relative check passed"
    except Exception as e:
        return False, str(e)


def test_comprehensive_check_passes() -> Tuple[bool, str]:
    """Test comprehensive gradient checker on simple function."""
    try:
        def expr(x):
            return x ** 2 + 2 * x
        
        result = comprehensive_gradient_check(expr, 3.0, "x²+2x")
        
        if not result.get('overall_pass', False):
            return False, "Should pass overall"
        
        checks = result.get('checks', [])
        if len(checks) < 3:
            return False, "Not enough checks performed"
        
        return True, "Comprehensive check passed"
    except Exception as e:
        return False, str(e)


def test_comprehensive_check_relu() -> Tuple[bool, str]:
    """Test comprehensive gradient checker on ReLU."""
    try:
        def expr(x):
            return x.relu()
        
        # Test at positive value (gradient should be 1)
        result = comprehensive_gradient_check(expr, 2.0, "relu")
        
        if not result.get('overall_pass', False):
            return False, "ReLU check should pass at x=2"
        return True, "ReLU gradient verified"
    except Exception as e:
        return False, str(e)


def test_comprehensive_check_sigmoid() -> Tuple[bool, str]:
    """Test comprehensive gradient checker on sigmoid."""
    try:
        def expr(x):
            return x.sigmoid()
        
        result = comprehensive_gradient_check(expr, 0.0, "sigmoid")
        
        if not result.get('overall_pass', False):
            return False, "Sigmoid check should pass"
        
        # Check specific gradient at 0
        analytical = result['checks'][0]['analytical']
        expected = 0.25  # sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5
        if abs(analytical - expected) > 1e-4:
            return False, f"sigmoid'(0) = {analytical}, expected {expected}"
        
        return True, "Sigmoid gradient = 0.25 at 0"
    except Exception as e:
        return False, str(e)


def test_gradient_check_chain_rule() -> Tuple[bool, str]:
    """Test gradient check with chain rule."""
    try:
        def expr(x):
            return (x ** 2).sigmoid()
        
        result = gradient_check_single(expr, 1.0)
        
        if not result.get('matches', False):
            return False, f"Chain rule check failed: {result}"
        return True, "sigmoid(x²) gradient verified"
    except Exception as e:
        return False, str(e)


def test_step_size_effect() -> Tuple[bool, str]:
    """Test that step size affects accuracy appropriately."""
    try:
        def f(x):
            return x ** 2
        
        x, true_grad = 3.0, 6.0
        
        # Larger h should give worse results
        h_large = central_difference(f, x, h=1e-2)
        h_small = central_difference(f, x, h=1e-6)
        
        if h_large is None or h_small is None:
            return False, "Returned None"
        
        err_large = abs(h_large - true_grad)
        err_small = abs(h_small - true_grad)
        
        # Small h should generally be more accurate (up to numerical precision)
        if err_large < err_small * 10:
            return False, "Larger h should have larger error"
        
        return True, "Step size affects accuracy correctly"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("forward_diff_quadratic", test_forward_difference_quadratic),
        ("forward_diff_cubic", test_forward_difference_cubic),
        ("central_diff_quadratic", test_central_difference_quadratic),
        ("central_diff_trig", test_central_difference_trig),
        ("central_more_accurate", test_central_more_accurate),
        ("grad_check_single_basic", test_gradient_check_single_basic),
        ("grad_check_single_complex", test_gradient_check_single_complex),
        ("grad_check_multi_basic", test_gradient_check_multi_basic),
        ("grad_check_multi_complex", test_gradient_check_multi_complex),
        ("relative_error_basic", test_relative_error_basic),
        ("relative_error_small", test_relative_error_small_values),
        ("grad_check_relative", test_gradient_check_relative_basic),
        ("comprehensive_basic", test_comprehensive_check_passes),
        ("comprehensive_relu", test_comprehensive_check_relu),
        ("comprehensive_sigmoid", test_comprehensive_check_sigmoid),
        ("chain_rule", test_gradient_check_chain_rule),
        ("step_size_effect", test_step_size_effect),
    ]
    
    print(f"\n{'='*50}\nDay 10: Numerical Gradient Checking - Tests\n{'='*50}")
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    print(f"\nSummary: {passed}/{len(tests)}")


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    run_all_tests()
