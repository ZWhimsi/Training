"""Test Suite for Day 9: Activation Functions Backward"""

import math
import sys
from typing import Tuple

try:
    from day09 import Value, compare_activations, check_activation_gradient
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_relu_positive() -> Tuple[bool, str]:
    """Test ReLU with positive input."""
    try:
        x = Value(3.0)
        y = x.relu()
        
        if y is None:
            return False, "ReLU returned None"
        if abs(y.data - 3.0) > 1e-6:
            return False, f"relu(3) = {y.data}, expected 3.0"
        
        y.backward()
        if abs(x.grad - 1.0) > 1e-6:
            return False, f"d relu(3)/dx = {x.grad}, expected 1.0"
        return True, "relu(3)=3, grad=1"
    except Exception as e:
        return False, str(e)


def test_relu_negative() -> Tuple[bool, str]:
    """Test ReLU with negative input."""
    try:
        x = Value(-3.0)
        y = x.relu()
        
        if y is None:
            return False, "ReLU returned None"
        if abs(y.data - 0.0) > 1e-6:
            return False, f"relu(-3) = {y.data}, expected 0.0"
        
        y.backward()
        if abs(x.grad - 0.0) > 1e-6:
            return False, f"d relu(-3)/dx = {x.grad}, expected 0.0"
        return True, "relu(-3)=0, grad=0"
    except Exception as e:
        return False, str(e)


def test_relu_zero() -> Tuple[bool, str]:
    """Test ReLU at zero."""
    try:
        x = Value(0.0)
        y = x.relu()
        
        if y is None:
            return False, "ReLU returned None"
        if abs(y.data - 0.0) > 1e-6:
            return False, f"relu(0) = {y.data}, expected 0.0"
        
        y.backward()
        # Gradient at 0 is commonly defined as 0
        if x.grad not in [0.0, 1.0]:  # Either convention is acceptable
            return False, f"d relu(0)/dx = {x.grad}, expected 0 or 1"
        return True, "relu(0)=0"
    except Exception as e:
        return False, str(e)


def test_leaky_relu_positive() -> Tuple[bool, str]:
    """Test Leaky ReLU with positive input."""
    try:
        x = Value(2.0)
        y = x.leaky_relu(alpha=0.1)
        
        if y is None:
            return False, "Leaky ReLU returned None"
        if abs(y.data - 2.0) > 1e-6:
            return False, f"leaky_relu(2) = {y.data}, expected 2.0"
        
        y.backward()
        if abs(x.grad - 1.0) > 1e-6:
            return False, f"grad = {x.grad}, expected 1.0"
        return True, "leaky_relu(2)=2"
    except Exception as e:
        return False, str(e)


def test_leaky_relu_negative() -> Tuple[bool, str]:
    """Test Leaky ReLU with negative input."""
    try:
        x = Value(-2.0)
        alpha = 0.1
        y = x.leaky_relu(alpha=alpha)
        
        if y is None:
            return False, "Leaky ReLU returned None"
        expected = alpha * (-2.0)  # -0.2
        if abs(y.data - expected) > 1e-6:
            return False, f"leaky_relu(-2) = {y.data}, expected {expected}"
        
        y.backward()
        if abs(x.grad - alpha) > 1e-6:
            return False, f"grad = {x.grad}, expected {alpha}"
        return True, f"leaky_relu(-2)={expected}"
    except Exception as e:
        return False, str(e)


def test_exp_forward() -> Tuple[bool, str]:
    """Test exp forward pass."""
    try:
        x = Value(1.0)
        y = x.exp()
        
        if y is None:
            return False, "exp returned None"
        expected = math.exp(1.0)
        if abs(y.data - expected) > 1e-6:
            return False, f"exp(1) = {y.data}, expected {expected}"
        return True, f"exp(1) = e"
    except Exception as e:
        return False, str(e)


def test_exp_backward() -> Tuple[bool, str]:
    """Test exp backward pass."""
    try:
        x = Value(2.0)
        y = x.exp()
        y.backward()
        
        expected_grad = math.exp(2.0)
        if abs(x.grad - expected_grad) > 1e-5:
            return False, f"d exp(2)/dx = {x.grad}, expected {expected_grad}"
        return True, "d exp(x)/dx = exp(x)"
    except Exception as e:
        return False, str(e)


def test_sigmoid_values() -> Tuple[bool, str]:
    """Test sigmoid at known values."""
    try:
        # sigmoid(0) = 0.5
        x = Value(0.0)
        y = x.sigmoid()
        
        if y is None:
            return False, "sigmoid returned None"
        if abs(y.data - 0.5) > 1e-6:
            return False, f"sigmoid(0) = {y.data}, expected 0.5"
        return True, "sigmoid(0) = 0.5"
    except Exception as e:
        return False, str(e)


def test_sigmoid_backward() -> Tuple[bool, str]:
    """Test sigmoid backward pass."""
    try:
        x = Value(0.0)
        y = x.sigmoid()
        y.backward()
        
        # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        if abs(x.grad - 0.25) > 1e-6:
            return False, f"sigmoid'(0) = {x.grad}, expected 0.25"
        return True, "sigmoid'(0) = 0.25"
    except Exception as e:
        return False, str(e)


def test_sigmoid_saturation() -> Tuple[bool, str]:
    """Test sigmoid behavior at extreme values."""
    try:
        # Large positive value
        x_pos = Value(10.0)
        y_pos = x_pos.sigmoid()
        
        if abs(y_pos.data - 1.0) > 0.001:
            return False, f"sigmoid(10) = {y_pos.data}, expected ~1.0"
        
        y_pos.backward()
        # Gradient should be very small (vanishing gradient)
        if x_pos.grad > 0.01:
            return False, f"sigmoid'(10) = {x_pos.grad}, expected ~0"
        
        # Large negative value
        x_neg = Value(-10.0)
        y_neg = x_neg.sigmoid()
        
        if abs(y_neg.data - 0.0) > 0.001:
            return False, f"sigmoid(-10) = {y_neg.data}, expected ~0.0"
        
        return True, "Saturation behavior correct"
    except Exception as e:
        return False, str(e)


def test_tanh_values() -> Tuple[bool, str]:
    """Test tanh at known values."""
    try:
        # tanh(0) = 0
        x = Value(0.0)
        y = x.tanh()
        
        if y is None:
            return False, "tanh returned None"
        if abs(y.data - 0.0) > 1e-6:
            return False, f"tanh(0) = {y.data}, expected 0.0"
        return True, "tanh(0) = 0"
    except Exception as e:
        return False, str(e)


def test_tanh_backward() -> Tuple[bool, str]:
    """Test tanh backward pass."""
    try:
        x = Value(0.0)
        y = x.tanh()
        y.backward()
        
        # tanh'(0) = 1 - tanh²(0) = 1 - 0 = 1
        if abs(x.grad - 1.0) > 1e-6:
            return False, f"tanh'(0) = {x.grad}, expected 1.0"
        return True, "tanh'(0) = 1"
    except Exception as e:
        return False, str(e)


def test_tanh_range() -> Tuple[bool, str]:
    """Test tanh output range."""
    try:
        # tanh output should be in [-1, 1]
        for val in [-100, -1, 0, 1, 100]:
            x = Value(float(val))
            y = x.tanh()
            if y.data < -1.0001 or y.data > 1.0001:
                return False, f"tanh({val}) = {y.data}, out of range"
        return True, "tanh in [-1, 1]"
    except Exception as e:
        return False, str(e)


def test_softplus() -> Tuple[bool, str]:
    """Test softplus activation."""
    try:
        x = Value(0.0)
        y = x.softplus()
        
        if y is None:
            return False, "softplus returned None"
        
        # softplus(0) = log(1 + e^0) = log(2) ≈ 0.693
        expected = math.log(2)
        if abs(y.data - expected) > 1e-5:
            return False, f"softplus(0) = {y.data}, expected {expected}"
        
        y.backward()
        # d/dx softplus(x) = sigmoid(x) = 0.5 at x=0
        if abs(x.grad - 0.5) > 1e-5:
            return False, f"softplus'(0) = {x.grad}, expected 0.5"
        
        return True, "softplus(0) = log(2)"
    except Exception as e:
        return False, str(e)


def test_chain_rule_activation() -> Tuple[bool, str]:
    """Test chain rule with activation function."""
    try:
        # f(x) = sigmoid(x^2) at x = 1
        x = Value(1.0)
        y = (x ** 2).sigmoid()
        y.backward()
        
        # f(x) = sigmoid(x^2)
        # f'(x) = sigmoid'(x^2) * 2x = sigmoid(x^2) * (1 - sigmoid(x^2)) * 2x
        sig_1 = 1 / (1 + math.exp(-1))  # sigmoid(1)
        expected_grad = sig_1 * (1 - sig_1) * 2
        
        if abs(x.grad - expected_grad) > 1e-4:
            return False, f"grad = {x.grad}, expected {expected_grad}"
        return True, "Chain rule with sigmoid"
    except Exception as e:
        return False, str(e)


def test_numerical_gradient_relu() -> Tuple[bool, str]:
    """Verify ReLU gradient numerically."""
    try:
        analytical, numerical, matches = check_activation_gradient('relu', 2.0)
        if not matches:
            return False, f"analytical={analytical}, numerical={numerical}"
        return True, "ReLU gradient verified"
    except Exception as e:
        return False, str(e)


def test_numerical_gradient_sigmoid() -> Tuple[bool, str]:
    """Verify sigmoid gradient numerically."""
    try:
        analytical, numerical, matches = check_activation_gradient('sigmoid', 0.5)
        if not matches:
            return False, f"analytical={analytical}, numerical={numerical}"
        return True, "Sigmoid gradient verified"
    except Exception as e:
        return False, str(e)


def test_numerical_gradient_tanh() -> Tuple[bool, str]:
    """Verify tanh gradient numerically."""
    try:
        analytical, numerical, matches = check_activation_gradient('tanh', 0.5)
        if not matches:
            return False, f"analytical={analytical}, numerical={numerical}"
        return True, "Tanh gradient verified"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("relu_positive", test_relu_positive),
        ("relu_negative", test_relu_negative),
        ("relu_zero", test_relu_zero),
        ("leaky_relu_positive", test_leaky_relu_positive),
        ("leaky_relu_negative", test_leaky_relu_negative),
        ("exp_forward", test_exp_forward),
        ("exp_backward", test_exp_backward),
        ("sigmoid_values", test_sigmoid_values),
        ("sigmoid_backward", test_sigmoid_backward),
        ("sigmoid_saturation", test_sigmoid_saturation),
        ("tanh_values", test_tanh_values),
        ("tanh_backward", test_tanh_backward),
        ("tanh_range", test_tanh_range),
        ("softplus", test_softplus),
        ("chain_rule_activation", test_chain_rule_activation),
        ("numerical_gradient_relu", test_numerical_gradient_relu),
        ("numerical_gradient_sigmoid", test_numerical_gradient_sigmoid),
        ("numerical_gradient_tanh", test_numerical_gradient_tanh),
    ]
    
    print(f"\n{'='*50}\nDay 9: Activation Functions - Tests\n{'='*50}")
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
