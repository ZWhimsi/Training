"""Test Suite for Day 16: Matrix Multiplication Gradient"""

import numpy as np
import sys
from typing import Tuple

try:
    from day16 import (
        Tensor,
        Linear,
        test_transpose,
        test_matmul_2d,
        test_matmul_gradient,
        test_matmul_chain,
        test_linear_layer
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_transpose_forward() -> Tuple[bool, str]:
    """Test transpose forward pass."""
    try:
        A = Tensor([[1, 2, 3], [4, 5, 6]])
        AT = A.T
        
        if AT is None or AT.data is None:
            return False, "Returned None"
        if AT.shape != (3, 2):
            return False, f"shape = {AT.shape}"
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        if not np.allclose(AT.data, expected):
            return False, "Values incorrect"
        return True, "(2,3).T = (3,2)"
    except Exception as e:
        return False, str(e)


def test_transpose_backward() -> Tuple[bool, str]:
    """Test transpose backward pass."""
    try:
        A = Tensor([[1.0, 2.0], [3.0, 4.0]])
        AT = A.T
        loss = AT.sum()
        loss.backward()
        
        if A.grad is None:
            return False, "grad is None"
        if A.grad.shape != (2, 2):
            return False, f"grad shape = {A.grad.shape}"
        if not np.allclose(A.grad, np.ones((2, 2))):
            return False, "grad values incorrect"
        return True, "Transpose gradient OK"
    except Exception as e:
        return False, str(e)


def test_matmul_forward_2x2() -> Tuple[bool, str]:
    """Test 2x2 matmul forward."""
    try:
        A = Tensor([[1, 2], [3, 4]])
        B = Tensor([[5, 6], [7, 8]])
        C = A @ B
        
        if C is None or C.data is None:
            return False, "Returned None"
        expected = np.array([[19, 22], [43, 50]])
        if not np.allclose(C.data, expected):
            return False, f"Got {C.data}"
        return True, "2x2 @ 2x2 correct"
    except Exception as e:
        return False, str(e)


def test_matmul_forward_rect() -> Tuple[bool, str]:
    """Test rectangular matmul."""
    try:
        A = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        B = Tensor([[7, 8], [9, 10], [11, 12]])  # (3, 2)
        C = A @ B  # (2, 2)
        
        if C is None or C.data is None:
            return False, "Returned None"
        if C.shape != (2, 2):
            return False, f"shape = {C.shape}"
        expected = np.array([[58, 64], [139, 154]])
        if not np.allclose(C.data, expected):
            return False, f"Got {C.data}"
        return True, "(2,3) @ (3,2) = (2,2)"
    except Exception as e:
        return False, str(e)


def test_matmul_gradient_simple() -> Tuple[bool, str]:
    """Test matmul gradient with ones output gradient."""
    try:
        A = Tensor([[1.0, 2.0], [3.0, 4.0]])
        B = Tensor([[5.0, 6.0], [7.0, 8.0]])
        C = A @ B
        loss = C.sum()
        loss.backward()
        
        if A.grad is None or B.grad is None:
            return False, "Gradients are None"
        
        # dL/dA = ones @ B.T = [[11,15],[11,15]]
        expected_A = np.array([[11, 15], [11, 15]])
        # dL/dB = A.T @ ones = [[4,4],[6,6]]
        expected_B = np.array([[4, 4], [6, 6]])
        
        if not np.allclose(A.grad, expected_A):
            return False, f"grad_A = {A.grad}"
        if not np.allclose(B.grad, expected_B):
            return False, f"grad_B = {B.grad}"
        return True, "Gradients match"
    except Exception as e:
        return False, str(e)


def test_matmul_gradient_rect() -> Tuple[bool, str]:
    """Test matmul gradient with rectangular matrices."""
    try:
        A = Tensor(np.ones((2, 3)))  # (2, 3)
        B = Tensor(np.ones((3, 4)))  # (3, 4)
        C = A @ B  # (2, 4)
        loss = C.sum()
        loss.backward()
        
        if A.grad is None or B.grad is None:
            return False, "Gradients are None"
        
        # Check gradient shapes
        if A.grad.shape != (2, 3):
            return False, f"grad_A shape = {A.grad.shape}"
        if B.grad.shape != (3, 4):
            return False, f"grad_B shape = {B.grad.shape}"
        
        # dL/dA = ones(2,4) @ ones(4,3) = 4*ones(2,3)
        if not np.allclose(A.grad, 4 * np.ones((2, 3))):
            return False, f"grad_A values = {A.grad}"
        # dL/dB = ones(3,2) @ ones(2,4) = 2*ones(3,4)
        if not np.allclose(B.grad, 2 * np.ones((3, 4))):
            return False, f"grad_B values = {B.grad}"
        return True, "Rectangular gradients OK"
    except Exception as e:
        return False, str(e)


def test_matmul_vector() -> Tuple[bool, str]:
    """Test matrix-vector multiplication."""
    try:
        A = Tensor([[1, 2], [3, 4], [5, 6]])  # (3, 2)
        x = Tensor([1.0, 1.0])  # (2,)
        y = A @ x.data.reshape(-1, 1)  # Make it (2,1) -> (3,1)
        
        if y is None or y.data is None:
            return False, "Returned None"
        
        expected = np.array([[3], [7], [11]])
        if not np.allclose(y.data, expected):
            return False, f"Got {y.data}"
        return True, "Matrix-vector works"
    except Exception as e:
        return False, str(e)


def test_matmul_chain_rule() -> Tuple[bool, str]:
    """Test chain rule through multiple matmuls."""
    try:
        A = Tensor([[2.0, 0.0], [0.0, 3.0]])  # Diagonal
        B = Tensor([[1.0, 2.0], [3.0, 4.0]])
        x = Tensor([[1.0], [1.0]])
        
        # y = A @ B @ x
        y = A @ (B @ x)
        loss = y.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "x.grad is None"
        
        # Verify with manual calculation
        # y = A @ B @ x
        # dy/dx = (A @ B).T = B.T @ A.T
        AB = A.data @ B.data
        expected_grad = AB.T @ np.ones((2, 1))
        
        if not np.allclose(x.grad, expected_grad):
            return False, f"x.grad = {x.grad}"
        return True, "Chain rule works"
    except Exception as e:
        return False, str(e)


def test_batch_matmul() -> Tuple[bool, str]:
    """Test batch matrix multiplication."""
    try:
        # Batch of 2 matrices
        A = Tensor(np.random.randn(2, 3, 4))
        B = Tensor(np.random.randn(2, 4, 5))
        
        if not hasattr(Tensor, 'bmm'):
            return True, "bmm not implemented (optional)"
        
        C = A.bmm(B)
        
        if C is None or C.data is None:
            return False, "Returned None"
        if C.shape != (2, 3, 5):
            return False, f"shape = {C.shape}"
        
        # Verify with numpy
        expected = np.matmul(A.data, B.data)
        if not np.allclose(C.data, expected):
            return False, "Values don't match numpy"
        return True, "Batch matmul works"
    except Exception as e:
        return False, str(e)


def test_batch_matmul_gradient() -> Tuple[bool, str]:
    """Test batch matmul gradient."""
    try:
        if not hasattr(Tensor, 'bmm'):
            return True, "bmm not implemented (optional)"
        
        A = Tensor(np.ones((2, 2, 2)))
        B = Tensor(np.ones((2, 2, 2)))
        C = A.bmm(B)
        loss = C.sum()
        loss.backward()
        
        if A.grad is None or B.grad is None:
            return False, "Gradients are None"
        
        # For ones matrices, gradient should be 2*ones (inner dim)
        if not np.allclose(A.grad, 2 * np.ones((2, 2, 2))):
            return False, f"A.grad = {A.grad}"
        if not np.allclose(B.grad, 2 * np.ones((2, 2, 2))):
            return False, f"B.grad = {B.grad}"
        return True, "Batch gradient OK"
    except Exception as e:
        return False, str(e)


def test_linear_forward() -> Tuple[bool, str]:
    """Test Linear layer forward pass."""
    try:
        linear = Linear(3, 2, bias=False)
        linear.weight = Tensor(np.ones((3, 2)))
        
        x = Tensor([[1.0, 2.0, 3.0]])
        y = linear(x)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (1, 2):
            return False, f"shape = {y.shape}"
        # Sum of [1,2,3] = 6 for each output
        if not np.allclose(y.data, [[6, 6]]):
            return False, f"values = {y.data}"
        return True, "Linear forward OK"
    except Exception as e:
        return False, str(e)


def test_linear_with_bias() -> Tuple[bool, str]:
    """Test Linear layer with bias."""
    try:
        linear = Linear(2, 2, bias=True)
        linear.weight = Tensor(np.eye(2))
        linear.bias = Tensor(np.array([1.0, 2.0]))
        
        x = Tensor([[1.0, 1.0]])
        y = linear(x)
        
        if y is None or y.data is None:
            return False, "Returned None"
        # y = x @ I + bias = [1,1] + [1,2] = [2,3]
        if not np.allclose(y.data, [[2, 3]]):
            return False, f"values = {y.data}"
        return True, "Linear with bias OK"
    except Exception as e:
        return False, str(e)


def test_linear_gradient() -> Tuple[bool, str]:
    """Test Linear layer gradient."""
    try:
        linear = Linear(2, 2, bias=True)
        linear.weight = Tensor(np.ones((2, 2)))
        linear.bias = Tensor(np.zeros(2))
        
        x = Tensor([[1.0, 2.0]])
        y = linear(x)
        loss = y.sum()
        loss.backward()
        
        if linear.weight.grad is None:
            return False, "weight.grad is None"
        if x.grad is None:
            return False, "x.grad is None"
        
        # dL/dW = x.T @ ones = [[1,1],[2,2]]
        expected_w_grad = np.array([[1, 1], [2, 2]])
        if not np.allclose(linear.weight.grad, expected_w_grad):
            return False, f"weight.grad = {linear.weight.grad}"
        
        # dL/dx = ones @ W.T = [2, 2]
        if not np.allclose(x.grad, [[2, 2]]):
            return False, f"x.grad = {x.grad}"
        return True, "Linear gradient OK"
    except Exception as e:
        return False, str(e)


def test_mlp_forward_backward() -> Tuple[bool, str]:
    """Test simple MLP forward and backward."""
    try:
        # Two layer MLP
        layer1 = Linear(2, 3, bias=False)
        layer2 = Linear(3, 1, bias=False)
        
        x = Tensor([[1.0, 1.0]])
        h = layer1(x)
        
        if h is None:
            return False, "Layer 1 returned None"
        
        y = layer2(h)
        
        if y is None:
            return False, "Layer 2 returned None"
        
        loss = y.sum()
        loss.backward()
        
        if layer1.weight.grad is None:
            return False, "Layer 1 weight grad is None"
        if layer2.weight.grad is None:
            return False, "Layer 2 weight grad is None"
        return True, "MLP forward/backward works"
    except Exception as e:
        return False, str(e)


def test_against_pytorch() -> Tuple[bool, str]:
    """Test against PyTorch reference."""
    try:
        import torch
        
        np_A = np.array([[1.0, 2.0], [3.0, 4.0]])
        np_B = np.array([[5.0, 6.0], [7.0, 8.0]])
        
        # Our implementation
        A = Tensor(np_A)
        B = Tensor(np_B)
        C = A @ B
        loss = C.sum()
        loss.backward()
        
        # PyTorch
        tA = torch.tensor(np_A, requires_grad=True)
        tB = torch.tensor(np_B, requires_grad=True)
        tC = tA @ tB
        tLoss = tC.sum()
        tLoss.backward()
        
        if not np.allclose(C.data, tC.detach().numpy()):
            return False, "Forward mismatch"
        if not np.allclose(A.grad, tA.grad.numpy()):
            return False, "A gradient mismatch"
        if not np.allclose(B.grad, tB.grad.numpy()):
            return False, "B gradient mismatch"
        return True, "Matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def test_identity_matmul() -> Tuple[bool, str]:
    """Test matmul with identity matrix."""
    try:
        I = Tensor(np.eye(3))
        A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        # A @ I should equal A
        B = A @ I
        
        if B is None or B.data is None:
            return False, "Returned None"
        if not np.allclose(B.data, A.data):
            return False, "A @ I != A"
        
        B.backward()
        
        # Gradient should flow through unchanged
        if not np.allclose(A.grad, np.ones_like(A.data)):
            return False, f"A.grad = {A.grad}"
        return True, "Identity matmul OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("transpose_forward", test_transpose_forward),
        ("transpose_backward", test_transpose_backward),
        ("matmul_forward_2x2", test_matmul_forward_2x2),
        ("matmul_forward_rect", test_matmul_forward_rect),
        ("matmul_gradient_simple", test_matmul_gradient_simple),
        ("matmul_gradient_rect", test_matmul_gradient_rect),
        ("matmul_vector", test_matmul_vector),
        ("matmul_chain_rule", test_matmul_chain_rule),
        ("batch_matmul", test_batch_matmul),
        ("batch_matmul_gradient", test_batch_matmul_gradient),
        ("linear_forward", test_linear_forward),
        ("linear_with_bias", test_linear_with_bias),
        ("linear_gradient", test_linear_gradient),
        ("mlp_forward_backward", test_mlp_forward_backward),
        ("against_pytorch", test_against_pytorch),
        ("identity_matmul", test_identity_matmul),
    ]
    
    print(f"\n{'='*50}\nDay 16: Matrix Multiplication - Tests\n{'='*50}")
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
