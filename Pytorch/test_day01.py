"""Test Suite for Day 1: Tensor Basics"""

import torch
import sys
from typing import Tuple

try:
    from day01 import create_tensors, get_tensor_properties, tensor_indexing, reshape_tensors, device_operations
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_create_tensors() -> Tuple[bool, str]:
    try:
        result = create_tensors()
        
        if result['from_list'] is None:
            return False, "from_list is None"
        if not torch.equal(result['from_list'], torch.tensor([1, 2, 3, 4, 5])):
            return False, "from_list incorrect"
        
        if result['zeros'] is None or result['zeros'].shape != (3, 4):
            return False, "zeros shape wrong"
        if result['zeros'].sum() != 0:
            return False, "zeros not all zero"
        
        if result['ones'] is None or result['ones'].shape != (2, 3):
            return False, "ones shape wrong"
        if result['ones'].sum() != 6:
            return False, "ones not all ones"
        
        if result['random'] is None or result['random'].shape != (5, 5):
            return False, "random shape wrong"
        
        if result['range'] is None:
            return False, "range is None"
        if not torch.equal(result['range'], torch.tensor([0, 2, 4, 6, 8])):
            return False, "range values wrong"
        
        return True, "All tensors created correctly"
    except Exception as e:
        return False, str(e)


def test_tensor_properties() -> Tuple[bool, str]:
    try:
        t = torch.randn(3, 4, 5)
        result = get_tensor_properties(t)
        
        if result['shape'] != (3, 4, 5):
            return False, f"Shape: expected (3,4,5), got {result['shape']}"
        if result['dtype'] != torch.float32:
            return False, f"Dtype wrong"
        if result['ndim'] != 3:
            return False, f"Ndim: expected 3, got {result['ndim']}"
        if result['numel'] != 60:
            return False, f"Numel: expected 60, got {result['numel']}"
        
        return True, "Properties extracted correctly"
    except Exception as e:
        return False, str(e)


def test_tensor_indexing() -> Tuple[bool, str]:
    try:
        t = torch.arange(20).reshape(4, 5).float()
        # t is:
        # [[ 0,  1,  2,  3,  4],
        #  [ 5,  6,  7,  8,  9],
        #  [10, 11, 12, 13, 14],
        #  [15, 16, 17, 18, 19]]
        result = tensor_indexing(t)
        
        if result['first_row'] is None:
            return False, "first_row is None"
        expected_first_row = torch.tensor([0., 1., 2., 3., 4.])
        if not torch.equal(result['first_row'], expected_first_row):
            return False, f"first_row: got {result['first_row']}, expected {expected_first_row}"
        
        if result['last_col'] is None:
            return False, "last_col is None"
        expected_last_col = torch.tensor([4., 9., 14., 19.])
        if not torch.equal(result['last_col'], expected_last_col):
            return False, f"last_col: got {result['last_col']}, expected {expected_last_col}"
        
        if result['top_left_2x2'] is None:
            return False, "top_left_2x2 is None"
        expected_top_left = torch.tensor([[0., 1.], [5., 6.]])
        if not torch.equal(result['top_left_2x2'], expected_top_left):
            return False, f"top_left_2x2: got {result['top_left_2x2']}, expected {expected_top_left}"
        
        if result['every_other_row'] is None:
            return False, "every_other_row is None"
        expected_every_other = torch.tensor([[0., 1., 2., 3., 4.], [10., 11., 12., 13., 14.]])
        if not torch.equal(result['every_other_row'], expected_every_other):
            return False, f"every_other_row: got {result['every_other_row']}, expected {expected_every_other}"
        
        return True, "Indexing correct"
    except Exception as e:
        return False, str(e)


def test_reshape_tensors() -> Tuple[bool, str]:
    try:
        t = torch.arange(12).float()  # [0, 1, 2, ..., 11]
        result = reshape_tensors(t)
        
        # Test 3x4 reshape - must have correct shape AND values
        if result['as_3x4'] is None:
            return False, "as_3x4 is None"
        if result['as_3x4'].shape != (3, 4):
            return False, f"as_3x4 shape: got {result['as_3x4'].shape}, expected (3, 4)"
        expected_3x4 = torch.tensor([[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]])
        if not torch.equal(result['as_3x4'], expected_3x4):
            return False, f"as_3x4 values wrong: got {result['as_3x4']}, expected {expected_3x4}"
        
        # Test 2x6 reshape - must have correct shape AND values
        if result['as_2x6'] is None:
            return False, "as_2x6 is None"
        if result['as_2x6'].shape != (2, 6):
            return False, f"as_2x6 shape: got {result['as_2x6'].shape}, expected (2, 6)"
        expected_2x6 = torch.tensor([[0., 1., 2., 3., 4., 5.], [6., 7., 8., 9., 10., 11.]])
        if not torch.equal(result['as_2x6'], expected_2x6):
            return False, f"as_2x6 values wrong: got {result['as_2x6']}, expected {expected_2x6}"
        
        # Test flatten - must have correct shape AND values
        if result['as_flat'] is None:
            return False, "as_flat is None"
        if result['as_flat'].shape != (12,):
            return False, f"as_flat shape: got {result['as_flat'].shape}, expected (12,)"
        if not torch.equal(result['as_flat'], t):
            return False, f"as_flat values wrong"
        
        # Test batch dim - must have correct shape AND values
        if result['with_batch_dim'] is None:
            return False, "with_batch_dim is None"
        if result['with_batch_dim'].shape != (1, 12):
            return False, f"with_batch_dim shape: got {result['with_batch_dim'].shape}, expected (1, 12)"
        if not torch.equal(result['with_batch_dim'].squeeze(), t):
            return False, f"with_batch_dim values wrong"
        
        return True, "Reshaping correct"
    except Exception as e:
        return False, str(e)


def test_device_operations() -> Tuple[bool, str]:
    try:
        result = device_operations()
        
        if result['cpu_tensor'] is None:
            return False, "cpu_tensor is None"
        if str(result['cpu_tensor'].device) != 'cpu':
            return False, "cpu_tensor not on CPU"
        
        if torch.cuda.is_available():
            if result['gpu_tensor'] is None:
                return False, "gpu_tensor is None but CUDA available"
            if 'cuda' not in str(result['gpu_tensor'].device):
                return False, "gpu_tensor not on CUDA"
        
        if result['back_to_cpu'] is None:
            return False, "back_to_cpu is None"
        
        return True, "Device operations correct"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("create_tensors", test_create_tensors),
        ("tensor_properties", test_tensor_properties),
        ("tensor_indexing", test_tensor_indexing),
        ("reshape_tensors", test_reshape_tensors),
        ("device_operations", test_device_operations),
    ]
    
    print(f"\n{'='*50}\nDay 1: Tensor Basics - Tests\n{'='*50}")
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
