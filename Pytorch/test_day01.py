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
        result = tensor_indexing(t)
        
        if result['first_row'] is None:
            return False, "first_row is None"
        if not torch.equal(result['first_row'], t[0]):
            return False, "first_row wrong"
        
        if result['last_col'] is None:
            return False, "last_col is None"
        if not torch.equal(result['last_col'], t[:, -1]):
            return False, "last_col wrong"
        
        if result['top_left_2x2'] is None:
            return False, "top_left_2x2 is None"
        if result['top_left_2x2'].shape != (2, 2):
            return False, "top_left_2x2 shape wrong"
        
        if result['every_other_row'] is None:
            return False, "every_other_row is None"
        if result['every_other_row'].shape[0] != 2:
            return False, "every_other_row wrong"
        
        return True, "Indexing correct"
    except Exception as e:
        return False, str(e)


def test_reshape_tensors() -> Tuple[bool, str]:
    try:
        t = torch.arange(12).float()
        result = reshape_tensors(t)
        
        if result['as_3x4'] is None or result['as_3x4'].shape != (3, 4):
            return False, "3x4 reshape failed"
        
        if result['as_2x6'] is None or result['as_2x6'].shape != (2, 6):
            return False, "2x6 reshape failed"
        
        if result['as_flat'] is None or result['as_flat'].shape != (12,):
            return False, "flatten failed"
        
        if result['with_batch_dim'] is None or result['with_batch_dim'].shape != (1, 12):
            return False, "batch dim failed"
        
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
