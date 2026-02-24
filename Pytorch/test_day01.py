"""Test Suite for Day 1: Tensor Basics"""

import torch
import pytest

from day01 import create_tensors, get_tensor_properties, tensor_indexing, reshape_tensors, device_operations


def test_create_tensors():
    result = create_tensors()
    
    assert result['from_list'] is not None, "from_list is None"
    assert torch.equal(result['from_list'], torch.tensor([1, 2, 3, 4, 5])), "from_list incorrect"
    
    assert result['zeros'] is not None, "zeros is None"
    assert result['zeros'].shape == (3, 4), f"zeros shape: got {result['zeros'].shape}, expected (3, 4)"
    assert result['zeros'].sum() == 0, "zeros not all zero"
    
    assert result['ones'] is not None, "ones is None"
    assert result['ones'].shape == (2, 3), f"ones shape: got {result['ones'].shape}, expected (2, 3)"
    assert result['ones'].sum() == 6, "ones not all ones"
    
    assert result['random'] is not None, "random is None"
    assert result['random'].shape == (5, 5), f"random shape: got {result['random'].shape}, expected (5, 5)"
    
    assert result['range'] is not None, "range is None"
    assert torch.equal(result['range'], torch.tensor([0, 2, 4, 6, 8])), "range values wrong"


def test_tensor_properties():
    t = torch.randn(3, 4, 5)
    result = get_tensor_properties(t)
    
    assert result['shape'] == (3, 4, 5), f"Shape: expected (3,4,5), got {result['shape']}"
    assert result['dtype'] == torch.float32, f"Dtype: expected torch.float32, got {result['dtype']}"
    assert result['ndim'] == 3, f"Ndim: expected 3, got {result['ndim']}"
    assert result['numel'] == 60, f"Numel: expected 60, got {result['numel']}"


def test_tensor_indexing():
    t = torch.arange(20).reshape(4, 5).float()
    result = tensor_indexing(t)
    
    assert result['first_row'] is not None, "first_row is None"
    expected_first_row = torch.tensor([0., 1., 2., 3., 4.])
    assert torch.equal(result['first_row'], expected_first_row), f"first_row: got {result['first_row']}, expected {expected_first_row}"
    
    assert result['last_col'] is not None, "last_col is None"
    expected_last_col = torch.tensor([4., 9., 14., 19.])
    assert torch.equal(result['last_col'], expected_last_col), f"last_col: got {result['last_col']}, expected {expected_last_col}"
    
    assert result['top_left_2x2'] is not None, "top_left_2x2 is None"
    expected_top_left = torch.tensor([[0., 1.], [5., 6.]])
    assert torch.equal(result['top_left_2x2'], expected_top_left), f"top_left_2x2: got {result['top_left_2x2']}, expected {expected_top_left}"
    
    assert result['every_other_row'] is not None, "every_other_row is None"
    expected_every_other = torch.tensor([[0., 1., 2., 3., 4.], [10., 11., 12., 13., 14.]])
    assert torch.equal(result['every_other_row'], expected_every_other), f"every_other_row: got {result['every_other_row']}, expected {expected_every_other}"


def test_reshape_tensors():
    t = torch.arange(12).float()
    result = reshape_tensors(t)
    
    assert result['as_3x4'] is not None, "as_3x4 is None"
    assert result['as_3x4'].shape == (3, 4), f"as_3x4 shape: got {result['as_3x4'].shape}, expected (3, 4)"
    expected_3x4 = torch.tensor([[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]])
    assert torch.equal(result['as_3x4'], expected_3x4), f"as_3x4 values wrong"
    
    assert result['as_2x6'] is not None, "as_2x6 is None"
    assert result['as_2x6'].shape == (2, 6), f"as_2x6 shape: got {result['as_2x6'].shape}, expected (2, 6)"
    expected_2x6 = torch.tensor([[0., 1., 2., 3., 4., 5.], [6., 7., 8., 9., 10., 11.]])
    assert torch.equal(result['as_2x6'], expected_2x6), f"as_2x6 values wrong"
    
    assert result['as_flat'] is not None, "as_flat is None"
    assert result['as_flat'].shape == (12,), f"as_flat shape: got {result['as_flat'].shape}, expected (12,)"
    assert torch.equal(result['as_flat'], t), "as_flat values wrong"
    
    assert result['with_batch_dim'] is not None, "with_batch_dim is None"
    assert result['with_batch_dim'].shape == (1, 12), f"with_batch_dim shape: got {result['with_batch_dim'].shape}, expected (1, 12)"
    assert torch.equal(result['with_batch_dim'].squeeze(), t), "with_batch_dim values wrong"


def test_device_operations():
    result = device_operations()
    
    assert result['cpu_tensor'] is not None, "cpu_tensor is None"
    assert str(result['cpu_tensor'].device) == 'cpu', "cpu_tensor not on CPU"
    
    if torch.cuda.is_available():
        assert result['gpu_tensor'] is not None, "gpu_tensor is None but CUDA available"
        assert 'cuda' in str(result['gpu_tensor'].device), "gpu_tensor not on CUDA"
    
    assert result['back_to_cpu'] is not None, "back_to_cpu is None"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
