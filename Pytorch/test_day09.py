"""Test Suite for Day 9: Data Loading"""

import torch
import pytest
from torch.utils.data import DataLoader
try:
    from day09 import (SimpleDataset, TransformDataset, custom_collate,
                       SequenceDataset, BalancedSampler, demonstrate_dataloader)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_simple_dataset():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    dataset = SimpleDataset(X, y)
    
    assert len(dataset) != 0, "Not implemented"
    assert len(dataset) == 100, f"Length: got {len(dataset)}, expected 100"
    
    for idx in [0, 50, 99]:
        sample_x, sample_y = dataset[idx]
        assert sample_x is not None, f"__getitem__[{idx}] returned None"
        assert sample_x.shape == torch.Size([10]), f"Sample shape: got {sample_x.shape}, expected (10,)"
        assert torch.equal(sample_x, X[idx]), f"X[{idx}] doesn't match original: got {sample_x[:3]}, expected {X[idx][:3]}"
        assert sample_y == y[idx], f"y[{idx}] doesn't match: got {sample_y}, expected {y[idx]}"

def test_transform_dataset():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    
    transform = lambda x: x + 1
    dataset = TransformDataset(X, y, transform=transform)
    
    sample_x, _ = dataset[0]
    assert sample_x is not None, "Not implemented"
    
    expected = X[0] + 1
    assert torch.allclose(sample_x, expected), "Transform not applied"

def test_custom_collate():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    x0, y0 = torch.randn(10), torch.tensor(0)
    x1, y1 = torch.randn(10), torch.tensor(1)
    x2, y2 = torch.randn(10), torch.tensor(2)
    batch = [(x0, y0), (x1, y1), (x2, y2)]
    
    x_batch, y_batch = custom_collate(batch)
    
    assert x_batch is not None, "Not implemented"
    assert x_batch.shape == torch.Size([3, 10]), f"X shape: got {x_batch.shape}, expected (3, 10)"
    assert y_batch.shape == torch.Size([3]), f"Y shape: got {y_batch.shape}, expected (3,)"
    
    assert torch.equal(x_batch[0], x0), "x_batch[0] doesn't match original"
    assert torch.equal(x_batch[1], x1), "x_batch[1] doesn't match original"
    assert torch.equal(x_batch[2], x2), "x_batch[2] doesn't match original"
    
    expected_y = torch.tensor([0, 1, 2])
    assert torch.equal(y_batch, expected_y), f"y_batch: got {y_batch}, expected {expected_y}"

def test_sequence_dataset():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    data = torch.arange(20).float()
    seq_length = 5
    dataset = SequenceDataset(data, seq_length)
    
    assert len(dataset) != 0, "Not implemented"
    assert len(dataset) == 15, f"Length: {len(dataset)}"
    
    x, y = dataset[0]
    assert x is not None, "__getitem__ not implemented"
    
    expected_x = torch.arange(5).float()
    expected_y = torch.arange(1, 6).float()
    
    assert torch.allclose(x, expected_x), f"x: {x}"
    assert torch.allclose(y, expected_y), f"y: {y}"

def test_balanced_sampler():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    labels = torch.cat([torch.zeros(90), torch.ones(10)]).long()
    sampler = BalancedSampler(labels, batch_size=16)
    
    indices = list(sampler)
    
    assert len(indices) != 0, "Not implemented"
    
    sampled_labels = labels[indices]
    class_0_count = (sampled_labels == 0).sum().item()
    class_1_count = (sampled_labels == 1).sum().item()
    
    ratio = max(class_0_count, class_1_count) / min(class_0_count, class_1_count)
    assert ratio <= 2, f"Not balanced: {class_0_count} vs {class_1_count}"

def test_dataloader_integration():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    dataset = SimpleDataset(X, y)
    
    assert len(dataset) != 0, "Dataset not implemented"
    
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    batch_count = 0
    for batch_x, batch_y in loader:
        batch_count += 1
        assert batch_x.shape[0] <= 16, f"Batch too large: {batch_x.shape}"
    
    assert batch_count != 0, "No batches"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
