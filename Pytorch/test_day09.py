"""Test Suite for Day 9: Data Loading"""

import torch
from torch.utils.data import DataLoader
from typing import Tuple

try:
    from day09 import (SimpleDataset, TransformDataset, custom_collate,
                       SequenceDataset, BalancedSampler, demonstrate_dataloader)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_simple_dataset() -> Tuple[bool, str]:
    try:
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        dataset = SimpleDataset(X, y)
        
        if len(dataset) == 0:
            return False, "Not implemented"
        
        if len(dataset) != 100:
            return False, f"Length: {len(dataset)}"
        
        sample_x, sample_y = dataset[0]
        if sample_x is None:
            return False, "__getitem__ not implemented"
        
        if sample_x.shape != torch.Size([10]):
            return False, f"Sample shape: {sample_x.shape}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_transform_dataset() -> Tuple[bool, str]:
    try:
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        
        # Transform: add 1 to all values
        transform = lambda x: x + 1
        dataset = TransformDataset(X, y, transform=transform)
        
        sample_x, _ = dataset[0]
        if sample_x is None:
            return False, "Not implemented"
        
        # Check transform was applied
        expected = X[0] + 1
        if not torch.allclose(sample_x, expected):
            return False, "Transform not applied"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_custom_collate() -> Tuple[bool, str]:
    try:
        batch = [
            (torch.randn(10), torch.tensor(0)),
            (torch.randn(10), torch.tensor(1)),
            (torch.randn(10), torch.tensor(2)),
        ]
        
        x_batch, y_batch = custom_collate(batch)
        
        if x_batch is None:
            return False, "Not implemented"
        
        if x_batch.shape != torch.Size([3, 10]):
            return False, f"X shape: {x_batch.shape}"
        
        if y_batch.shape != torch.Size([3]):
            return False, f"Y shape: {y_batch.shape}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_sequence_dataset() -> Tuple[bool, str]:
    try:
        data = torch.arange(20).float()
        seq_length = 5
        dataset = SequenceDataset(data, seq_length)
        
        if len(dataset) == 0:
            return False, "Not implemented"
        
        # Should have 20 - 5 = 15 sequences
        if len(dataset) != 15:
            return False, f"Length: {len(dataset)}"
        
        x, y = dataset[0]
        if x is None:
            return False, "__getitem__ not implemented"
        
        # x should be [0, 1, 2, 3, 4]
        # y should be [1, 2, 3, 4, 5]
        expected_x = torch.arange(5).float()
        expected_y = torch.arange(1, 6).float()
        
        if not torch.allclose(x, expected_x):
            return False, f"x: {x}"
        if not torch.allclose(y, expected_y):
            return False, f"y: {y}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_balanced_sampler() -> Tuple[bool, str]:
    try:
        # Imbalanced labels: 90 class 0, 10 class 1
        labels = torch.cat([torch.zeros(90), torch.ones(10)]).long()
        sampler = BalancedSampler(labels, batch_size=16)
        
        indices = list(sampler)
        
        if len(indices) == 0:
            return False, "Not implemented"
        
        # Should have roughly equal samples from each class
        sampled_labels = labels[indices]
        class_0_count = (sampled_labels == 0).sum().item()
        class_1_count = (sampled_labels == 1).sum().item()
        
        # Should be roughly balanced (within 2x)
        ratio = max(class_0_count, class_1_count) / min(class_0_count, class_1_count)
        if ratio > 2:
            return False, f"Not balanced: {class_0_count} vs {class_1_count}"
        
        return True, f"OK ({class_0_count}:{class_1_count})"
    except Exception as e:
        return False, str(e)


def test_dataloader_integration() -> Tuple[bool, str]:
    try:
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        dataset = SimpleDataset(X, y)
        
        if len(dataset) == 0:
            return False, "Dataset not implemented"
        
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        batch_count = 0
        for batch_x, batch_y in loader:
            batch_count += 1
            if batch_x.shape[0] > 16:
                return False, f"Batch too large: {batch_x.shape}"
        
        if batch_count == 0:
            return False, "No batches"
        
        return True, f"OK ({batch_count} batches)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("simple_dataset", test_simple_dataset),
        ("transform_dataset", test_transform_dataset),
        ("custom_collate", test_custom_collate),
        ("sequence_dataset", test_sequence_dataset),
        ("balanced_sampler", test_balanced_sampler),
        ("dataloader_integration", test_dataloader_integration),
    ]
    
    print(f"\n{'='*50}\nDay 9: Data Loading - Tests\n{'='*50}")
    
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        return
    
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    print(f"\nSummary: {passed}/{len(tests)}")


if __name__ == "__main__":
    run_all_tests()
