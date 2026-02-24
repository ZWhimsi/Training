"""Test Suite for Day 22: DataLoader"""

import numpy as np
import pytest

from day22 import Tensor, DataLoader


def test_dataloader_creation():
    """Test DataLoader creation."""
    X = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = Tensor([0, 1, 0, 1])
    
    loader = DataLoader(X, y, batch_size=2)
    
    assert loader is not None, "DataLoader returned None"


def test_dataloader_iteration():
    """Test DataLoader iteration."""
    X = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = Tensor([0, 1, 2, 3])
    
    loader = DataLoader(X, y, batch_size=2)
    
    batches = list(loader)
    
    assert len(batches) == 2, f"Should have 2 batches, got {len(batches)}"


def test_dataloader_batch_contents():
    """Test DataLoader batch contents."""
    X = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = Tensor([0, 1])
    
    loader = DataLoader(X, y, batch_size=2, shuffle=False)
    
    for X_batch, y_batch in loader:
        assert X_batch is not None and y_batch is not None, "Batch is None"
        assert np.allclose(X_batch.data, [[1, 2], [3, 4]]), f"X_batch = {X_batch.data}"
        assert np.allclose(y_batch.data, [0, 1]), f"y_batch = {y_batch.data}"


def test_dataloader_batch_size():
    """Test DataLoader respects batch size."""
    X = Tensor(np.arange(10).reshape(10, 1))
    y = Tensor(np.arange(10))
    
    loader = DataLoader(X, y, batch_size=3)
    
    for X_batch, y_batch in loader:
        assert X_batch.shape[0] <= 3, f"Batch size = {X_batch.shape[0]}, expected <= 3"


def test_dataloader_shuffle():
    """Test DataLoader shuffling."""
    X = Tensor(np.arange(100).reshape(100, 1))
    y = Tensor(np.arange(100))
    
    loader1 = DataLoader(X, y, batch_size=10, shuffle=True)
    loader2 = DataLoader(X, y, batch_size=10, shuffle=True)
    
    # With different seeds, order should be different
    # (Unless we get very unlucky)
    batches1 = [X_batch.data.flatten().tolist() for X_batch, _ in loader1]
    batches2 = [X_batch.data.flatten().tolist() for X_batch, _ in loader2]
    
    # At least one batch should be different (with high probability)
    # Or all batches should be shuffled from original order
    first_batch1 = batches1[0]
    expected_unshuffled = list(range(10))
    
    # Either shuffled from expected OR different between runs
    is_shuffled = first_batch1 != expected_unshuffled or batches1 != batches2
    assert is_shuffled, "DataLoader doesn't seem to shuffle"


def test_dataloader_no_shuffle():
    """Test DataLoader without shuffling."""
    X = Tensor(np.arange(6).reshape(6, 1))
    y = Tensor(np.arange(6))
    
    loader = DataLoader(X, y, batch_size=3, shuffle=False)
    
    batches = list(loader)
    
    X1, _ = batches[0]
    X2, _ = batches[1]
    
    assert np.allclose(X1.data.flatten(), [0, 1, 2]), f"First batch = {X1.data.flatten()}"
    assert np.allclose(X2.data.flatten(), [3, 4, 5]), f"Second batch = {X2.data.flatten()}"


def test_dataloader_drop_last():
    """Test DataLoader drop_last option."""
    X = Tensor(np.arange(10).reshape(10, 1))
    y = Tensor(np.arange(10))
    
    # With drop_last=True, 10 samples / batch_size 3 = 3 full batches (9 samples)
    loader = DataLoader(X, y, batch_size=3, drop_last=True)
    batches = list(loader)
    
    assert len(batches) == 3, f"With drop_last, should have 3 batches, got {len(batches)}"
    
    # Without drop_last, should have 4 batches (last one has 1 sample)
    loader2 = DataLoader(X, y, batch_size=3, drop_last=False)
    batches2 = list(loader2)
    
    assert len(batches2) == 4, f"Without drop_last, should have 4 batches, got {len(batches2)}"


def test_dataloader_len():
    """Test DataLoader __len__."""
    X = Tensor(np.arange(10).reshape(10, 1))
    y = Tensor(np.arange(10))
    
    loader = DataLoader(X, y, batch_size=3)
    
    assert len(loader) == 4, f"len = {len(loader)}, expected 4"


def test_dataloader_multiple_epochs():
    """Test DataLoader over multiple epochs."""
    X = Tensor([[1, 2], [3, 4]])
    y = Tensor([0, 1])
    
    loader = DataLoader(X, y, batch_size=2)
    
    # Should be able to iterate multiple times
    for epoch in range(3):
        count = 0
        for batch in loader:
            count += 1
        assert count > 0, f"Epoch {epoch} had no batches"


def test_dataloader_X_only():
    """Test DataLoader with X only (no y)."""
    X = Tensor([[1, 2], [3, 4], [5, 6]])
    
    try:
        loader = DataLoader(X, batch_size=2)
        
        for batch in loader:
            assert batch is not None, "Batch is None"
    except TypeError:
        pytest.skip("DataLoader requires y argument")


def test_dataloader_single_sample():
    """Test DataLoader with single sample."""
    X = Tensor([[1, 2, 3]])
    y = Tensor([0])
    
    loader = DataLoader(X, y, batch_size=1)
    
    batches = list(loader)
    assert len(batches) == 1, f"Should have 1 batch, got {len(batches)}"
    
    X_batch, y_batch = batches[0]
    assert X_batch.shape == (1, 3), f"X shape = {X_batch.shape}"


def test_dataloader_large_batch():
    """Test DataLoader with batch_size larger than dataset."""
    X = Tensor([[1, 2], [3, 4]])
    y = Tensor([0, 1])
    
    loader = DataLoader(X, y, batch_size=100)
    
    batches = list(loader)
    assert len(batches) == 1, f"Should have 1 batch, got {len(batches)}"
    
    X_batch, _ = batches[0]
    assert X_batch.shape[0] == 2, f"Batch should have all 2 samples"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
