"""
Day 9: Data Loading
===================
Estimated time: 1-2 hours
Prerequisites: Day 8 (training loop)

Learning objectives:
- Understand Dataset and DataLoader
- Create custom datasets
- Implement data transforms
- Handle batching, shuffling, and workers
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from typing import List, Tuple, Callable, Any


# ============================================================================
# Exercise 1: Custom Dataset Class
# ============================================================================

class SimpleDataset(Dataset):
    """
    Basic custom dataset.
    
    TODO: Implement __len__ and __getitem__
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        # API hints:
        # - len(tensor) -> returns the size of the first dimension
        return None
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the (input, target) tuple at the given index."""
        # API hints:
        # - tensor[idx] -> index into tensor to get a single sample
        return None


# ============================================================================
# Exercise 2: Dataset with Transform
# ============================================================================

class TransformDataset(Dataset):
    """
    Dataset that applies transforms to data.
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor, 
                 transform: Callable = None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the sample at index, applying transform to input if provided.
        """
        # API hints:
        # - self.transform(x) -> apply a callable transform to input tensor
        # - Check if self.transform is not None before applying
        return None


# ============================================================================
# Exercise 3: Collate Function
# ============================================================================

def custom_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function to combine samples into a batch.
    
    Args:
        batch: List of (x, y) tuples
    
    Returns:
        Tuple of (batched_x, batched_y)
    """
    # API hints:
    # - torch.stack(tensors, dim=0) -> concatenate tensors along a new dimension
    # - List comprehension to extract x's and y's from batch tuples
    return None


# ============================================================================
# Exercise 4: Sequence Dataset
# ============================================================================

class SequenceDataset(Dataset):
    """
    Dataset for sequential data (like text or time series).
    Returns overlapping sequences of fixed length.
    """
    def __init__(self, data: torch.Tensor, seq_length: int):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self) -> int:
        """Return the number of valid overlapping sequences that can be extracted."""
        # API hints:
        # - Total length minus sequence length gives number of valid start positions
        return None
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (input_sequence, target_sequence) for sequence prediction.
        Input is data[idx:idx+seq_length], target is shifted by 1 position.
        """
        # API hints:
        # - tensor[start:end] -> slice tensor to get a subsequence
        # - Input: positions idx to idx+seq_length
        # - Target: positions idx+1 to idx+seq_length+1
        return None


# ============================================================================
# Exercise 5: Custom Sampler
# ============================================================================

class BalancedSampler(Sampler):
    """
    Sampler that balances classes by oversampling minority classes.
    """
    def __init__(self, labels: torch.Tensor, batch_size: int):
        self.labels = labels
        self.batch_size = batch_size
        
        # Count samples per class
        self.class_counts = {}
        for label in labels.unique():
            self.class_counts[label.item()] = (labels == label).sum().item()
        
        self.max_count = max(self.class_counts.values())
        self.indices_per_class = {
            c: (labels == c).nonzero().squeeze().tolist()
            for c in self.class_counts
        }
    
    def __iter__(self):
        """
        Yield sample indices that balance classes by oversampling minority classes.
        """
        # API hints:
        # - self.indices_per_class[c] -> list of indices for class c
        # - self.max_count -> target count to oversample each class to
        # - Duplicate lists to oversample: list + list
        # - random.shuffle(indices) -> shuffle list in-place
        # - iter(indices) -> create iterator from list
        return iter([])
    
    def __len__(self) -> int:
        return self.max_count * len(self.class_counts)


# ============================================================================
# Exercise 6: DataLoader Usage
# ============================================================================

def demonstrate_dataloader():
    """
    Create and return a DataLoader with common configuration options.
    Should create sample data, wrap in SimpleDataset, and create DataLoader.
    """
    # API hints:
    # - torch.randn(size) -> create random tensor for features
    # - torch.randint(low, high, size) -> create random integer tensor for labels
    # - SimpleDataset(X, y) -> wrap data in custom dataset
    # - DataLoader(dataset, batch_size, shuffle, num_workers, drop_last) -> create batched iterator
    return None


if __name__ == "__main__":
    print("Day 9: Data Loading")
    print("=" * 50)
    
    # Test SimpleDataset
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    
    dataset = SimpleDataset(X, y)
    print(f"\nSimpleDataset length: {len(dataset)}")
    
    if dataset[0][0] is not None:
        sample_x, sample_y = dataset[0]
        print(f"Sample shape: x={sample_x.shape}, y={sample_y}")
    
    # Test DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for batch_x, batch_y in loader:
        print(f"Batch shape: x={batch_x.shape}, y={batch_y.shape}")
        break
    
    print("\nRun test_day09.py to verify all implementations!")
