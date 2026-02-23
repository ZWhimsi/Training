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
        """
        Return the number of samples.
        HINT: return len(self.X)
        """
        return 0  # Replace
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return sample at index.
        HINT: return self.X[idx], self.y[idx]
        """
        return None, None  # Replace


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
        TODO: Apply transform if provided
        HINT:
            x = self.X[idx]
            if self.transform:
                x = self.transform(x)
            return x, self.y[idx]
        """
        return None, None  # Replace


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
    
    TODO: Stack samples into batches
    HINT:
        xs = [item[0] for item in batch]
        ys = [item[1] for item in batch]
        return torch.stack(xs), torch.stack(ys)
    """
    return None, None  # Replace


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
        """
        TODO: Return number of valid sequences
        HINT: return len(self.data) - self.seq_length
        """
        return 0  # Replace
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (input_sequence, target_sequence).
        Input is data[idx:idx+seq_length]
        Target is data[idx+1:idx+seq_length+1]
        
        TODO: Implement sequence extraction
        """
        return None, None  # Replace


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
        TODO: Yield indices that balance classes
        HINT:
            indices = []
            for c, idx_list in self.indices_per_class.items():
                # Oversample to match max_count
                while len(idx_list) < self.max_count:
                    idx_list = idx_list + idx_list
                indices.extend(idx_list[:self.max_count])
            
            # Shuffle
            import random
            random.shuffle(indices)
            return iter(indices)
        """
        return iter([])  # Replace
    
    def __len__(self) -> int:
        return self.max_count * len(self.class_counts)


# ============================================================================
# Exercise 6: DataLoader Usage
# ============================================================================

def demonstrate_dataloader():
    """
    Demonstrate DataLoader features.
    
    TODO: Create dataloader with various options
    HINT:
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        dataset = SimpleDataset(X, y)
        
        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,  # Set to 0 for simplicity
            drop_last=True,  # Drop incomplete batches
        )
        
        return loader
    """
    return None  # Replace


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
