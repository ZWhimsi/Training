"""
Day 8: Training Loop
====================
Estimated time: 1-2 hours
Prerequisites: Day 7 (optimizers)

Learning objectives:
- Build a complete training loop
- Implement validation
- Track metrics and losses
- Handle batching and epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Callable


# ============================================================================
# Exercise 1: Simple Training Step
# ============================================================================

def train_step(model: nn.Module, batch: tuple, optimizer: torch.optim.Optimizer,
               loss_fn: Callable) -> float:
    """
    Perform a single training step.
    
    Args:
        model: Neural network
        batch: Tuple of (inputs, targets)
        optimizer: Optimizer instance
        loss_fn: Loss function
    
    Returns:
        Loss value as float
    """
    # API hints:
    # - model.train() -> set model to training mode
    # - optimizer.zero_grad() -> clear gradients
    # - model(inputs) -> forward pass
    # - loss_fn(outputs, targets) -> compute loss
    # - loss.backward() -> backpropagation
    # - optimizer.step() -> update weights
    # - loss.item() -> get Python scalar
    
    return 0.0


# ============================================================================
# Exercise 2: Validation Step
# ============================================================================

def val_step(model: nn.Module, batch: tuple, loss_fn: Callable) -> Dict[str, float]:
    """
    Perform a single validation step.
    
    Args:
        model: Neural network
        batch: Tuple of (inputs, targets)
        loss_fn: Loss function
    
    Returns:
        Dictionary with 'loss' and 'accuracy'
    """
    # API hints:
    # - model.eval() -> set model to evaluation mode
    # - torch.no_grad() -> disable gradient computation
    # - outputs.argmax(dim=-1) -> predicted class indices
    # - (preds == targets).float().mean() -> accuracy calculation
    # - tensor.item() -> get Python scalar
    
    return {'loss': 0.0, 'accuracy': 0.0}


# ============================================================================
# Exercise 3: Full Epoch Training
# ============================================================================

def train_epoch(model: nn.Module, dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, loss_fn: Callable) -> float:
    """
    Train for one full epoch.
    
    Returns:
        Average loss over epoch
    """
    # API hints:
    # - model.train() -> set training mode
    # - for batch in dataloader -> iterate over batches
    # - train_step(model, batch, optimizer, loss_fn) -> single training step
    # - total_loss / len(dataloader) -> average loss
    
    return 0.0


def validate_epoch(model: nn.Module, dataloader: DataLoader, 
                   loss_fn: Callable) -> Dict[str, float]:
    """
    Validate for one full epoch.
    
    Returns:
        Dictionary with average loss and accuracy
    """
    # API hints:
    # - model.eval() -> set evaluation mode
    # - for batch in dataloader -> iterate over batches
    # - val_step(model, batch, loss_fn) -> single validation step
    # - Accumulate and average loss and accuracy
    
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    
    return {'loss': 0.0, 'accuracy': 0.0}


# ============================================================================
# Exercise 4: Complete Training Function
# ============================================================================

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                optimizer: torch.optim.Optimizer, loss_fn: Callable,
                n_epochs: int = 10) -> Dict[str, List[float]]:
    """
    Complete training function with validation.
    
    Returns:
        History dictionary with 'train_loss', 'val_loss', 'val_accuracy'
    """
    # API hints:
    # - for epoch in range(n_epochs) -> training loop
    # - train_epoch(model, train_loader, optimizer, loss_fn) -> one epoch of training
    # - validate_epoch(model, val_loader, loss_fn) -> one epoch of validation
    # - history['key'].append(value) -> track metrics
    # - print() -> log progress
    
    return {'train_loss': [], 'val_loss': [], 'val_accuracy': []}


# ============================================================================
# Exercise 5: Early Stopping
# ============================================================================

class EarlyStopping:
    """
    Stop training if validation loss doesn't improve.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop based on validation loss."""
        # API hints:
        # - Compare val_loss to self.best_loss - self.min_delta
        # - If improved: update best_loss, reset counter
        # - If not improved: increment counter
        # - If counter >= patience: set should_stop = True
        # - Return self.should_stop
        
        return False


# ============================================================================
# Helper: Create Toy Dataset
# ============================================================================

def create_toy_dataset(n_samples: int = 1000, n_features: int = 10, n_classes: int = 3):
    """Create a simple classification dataset."""
    X = torch.randn(n_samples, n_features)
    # Simple rule: class determined by first feature
    y = (X[:, 0] * 3 + 0.5).long().clamp(0, n_classes - 1)
    return TensorDataset(X, y)


if __name__ == "__main__":
    print("Day 8: Training Loop")
    print("=" * 50)
    
    # Create model and data
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 3)
    )
    
    train_data = create_toy_dataset(800)
    val_data = create_toy_dataset(200)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    print("\nTraining model:")
    history = train_model(model, train_loader, val_loader, optimizer, loss_fn, n_epochs=5)
    
    if history['train_loss']:
        print(f"\nFinal: train_loss={history['train_loss'][-1]:.4f}, val_acc={history['val_accuracy'][-1]:.4f}")
    
    print("\nRun test_day08.py to verify all implementations!")
