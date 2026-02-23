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
    
    TODO: Implement training step
    HINT:
        model.train()
        optimizer.zero_grad()
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()
    """
    return 0.0  # Replace


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
    
    TODO: Implement validation step
    HINT:
        model.eval()
        with torch.no_grad():
            inputs, targets = batch
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # For classification
            preds = outputs.argmax(dim=-1)
            accuracy = (preds == targets).float().mean()
            
            return {'loss': loss.item(), 'accuracy': accuracy.item()}
    """
    return {'loss': 0.0, 'accuracy': 0.0}  # Replace


# ============================================================================
# Exercise 3: Full Epoch Training
# ============================================================================

def train_epoch(model: nn.Module, dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, loss_fn: Callable) -> float:
    """
    Train for one full epoch.
    
    Returns:
        Average loss over epoch
    
    TODO: Implement epoch training
    HINT:
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            loss = train_step(model, batch, optimizer, loss_fn)
            total_loss += loss
        return total_loss / len(dataloader)
    """
    return 0.0  # Replace


def validate_epoch(model: nn.Module, dataloader: DataLoader, 
                   loss_fn: Callable) -> Dict[str, float]:
    """
    Validate for one full epoch.
    
    Returns:
        Dictionary with average loss and accuracy
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    
    # TODO: Implement validation epoch
    # HINT: Similar to train_epoch but use val_step
    
    return {'loss': 0.0, 'accuracy': 0.0}  # Replace


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
    
    TODO: Implement complete training
    HINT:
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
            val_metrics = validate_epoch(model, val_loader, loss_fn)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.4f}")
        
        return history
    """
    return {'train_loss': [], 'val_loss': [], 'val_accuracy': []}  # Replace


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
        """
        TODO: Check if training should stop
        HINT:
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True
            return self.should_stop
        """
        return False  # Replace


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
