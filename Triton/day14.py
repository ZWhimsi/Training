"""
Day 14: Cross-Entropy Loss Kernel
=================================
Estimated time: 1-2 hours
Prerequisites: Day 13 (softmax)

Learning objectives:
- Understand cross-entropy loss mathematically
- Implement log-softmax efficiently
- Compute cross-entropy with numerical stability
- Handle class indices and one-hot targets
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# CONCEPT: Cross-Entropy Loss
# ============================================================================
# For classification, cross-entropy loss is:
# L = -sum(y_true * log(y_pred))
#
# For hard labels (class indices):
# L = -log(softmax(logits)[correct_class])
#
# Numerically stable version uses log-softmax:
# log_softmax(x) = x - log(sum(exp(x)))
#                = x - max(x) - log(sum(exp(x - max(x))))
# ============================================================================


# ============================================================================
# Exercise 1: Log-Softmax for a Single Row
# ============================================================================

@triton.jit
def log_softmax_row_kernel(
    input_ptr, output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax for a single row.
    
    log_softmax(x_i) = x_i - max(x) - log(sum(exp(x - max(x))))
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load row
    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    
    # TODO: Find max for stability
    # HINT: x_max = tl.max(x, axis=0)
    x_max = None  # Replace
    
    # TODO: Compute shifted values
    # HINT: x_shifted = x - x_max
    x_shifted = None  # Replace
    
    # TODO: Compute log(sum(exp(x_shifted)))
    # HINT: log_sum_exp = tl.log(tl.sum(tl.exp(x_shifted), axis=0))
    log_sum_exp = None  # Replace
    
    # TODO: Compute log_softmax
    # HINT: log_softmax = x_shifted - log_sum_exp
    log_softmax = None  # Replace
    
    # TODO: Store
    # HINT: tl.store(output_ptr + row_start + col_offsets, log_softmax, mask=mask)
    pass  # Replace


def log_softmax_rows(x: torch.Tensor) -> torch.Tensor:
    """Compute log_softmax along last dimension."""
    assert x.dim() == 2, "Expected 2D input"
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    log_softmax_row_kernel[(n_rows,)](x, output, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return output


# ============================================================================
# Exercise 2: Cross-Entropy Loss (Per Sample)
# ============================================================================

@triton.jit
def cross_entropy_kernel(
    logits_ptr, targets_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute cross-entropy loss for each sample.
    
    loss[i] = -log_softmax(logits[i])[targets[i]]
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load logits row
    logits = tl.load(logits_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    
    # Load target class for this row
    target = tl.load(targets_ptr + row_idx)
    
    # TODO: Compute log_softmax
    logits_max = tl.max(logits, axis=0)
    logits_shifted = logits - logits_max
    log_sum_exp = tl.log(tl.sum(tl.exp(logits_shifted), axis=0))
    log_softmax = logits_shifted - log_sum_exp
    
    # TODO: Get log_softmax at target index
    # HINT: target_log_prob = tl.sum(tl.where(col_offsets == target, log_softmax, 0.0), axis=0)
    target_log_prob = None  # Replace
    
    # TODO: Loss is negative log probability
    # HINT: loss = -target_log_prob
    loss = None  # Replace
    
    # TODO: Store
    # HINT: tl.store(output_ptr + row_idx, loss)
    pass  # Replace


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss per sample."""
    assert logits.dim() == 2
    n_rows, n_cols = logits.shape
    output = torch.empty(n_rows, device=logits.device, dtype=logits.dtype)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    cross_entropy_kernel[(n_rows,)](logits, targets, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return output


# ============================================================================
# Exercise 3: Mean Cross-Entropy Loss
# ============================================================================

def cross_entropy_mean(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute mean cross-entropy loss over batch.
    
    This is what's typically used for training.
    """
    # TODO: Compute per-sample loss and take mean
    # HINT: return cross_entropy_loss(logits, targets).mean()
    return None  # Replace


# ============================================================================
# Exercise 4: Cross-Entropy with Label Smoothing
# ============================================================================

@triton.jit
def cross_entropy_smooth_kernel(
    logits_ptr, targets_ptr, output_ptr,
    n_rows, n_cols,
    smoothing,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Cross-entropy with label smoothing.
    
    Instead of one-hot [0, 0, 1, 0], use:
    [(1-smoothing) * one_hot + smoothing / n_classes]
    
    This regularizes the model and prevents overconfidence.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load logits
    logits = tl.load(logits_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    target = tl.load(targets_ptr + row_idx)
    
    # Compute log_softmax
    logits_max = tl.max(logits, axis=0)
    logits_shifted = logits - logits_max
    log_sum_exp = tl.log(tl.sum(tl.exp(logits_shifted), axis=0))
    log_softmax = logits_shifted - log_sum_exp
    
    # TODO: Create smoothed labels
    # smooth_label = smoothing / n_cols for all, plus (1 - smoothing) for target
    uniform_prob = smoothing / n_cols
    is_target = tl.where(col_offsets == target, 1.0, 0.0)
    smooth_labels = None  # Replace: uniform_prob + (1.0 - smoothing) * is_target
    
    # TODO: Compute loss: -sum(smooth_labels * log_softmax)
    loss = None  # Replace: -tl.sum(smooth_labels * log_softmax, axis=0)
    
    # TODO: Store (only for valid mask)
    pass  # Replace


def cross_entropy_smooth(logits: torch.Tensor, targets: torch.Tensor, 
                         smoothing: float = 0.1) -> torch.Tensor:
    """Cross-entropy with label smoothing."""
    n_rows, n_cols = logits.shape
    output = torch.empty(n_rows, device=logits.device, dtype=logits.dtype)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    cross_entropy_smooth_kernel[(n_rows,)](
        logits, targets, output, n_rows, n_cols, smoothing, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


if __name__ == "__main__":
    print("Day 14: Cross-Entropy Loss")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        # Test setup
        batch_size, n_classes = 4, 10
        logits = torch.randn(batch_size, n_classes, device=device)
        targets = torch.randint(0, n_classes, (batch_size,), device=device)
        
        print("\nTesting log_softmax:")
        result = log_softmax_rows(logits)
        expected = torch.nn.functional.log_softmax(logits, dim=-1)
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
        
        print("\nTesting cross_entropy:")
        result = cross_entropy_loss(logits, targets)
        expected = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day14.py to verify!")
