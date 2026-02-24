"""Test Suite for Day 25: Advanced Attention Patterns"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Tuple

try:
    from day25 import (
        create_sliding_window_mask, SlidingWindowAttention,
        create_dilated_mask, create_block_sparse_mask, BlockSparseAttention,
        AttentionWithSinks, LocalGlobalAttention,
        compute_attention_memory, analyze_attention_patterns
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_sliding_window_mask_shape() -> Tuple[bool, str]:
    """Test sliding window mask shape."""
    try:
        seq_len, window_size = 16, 4
        mask = create_sliding_window_mask(seq_len, window_size)
        
        if mask.shape != (seq_len, seq_len):
            return False, f"Shape {mask.shape} != {(seq_len, seq_len)}"
        
        return True, f"Shape correct: {mask.shape}"
    except Exception as e:
        return False, str(e)


def test_sliding_window_mask_causal() -> Tuple[bool, str]:
    """Test that causal sliding window mask is lower triangular-ish."""
    try:
        seq_len, window_size = 8, 3
        mask = create_sliding_window_mask(seq_len, window_size, causal=True)
        
        # Check upper triangle is zero (causal)
        upper = torch.triu(mask, diagonal=1)
        if upper.sum() > 0:
            return False, "Causal mask has non-zero upper triangle"
        
        # Check diagonal is 1 (can attend to self)
        diag = torch.diag(mask)
        if not torch.all(diag == 1):
            return False, "Diagonal should be 1"
        
        return True, "Causal mask is correct"
    except Exception as e:
        return False, str(e)


def test_sliding_window_mask_window() -> Tuple[bool, str]:
    """Test that window size is enforced."""
    try:
        seq_len, window_size = 10, 3
        mask = create_sliding_window_mask(seq_len, window_size, causal=True)
        
        # Position 7 should attend to positions 4, 5, 6, 7 (window of 3 + self)
        # So positions 0, 1, 2, 3 should be masked
        if mask[7, 0] != 0:
            return False, "Position outside window should be masked"
        if mask[7, 7] != 1:
            return False, "Current position should not be masked"
        if mask[7, 4] != 1:
            return False, "Position within window should not be masked"
        
        return True, "Window size enforced correctly"
    except Exception as e:
        return False, str(e)


def test_sliding_window_mask_bidirectional() -> Tuple[bool, str]:
    """Test bidirectional sliding window mask."""
    try:
        seq_len, window_size = 8, 2
        mask = create_sliding_window_mask(seq_len, window_size, causal=False)
        
        # Position 4 should attend to positions 2, 3, 4, 5, 6
        # (2 before, self, 2 after)
        for i in range(2, 7):
            if mask[4, i] != 1:
                return False, f"Position {i} should be attended from position 4"
        
        if mask[4, 0] != 0:
            return False, "Position 0 should be masked from position 4"
        
        return True, "Bidirectional window correct"
    except Exception as e:
        return False, str(e)


def test_sliding_window_attention_init() -> Tuple[bool, str]:
    """Test SlidingWindowAttention initialization."""
    try:
        d_model, num_heads, window = 128, 4, 8
        attn = SlidingWindowAttention(d_model, num_heads, window)
        
        if attn.W_q is None:
            return False, "W_q not initialized"
        if attn.W_k is None:
            return False, "W_k not initialized"
        if attn.window_size != window:
            return False, "window_size not set"
        
        return True, "SlidingWindowAttention initialized"
    except Exception as e:
        return False, str(e)


def test_sliding_window_attention_forward() -> Tuple[bool, str]:
    """Test SlidingWindowAttention forward pass with window pattern verification."""
    try:
        d_model, num_heads, window = 128, 4, 8
        attn = SlidingWindowAttention(d_model, num_heads, window)
        
        if attn.W_q is None:
            return False, "Attention not initialized"
        
        batch, seq_len = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_model)
        
        output, weights = attn(x)
        
        if output.shape != x.shape:
            return False, f"Output shape {output.shape} != {x.shape}"
        
        expected_weights = (batch, num_heads, seq_len, seq_len)
        if weights.shape != expected_weights:
            return False, f"Weights shape wrong"
        
        # Verify attention weights follow window pattern
        # For causal sliding window, position i should only attend to [max(0, i-window), i]
        for i in range(seq_len):
            # Should attend to self
            if weights[0, 0, i, i] < 1e-6:
                return False, f"Position {i} should attend to self"
            # Should NOT attend to future (if causal)
            if i < seq_len - 1:
                future_attn = weights[0, 0, i, i+1:].sum()
                if future_attn > 1e-5:
                    return False, f"Position {i} should not attend to future positions"
        
        # Verify output is not zeros
        if output.abs().sum() == 0:
            return False, "Output is all zeros"
        
        # Verify attention weights sum to 1 per query position
        attn_sums = weights.sum(dim=-1)
        if not torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4):
            return False, "Attention weights don't sum to 1"
        
        return True, f"Forward pass works with valid window pattern"
    except Exception as e:
        return False, str(e)


def test_sliding_window_attention_sparse() -> Tuple[bool, str]:
    """Test that sliding window attention has sparse pattern."""
    try:
        d_model, num_heads, window = 128, 4, 4
        attn = SlidingWindowAttention(d_model, num_heads, window)
        
        if attn.W_q is None:
            return False, "Attention not initialized"
        
        batch, seq_len = 1, 32
        x = torch.randn(batch, seq_len, d_model)
        
        _, weights = attn(x)
        
        # Check that many positions have zero attention (sparse)
        # For window=4, each position attends to at most 5 positions
        # So sparsity should be high for long sequences
        zero_ratio = (weights < 1e-5).float().mean()
        
        # Should have significant sparsity
        if zero_ratio < 0.5:
            return False, f"Not sparse enough: {zero_ratio:.2%} zeros"
        
        return True, f"Attention is {zero_ratio:.1%} sparse"
    except Exception as e:
        return False, str(e)


def test_dilated_mask_shape() -> Tuple[bool, str]:
    """Test dilated attention mask shape."""
    try:
        seq_len, window, dilation = 16, 4, 2
        mask = create_dilated_mask(seq_len, window, dilation)
        
        if mask.shape != (seq_len, seq_len):
            return False, f"Shape {mask.shape} != {(seq_len, seq_len)}"
        
        return True, "Dilated mask shape correct"
    except Exception as e:
        return False, str(e)


def test_dilated_mask_pattern() -> Tuple[bool, str]:
    """Test dilated mask attends to strided positions."""
    try:
        seq_len, window, dilation = 16, 3, 2
        mask = create_dilated_mask(seq_len, window, dilation, causal=True)
        
        # Position 10 should attend to: 10, 8, 6 (every 2nd, 3 total)
        # Should NOT attend to: 9, 7, 5 (odd positions)
        
        if mask[10, 10] != 1:
            return False, "Should attend to self"
        if mask[10, 8] != 1:
            return False, "Should attend to position 8 (dilation step)"
        if mask[10, 6] != 1:
            return False, "Should attend to position 6 (2 dilation steps)"
        
        # Should not attend to odd positions
        if mask[10, 9] != 0:
            return False, "Should NOT attend to position 9 (not on stride)"
        
        return True, "Dilated pattern correct"
    except Exception as e:
        return False, str(e)


def test_block_sparse_mask() -> Tuple[bool, str]:
    """Test block sparse attention mask."""
    try:
        seq_len, block_size = 16, 4
        mask = create_block_sparse_mask(seq_len, block_size, num_global_blocks=1)
        
        if mask.shape != (seq_len, seq_len):
            return False, f"Shape wrong"
        
        # Block diagonal should be 1
        if mask[0, 0] != 1 or mask[5, 5] != 1:
            return False, "Diagonal blocks should be attended"
        
        # First block should be global
        if mask[0, 15] != 1:
            return False, "Global block should attend everywhere"
        
        return True, "Block sparse mask correct"
    except Exception as e:
        return False, str(e)


def test_block_sparse_attention() -> Tuple[bool, str]:
    """Test BlockSparseAttention produces valid output."""
    try:
        d_model, num_heads, block_size = 128, 4, 8
        attn = BlockSparseAttention(d_model, num_heads, block_size)
        
        if attn.W_q is None:
            return False, "BlockSparseAttention not initialized"
        
        batch, seq_len = 2, 32
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_model)
        
        output, weights = attn(x)
        
        if output.shape != x.shape:
            return False, "Output shape wrong"
        
        # Verify output is not zeros
        if output.abs().sum() == 0:
            return False, "Output is all zeros"
        
        # Verify output has reasonable values
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, "Output contains NaN or Inf"
        
        # Verify attention weights sum to 1
        attn_sums = weights.sum(dim=-1)
        if not torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4):
            return False, "Attention weights don't sum to 1"
        
        # Verify attention has some sparsity (block pattern)
        zero_ratio = (weights < 1e-5).float().mean()
        if zero_ratio < 0.3:
            return False, f"Block sparse should have some sparsity, got {zero_ratio:.2%} zeros"
        
        return True, f"BlockSparseAttention works, sparsity={zero_ratio:.1%}"
    except Exception as e:
        return False, str(e)


def test_attention_sinks_init() -> Tuple[bool, str]:
    """Test AttentionWithSinks initialization."""
    try:
        d_model, num_heads = 128, 4
        attn = AttentionWithSinks(d_model, num_heads, num_sink_tokens=4, window_size=16)
        
        if attn.W_q is None:
            return False, "W_q not initialized"
        if attn.num_sink_tokens != 4:
            return False, "num_sink_tokens not set"
        
        return True, "AttentionWithSinks initialized"
    except Exception as e:
        return False, str(e)


def test_attention_sinks_mask() -> Tuple[bool, str]:
    """Test sink attention mask pattern."""
    try:
        d_model, num_heads = 128, 4
        attn = AttentionWithSinks(d_model, num_heads, num_sink_tokens=4, window_size=8)
        
        if attn.W_q is None:
            return False, "Attention not initialized"
        
        seq_len = 20
        mask = attn.create_sink_window_mask(seq_len)
        
        # All positions should attend to sink tokens (0-3)
        if mask[15, 0] != 1:
            return False, "Should attend to sink token 0"
        if mask[15, 3] != 1:
            return False, "Should attend to sink token 3"
        
        # Position 15 should attend to recent window (positions 8-15)
        if mask[15, 15] != 1:
            return False, "Should attend to self"
        if mask[15, 8] != 1:
            return False, "Should attend to position in window"
        
        return True, "Sink mask pattern correct"
    except Exception as e:
        return False, str(e)


def test_attention_sinks_forward() -> Tuple[bool, str]:
    """Test AttentionWithSinks forward pass with sink attention verification."""
    try:
        d_model, num_heads = 128, 4
        num_sinks = 4
        attn = AttentionWithSinks(d_model, num_heads, num_sink_tokens=num_sinks, window_size=16)
        
        if attn.W_q is None:
            return False, "Attention not initialized"
        
        batch, seq_len = 2, 32
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_model)
        
        output, weights = attn(x)
        
        if output.shape != x.shape:
            return False, "Output shape wrong"
        
        # Verify output is not zeros
        if output.abs().sum() == 0:
            return False, "Output is all zeros"
        
        # Verify output has reasonable values
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, "Output contains NaN or Inf"
        
        # Verify attention weights sum to 1
        attn_sums = weights.sum(dim=-1)
        if not torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4):
            return False, "Attention weights don't sum to 1"
        
        # Verify sink tokens receive attention (positions near end should attend to sinks)
        sink_attn = weights[0, 0, -1, :num_sinks].sum()  # Last position's attention to sinks
        if sink_attn < 0.01:
            return False, f"Sink tokens should receive some attention, got {sink_attn:.4f}"
        
        return True, f"AttentionWithSinks works, sink attention={sink_attn:.3f}"
    except Exception as e:
        return False, str(e)


def test_local_global_attention() -> Tuple[bool, str]:
    """Test LocalGlobalAttention with global token verification."""
    try:
        d_model, num_heads, local_window = 128, 4, 8
        attn = LocalGlobalAttention(d_model, num_heads, local_window)
        
        if attn.W_q is None:
            return False, "LocalGlobalAttention not initialized"
        
        batch, seq_len = 2, 32
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_model)
        global_indices = torch.tensor([0])  # CLS token
        
        output, weights = attn(x, global_indices)
        
        if output.shape != x.shape:
            return False, "Output shape wrong"
        
        # Verify output is not zeros
        if output.abs().sum() == 0:
            return False, "Output is all zeros"
        
        # Verify output has reasonable values
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, "Output contains NaN or Inf"
        
        # Verify attention weights sum to 1
        attn_sums = weights.sum(dim=-1)
        if not torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4):
            return False, "Attention weights don't sum to 1"
        
        # Verify global token (position 0) receives attention from all positions
        global_attn_received = weights[0, 0, :, 0].mean()  # Mean attention to global token
        if global_attn_received < 0.01:
            return False, f"Global token should receive attention, got mean={global_attn_received:.4f}"
        
        return True, f"LocalGlobalAttention works, global attn received={global_attn_received:.3f}"
    except Exception as e:
        return False, str(e)


def test_local_global_mask() -> Tuple[bool, str]:
    """Test local + global mask pattern."""
    try:
        d_model, num_heads, local_window = 128, 4, 4
        attn = LocalGlobalAttention(d_model, num_heads, local_window)
        
        seq_len = 16
        global_indices = torch.tensor([0])
        mask = attn.create_local_global_mask(seq_len, global_indices)
        
        # Global token (0) should attend to all
        if not torch.all(mask[0, :] == 1):
            return False, "Global token should attend to all"
        
        # All should attend to global token
        if not torch.all(mask[:, 0] == 1):
            return False, "All should attend to global token"
        
        return True, "Local + global mask correct"
    except Exception as e:
        return False, str(e)


def test_memory_comparison() -> Tuple[bool, str]:
    """Test memory comparison function."""
    try:
        mem = compute_attention_memory(8192, num_heads=32, head_dim=128)
        
        if mem['standard_bytes'] == 0:
            return False, "Standard memory not computed"
        if mem['sliding_window_bytes'] == 0:
            return False, "Sliding window memory not computed"
        
        # Sliding window should use less memory
        if mem['sliding_window_bytes'] >= mem['standard_bytes']:
            return False, "Sliding window should use less memory"
        
        return True, f"Sliding window is {mem['sliding_vs_standard']:.1f}x more efficient"
    except Exception as e:
        return False, str(e)


def test_attention_pattern_analysis() -> Tuple[bool, str]:
    """Test attention pattern analysis."""
    try:
        batch, heads, seq = 1, 4, 16
        
        # Create random attention weights
        logits = torch.randn(batch, heads, seq, seq)
        weights = F.softmax(logits, dim=-1)
        
        stats = analyze_attention_patterns(weights)
        
        if 'entropy' not in stats:
            return False, "Missing entropy"
        if 'sink_attention' not in stats:
            return False, "Missing sink_attention"
        
        return True, f"Entropy: {stats['entropy']:.2f}"
    except Exception as e:
        return False, str(e)


def test_gradient_flow_sliding() -> Tuple[bool, str]:
    """Test gradient flow through sliding window attention."""
    try:
        d_model, num_heads, window = 128, 4, 8
        attn = SlidingWindowAttention(d_model, num_heads, window)
        
        if attn.W_q is None:
            return False, "Attention not initialized"
        
        x = torch.randn(2, 16, d_model, requires_grad=True)
        output, _ = attn(x)
        
        loss = output.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "No gradient to input"
        if x.grad.abs().sum() == 0:
            return False, "Gradients are zero"
        
        return True, "Gradients flow correctly"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("sliding_window_mask_shape", test_sliding_window_mask_shape),
        ("sliding_window_mask_causal", test_sliding_window_mask_causal),
        ("sliding_window_mask_window", test_sliding_window_mask_window),
        ("sliding_window_mask_bidir", test_sliding_window_mask_bidirectional),
        ("sliding_window_attn_init", test_sliding_window_attention_init),
        ("sliding_window_attn_forward", test_sliding_window_attention_forward),
        ("sliding_window_attn_sparse", test_sliding_window_attention_sparse),
        ("dilated_mask_shape", test_dilated_mask_shape),
        ("dilated_mask_pattern", test_dilated_mask_pattern),
        ("block_sparse_mask", test_block_sparse_mask),
        ("block_sparse_attention", test_block_sparse_attention),
        ("attention_sinks_init", test_attention_sinks_init),
        ("attention_sinks_mask", test_attention_sinks_mask),
        ("attention_sinks_forward", test_attention_sinks_forward),
        ("local_global_attention", test_local_global_attention),
        ("local_global_mask", test_local_global_mask),
        ("memory_comparison", test_memory_comparison),
        ("attention_pattern_analysis", test_attention_pattern_analysis),
        ("gradient_flow_sliding", test_gradient_flow_sliding),
    ]
    
    print(f"\n{'='*50}\nDay 25: Advanced Attention Patterns - Tests\n{'='*50}")
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
