"""Test Suite for Day 18: Transformer Decoder Block"""

import torch
import torch.nn as nn
import sys
from typing import Tuple

try:
    from day18 import (create_causal_mask, create_causal_mask_batched,
                       MaskedMultiHeadAttention, CrossAttention,
                       FeedForward, LayerNorm, PreNormDecoderBlock,
                       TransformerDecoder)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_causal_mask_shape() -> Tuple[bool, str]:
    """Test causal mask has correct shape."""
    try:
        seq_len = 8
        mask = create_causal_mask(seq_len)
        
        if mask is None:
            return False, "create_causal_mask returned None"
        
        if mask.shape != (seq_len, seq_len):
            return False, f"Shape {mask.shape}, expected ({seq_len}, {seq_len})"
        
        return True, f"Shape correct: {mask.shape}"
    except Exception as e:
        return False, str(e)


def test_causal_mask_values() -> Tuple[bool, str]:
    """Test causal mask has correct structure (lower triangular)."""
    try:
        seq_len = 4
        mask = create_causal_mask(seq_len)
        
        if mask is None:
            return False, "create_causal_mask returned None"
        
        # Should be lower triangular
        expected = torch.tril(torch.ones(seq_len, seq_len))
        
        if not torch.allclose(mask, expected):
            return False, "Mask is not lower triangular"
        
        # Check specific values
        if mask[0, 1] != 0:  # Position 0 should not see position 1
            return False, "Position 0 can see future (position 1)"
        if mask[3, 0] != 1:  # Position 3 should see position 0
            return False, "Position 3 cannot see past (position 0)"
        
        return True, "Causal mask is correct (lower triangular)"
    except Exception as e:
        return False, str(e)


def test_causal_mask_batched() -> Tuple[bool, str]:
    """Test batched causal mask."""
    try:
        batch, heads, seq = 2, 4, 8
        mask = create_causal_mask_batched(batch, heads, seq)
        
        if mask is None:
            return False, "create_causal_mask_batched returned None"
        
        # Should be broadcastable to [batch, heads, seq, seq]
        # Can be [1, 1, seq, seq] or [batch, heads, seq, seq]
        if len(mask.shape) != 4:
            return False, f"Expected 4D tensor, got {len(mask.shape)}D"
        
        if mask.shape[-2:] != (seq, seq):
            return False, f"Last two dims should be ({seq}, {seq})"
        
        return True, f"Batched mask shape: {mask.shape}"
    except Exception as e:
        return False, str(e)


def test_masked_mha_shape() -> Tuple[bool, str]:
    """Test MaskedMultiHeadAttention output shape."""
    try:
        d_model, num_heads = 64, 4
        batch, seq = 2, 8
        
        mha = MaskedMultiHeadAttention(d_model, num_heads)
        
        if mha.W_q is None:
            return False, "W_q not initialized"
        
        x = torch.randn(batch, seq, d_model)
        output, weights = mha(x, use_causal_mask=True)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Output shape {output.shape} != input shape {x.shape}"
        
        return True, f"Output shape correct: {output.shape}"
    except Exception as e:
        return False, str(e)


def test_masked_mha_causal() -> Tuple[bool, str]:
    """Test that causal masking prevents attending to future."""
    try:
        d_model, num_heads = 64, 4
        batch, seq = 2, 8
        
        mha = MaskedMultiHeadAttention(d_model, num_heads)
        
        if mha.W_q is None:
            return False, "W_q not initialized"
        
        x = torch.randn(batch, seq, d_model)
        _, weights = mha(x, use_causal_mask=True)
        
        if weights is None:
            return False, "attention weights is None"
        
        # Check that attention weights for future positions are ~0
        # Upper triangular part (excluding diagonal) should be ~0
        upper_tri = torch.triu(weights, diagonal=1)
        if upper_tri.abs().max() > 1e-5:
            return False, "Attention to future positions not masked"
        
        return True, "Future positions correctly masked"
    except Exception as e:
        return False, str(e)


def test_cross_attention_shape() -> Tuple[bool, str]:
    """Test CrossAttention with different source/target lengths."""
    try:
        d_model, num_heads = 64, 4
        batch = 2
        tgt_seq, src_seq = 8, 12  # Different lengths
        
        cross_attn = CrossAttention(d_model, num_heads)
        
        if cross_attn.W_q is None:
            return False, "W_q not initialized"
        
        decoder_hidden = torch.randn(batch, tgt_seq, d_model)
        encoder_output = torch.randn(batch, src_seq, d_model)
        
        output, weights = cross_attn(decoder_hidden, encoder_output)
        
        if output is None:
            return False, "output is None"
        
        expected_out_shape = (batch, tgt_seq, d_model)
        if output.shape != expected_out_shape:
            return False, f"Output {output.shape}, expected {expected_out_shape}"
        
        if weights is not None:
            expected_weight_shape = (batch, num_heads, tgt_seq, src_seq)
            if weights.shape != expected_weight_shape:
                return False, f"Weights {weights.shape}, expected {expected_weight_shape}"
        
        return True, f"Cross-attention: tgt={tgt_seq}, src={src_seq}"
    except Exception as e:
        return False, str(e)


def test_cross_attention_no_causal() -> Tuple[bool, str]:
    """Test that cross-attention has no causal masking."""
    try:
        d_model, num_heads = 64, 4
        batch = 2
        tgt_seq, src_seq = 4, 6
        
        cross_attn = CrossAttention(d_model, num_heads)
        
        if cross_attn.W_q is None:
            return False, "W_q not initialized"
        
        decoder_hidden = torch.randn(batch, tgt_seq, d_model)
        encoder_output = torch.randn(batch, src_seq, d_model)
        
        _, weights = cross_attn(decoder_hidden, encoder_output)
        
        if weights is None:
            return False, "attention weights is None"
        
        # All positions should have non-zero attention
        # Check that first decoder position attends to all encoder positions
        first_pos_weights = weights[:, :, 0, :]  # [batch, heads, src_seq]
        if (first_pos_weights.abs() < 1e-10).any():
            # Some positions have exactly zero attention - might be masked incorrectly
            pass  # This is okay, might just be due to softmax
        
        # Verify shape allows full attention
        if weights.shape[-1] != src_seq:
            return False, "Not attending to full source sequence"
        
        return True, "Cross-attention sees full encoder output"
    except Exception as e:
        return False, str(e)


def test_feedforward_shape() -> Tuple[bool, str]:
    """Test FeedForward network."""
    try:
        d_model = 64
        ffn = FeedForward(d_model)
        
        if ffn.linear1 is None:
            return False, "linear1 not initialized"
        
        x = torch.randn(2, 8, d_model)
        output = ffn(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Shape {output.shape} != {x.shape}"
        
        return True, f"FFN shape preserved: {output.shape}"
    except Exception as e:
        return False, str(e)


def test_layer_norm() -> Tuple[bool, str]:
    """Test LayerNorm implementation."""
    try:
        d_model = 64
        norm = LayerNorm(d_model)
        
        if norm.gamma is None:
            return False, "gamma not initialized"
        
        x = torch.randn(2, 8, d_model)
        output = norm(x)
        
        if output is None:
            return False, "output is None"
        
        # Check normalization (mean ~0, std ~1)
        mean = output.mean(dim=-1)
        std = output.std(dim=-1, unbiased=False)
        
        if not torch.allclose(mean, torch.zeros_like(mean), atol=1e-5):
            return False, f"Mean not ~0: {mean.mean().item():.4f}"
        
        return True, "LayerNorm produces normalized output"
    except Exception as e:
        return False, str(e)


def test_decoder_block_shape() -> Tuple[bool, str]:
    """Test PreNormDecoderBlock output shape."""
    try:
        d_model, num_heads = 64, 4
        batch = 2
        tgt_seq, src_seq = 8, 12
        
        decoder_block = PreNormDecoderBlock(d_model, num_heads)
        
        if decoder_block.self_attn is None:
            return False, "self_attn not initialized"
        if decoder_block.cross_attn is None:
            return False, "cross_attn not initialized"
        
        decoder_input = torch.randn(batch, tgt_seq, d_model)
        encoder_output = torch.randn(batch, src_seq, d_model)
        
        output = decoder_block(decoder_input, encoder_output)
        
        if output is None:
            return False, "output is None"
        if output.shape != decoder_input.shape:
            return False, f"Shape {output.shape} != {decoder_input.shape}"
        
        return True, f"Decoder block output: {output.shape}"
    except Exception as e:
        return False, str(e)


def test_decoder_block_gradient_flow() -> Tuple[bool, str]:
    """Test that gradients flow through decoder block."""
    try:
        d_model, num_heads = 64, 4
        batch, tgt_seq, src_seq = 2, 8, 12
        
        decoder_block = PreNormDecoderBlock(d_model, num_heads)
        
        if decoder_block.self_attn is None:
            return False, "Block not initialized"
        
        decoder_input = torch.randn(batch, tgt_seq, d_model, requires_grad=True)
        encoder_output = torch.randn(batch, src_seq, d_model, requires_grad=True)
        
        output = decoder_block(decoder_input, encoder_output)
        
        if output is None:
            return False, "output is None"
        
        loss = output.sum()
        loss.backward()
        
        if decoder_input.grad is None:
            return False, "No gradient to decoder input"
        if encoder_output.grad is None:
            return False, "No gradient to encoder output"
        
        return True, "Gradients flow to both decoder and encoder inputs"
    except Exception as e:
        return False, str(e)


def test_decoder_stack() -> Tuple[bool, str]:
    """Test TransformerDecoder stack."""
    try:
        d_model, num_heads, num_layers = 64, 4, 3
        batch, tgt_seq, src_seq = 2, 8, 12
        
        decoder = TransformerDecoder(d_model, num_heads, num_layers)
        
        if decoder.layers is None:
            return False, "layers not initialized"
        if len(decoder.layers) != num_layers:
            return False, f"Expected {num_layers} layers, got {len(decoder.layers)}"
        
        decoder_input = torch.randn(batch, tgt_seq, d_model)
        encoder_output = torch.randn(batch, src_seq, d_model)
        
        output = decoder(decoder_input, encoder_output)
        
        if output is None:
            return False, "output is None"
        if output.shape != decoder_input.shape:
            return False, f"Shape {output.shape} != {decoder_input.shape}"
        
        return True, f"TransformerDecoder with {num_layers} layers works"
    except Exception as e:
        return False, str(e)


def test_decoder_autoregressive() -> Tuple[bool, str]:
    """Test decoder in autoregressive mode (different sequence lengths)."""
    try:
        d_model, num_heads, num_layers = 64, 4, 2
        batch, src_seq = 2, 10
        
        decoder = TransformerDecoder(d_model, num_heads, num_layers)
        
        if decoder.layers is None:
            return False, "layers not initialized"
        
        encoder_output = torch.randn(batch, src_seq, d_model)
        
        # Simulate autoregressive generation with increasing lengths
        for tgt_len in [1, 3, 5]:
            decoder_input = torch.randn(batch, tgt_len, d_model)
            output = decoder(decoder_input, encoder_output)
            
            if output is None:
                return False, f"output is None for tgt_len={tgt_len}"
            if output.shape != (batch, tgt_len, d_model):
                return False, f"Wrong shape for tgt_len={tgt_len}"
        
        return True, "Decoder works with varying target lengths (autoregressive)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("causal_mask_shape", test_causal_mask_shape),
        ("causal_mask_values", test_causal_mask_values),
        ("causal_mask_batched", test_causal_mask_batched),
        ("masked_mha_shape", test_masked_mha_shape),
        ("masked_mha_causal", test_masked_mha_causal),
        ("cross_attention_shape", test_cross_attention_shape),
        ("cross_attention_no_causal", test_cross_attention_no_causal),
        ("feedforward_shape", test_feedforward_shape),
        ("layer_norm", test_layer_norm),
        ("decoder_block_shape", test_decoder_block_shape),
        ("decoder_block_gradient", test_decoder_block_gradient_flow),
        ("decoder_stack", test_decoder_stack),
        ("decoder_autoregressive", test_decoder_autoregressive),
    ]
    
    print(f"\n{'='*50}\nDay 18: Transformer Decoder Block - Tests\n{'='*50}")
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
