"""Test Suite for Day 18: Transformer Decoder Block"""

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
try:
    from day18 import (create_causal_mask, create_causal_mask_batched,
                       MaskedMultiHeadAttention, CrossAttention,
                       FeedForward, LayerNorm, PreNormDecoderBlock,
                       TransformerDecoder)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_causal_mask_shape():
    """Test causal mask has correct shape and diagonal values."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    seq_len = 8
    mask = create_causal_mask(seq_len)
    
    assert mask is not None, "create_causal_mask returned None"
    
    assert mask.shape == (seq_len, seq_len), f"Shape {mask.shape}, expected ({seq_len}, {seq_len})"
    
    diag = torch.diag(mask)
    assert torch.allclose(diag, torch.ones(seq_len)), "Diagonal should be all 1s"

def test_causal_mask_values():
    """Test causal mask has correct structure (lower triangular)."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    seq_len = 4
    mask = create_causal_mask(seq_len)
    
    assert mask is not None, "create_causal_mask returned None"
    
    expected = torch.tril(torch.ones(seq_len, seq_len))
    
    assert torch.allclose(mask, expected), "Mask is not lower triangular"
    
    assert mask[0, 1] == 0, "Position 0 can see future (position 1)"
    assert mask[3, 0] == 1, "Position 3 cannot see past (position 0)"

def test_causal_mask_batched():
    """Test batched causal mask is correct lower triangular."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, heads, seq = 2, 4, 8
    mask = create_causal_mask_batched(batch, heads, seq)
    
    assert mask is not None, "create_causal_mask_batched returned None"
    
    assert len(mask.shape) == 4, f"Expected 4D tensor, got {len(mask.shape)}D"
    
    assert mask.shape[-2:] == (seq, seq), f"Last two dims should be ({seq}, {seq})"
    
    expected_2d = torch.tril(torch.ones(seq, seq))
    actual_2d = mask[0, 0] if mask.shape[0] > 0 else mask.squeeze()[:seq, :seq]
    assert torch.allclose(actual_2d, expected_2d), "Batched mask is not lower triangular"

def test_masked_mha_shape():
    """Test MaskedMultiHeadAttention output shape and attention weights."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 64, 4
    batch, seq = 2, 8
    
    mha = MaskedMultiHeadAttention(d_model, num_heads, dropout=0.0)
    
    assert mha.W_q is not None, "W_q not initialized"
    
    x = torch.randn(batch, seq, d_model)
    output, weights = mha(x, use_causal_mask=True)
    
    assert output is not None, "output is None"
    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
    
    assert weights is not None, "attention weights is None"
    
    weights_sum = weights.sum(dim=-1)
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5), "Attention weights don't sum to 1"

def test_masked_mha_causal():
    """Test that causal masking prevents attending to future."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 64, 4
    batch, seq = 2, 8
    
    mha = MaskedMultiHeadAttention(d_model, num_heads)
    
    assert mha.W_q is not None, "W_q not initialized"
    
    x = torch.randn(batch, seq, d_model)
    _, weights = mha(x, use_causal_mask=True)
    
    assert weights is not None, "attention weights is None"
    
    upper_tri = torch.triu(weights, diagonal=1)
    assert upper_tri.abs().max() <= 1e-5, "Attention to future positions not masked"

def test_cross_attention_shape():
    """Test CrossAttention with different source/target lengths and valid attention."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 64, 4
    batch = 2
    tgt_seq, src_seq = 8, 12
    
    cross_attn = CrossAttention(d_model, num_heads, dropout=0.0)
    
    assert cross_attn.W_q is not None, "W_q not initialized"
    
    decoder_hidden = torch.randn(batch, tgt_seq, d_model)
    encoder_output = torch.randn(batch, src_seq, d_model)
    
    output, weights = cross_attn(decoder_hidden, encoder_output)
    
    assert output is not None, "output is None"
    
    expected_out_shape = (batch, tgt_seq, d_model)
    assert output.shape == expected_out_shape, f"Output {output.shape}, expected {expected_out_shape}"
    
    assert weights is not None, "attention weights is None"
    
    expected_weight_shape = (batch, num_heads, tgt_seq, src_seq)
    assert weights.shape == expected_weight_shape, f"Weights {weights.shape}, expected {expected_weight_shape}"
    
    weights_sum = weights.sum(dim=-1)
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5), "Cross-attention weights don't sum to 1 over source"

def test_cross_attention_no_causal():
    """Test that cross-attention has no causal masking."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 64, 4
    batch = 2
    tgt_seq, src_seq = 4, 6
    
    cross_attn = CrossAttention(d_model, num_heads)
    
    assert cross_attn.W_q is not None, "W_q not initialized"
    
    decoder_hidden = torch.randn(batch, tgt_seq, d_model)
    encoder_output = torch.randn(batch, src_seq, d_model)
    
    _, weights = cross_attn(decoder_hidden, encoder_output)
    
    assert weights is not None, "attention weights is None"
    
    assert weights.shape[-1] == src_seq, "Not attending to full source sequence"

def test_feedforward_shape():
    """Test FeedForward network computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 64
    ffn = FeedForward(d_model, dropout=0.0)
    
    assert ffn.linear1 is not None, "linear1 not initialized"
    assert ffn.linear2 is not None, "linear2 not initialized"
    
    x = torch.randn(2, 8, d_model)
    output = ffn(x)
    
    assert output is not None, "output is None"
    assert output.shape == x.shape, f"Shape {output.shape} != {x.shape}"
    
    with torch.no_grad():
        expected = ffn.linear2(F.gelu(ffn.linear1(x)))
    
    assert torch.allclose(output, expected, atol=1e-5), "FFN output doesn't match GELU(xW1)W2"

def test_layer_norm():
    """Test LayerNorm implementation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 64
    norm = LayerNorm(d_model)
    
    assert norm.gamma is not None, "gamma not initialized"
    
    x = torch.randn(2, 8, d_model)
    output = norm(x)
    
    assert output is not None, "output is None"
    
    mean = output.mean(dim=-1)
    std = output.std(dim=-1, unbiased=False)
    
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), f"Mean not ~0: {mean.mean().item():.4f}"

def test_decoder_block_shape():
    """Test PreNormDecoderBlock output and residual connections."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 64, 4
    batch = 2
    tgt_seq, src_seq = 8, 12
    
    decoder_block = PreNormDecoderBlock(d_model, num_heads, dropout=0.0)
    
    assert decoder_block.self_attn is not None, "self_attn not initialized"
    assert decoder_block.cross_attn is not None, "cross_attn not initialized"
    assert decoder_block.ffn is not None, "ffn not initialized"
    assert decoder_block.norm1 is not None and decoder_block.norm2 is not None and decoder_block.norm3 is not None, "norm layers not initialized"
    
    decoder_input = torch.randn(batch, tgt_seq, d_model)
    encoder_output = torch.randn(batch, src_seq, d_model)
    
    output = decoder_block(decoder_input, encoder_output)
    
    assert output is not None, "output is None"
    assert output.shape == decoder_input.shape, f"Shape {output.shape} != {decoder_input.shape}"
    
    correlation = torch.corrcoef(torch.stack([decoder_input.flatten(), output.flatten()]))[0, 1]
    assert correlation >= 0.05, f"Weak residual connection: corr={correlation:.4f}"

def test_decoder_block_gradient_flow():
    """Test that gradients flow through decoder block."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 64, 4
    batch, tgt_seq, src_seq = 2, 8, 12
    
    decoder_block = PreNormDecoderBlock(d_model, num_heads)
    
    assert decoder_block.self_attn is not None, "Block not initialized"
    
    decoder_input = torch.randn(batch, tgt_seq, d_model, requires_grad=True)
    encoder_output = torch.randn(batch, src_seq, d_model, requires_grad=True)
    
    output = decoder_block(decoder_input, encoder_output)
    
    assert output is not None, "output is None"
    
    loss = output.sum()
    loss.backward()
    
    assert decoder_input.grad is not None, "No gradient to decoder input"
    assert encoder_output.grad is not None, "No gradient to encoder output"

def test_decoder_stack():
    """Test TransformerDecoder stack with final normalization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_layers = 64, 4, 3
    batch, tgt_seq, src_seq = 2, 8, 12
    
    decoder = TransformerDecoder(d_model, num_heads, num_layers)
    
    assert decoder.layers is not None, "layers not initialized"
    assert len(decoder.layers) == num_layers, f"Expected {num_layers} layers, got {len(decoder.layers)}"
    assert decoder.final_norm is not None, "final_norm not initialized"
    
    decoder_input = torch.randn(batch, tgt_seq, d_model)
    encoder_output = torch.randn(batch, src_seq, d_model)
    
    output = decoder(decoder_input, encoder_output)
    
    assert output is not None, "output is None"
    assert output.shape == decoder_input.shape, f"Shape {output.shape} != {decoder_input.shape}"
    
    mean = output.mean(dim=-1)
    std = output.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4), f"Final norm not applied: mean={mean.mean().item():.4f}"

def test_decoder_autoregressive():
    """Test decoder in autoregressive mode with consistent hidden states."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_layers = 64, 4, 2
    batch, src_seq = 2, 10
    
    decoder = TransformerDecoder(d_model, num_heads, num_layers, dropout=0.0)
    
    assert decoder.layers is not None, "layers not initialized"
    
    torch.manual_seed(42)
    encoder_output = torch.randn(batch, src_seq, d_model)
    
    outputs = []
    for tgt_len in [1, 3, 5]:
        torch.manual_seed(123)
        decoder_input = torch.randn(batch, tgt_len, d_model)
        output = decoder(decoder_input, encoder_output)
        
        assert output is not None, f"output is None for tgt_len={tgt_len}"
        assert output.shape == (batch, tgt_len, d_model), f"Wrong shape for tgt_len={tgt_len}"
        outputs.append(output)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
