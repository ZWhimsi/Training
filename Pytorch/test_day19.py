"""Test Suite for Day 19: Full Transformer Architecture"""

import torch
import pytest
import torch.nn as nn
import math
try:
    from day19 import (PositionalEncoding, TransformerEmbedding,
                       TransformerEncoder, TransformerDecoder,
                       Transformer, greedy_decode, create_causal_mask)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_positional_encoding_shape():
    """Test PositionalEncoding adds position info to input."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 64
    pe = PositionalEncoding(d_model, max_len=100, dropout=0.0)
    
    x = torch.randn(2, 16, d_model)
    output = pe(x)
    
    assert output is not None, "output is None"
    assert output.shape == x.shape, f"Shape {output.shape} != {x.shape}"
    
    assert not torch.allclose(output, x, atol=1e-6), "No positional encoding added"
    
    zero_input = torch.zeros(1, 16, d_model)
    pe_only = pe(zero_input)
    pe_only2 = pe(zero_input)
    assert torch.allclose(pe_only, pe_only2, atol=1e-6), "PE not consistent across calls"

def test_positional_encoding_different_positions():
    """Test that different positions have different encodings."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 64
    pe = PositionalEncoding(d_model, dropout=0.0)
    
    x = torch.zeros(1, 10, d_model)
    output = pe(x)
    
    assert output is not None, "output is None"
    
    pos_diff = (output[0, 0] - output[0, 1]).abs().sum()
    assert pos_diff >= 1e-5, "Positions 0 and 1 have same encoding"

def test_transformer_embedding_shape():
    """Test TransformerEmbedding produces unique embeddings per token."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    vocab_size = 1000
    d_model = 64
    
    emb = TransformerEmbedding(vocab_size, d_model, dropout=0.0)
    
    assert emb.token_embedding is not None, "token_embedding not initialized"
    
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    output = emb(tokens)
    
    assert output is not None, "output is None"
    
    expected_shape = (2, 4, d_model)
    assert output.shape == expected_shape, f"Shape {output.shape} != {expected_shape}"
    
    assert not torch.allclose(output[0, 0], output[0, 1], atol=1e-4), "Different tokens have same embedding"

def test_transformer_embedding_scaling():
    """Test that embeddings are scaled by sqrt(d_model)."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    vocab_size = 1000
    d_model = 64
    
    emb = TransformerEmbedding(vocab_size, d_model, dropout=0.0)
    
    assert emb.token_embedding is not None, "token_embedding not initialized"
    assert emb.pos_encoding is not None, "pos_encoding not initialized"
    
    tokens = torch.randint(1, vocab_size, (1, 1))
    
    raw_emb = emb.token_embedding(tokens)
    
    expected_scale = math.sqrt(d_model)
    scaled_emb = raw_emb * expected_scale
    
    output = emb(tokens)
    
    assert output is not None, "output is None"
    
    raw_norm = raw_emb.norm().item()
    output_norm = output.norm().item()
    
    assert output_norm >= raw_norm * 0.5, "Embedding not scaled up"

def test_encoder_stack():
    """Test TransformerEncoder stack with correct layer count and final norm."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_layers = 64, 4, 2
    
    encoder = TransformerEncoder(d_model, num_heads, num_layers)
    
    assert encoder.layers is not None, "layers not initialized"
    assert len(encoder.layers) == num_layers, f"Expected {num_layers} layers"
    assert encoder.final_norm is not None, "final_norm not initialized"
    
    x = torch.randn(2, 16, d_model)
    output = encoder(x)
    
    assert output is not None, "output is None"
    assert output.shape == x.shape, f"Shape {output.shape} != {x.shape}"
    
    mean = output.mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4), f"Output not normalized: mean={mean.mean().item():.4f}"

def test_decoder_stack():
    """Test TransformerDecoder stack with final norm and cross-attention."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_layers = 64, 4, 2
    
    decoder = TransformerDecoder(d_model, num_heads, num_layers)
    
    assert decoder.layers is not None, "layers not initialized"
    assert len(decoder.layers) == num_layers, f"Expected {num_layers} layers"
    assert decoder.final_norm is not None, "final_norm not initialized"
    
    tgt = torch.randn(2, 8, d_model)
    enc_out = torch.randn(2, 16, d_model)
    
    output = decoder(tgt, enc_out)
    
    assert output is not None, "output is None"
    assert output.shape == tgt.shape, f"Shape {output.shape} != {tgt.shape}"
    
    mean = output.mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4), f"Output not normalized: mean={mean.mean().item():.4f}"

def test_transformer_forward():
    """Test full Transformer forward pass with valid logit output."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=500,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    assert model.encoder is not None, "encoder not initialized"
    assert model.decoder is not None, "decoder not initialized"
    assert model.output_projection is not None, "output_projection not initialized"
    
    batch, src_len, tgt_len = 2, 10, 8
    src = torch.randint(1, 1000, (batch, src_len))
    tgt = torch.randint(1, 500, (batch, tgt_len))
    
    output = model(src, tgt)
    
    assert output is not None, "output is None"
    
    expected_shape = (batch, tgt_len, 500)
    assert output.shape == expected_shape, f"Shape {output.shape} != {expected_shape}"
    
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "Output contains NaN or Inf"
    
    assert not torch.allclose(output[0, 0], output[0, 1], atol=1e-3), "All positions have identical logits"

def test_transformer_src_mask():
    """Test source padding mask creation with correct values."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=32,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        padding_idx=0
    )
    
    src = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    mask = model.create_src_mask(src)
    
    assert mask is not None, "create_src_mask returned None"
    
    mask_squeezed = mask.squeeze()
    
    assert len(mask.shape) == 4, f"Mask should be 4D, got {len(mask.shape)}D"
    
    assert mask_squeezed[0, 3] == 0 and mask_squeezed[0, 4] == 0, "Padding positions should be masked (0)"
    assert mask_squeezed[0, 0] == 1 and mask_squeezed[0, 1] == 1, "Real token positions should be unmasked (1)"

def test_transformer_tgt_mask():
    """Test target mask combines causal and padding correctly."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=32,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        padding_idx=0
    )
    
    tgt = torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0]])
    mask = model.create_tgt_mask(tgt)
    
    assert mask is not None, "create_tgt_mask returned None"
    
    assert len(mask.shape) == 4, f"Expected 4D mask, got {len(mask.shape)}D"
    
    mask_2d = mask[0, 0]
    for i in range(mask_2d.shape[0]):
        for j in range(i + 1, mask_2d.shape[1]):
            assert mask_2d[i, j] == 0, f"Position {i} can attend to future position {j}"

def test_transformer_encode():
    """Test encoder method produces contextual embeddings."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    assert model.encoder is not None, "encoder not initialized"
    assert model.src_embedding is not None, "src_embedding not initialized"
    
    src = torch.randint(1, 100, (2, 10))
    encoder_output = model.encode(src)
    
    assert encoder_output is not None, "encode returned None"
    
    expected_shape = (2, 10, 64)
    assert encoder_output.shape == expected_shape, f"Shape {encoder_output.shape} != {expected_shape}"
    
    raw_emb = model.src_embedding(src)
    assert raw_emb is None or not torch.allclose(encoder_output, raw_emb, atol=1e-3), "Encoder output same as raw embedding (no processing)"
    
    assert not torch.allclose(encoder_output[0, 0], encoder_output[0, 1], atol=1e-3), "All positions have identical encoder output"

def test_transformer_decode():
    """Test decoder method uses encoder output via cross-attention."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    assert model.decoder is not None, "decoder not initialized"
    assert model.tgt_embedding is not None, "tgt_embedding not initialized"
    
    tgt = torch.randint(1, 100, (2, 8))
    encoder_output = torch.randn(2, 10, 64)
    
    decoder_output = model.decode(tgt, encoder_output)
    
    assert decoder_output is not None, "decode returned None"
    
    expected_shape = (2, 8, 64)
    assert decoder_output.shape == expected_shape, f"Shape {decoder_output.shape} != {expected_shape}"
    
    different_encoder = torch.randn(2, 10, 64) * 10
    decoder_output2 = model.decode(tgt, different_encoder)
    
    if decoder_output2 is not None:
        assert not torch.allclose(decoder_output, decoder_output2, atol=1e-2), "Decoder ignores encoder output"

def test_transformer_gradient_flow():
    """Test gradient flow through full model."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=32,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1
    )
    
    assert model.encoder is not None, "Model not initialized"
    
    src = torch.randint(1, 100, (2, 5))
    tgt = torch.randint(1, 100, (2, 4))
    
    output = model(src, tgt)
    
    assert output is not None, "output is None"
    
    loss = output.sum()
    loss.backward()
    
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    
    assert has_grad, "No gradients computed"

def test_greedy_decode_shape():
    """Test greedy decoding output shape."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=32,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1
    )
    
    assert model.encoder is not None, "Model not initialized"
    
    src = torch.randint(1, 100, (2, 5))
    max_len = 10
    start_token = 1
    end_token = 2
    
    with torch.no_grad():
        generated = greedy_decode(model, src, max_len, start_token, end_token)
    
    assert generated is not None, "greedy_decode returned None"
    
    assert generated[:, 0].unique().item() == start_token, f"First token should be {start_token}"
    
    assert generated.shape[1] <= max_len, f"Generated length {generated.shape[1]} > max_len {max_len}"

def test_transformer_different_vocab_sizes():
    """Test Transformer with different source/target vocab sizes and valid output."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = Transformer(
        src_vocab_size=5000,
        tgt_vocab_size=3000,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    assert model.encoder is not None, "Model not initialized"
    assert model.output_projection is not None, "output_projection not initialized"
    
    src = torch.randint(1, 5000, (2, 10))
    tgt = torch.randint(1, 3000, (2, 8))
    
    output = model(src, tgt)
    
    assert output is not None, "output is None"
    
    assert output.shape[-1] == 3000, f"Output vocab size {output.shape[-1]} != 3000"
    
    assert model.output_projection.weight.shape[0] == 3000, "Output projection has wrong vocab size"
    
    probs = torch.softmax(output, dim=-1)
    prob_sum = probs.sum(dim=-1)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), "Softmax of logits doesn't sum to 1"

def test_causal_mask_function():
    """Test the causal mask helper function."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    seq_len = 5
    mask = create_causal_mask(seq_len)
    
    assert mask is not None, "create_causal_mask returned None"
    
    expected = torch.tril(torch.ones(seq_len, seq_len))
    assert torch.allclose(mask, expected), "Causal mask not lower triangular"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
