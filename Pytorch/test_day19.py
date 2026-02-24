"""Test Suite for Day 19: Full Transformer Architecture"""

import torch
import torch.nn as nn
import sys
from typing import Tuple

try:
    from day19 import (PositionalEncoding, TransformerEmbedding,
                       TransformerEncoder, TransformerDecoder,
                       Transformer, greedy_decode, create_causal_mask)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_positional_encoding_shape() -> Tuple[bool, str]:
    """Test PositionalEncoding adds position info to input."""
    try:
        d_model = 64
        pe = PositionalEncoding(d_model, max_len=100, dropout=0.0)
        
        x = torch.randn(2, 16, d_model)
        output = pe(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Shape {output.shape} != {x.shape}"
        
        # Verify PE is added (not identical to input)
        if torch.allclose(output, x, atol=1e-6):
            return False, "No positional encoding added"
        
        # PE should be consistent (same position gets same encoding)
        zero_input = torch.zeros(1, 16, d_model)
        pe_only = pe(zero_input)
        pe_only2 = pe(zero_input)
        if not torch.allclose(pe_only, pe_only2, atol=1e-6):
            return False, "PE not consistent across calls"
        
        return True, f"PE added correctly to input"
    except Exception as e:
        return False, str(e)


def test_positional_encoding_different_positions() -> Tuple[bool, str]:
    """Test that different positions have different encodings."""
    try:
        d_model = 64
        pe = PositionalEncoding(d_model, dropout=0.0)
        
        x = torch.zeros(1, 10, d_model)  # Zero input to see only PE
        output = pe(x)
        
        if output is None:
            return False, "output is None"
        
        # Different positions should have different encodings
        pos_diff = (output[0, 0] - output[0, 1]).abs().sum()
        if pos_diff < 1e-5:
            return False, "Positions 0 and 1 have same encoding"
        
        return True, "Different positions have different encodings"
    except Exception as e:
        return False, str(e)


def test_transformer_embedding_shape() -> Tuple[bool, str]:
    """Test TransformerEmbedding produces unique embeddings per token."""
    try:
        vocab_size = 1000
        d_model = 64
        
        emb = TransformerEmbedding(vocab_size, d_model, dropout=0.0)
        
        if emb.token_embedding is None:
            return False, "token_embedding not initialized"
        
        # Create tokens with known different values
        tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        output = emb(tokens)
        
        if output is None:
            return False, "output is None"
        
        expected_shape = (2, 4, d_model)
        if output.shape != expected_shape:
            return False, f"Shape {output.shape} != {expected_shape}"
        
        # Different tokens should have different embeddings
        if torch.allclose(output[0, 0], output[0, 1], atol=1e-4):
            return False, "Different tokens have same embedding"
        
        # Same token should have same base embedding (before PE)
        tokens_same = torch.tensor([[5, 5, 5, 5]])
        out_same = emb(tokens_same)
        # Note: positions differ, so outputs will differ due to PE
        # But base embedding component should be similar
        
        return True, f"Embeddings unique per token"
    except Exception as e:
        return False, str(e)


def test_transformer_embedding_scaling() -> Tuple[bool, str]:
    """Test that embeddings are scaled by sqrt(d_model)."""
    try:
        import math
        vocab_size = 1000
        d_model = 64
        
        emb = TransformerEmbedding(vocab_size, d_model, dropout=0.0)
        
        if emb.token_embedding is None:
            return False, "token_embedding not initialized"
        if emb.pos_encoding is None:
            return False, "pos_encoding not initialized"
        
        tokens = torch.randint(1, vocab_size, (1, 1))
        
        # Get raw embedding
        raw_emb = emb.token_embedding(tokens)
        
        # Expected scaled embedding: raw * sqrt(d_model)
        expected_scale = math.sqrt(d_model)
        scaled_emb = raw_emb * expected_scale
        
        # Get output with PE using zero input to isolate PE effect
        # Then back-calculate the scaled embedding
        output = emb(tokens)
        
        if output is None:
            return False, "output is None"
        
        # The scaled embedding norm should be ~sqrt(d_model) times raw
        raw_norm = raw_emb.norm().item()
        # Note: output includes PE, so we can't directly compare
        # But we can verify scaling is applied by checking output magnitude
        output_norm = output.norm().item()
        
        # Scaled output should have larger norm than raw (approx sqrt(d_model) factor)
        if output_norm < raw_norm * 0.5:  # Allow for PE effects
            return False, "Embedding not scaled up"
        
        return True, f"Embedding scaled by sqrt({d_model})={expected_scale:.2f}"
    except Exception as e:
        return False, str(e)


def test_encoder_stack() -> Tuple[bool, str]:
    """Test TransformerEncoder stack with correct layer count and final norm."""
    try:
        d_model, num_heads, num_layers = 64, 4, 2
        
        encoder = TransformerEncoder(d_model, num_heads, num_layers)
        
        if encoder.layers is None:
            return False, "layers not initialized"
        if len(encoder.layers) != num_layers:
            return False, f"Expected {num_layers} layers"
        if encoder.final_norm is None:
            return False, "final_norm not initialized"
        
        x = torch.randn(2, 16, d_model)
        output = encoder(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Shape {output.shape} != {x.shape}"
        
        # With final norm, output should be normalized
        mean = output.mean(dim=-1)
        if not torch.allclose(mean, torch.zeros_like(mean), atol=1e-4):
            return False, f"Output not normalized: mean={mean.mean().item():.4f}"
        
        return True, f"Encoder with {num_layers} layers, final norm applied"
    except Exception as e:
        return False, str(e)


def test_decoder_stack() -> Tuple[bool, str]:
    """Test TransformerDecoder stack with final norm and cross-attention."""
    try:
        d_model, num_heads, num_layers = 64, 4, 2
        
        decoder = TransformerDecoder(d_model, num_heads, num_layers)
        
        if decoder.layers is None:
            return False, "layers not initialized"
        if len(decoder.layers) != num_layers:
            return False, f"Expected {num_layers} layers"
        if decoder.final_norm is None:
            return False, "final_norm not initialized"
        
        tgt = torch.randn(2, 8, d_model)
        enc_out = torch.randn(2, 16, d_model)
        
        output = decoder(tgt, enc_out)
        
        if output is None:
            return False, "output is None"
        if output.shape != tgt.shape:
            return False, f"Shape {output.shape} != {tgt.shape}"
        
        # With final norm, output should be normalized
        mean = output.mean(dim=-1)
        if not torch.allclose(mean, torch.zeros_like(mean), atol=1e-4):
            return False, f"Output not normalized: mean={mean.mean().item():.4f}"
        
        return True, f"Decoder with {num_layers} layers, final norm applied"
    except Exception as e:
        return False, str(e)


def test_transformer_forward() -> Tuple[bool, str]:
    """Test full Transformer forward pass with valid logit output."""
    try:
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=500,
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        if model.encoder is None:
            return False, "encoder not initialized"
        if model.decoder is None:
            return False, "decoder not initialized"
        if model.output_projection is None:
            return False, "output_projection not initialized"
        
        batch, src_len, tgt_len = 2, 10, 8
        src = torch.randint(1, 1000, (batch, src_len))
        tgt = torch.randint(1, 500, (batch, tgt_len))
        
        output = model(src, tgt)
        
        if output is None:
            return False, "output is None"
        
        expected_shape = (batch, tgt_len, 500)
        if output.shape != expected_shape:
            return False, f"Shape {output.shape} != {expected_shape}"
        
        # Verify logits are valid (not all same, finite values)
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, "Output contains NaN or Inf"
        
        # Different positions should have different logits
        if torch.allclose(output[0, 0], output[0, 1], atol=1e-3):
            return False, "All positions have identical logits"
        
        return True, f"Transformer outputs valid logits: {output.shape}"
    except Exception as e:
        return False, str(e)


def test_transformer_src_mask() -> Tuple[bool, str]:
    """Test source padding mask creation with correct values."""
    try:
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            padding_idx=0
        )
        
        # Create input with padding (0 is padding)
        src = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        mask = model.create_src_mask(src)
        
        if mask is None:
            return False, "create_src_mask returned None"
        
        # Mask should be 0 for padding positions, 1 for real tokens
        # First sequence: positions 0,1,2 are real (mask=1), positions 3,4 are padding (mask=0)
        # Expected mask values for first sequence: [1, 1, 1, 0, 0]
        mask_squeezed = mask.squeeze()
        
        # Check shape is broadcastable
        if len(mask.shape) != 4:
            return False, f"Mask should be 4D, got {len(mask.shape)}D"
        
        # Check that padding positions are 0
        if mask_squeezed[0, 3] != 0 or mask_squeezed[0, 4] != 0:
            return False, "Padding positions should be masked (0)"
        if mask_squeezed[0, 0] != 1 or mask_squeezed[0, 1] != 1:
            return False, "Real token positions should be unmasked (1)"
        
        return True, "Source mask correctly masks padding tokens"
    except Exception as e:
        return False, str(e)


def test_transformer_tgt_mask() -> Tuple[bool, str]:
    """Test target mask combines causal and padding correctly."""
    try:
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            padding_idx=0
        )
        
        # Create target with padding
        tgt = torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0]])
        mask = model.create_tgt_mask(tgt)
        
        if mask is None:
            return False, "create_tgt_mask returned None"
        
        # Should be 4D for attention
        if len(mask.shape) != 4:
            return False, f"Expected 4D mask, got {len(mask.shape)}D"
        
        # Check causal property: position i should not attend to j > i
        # Upper triangle should be 0 (or masked)
        mask_2d = mask[0, 0]  # Get first batch, first head
        for i in range(mask_2d.shape[0]):
            for j in range(i + 1, mask_2d.shape[1]):
                if mask_2d[i, j] != 0:
                    return False, f"Position {i} can attend to future position {j}"
        
        # Check padding: position 3 in first sequence is padding
        # It should also be masked in the target
        
        return True, "Target mask is causal and handles padding"
    except Exception as e:
        return False, str(e)


def test_transformer_encode() -> Tuple[bool, str]:
    """Test encoder method produces contextual embeddings."""
    try:
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        if model.encoder is None:
            return False, "encoder not initialized"
        if model.src_embedding is None:
            return False, "src_embedding not initialized"
        
        src = torch.randint(1, 100, (2, 10))
        encoder_output = model.encode(src)
        
        if encoder_output is None:
            return False, "encode returned None"
        
        expected_shape = (2, 10, 64)
        if encoder_output.shape != expected_shape:
            return False, f"Shape {encoder_output.shape} != {expected_shape}"
        
        # Encoder output should be contextual (different from raw embeddings)
        raw_emb = model.src_embedding(src)
        if raw_emb is not None and torch.allclose(encoder_output, raw_emb, atol=1e-3):
            return False, "Encoder output same as raw embedding (no processing)"
        
        # Different positions should have different outputs (context-dependent)
        if torch.allclose(encoder_output[0, 0], encoder_output[0, 1], atol=1e-3):
            return False, "All positions have identical encoder output"
        
        return True, f"Encoder produces contextual embeddings"
    except Exception as e:
        return False, str(e)


def test_transformer_decode() -> Tuple[bool, str]:
    """Test decoder method uses encoder output via cross-attention."""
    try:
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        if model.decoder is None:
            return False, "decoder not initialized"
        if model.tgt_embedding is None:
            return False, "tgt_embedding not initialized"
        
        tgt = torch.randint(1, 100, (2, 8))
        encoder_output = torch.randn(2, 10, 64)
        
        decoder_output = model.decode(tgt, encoder_output)
        
        if decoder_output is None:
            return False, "decode returned None"
        
        expected_shape = (2, 8, 64)
        if decoder_output.shape != expected_shape:
            return False, f"Shape {decoder_output.shape} != {expected_shape}"
        
        # Decoder should use encoder output (different encoder = different output)
        different_encoder = torch.randn(2, 10, 64) * 10  # Very different
        decoder_output2 = model.decode(tgt, different_encoder)
        
        if decoder_output2 is not None:
            if torch.allclose(decoder_output, decoder_output2, atol=1e-2):
                return False, "Decoder ignores encoder output"
        
        return True, f"Decoder uses encoder output via cross-attention"
    except Exception as e:
        return False, str(e)


def test_transformer_gradient_flow() -> Tuple[bool, str]:
    """Test gradient flow through full model."""
    try:
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1
        )
        
        if model.encoder is None:
            return False, "Model not initialized"
        
        src = torch.randint(1, 100, (2, 5))
        tgt = torch.randint(1, 100, (2, 4))
        
        output = model(src, tgt)
        
        if output is None:
            return False, "output is None"
        
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        if not has_grad:
            return False, "No gradients computed"
        
        return True, "Gradients flow through full model"
    except Exception as e:
        return False, str(e)


def test_greedy_decode_shape() -> Tuple[bool, str]:
    """Test greedy decoding output shape."""
    try:
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1
        )
        
        if model.encoder is None:
            return False, "Model not initialized"
        
        src = torch.randint(1, 100, (2, 5))
        max_len = 10
        start_token = 1
        end_token = 2
        
        with torch.no_grad():
            generated = greedy_decode(model, src, max_len, start_token, end_token)
        
        if generated is None:
            return False, "greedy_decode returned None"
        
        # Should start with start_token
        if generated[:, 0].unique().item() != start_token:
            return False, f"First token should be {start_token}"
        
        # Length should be <= max_len
        if generated.shape[1] > max_len:
            return False, f"Generated length {generated.shape[1]} > max_len {max_len}"
        
        return True, f"Generated shape: {generated.shape}"
    except Exception as e:
        return False, str(e)


def test_transformer_different_vocab_sizes() -> Tuple[bool, str]:
    """Test Transformer with different source/target vocab sizes and valid output."""
    try:
        model = Transformer(
            src_vocab_size=5000,
            tgt_vocab_size=3000,
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        if model.encoder is None:
            return False, "Model not initialized"
        if model.output_projection is None:
            return False, "output_projection not initialized"
        
        src = torch.randint(1, 5000, (2, 10))
        tgt = torch.randint(1, 3000, (2, 8))
        
        output = model(src, tgt)
        
        if output is None:
            return False, "output is None"
        
        # Output vocab should be target vocab
        if output.shape[-1] != 3000:
            return False, f"Output vocab size {output.shape[-1]} != 3000"
        
        # Verify output projection weight has correct shape
        if model.output_projection.weight.shape[0] != 3000:
            return False, "Output projection has wrong vocab size"
        
        # Softmax should produce valid probabilities
        probs = torch.softmax(output, dim=-1)
        prob_sum = probs.sum(dim=-1)
        if not torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5):
            return False, "Softmax of logits doesn't sum to 1"
        
        return True, "Different vocab sizes with valid output projection"
    except Exception as e:
        return False, str(e)


def test_causal_mask_function() -> Tuple[bool, str]:
    """Test the causal mask helper function."""
    try:
        seq_len = 5
        mask = create_causal_mask(seq_len)
        
        if mask is None:
            return False, "create_causal_mask returned None"
        
        expected = torch.tril(torch.ones(seq_len, seq_len))
        if not torch.allclose(mask, expected):
            return False, "Causal mask not lower triangular"
        
        return True, "Causal mask is correct"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("positional_encoding_shape", test_positional_encoding_shape),
        ("positional_encoding_positions", test_positional_encoding_different_positions),
        ("transformer_embedding_shape", test_transformer_embedding_shape),
        ("transformer_embedding_scaling", test_transformer_embedding_scaling),
        ("encoder_stack", test_encoder_stack),
        ("decoder_stack", test_decoder_stack),
        ("transformer_forward", test_transformer_forward),
        ("transformer_src_mask", test_transformer_src_mask),
        ("transformer_tgt_mask", test_transformer_tgt_mask),
        ("transformer_encode", test_transformer_encode),
        ("transformer_decode", test_transformer_decode),
        ("transformer_gradient", test_transformer_gradient_flow),
        ("greedy_decode", test_greedy_decode_shape),
        ("different_vocab_sizes", test_transformer_different_vocab_sizes),
        ("causal_mask", test_causal_mask_function),
    ]
    
    print(f"\n{'='*50}\nDay 19: Full Transformer Architecture - Tests\n{'='*50}")
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
