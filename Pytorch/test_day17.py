"""Test Suite for Day 17: Transformer Encoder Block"""

import torch
import pytest
import torch.nn as nn
import math
try:
    from day17 import (LayerNorm, FeedForward, PostNormEncoderBlock, 
                       PreNormEncoderBlock, TransformerEncoder,
                       TransformerEncoderWithEmbedding, PositionalEncoding)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_layer_norm_shape():
    """Test LayerNorm output shape and parameter initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 64
    norm = LayerNorm(d_model)
    
    assert norm.gamma is not None, "gamma not initialized"
    assert norm.beta is not None, "beta not initialized"
    
    assert torch.allclose(norm.gamma.data, torch.ones(d_model)), "gamma should be initialized to ones"
    assert torch.allclose(norm.beta.data, torch.zeros(d_model)), "beta should be initialized to zeros"
    
    x = torch.randn(2, 8, d_model)
    output = norm(x)
    
    assert output is not None, "output is None"
    assert output.shape == x.shape, f"Shape {output.shape} != {x.shape}"

def test_layer_norm_stats():
    """Test that LayerNorm produces correct mean/variance."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 64
    norm = LayerNorm(d_model)
    
    x = torch.randn(2, 8, d_model)
    output = norm(x)
    
    assert output is not None, "output is None"
    
    mean = output.mean(dim=-1)
    std = output.std(dim=-1, unbiased=False)
    
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), f"Mean not close to 0: {mean.mean().item():.4f}"
    assert torch.allclose(std, torch.ones_like(std), atol=1e-1), f"Std not close to 1: {std.mean().item():.4f}"

def test_layer_norm_vs_pytorch():
    """Compare custom LayerNorm with PyTorch's."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 64
    our_norm = LayerNorm(d_model)
    pytorch_norm = nn.LayerNorm(d_model)
    
    with torch.no_grad():
        pytorch_norm.weight.copy_(our_norm.gamma)
        pytorch_norm.bias.copy_(our_norm.beta)
    
    x = torch.randn(2, 8, d_model)
    our_output = our_norm(x)
    pytorch_output = pytorch_norm(x)
    
    assert our_output is not None, "output is None"
    
    assert torch.allclose(our_output, pytorch_output, atol=1e-5), "Doesn't match PyTorch LayerNorm"

def test_feedforward_shape():
    """Test FeedForward network output against reference implementation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 64
    d_ff = d_model * 4
    ffn = FeedForward(d_model)
    
    assert ffn.linear1 is not None, "linear1 not initialized"
    assert ffn.linear2 is not None, "linear2 not initialized"
    
    x = torch.randn(2, 8, d_model)
    output = ffn(x)
    
    assert output is not None, "output is None"
    assert output.shape == x.shape, f"Shape {output.shape} != {x.shape}"
    
    with torch.no_grad():
        expected = ffn.linear2(torch.nn.functional.gelu(ffn.linear1(x)))
    
    assert torch.allclose(output, expected, atol=1e-5), "FFN output doesn't match expected GELU(xW1)W2"

def test_feedforward_expansion():
    """Test that FFN has correct expansion factor."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 64
    ffn = FeedForward(d_model)
    
    assert ffn.linear1 is not None, "linear1 not initialized"
    
    expected_d_ff = d_model * 4
    actual_d_ff = ffn.linear1.out_features
    
    assert actual_d_ff == expected_d_ff, f"d_ff={actual_d_ff}, expected {expected_d_ff}"

def test_post_norm_encoder_block():
    """Test PostNormEncoderBlock produces normalized output."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 64, 4
    block = PostNormEncoderBlock(d_model, num_heads)
    
    assert block.self_attn is not None, "self_attn not initialized"
    assert block.norm1 is not None, "norm1 not initialized"
    
    x = torch.randn(2, 8, d_model)
    output = block(x)
    
    assert output is not None, "output is None"
    assert output.shape == x.shape, f"Shape {output.shape} != {x.shape}"
    
    mean = output.mean(dim=-1)
    std = output.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4), f"Post-norm output mean not ~0: {mean.mean().item():.4f}"
    assert torch.allclose(std, torch.ones_like(std), atol=0.15), f"Post-norm output std not ~1: {std.mean().item():.4f}"

def test_pre_norm_encoder_block():
    """Test PreNormEncoderBlock has residual connection."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 64, 4
    block = PreNormEncoderBlock(d_model, num_heads, dropout=0.0)
    
    assert block.self_attn is not None, "self_attn not initialized"
    assert block.norm1 is not None, "norm1 not initialized"
    
    x = torch.randn(2, 8, d_model)
    output = block(x)
    
    assert output is not None, "output is None"
    assert output.shape == x.shape, f"Shape {output.shape} != {x.shape}"
    
    assert not torch.allclose(output, x, atol=1e-3), "Output identical to input - sublayers not applied"
    
    correlation = torch.corrcoef(torch.stack([x.flatten(), output.flatten()]))[0, 1]
    assert correlation >= 0.1, f"Weak residual: correlation={correlation:.4f}"

def test_residual_connection():
    """Test that residual connections are working (gradient flows through)."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 64, 4
    block = PreNormEncoderBlock(d_model, num_heads)
    
    assert block.self_attn is not None, "Block not initialized"
    
    x = torch.randn(2, 8, d_model, requires_grad=True)
    output = block(x)
    
    assert output is not None, "output is None"
    
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient flow to input"
    assert x.grad.abs().sum() != 0, "Gradients are all zero"

def test_encoder_stack():
    """Test stacked encoder with final normalization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_layers = 64, 4, 4
    encoder = TransformerEncoder(d_model, num_heads, num_layers, pre_norm=True)
    
    assert encoder.layers is not None, "layers not initialized"
    assert len(encoder.layers) == num_layers, f"Expected {num_layers} layers, got {len(encoder.layers)}"
    assert encoder.final_norm is not None, "final_norm not initialized for pre_norm encoder"
    
    x = torch.randn(2, 8, d_model)
    output = encoder(x)
    
    assert output is not None, "output is None"
    assert output.shape == x.shape, f"Shape {output.shape} != {x.shape}"
    
    mean = output.mean(dim=-1)
    std = output.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4), f"Final norm not applied: mean={mean.mean().item():.4f}"
    assert torch.allclose(std, torch.ones_like(std), atol=0.15), f"Final norm not applied: std={std.mean().item():.4f}"

def test_encoder_with_embedding():
    """Test complete encoder with embeddings and scaling."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    vocab_size = 1000
    d_model, num_heads, num_layers = 64, 4, 2
    
    encoder = TransformerEncoderWithEmbedding(
        vocab_size, d_model, num_heads, num_layers, dropout=0.0
    )
    
    assert encoder.token_emb is not None, "token_emb not initialized"
    assert encoder.encoder is not None, "encoder not initialized"
    assert encoder.pos_enc is not None, "pos_enc not initialized"
    
    x = torch.randint(0, vocab_size, (2, 16))
    output = encoder(x)
    
    assert output is not None, "output is None"
    
    expected_shape = (2, 16, d_model)
    assert output.shape == expected_shape, f"Shape {output.shape} != {expected_shape}"
    
    raw_emb = encoder.token_emb(x)
    expected_scale = math.sqrt(d_model)
    
    assert not (torch.allclose(raw_emb[0, 0], raw_emb[0, 1]) and x[0, 0] != x[0, 1]), "Different tokens should have different embeddings"

def test_positional_encoding():
    """Test sinusoidal positional encoding values."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 64
    pos_enc = PositionalEncoding(d_model, dropout=0.0)
    
    x = torch.zeros(1, 4, d_model)
    output = pos_enc(x)
    
    assert output.shape == x.shape, f"Shape changed: {output.shape}"
    
    pos = 1
    i = 0
    expected_sin = math.sin(pos / (10000 ** (2 * i / d_model)))
    actual = output[0, pos, 2 * i].item()
    assert abs(actual - expected_sin) <= 1e-5, f"PE[{pos}, {2*i}] = {actual:.4f}, expected sin={expected_sin:.4f}"
    
    expected_cos = math.cos(pos / (10000 ** (2 * i / d_model)))
    actual_cos = output[0, pos, 2 * i + 1].item()
    assert abs(actual_cos - expected_cos) <= 1e-5, f"PE[{pos}, {2*i+1}] = {actual_cos:.4f}, expected cos={expected_cos:.4f}"

def test_pre_vs_post_norm_different():
    """Test that pre-norm and post-norm produce different outputs."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 64, 4
    
    pre = PreNormEncoderBlock(d_model, num_heads)
    post = PostNormEncoderBlock(d_model, num_heads)
    
    assert pre.self_attn is not None and post.self_attn is not None, "Blocks not initialized"
    
    with torch.no_grad():
        post.self_attn.W_q.weight.copy_(pre.self_attn.W_q.weight)
        post.self_attn.W_q.bias.copy_(pre.self_attn.W_q.bias)
        post.self_attn.W_k.weight.copy_(pre.self_attn.W_k.weight)
        post.self_attn.W_k.bias.copy_(pre.self_attn.W_k.bias)
        post.self_attn.W_v.weight.copy_(pre.self_attn.W_v.weight)
        post.self_attn.W_v.bias.copy_(pre.self_attn.W_v.bias)
        post.self_attn.W_o.weight.copy_(pre.self_attn.W_o.weight)
        post.self_attn.W_o.bias.copy_(pre.self_attn.W_o.bias)
    
    x = torch.randn(2, 8, d_model)
    
    pre_out = pre(x)
    post_out = post(x)
    
    assert pre_out is not None and post_out is not None, "Output is None"
    
    assert not torch.allclose(pre_out, post_out, atol=1e-5), "Pre-norm and post-norm outputs are identical"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
