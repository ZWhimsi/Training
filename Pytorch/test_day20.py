"""Test Suite for Day 20: GPT-style Decoder-Only Model"""

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
try:
    from day20 import (RMSNorm, GPTBlock, GPT, compute_lm_loss,
                       prepare_lm_batch, get_gpt2_config)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_rms_norm_shape():
    """Test RMSNorm output shape and weight initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 64
    norm = RMSNorm(d_model)
    
    assert norm.weight is not None, "weight not initialized"
    
    assert torch.allclose(norm.weight.data, torch.ones(d_model)), "weight should be initialized to ones"
    
    x = torch.randn(2, 8, d_model)
    output = norm(x)
    
    assert output is not None, "output is None"
    assert output.shape == x.shape, f"Shape {output.shape} != {x.shape}"

def test_rms_norm_normalization():
    """Test that RMSNorm actually normalizes."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 64
    norm = RMSNorm(d_model)
    
    assert norm.weight is not None, "weight not initialized"
    
    x = torch.randn(2, 8, d_model) * 10
    output = norm(x)
    
    assert output is not None, "output is None"
    
    rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
    
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.2), f"RMS not ~1: {rms.mean().item():.4f}"

def test_gpt_block_shape():
    """Test GPTBlock output and residual connection."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 64, 4
    batch, seq = 2, 16
    
    block = GPTBlock(d_model, num_heads, dropout=0.0)
    
    assert block.W_q is not None, "W_q not initialized"
    assert block.ffn_linear1 is not None, "ffn_linear1 not initialized"
    assert block.norm1 is not None, "norm1 not initialized"
    
    x = torch.randn(batch, seq, d_model)
    output, kv_cache = block(x)
    
    assert output is not None, "output is None"
    assert output.shape == x.shape, f"Shape {output.shape} != {x.shape}"
    
    correlation = torch.corrcoef(torch.stack([x.flatten(), output.flatten()]))[0, 1]
    assert correlation >= 0.1, f"Weak residual connection: corr={correlation:.4f}"

def test_gpt_block_causal():
    """Test that GPTBlock respects causal masking."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 64, 4
    batch, seq = 1, 8
    
    block = GPTBlock(d_model, num_heads, dropout=0.0)
    
    assert block.W_q is not None, "Block not initialized"
    
    x = torch.randn(batch, seq, d_model)
    
    causal_mask = torch.tril(torch.ones(seq, seq))
    
    output1, _ = block(x, causal_mask)
    
    assert output1 is not None, "output is None"
    
    x_modified = x.clone()
    x_modified[:, -1, :] = torch.randn(d_model)
    
    output2, _ = block(x_modified, causal_mask)
    
    assert torch.allclose(output1[:, 0], output2[:, 0], atol=1e-5), "Causal masking not working - future affects past"

def test_gpt_block_kv_cache():
    """Test GPTBlock KV caching for generation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 64, 4
    batch = 2
    
    block = GPTBlock(d_model, num_heads, dropout=0.0)
    
    assert block.W_q is not None, "Block not initialized"
    
    x_full = torch.randn(batch, 5, d_model)
    mask_full = torch.tril(torch.ones(5, 5))
    output_full, cache = block(x_full, mask_full)
    
    assert output_full is not None, "output is None"
    assert cache is not None, "KV cache is None"
    
    k, v = cache
    assert k.shape[1] == 5, f"Cache K seq length {k.shape[1]} != 5"

def test_gpt_model_shape():
    """Test GPT model output shape and valid logits."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    vocab_size = 1000
    d_model, num_heads, num_layers = 64, 4, 2
    
    model = GPT(vocab_size, d_model, num_heads, num_layers, max_len=128)
    
    assert model.token_embedding is not None, "token_embedding not initialized"
    assert model.blocks is not None, "blocks not initialized"
    assert model.lm_head is not None, "lm_head not initialized"
    assert model.position_embedding is not None, "position_embedding not initialized"
    
    batch, seq = 2, 16
    input_ids = torch.randint(0, vocab_size, (batch, seq))
    
    logits, _ = model(input_ids)
    
    assert logits is not None, "logits is None"
    
    expected_shape = (batch, seq, vocab_size)
    assert logits.shape == expected_shape, f"Shape {logits.shape} != {expected_shape}"
    
    assert not torch.isnan(logits).any() and not torch.isinf(logits).any(), "Logits contain NaN or Inf"
    
    probs = F.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch, seq), atol=1e-5), "Softmax of logits doesn't sum to 1"

def test_gpt_generation():
    """Test GPT generation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    vocab_size = 100
    model = GPT(vocab_size, d_model=32, num_heads=2, num_layers=2, max_len=50)
    
    assert model.token_embedding is not None, "Model not initialized"
    
    prompt = torch.randint(0, vocab_size, (1, 5))
    max_new_tokens = 10
    
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0)
    
    assert generated is not None, "generate returned None"
    
    assert generated.shape[1] >= prompt.shape[1], "No new tokens generated"
    
    assert generated.shape[1] <= prompt.shape[1] + max_new_tokens, f"Generated too many tokens: {generated.shape[1]}"

def test_gpt_with_cache():
    """Test GPT with KV cache for efficient generation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    vocab_size = 100
    model = GPT(vocab_size, d_model=32, num_heads=2, num_layers=2, max_len=50)
    
    assert model.token_embedding is not None, "Model not initialized"
    
    input_ids = torch.randint(0, vocab_size, (1, 5))
    logits1, cache = model(input_ids, use_cache=True)
    
    assert logits1 is not None, "logits is None"
    assert cache is not None and len(cache) > 0, "Cache not returned"
    
    new_token = torch.randint(0, vocab_size, (1, 1))
    logits2, _ = model(new_token, use_cache=True, past_kv_cache=cache)
    
    assert logits2 is not None, "Second forward failed"
    assert logits2.shape == (1, 1, vocab_size), f"Wrong shape with cache: {logits2.shape}"

def test_compute_lm_loss():
    """Test language modeling loss computation matches PyTorch reference."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, seq, vocab_size = 2, 10, 100
    
    logits = torch.randn(batch, seq, vocab_size)
    targets = torch.randint(0, vocab_size, (batch, seq))
    
    loss = compute_lm_loss(logits, targets)
    
    assert loss is not None, "loss is None"
    
    assert loss.dim() == 0, f"Loss should be scalar, got dim={loss.dim()}"
    
    assert loss.item() >= 0, "Loss should be positive"
    
    expected_loss = F.cross_entropy(
        logits.view(-1, vocab_size), 
        targets.view(-1)
    )
    
    assert torch.allclose(loss, expected_loss, atol=1e-5), f"Loss {loss.item():.4f} != expected {expected_loss.item():.4f}"

def test_prepare_lm_batch():
    """Test LM batch preparation (shift by 1)."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, seq = 2, 10
    token_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    
    input_ids, target_ids = prepare_lm_batch(token_ids)
    
    assert input_ids is not None, "input_ids is None"
    assert target_ids is not None, "target_ids is None"
    
    assert input_ids.shape == (batch, seq - 1), f"Input shape {input_ids.shape} != ({batch}, {seq-1})"
    assert target_ids.shape == (batch, seq - 1), f"Target shape {target_ids.shape} != ({batch}, {seq-1})"
    
    assert input_ids[0, 0].item() == 0, "Input should start at 0"
    assert target_ids[0, 0].item() == 1, "Target should start at 1"

def test_gpt_gradient_flow():
    """Test gradient flow through GPT model."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = GPT(vocab_size=100, d_model=32, num_heads=2, num_layers=2)
    
    assert model.token_embedding is not None, "Model not initialized"
    
    input_ids = torch.randint(0, 100, (2, 8))
    logits, _ = model(input_ids)
    
    assert logits is not None, "logits is None"
    
    loss = logits.sum()
    loss.backward()
    
    assert model.token_embedding.weight.grad is not None, "No gradient to token embedding"

def test_gpt2_configs():
    """Test GPT-2 configuration function."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    for size in ['small', 'medium', 'large', 'xl']:
        config = get_gpt2_config(size)
        
        assert 'd_model' in config, f"Missing d_model in {size}"
        assert 'num_heads' in config, f"Missing num_heads in {size}"
        assert 'num_layers' in config, f"Missing num_layers in {size}"
        
        assert config['d_model'] % config['num_heads'] == 0, f"d_model not divisible by heads in {size}"

def test_gpt_with_rms_norm():
    """Test GPT model with RMSNorm produces valid output."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = GPT(
        vocab_size=100, 
        d_model=32, 
        num_heads=2, 
        num_layers=2,
        use_rms_norm=True
    )
    
    assert model.token_embedding is not None, "Model not initialized"
    
    if model.blocks is not None and len(model.blocks) > 0:
        block = model.blocks[0]
        if block.norm1 is not None:
            if hasattr(block.norm1, 'bias') and block.norm1.bias is not None:
                pytest.fail("Expected RMSNorm (no bias), got LayerNorm")
    
    input_ids = torch.randint(0, 100, (1, 8))
    logits, _ = model(input_ids)
    
    assert logits is not None, "logits is None"
    
    assert not torch.isnan(logits).any(), "NaN in output with RMSNorm"

def test_gpt_top_k_sampling():
    """Test top-k sampling in generation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = GPT(vocab_size=100, d_model=32, num_heads=2, num_layers=2)
    
    assert model.token_embedding is not None, "Model not initialized"
    
    prompt = torch.randint(0, 100, (1, 3))
    
    with torch.no_grad():
        gen_k5 = model.generate(prompt.clone(), max_new_tokens=5, top_k=5)
        gen_k50 = model.generate(prompt.clone(), max_new_tokens=5, top_k=50)
    
    assert gen_k5 is not None and gen_k50 is not None, "Generation failed"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
