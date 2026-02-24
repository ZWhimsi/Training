"""Test Suite for Day 29: DeepSeek Architecture Components"""

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
try:
    from day29 import (
        swish, Swish, SwiGLUFFN, SwiGLUExpert,
        DeepSeekRouter, DeepSeekMoE,
        DeepSeekConfig, DeepSeekBlock, DeepSeekModel,
        analyze_expert_usage, compute_activated_params
    )
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_swish_function():
    """Test swish activation function."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    x = torch.randn(2, 8, 64)
    y = swish(x)
    
    assert y.shape == x.shape, f"Output shape {y.shape} != {x.shape}"
    
    expected = x * torch.sigmoid(x)
    assert torch.allclose(y, expected, atol=1e-5), "Swish output doesn't match expected"

def test_swish_properties():
    """Test properties of swish activation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    zero = torch.zeros(1)
    assert abs(swish(zero).item()) <= 1e-6, "Swish(0) should be 0"
    
    x = torch.randn(10, requires_grad=True)
    y = swish(x)
    y.sum().backward()
    
    assert x.grad is not None, "Swish not differentiable"

def test_swiglu_ffn_init():
    """Test SwiGLU FFN initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 256
    ffn = SwiGLUFFN(d_model)
    
    assert ffn.W_gate is not None, "W_gate not initialized"
    assert ffn.W_up is not None, "W_up not initialized"
    assert ffn.W_down is not None, "W_down not initialized"

def test_swiglu_ffn_forward():
    """Test SwiGLU FFN forward pass."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 256
    ffn = SwiGLUFFN(d_model)
    
    assert ffn.W_gate is not None, "FFN not initialized"
    
    batch, seq = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq, d_model)
    out = ffn(x)
    
    assert out.shape == x.shape, f"Output shape {out.shape} != {x.shape}"
    
    gate = swish(ffn.W_gate(x))
    up = ffn.W_up(x)
    hidden = gate * up
    expected = ffn.W_down(hidden)
    
    assert torch.allclose(out, expected, atol=1e-5), "SwiGLU output doesn't match manual computation"

def test_swiglu_ffn_gating():
    """Test that SwiGLU gating mechanism works."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 64
    ffn = SwiGLUFFN(d_model)
    
    assert ffn.W_gate is not None, "FFN not initialized"
    
    torch.manual_seed(123)
    x = torch.randn(1, 1, d_model)
    out = ffn(x)
    
    gate_val = swish(ffn.W_gate(x))
    up_val = ffn.W_up(x)
    gated = gate_val * up_val
    expected = ffn.W_down(gated)
    
    assert torch.allclose(out, expected, atol=1e-5), "Gating computation doesn't match expected formula"
    
    assert gate_val.shape == up_val.shape, "Gate and up projections should have same shape"

def test_swiglu_expert_init():
    """Test SwiGLUExpert initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 256
    expert = SwiGLUExpert(d_model)
    
    assert expert.ffn is not None, "Expert FFN not initialized"

def test_swiglu_expert_forward():
    """Test SwiGLUExpert forward pass."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 256
    expert = SwiGLUExpert(d_model)
    
    assert expert.ffn is not None, "Expert not initialized"
    
    torch.manual_seed(42)
    x = torch.randn(2, 16, d_model)
    out = expert(x)
    
    assert out.shape == x.shape, "Output shape wrong"
    
    expected = expert.ffn(x)
    assert torch.allclose(out, expected, atol=1e-5), "Expert output doesn't match its FFN output"

def test_deepseek_router_init():
    """Test DeepSeekRouter initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_experts = 256, 8
    router = DeepSeekRouter(d_model, num_experts, top_k=2)
    
    assert router.gate is not None, "Router gate not initialized"

def test_deepseek_router_forward():
    """Test DeepSeekRouter forward pass."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_experts, top_k = 256, 8, 2
    router = DeepSeekRouter(d_model, num_experts, top_k)
    
    assert router.gate is not None, "Router not initialized"
    
    batch, seq = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq, d_model)
    
    gates, indices, logits = router(x, training=False)
    
    assert gates.shape == (batch, seq, top_k), f"Gates shape wrong: {gates.shape}"
    assert indices.shape == (batch, seq, top_k), f"Indices shape wrong: {indices.shape}"
    assert logits.shape == (batch, seq, num_experts), f"Logits shape wrong: {logits.shape}"
    
    gate_sums = gates.sum(dim=-1)
    assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=0.01), "Gates not normalized"
    
    expected_logits = router.gate(x)
    assert torch.allclose(logits, expected_logits, atol=1e-5), "Logits don't match gate projection"
    
    probs = F.softmax(logits, dim=-1)
    expected_probs, expected_indices = torch.topk(probs, top_k, dim=-1)
    assert torch.equal(indices, expected_indices), "Indices don't match top-k selection"

def test_deepseek_router_topk():
    """Test that router selects top-k experts."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_experts, top_k = 256, 8, 3
    router = DeepSeekRouter(d_model, num_experts, top_k)
    
    assert router.gate is not None, "Router not initialized"
    
    x = torch.randn(1, 1, d_model)
    gates, indices, logits = router(x, training=False)
    
    unique_count = len(torch.unique(indices[0, 0]))
    assert unique_count == top_k, f"Expected {top_k} unique experts, got {unique_count}"

def test_deepseek_moe_init():
    """Test DeepSeekMoE initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 256
    moe = DeepSeekMoE(d_model, num_shared_experts=2, num_routed_experts=4, top_k=2)
    
    assert moe.shared_experts is not None, "Shared experts not initialized"
    assert moe.routed_experts is not None, "Routed experts not initialized"
    assert moe.router is not None, "Router not initialized"
    
    assert len(moe.shared_experts) == 2, "Expected 2 shared experts"
    assert len(moe.routed_experts) == 4, "Expected 4 routed experts"

def test_deepseek_moe_forward():
    """Test DeepSeekMoE forward pass."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 256
    num_shared = 2
    num_routed = 4
    top_k = 2
    moe = DeepSeekMoE(d_model, num_shared_experts=num_shared, num_routed_experts=num_routed, top_k=top_k)
    
    assert moe.router is not None, "MoE not initialized"
    
    batch, seq = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq, d_model)
    
    out, aux = moe(x)
    
    assert out.shape == x.shape, "Output shape wrong"
    
    assert 'selected_experts' in aux, "Missing routing info"
    assert 'gate_weights' in aux, "Missing gate weights in aux"
    assert 'router_logits' in aux, "Missing router logits in aux"
    
    assert aux['selected_experts'].shape == (batch, seq, top_k), f"selected_experts shape wrong: {aux['selected_experts'].shape}"
    
    gate_sums = aux['gate_weights'].sum(dim=-1)
    assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=0.01), "Gate weights not normalized"

def test_deepseek_moe_shared_experts():
    """Test that shared experts are always active."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 64
    num_shared = 2
    moe = DeepSeekMoE(d_model, num_shared_experts=num_shared, num_routed_experts=4, top_k=1)
    
    assert moe.router is not None, "MoE not initialized"
    
    torch.manual_seed(42)
    x = torch.randn(2, 8, d_model)
    out, aux = moe(x)
    
    assert out.abs().sum() != 0, "Output is zero (shared experts not contributing)"
    
    shared_output = torch.zeros_like(x)
    for expert in moe.shared_experts:
        shared_output = shared_output + expert(x)
    shared_output = shared_output / num_shared
    
    routed_contribution = out - shared_output
    
    assert aux['gate_weights'].abs().sum() != 0, "Gate weights are all zero"

def test_deepseek_config():
    """Test DeepSeekConfig creation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = DeepSeekConfig(
        d_model=256,
        num_heads=4,
        head_dim=64,
        num_shared_experts=2,
        num_routed_experts=8,
        top_k=2
    )
    
    assert config.d_model == 256, "d_model not set"
    assert config.num_routed_experts == 8, "num_routed_experts not set"

def test_deepseek_block_init():
    """Test DeepSeekBlock initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = DeepSeekConfig(
        d_model=256,
        num_heads=4,
        head_dim=64,
        num_shared_experts=1,
        num_routed_experts=4,
        top_k=2
    )
    
    block = DeepSeekBlock(config, use_moe=True)
    
    assert block.attn is not None, "Attention not initialized"
    assert block.ffn is not None, "FFN not initialized"

def test_deepseek_block_forward():
    """Test DeepSeekBlock forward pass."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = DeepSeekConfig(
        d_model=256,
        num_heads=4,
        head_dim=64,
        num_shared_experts=1,
        num_routed_experts=4,
        top_k=2
    )
    
    block = DeepSeekBlock(config, use_moe=True)
    
    assert block.ffn is not None, "Block not initialized"
    
    batch, seq = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq, config.d_model)
    
    out, aux = block(x)
    
    assert out.shape == x.shape, "Output shape wrong"
    
    assert not torch.allclose(out, x, atol=1e-3), "Output too similar to input, block not transforming"
    
    assert 'selected_experts' in aux, "MoE block should have routing info in aux"

def test_deepseek_block_dense():
    """Test DeepSeekBlock with dense FFN (no MoE)."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = DeepSeekConfig(d_model=256, num_heads=4, head_dim=64)
    block = DeepSeekBlock(config, use_moe=False)
    
    assert block.ffn is not None, "Block not initialized"
    
    torch.manual_seed(42)
    x = torch.randn(2, 16, config.d_model)
    out, aux = block(x)
    
    assert out.shape == x.shape, "Output shape wrong"
    
    assert 'selected_experts' not in aux, "Dense block shouldn't have routing info"
    
    assert not torch.allclose(out, x, atol=1e-3), "Output too similar to input, block not transforming"
    
    assert isinstance(block.ffn, SwiGLUFFN), "Dense block should use SwiGLUFFN"

def test_deepseek_model_init():
    """Test DeepSeekModel initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = DeepSeekConfig(
        d_model=256,
        num_heads=4,
        head_dim=64,
        num_shared_experts=1,
        num_routed_experts=4,
        top_k=2
    )
    
    model = DeepSeekModel(config, num_layers=4, moe_layers=[1, 2, 3])
    
    assert model.layers is not None, "Layers not initialized"
    assert len(model.layers) == 4, "Expected 4 layers"
    
    moe_count = sum(1 for layer in model.layers if layer.use_moe)
    assert moe_count == 3, f"Expected 3 MoE layers, got {moe_count}"

def test_deepseek_model_forward():
    """Test DeepSeekModel forward pass."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = DeepSeekConfig(
        d_model=256,
        num_heads=4,
        head_dim=64,
        num_shared_experts=1,
        num_routed_experts=4,
        top_k=2
    )
    
    model = DeepSeekModel(config, num_layers=3, moe_layers=[1, 2])
    
    assert model.layers is not None, "Model not initialized"
    
    batch, seq = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq, config.d_model)
    
    out, aux_list = model(x)
    
    assert out.shape == x.shape, "Output shape wrong"
    
    assert len(aux_list) == 3, "Expected 3 aux_info dicts"
    
    rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
    assert rms.mean() <= 5.0, "Output seems unnormalized"
    
    assert 'selected_experts' not in aux_list[0], "Layer 0 should be dense (no routing info)"
    assert 'selected_experts' in aux_list[1], "Layer 1 should be MoE (have routing info)"
    assert 'selected_experts' in aux_list[2], "Layer 2 should be MoE (have routing info)"

def test_gradient_flow():
    """Test gradient flow through DeepSeek model."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = DeepSeekConfig(
        d_model=128,
        num_heads=2,
        head_dim=32,
        num_shared_experts=1,
        num_routed_experts=2,
        top_k=1
    )
    
    model = DeepSeekModel(config, num_layers=2, moe_layers=[1])
    
    assert model.layers is not None, "Model not initialized"
    
    x = torch.randn(2, 8, config.d_model, requires_grad=True)
    out, _ = model(x)
    
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient to input"
    assert x.grad.abs().sum() != 0, "Gradients are zero"

def test_compute_activated_params():
    """Test activated parameter computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = DeepSeekConfig(
        d_model=256,
        num_heads=4,
        head_dim=64,
        d_kv_latent=64,
        d_q_latent=48,
        num_shared_experts=2,
        num_routed_experts=8,
        top_k=2
    )
    
    params = compute_activated_params(config, num_layers=4, moe_layers=[1, 2, 3])
    
    assert params['total_activated_params'] != 0, "Parameter computation failed"
    
    assert params['num_moe_layers'] == 3, "Wrong MoE layer count"
    
    assert params['num_dense_layers'] == 1, f"Wrong dense layer count: {params['num_dense_layers']}"
    
    assert params['moe_activated_params_per_layer'] >= params['dense_ffn_params_per_layer'], "MoE should activate more params than dense FFN with shared experts"
    
    assert params['attention_params_per_layer'] > 0, "Attention params should be positive"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
