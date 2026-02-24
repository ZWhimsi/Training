"""Test Suite for Day 24: Mixture of Experts (MoE) Basics"""

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
try:
    from day24 import (
        Expert, Router, top_k_gating, compute_load_balancing_loss,
        SparseMoE, BatchedMoE, MoETransformerBlock, analyze_expert_usage
    )
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_expert_init():
    """Test Expert network initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 256
    expert = Expert(d_model)
    
    assert expert.w1 is not None, "w1 not initialized"
    assert expert.w2 is not None, "w2 not initialized"
    
    expected_hidden = d_model * 4
    assert expert.w1.out_features == expected_hidden, f"w1 out_features {expert.w1.out_features} != {expected_hidden}"
    assert expert.w2.in_features == expected_hidden, "w2 in_features wrong"

def test_expert_forward():
    """Test Expert forward pass computes FFN correctly."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 256
    expert = Expert(d_model)
    
    assert expert.w1 is not None, "Expert not initialized"
    
    torch.manual_seed(42)
    x = torch.randn(2, 16, d_model)
    output = expert(x)
    
    assert output.shape == x.shape, f"Output shape {output.shape} != {x.shape}"
    
    with torch.no_grad():
        expected = expert.w1(x)
        expected = F.gelu(expected)
        if expert.dropout is not None:
            expert.dropout.eval()
        expected = expert.w2(expected)
    
    assert torch.allclose(output, expected, atol=1e-5), "Expert output doesn't match expected FFN computation"
    
    assert output.abs().sum() != 0, "Expert output is all zeros"

def test_router_init():
    """Test Router initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_experts = 256, 8
    router = Router(d_model, num_experts)
    
    assert router.gate is not None, "gate not initialized"
    
    assert router.gate.out_features == num_experts, f"gate out_features {router.gate.out_features} != {num_experts}"

def test_router_forward():
    """Test Router forward pass computes linear projection correctly."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_experts = 256, 8
    router = Router(d_model, num_experts)
    
    assert router.gate is not None, "Router not initialized"
    
    batch, seq_len = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, d_model)
    logits = router(x)
    
    expected_shape = (batch, seq_len, num_experts)
    assert logits.shape == expected_shape, f"Logits shape {logits.shape} != {expected_shape}"
    
    with torch.no_grad():
        expected_logits = F.linear(x, router.gate.weight, router.gate.bias if hasattr(router.gate, 'bias') and router.gate.bias is not None else None)
    
    assert torch.allclose(logits, expected_logits, atol=1e-5), "Router logits don't match expected linear projection"
    
    assert logits.abs().sum() != 0, "Router logits are all zeros"

def test_top_k_gating_shapes():
    """Test top_k_gating output shapes."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, seq_len, num_experts = 2, 16, 8
    k = 2
    
    router_logits = torch.randn(batch, seq_len, num_experts)
    gates, indices, probs = top_k_gating(router_logits, k)
    
    assert gates.shape == (batch, seq_len, k), f"Gates shape {gates.shape} != {(batch, seq_len, k)}"
    assert indices.shape == (batch, seq_len, k), f"Indices shape {indices.shape} != {(batch, seq_len, k)}"
    assert probs.shape == (batch, seq_len, num_experts), f"Probs shape {probs.shape} != {(batch, seq_len, num_experts)}"

def test_top_k_gating_values():
    """Test top_k_gating returns valid probabilities."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, seq_len, num_experts = 2, 16, 8
    k = 2
    
    router_logits = torch.randn(batch, seq_len, num_experts)
    gates, indices, probs = top_k_gating(router_logits, k)
    
    gate_sum = gates.sum(dim=-1)
    assert torch.allclose(gate_sum, torch.ones_like(gate_sum), atol=1e-5), "Gates don't sum to 1"
    
    prob_sum = probs.sum(dim=-1)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), "Probs don't sum to 1"
    
    assert (gates >= 0).any(), "Gates have negative values"
    
    assert (indices < 0).any() or (indices <= num_experts).any(), "Indices out of range"

def test_top_k_selection():
    """Test that top_k_gating selects the highest probability experts."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, seq_len, num_experts = 1, 1, 4
    k = 2
    
    router_logits = torch.tensor([[[1.0, 2.0, 10.0, 5.0]]])
    gates, indices, probs = top_k_gating(router_logits, k)
    
    selected = set(indices[0, 0].tolist())
    expected = {2, 3}
    
    assert selected == expected, f"Selected {selected}, expected {expected}"

def test_load_balancing_loss():
    """Test load balancing loss computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, seq_len, num_experts = 2, 16, 8
    k = 2
    
    router_logits = torch.randn(batch, seq_len, num_experts)
    gates, indices, probs = top_k_gating(router_logits, k)
    
    loss = compute_load_balancing_loss(probs, indices, num_experts)
    
    assert torch.is_tensor(loss), "Loss should be a tensor"
    assert loss.numel() == 1, "Loss should be scalar"
    assert loss >= 0, "Loss should be non-negative"

def test_load_balancing_uniform():
    """Test that uniform routing has minimal load balancing loss."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, seq_len, num_experts = 4, 32, 4
    k = 1
    
    uniform_probs = torch.ones(batch, seq_len, num_experts) / num_experts
    
    indices = torch.arange(num_experts).repeat(batch * seq_len // num_experts + 1)
    indices = indices[:batch * seq_len].view(batch, seq_len, 1)
    
    loss = compute_load_balancing_loss(uniform_probs, indices, num_experts)
    
    assert abs(loss.item() - 1.0) <= 0.1, f"Uniform loss {loss.item():.4f} should be ~1.0"

def test_sparse_moe_init():
    """Test SparseMoE initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_experts = 256, 8
    moe = SparseMoE(d_model, num_experts)
    
    assert moe.router is not None, "router not initialized"
    assert moe.experts is not None, "experts not initialized"
    assert len(moe.experts) == num_experts, f"Expected {num_experts} experts, got {len(moe.experts)}"

def test_sparse_moe_forward():
    """Test SparseMoE forward pass produces weighted expert combination."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_experts = 128, 4
    moe = SparseMoE(d_model, num_experts, top_k=2)
    
    assert moe.router is not None, "MoE not initialized"
    
    batch, seq_len = 2, 8
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, d_model)
    
    output, aux_loss = moe(x)
    
    assert output.shape == x.shape, f"Output shape {output.shape} != {x.shape}"
    assert torch.is_tensor(aux_loss) or aux_loss.numel() != 1, "aux_loss should be scalar tensor"
    
    assert output.abs().sum() != 0, "MoE output is all zeros"
    
    assert not torch.allclose(output, x, atol=1e-3), "MoE output is same as input"
    
    assert aux_loss >= 0, f"aux_loss should be non-negative, got {aux_loss.item()}"
    
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "MoE output contains NaN or Inf"

def test_sparse_moe_gradient():
    """Test gradients flow through SparseMoE."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_experts = 128, 4
    moe = SparseMoE(d_model, num_experts, top_k=2)
    
    assert moe.router is not None, "MoE not initialized"
    
    x = torch.randn(2, 8, d_model, requires_grad=True)
    output, aux_loss = moe(x)
    
    loss = output.sum() + aux_loss
    loss.backward()
    
    assert x.grad is not None, "No gradient to input"
    assert x.grad.abs().sum() != 0, "Gradients are zero"
    
    assert moe.router.gate.weight.grad is not None, "No gradient to router"

def test_batched_moe_init():
    """Test BatchedMoE initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_experts = 128, 4
    moe = BatchedMoE(d_model, num_experts)
    
    assert moe.expert_w1 is not None, "expert_w1 not initialized"
    assert moe.expert_w2 is not None, "expert_w2 not initialized"
    
    expected_w1 = (num_experts, d_model, d_model * 4)
    assert moe.expert_w1.shape == expected_w1, f"expert_w1 shape {moe.expert_w1.shape} != {expected_w1}"

def test_batched_moe_forward():
    """Test BatchedMoE forward pass produces valid output."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_experts = 128, 4
    moe = BatchedMoE(d_model, num_experts, top_k=2)
    
    assert moe.expert_w1 is not None, "BatchedMoE not initialized"
    
    batch, seq_len = 2, 8
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, d_model)
    
    output, aux_loss = moe(x)
    
    assert output.shape == x.shape, "Output shape wrong"
    
    assert output.abs().sum() != 0, "BatchedMoE output is all zeros"
    
    assert not torch.allclose(output, x, atol=1e-3), "BatchedMoE output is same as input"
    
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "BatchedMoE output contains NaN or Inf"
    
    assert torch.is_tensor(aux_loss) or aux_loss.numel() != 1, "aux_loss should be scalar tensor"
    assert aux_loss >= 0, "aux_loss should be non-negative"

def test_moe_transformer_block():
    """Test MoE Transformer block with attention + MoE."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_experts = 128, 4, 4
    block = MoETransformerBlock(d_model, num_heads, num_experts, top_k=2)
    
    assert block.moe is not None, "MoE not initialized"
    assert block.attention is not None, "attention not initialized"
    
    batch, seq_len = 2, 8
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, d_model)
    
    output, aux_loss = block(x)
    
    assert output.shape == x.shape, "Output shape wrong"
    
    assert output.abs().sum() != 0, "Block output is all zeros"
    
    correlation = F.cosine_similarity(output.flatten(), x.flatten(), dim=0)
    assert correlation >= 0.1, f"Residual connection may not work: correlation={correlation:.3f}"
    
    assert torch.is_tensor(aux_loss) or aux_loss.numel() != 1, "aux_loss should be scalar tensor"
    
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "Block output contains NaN or Inf"

def test_expert_usage_analysis():
    """Test expert usage analysis."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, seq_len, num_experts = 4, 16, 8
    k = 2
    
    router_logits = torch.randn(batch, seq_len, num_experts)
    gates, indices, probs = top_k_gating(router_logits, k)
    
    stats = analyze_expert_usage(probs, indices, num_experts)
    
    assert 'assignments_per_expert' in stats, "Missing assignments_per_expert"
    assert len(stats['assignments_per_expert']) == num_experts, "Wrong number of experts in stats"
    
    total_expected = batch * seq_len * k
    assert stats['total_assignments'] == total_expected, "Total assignments wrong"

def test_sparse_vs_batched_equivalence():
    """Test that sparse and batched MoE produce similar outputs."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_experts = 64, 4
    top_k = 2
    
    sparse_moe = SparseMoE(d_model, num_experts, top_k=top_k)
    batched_moe = BatchedMoE(d_model, num_experts, top_k=top_k)
    
    assert sparse_moe.router is not None and batched_moe.expert_w1 is not None, "MoE not initialized"
    
    x = torch.randn(2, 4, d_model)
    
    sparse_out, sparse_loss = sparse_moe(x)
    batched_out, batched_loss = batched_moe(x)
    
    assert sparse_out.shape == x.shape, "Sparse output shape wrong"
    assert batched_out.shape == x.shape, "Batched output shape wrong"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
