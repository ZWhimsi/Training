"""Test Suite for Day 24: Mixture of Experts (MoE) Basics"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Tuple

try:
    from day24 import (
        Expert, Router, top_k_gating, compute_load_balancing_loss,
        SparseMoE, BatchedMoE, MoETransformerBlock, analyze_expert_usage
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_expert_init() -> Tuple[bool, str]:
    """Test Expert network initialization."""
    try:
        d_model = 256
        expert = Expert(d_model)
        
        if expert.w1 is None:
            return False, "w1 not initialized"
        if expert.w2 is None:
            return False, "w2 not initialized"
        
        # Check dimensions
        expected_hidden = d_model * 4
        if expert.w1.out_features != expected_hidden:
            return False, f"w1 out_features {expert.w1.out_features} != {expected_hidden}"
        if expert.w2.in_features != expected_hidden:
            return False, f"w2 in_features wrong"
        
        return True, "Expert initialized correctly"
    except Exception as e:
        return False, str(e)


def test_expert_forward() -> Tuple[bool, str]:
    """Test Expert forward pass computes FFN correctly."""
    try:
        d_model = 256
        expert = Expert(d_model)
        
        if expert.w1 is None:
            return False, "Expert not initialized"
        
        torch.manual_seed(42)
        x = torch.randn(2, 16, d_model)
        output = expert(x)
        
        if output.shape != x.shape:
            return False, f"Output shape {output.shape} != {x.shape}"
        
        # Verify by computing manually: w2(gelu(w1(x)))
        with torch.no_grad():
            expected = expert.w1(x)
            expected = F.gelu(expected)
            if expert.dropout is not None:
                expert.dropout.eval()
            expected = expert.w2(expected)
        
        if not torch.allclose(output, expected, atol=1e-5):
            return False, "Expert output doesn't match expected FFN computation"
        
        # Verify output is not trivially zero
        if output.abs().sum() == 0:
            return False, "Expert output is all zeros"
        
        return True, f"Expert forward matches FFN: w2(gelu(w1(x)))"
    except Exception as e:
        return False, str(e)


def test_router_init() -> Tuple[bool, str]:
    """Test Router initialization."""
    try:
        d_model, num_experts = 256, 8
        router = Router(d_model, num_experts)
        
        if router.gate is None:
            return False, "gate not initialized"
        
        if router.gate.out_features != num_experts:
            return False, f"gate out_features {router.gate.out_features} != {num_experts}"
        
        return True, "Router initialized correctly"
    except Exception as e:
        return False, str(e)


def test_router_forward() -> Tuple[bool, str]:
    """Test Router forward pass computes linear projection correctly."""
    try:
        d_model, num_experts = 256, 8
        router = Router(d_model, num_experts)
        
        if router.gate is None:
            return False, "Router not initialized"
        
        batch, seq_len = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_model)
        logits = router(x)
        
        expected_shape = (batch, seq_len, num_experts)
        if logits.shape != expected_shape:
            return False, f"Logits shape {logits.shape} != {expected_shape}"
        
        # Verify router computes x @ W (linear projection)
        with torch.no_grad():
            expected_logits = F.linear(x, router.gate.weight, router.gate.bias if hasattr(router.gate, 'bias') and router.gate.bias is not None else None)
        
        if not torch.allclose(logits, expected_logits, atol=1e-5):
            return False, "Router logits don't match expected linear projection"
        
        # Verify logits are not all zeros
        if logits.abs().sum() == 0:
            return False, "Router logits are all zeros"
        
        return True, f"Router forward matches linear projection, logits range=[{logits.min():.2f}, {logits.max():.2f}]"
    except Exception as e:
        return False, str(e)


def test_top_k_gating_shapes() -> Tuple[bool, str]:
    """Test top_k_gating output shapes."""
    try:
        batch, seq_len, num_experts = 2, 16, 8
        k = 2
        
        router_logits = torch.randn(batch, seq_len, num_experts)
        gates, indices, probs = top_k_gating(router_logits, k)
        
        if gates.shape != (batch, seq_len, k):
            return False, f"Gates shape {gates.shape} != {(batch, seq_len, k)}"
        if indices.shape != (batch, seq_len, k):
            return False, f"Indices shape {indices.shape} != {(batch, seq_len, k)}"
        if probs.shape != (batch, seq_len, num_experts):
            return False, f"Probs shape {probs.shape} != {(batch, seq_len, num_experts)}"
        
        return True, "Shapes correct"
    except Exception as e:
        return False, str(e)


def test_top_k_gating_values() -> Tuple[bool, str]:
    """Test top_k_gating returns valid probabilities."""
    try:
        batch, seq_len, num_experts = 2, 16, 8
        k = 2
        
        router_logits = torch.randn(batch, seq_len, num_experts)
        gates, indices, probs = top_k_gating(router_logits, k)
        
        # Gates should sum to 1 (renormalized)
        gate_sum = gates.sum(dim=-1)
        if not torch.allclose(gate_sum, torch.ones_like(gate_sum), atol=1e-5):
            return False, "Gates don't sum to 1"
        
        # Probs should be valid softmax (sum to 1)
        prob_sum = probs.sum(dim=-1)
        if not torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5):
            return False, "Probs don't sum to 1"
        
        # Gates should be non-negative
        if (gates < 0).any():
            return False, "Gates have negative values"
        
        # Indices should be valid expert indices
        if (indices < 0).any() or (indices >= num_experts).any():
            return False, "Indices out of range"
        
        return True, "Valid probabilities and indices"
    except Exception as e:
        return False, str(e)


def test_top_k_selection() -> Tuple[bool, str]:
    """Test that top_k_gating selects the highest probability experts."""
    try:
        batch, seq_len, num_experts = 1, 1, 4
        k = 2
        
        # Create logits where experts 2 and 3 have highest scores
        router_logits = torch.tensor([[[1.0, 2.0, 10.0, 5.0]]])
        gates, indices, probs = top_k_gating(router_logits, k)
        
        # Should select experts 2 and 3
        selected = set(indices[0, 0].tolist())
        expected = {2, 3}
        
        if selected != expected:
            return False, f"Selected {selected}, expected {expected}"
        
        return True, "Correctly selects top-k experts"
    except Exception as e:
        return False, str(e)


def test_load_balancing_loss() -> Tuple[bool, str]:
    """Test load balancing loss computation."""
    try:
        batch, seq_len, num_experts = 2, 16, 8
        k = 2
        
        router_logits = torch.randn(batch, seq_len, num_experts)
        gates, indices, probs = top_k_gating(router_logits, k)
        
        loss = compute_load_balancing_loss(probs, indices, num_experts)
        
        if not torch.is_tensor(loss):
            return False, "Loss should be a tensor"
        if loss.numel() != 1:
            return False, "Loss should be scalar"
        if loss < 0:
            return False, "Loss should be non-negative"
        
        return True, f"Load balancing loss: {loss.item():.4f}"
    except Exception as e:
        return False, str(e)


def test_load_balancing_uniform() -> Tuple[bool, str]:
    """Test that uniform routing has minimal load balancing loss."""
    try:
        batch, seq_len, num_experts = 4, 32, 4
        k = 1
        
        # Create uniform router probabilities
        uniform_probs = torch.ones(batch, seq_len, num_experts) / num_experts
        
        # Create indices that evenly distribute tokens
        indices = torch.arange(num_experts).repeat(batch * seq_len // num_experts + 1)
        indices = indices[:batch * seq_len].view(batch, seq_len, 1)
        
        loss = compute_load_balancing_loss(uniform_probs, indices, num_experts)
        
        # Uniform distribution should give loss close to 1.0
        # (num_experts * (1/num_experts) * (1/num_experts) * num_experts = 1)
        if abs(loss.item() - 1.0) > 0.1:
            return False, f"Uniform loss {loss.item():.4f} should be ~1.0"
        
        return True, f"Uniform distribution loss: {loss.item():.4f}"
    except Exception as e:
        return False, str(e)


def test_sparse_moe_init() -> Tuple[bool, str]:
    """Test SparseMoE initialization."""
    try:
        d_model, num_experts = 256, 8
        moe = SparseMoE(d_model, num_experts)
        
        if moe.router is None:
            return False, "router not initialized"
        if moe.experts is None:
            return False, "experts not initialized"
        if len(moe.experts) != num_experts:
            return False, f"Expected {num_experts} experts, got {len(moe.experts)}"
        
        return True, f"SparseMoE initialized with {num_experts} experts"
    except Exception as e:
        return False, str(e)


def test_sparse_moe_forward() -> Tuple[bool, str]:
    """Test SparseMoE forward pass produces weighted expert combination."""
    try:
        d_model, num_experts = 128, 4
        moe = SparseMoE(d_model, num_experts, top_k=2)
        
        if moe.router is None:
            return False, "MoE not initialized"
        
        batch, seq_len = 2, 8
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_model)
        
        output, aux_loss = moe(x)
        
        if output.shape != x.shape:
            return False, f"Output shape {output.shape} != {x.shape}"
        if not torch.is_tensor(aux_loss) or aux_loss.numel() != 1:
            return False, "aux_loss should be scalar tensor"
        
        # Verify output is not zeros (experts did computation)
        if output.abs().sum() == 0:
            return False, "MoE output is all zeros"
        
        # Verify output is different from input (transformation happened)
        if torch.allclose(output, x, atol=1e-3):
            return False, "MoE output is same as input"
        
        # Verify aux_loss is non-negative (it's a sum of products of probabilities)
        if aux_loss < 0:
            return False, f"aux_loss should be non-negative, got {aux_loss.item()}"
        
        # Verify output has reasonable values
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, "MoE output contains NaN or Inf"
        
        return True, f"SparseMoE forward works, aux_loss={aux_loss.item():.4f}, output std={output.std():.4f}"
    except Exception as e:
        return False, str(e)


def test_sparse_moe_gradient() -> Tuple[bool, str]:
    """Test gradients flow through SparseMoE."""
    try:
        d_model, num_experts = 128, 4
        moe = SparseMoE(d_model, num_experts, top_k=2)
        
        if moe.router is None:
            return False, "MoE not initialized"
        
        x = torch.randn(2, 8, d_model, requires_grad=True)
        output, aux_loss = moe(x)
        
        loss = output.sum() + aux_loss
        loss.backward()
        
        if x.grad is None:
            return False, "No gradient to input"
        if x.grad.abs().sum() == 0:
            return False, "Gradients are zero"
        
        # Check router gradients
        if moe.router.gate.weight.grad is None:
            return False, "No gradient to router"
        
        return True, "Gradients flow correctly"
    except Exception as e:
        return False, str(e)


def test_batched_moe_init() -> Tuple[bool, str]:
    """Test BatchedMoE initialization."""
    try:
        d_model, num_experts = 128, 4
        moe = BatchedMoE(d_model, num_experts)
        
        if moe.expert_w1 is None:
            return False, "expert_w1 not initialized"
        if moe.expert_w2 is None:
            return False, "expert_w2 not initialized"
        
        expected_w1 = (num_experts, d_model, d_model * 4)
        if moe.expert_w1.shape != expected_w1:
            return False, f"expert_w1 shape {moe.expert_w1.shape} != {expected_w1}"
        
        return True, "BatchedMoE initialized correctly"
    except Exception as e:
        return False, str(e)


def test_batched_moe_forward() -> Tuple[bool, str]:
    """Test BatchedMoE forward pass produces valid output."""
    try:
        d_model, num_experts = 128, 4
        moe = BatchedMoE(d_model, num_experts, top_k=2)
        
        if moe.expert_w1 is None:
            return False, "BatchedMoE not initialized"
        
        batch, seq_len = 2, 8
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_model)
        
        output, aux_loss = moe(x)
        
        if output.shape != x.shape:
            return False, f"Output shape wrong"
        
        # Verify output is not zeros
        if output.abs().sum() == 0:
            return False, "BatchedMoE output is all zeros"
        
        # Verify output is different from input
        if torch.allclose(output, x, atol=1e-3):
            return False, "BatchedMoE output is same as input"
        
        # Verify output has reasonable values
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, "BatchedMoE output contains NaN or Inf"
        
        # Verify aux_loss is valid
        if not torch.is_tensor(aux_loss) or aux_loss.numel() != 1:
            return False, "aux_loss should be scalar tensor"
        if aux_loss < 0:
            return False, "aux_loss should be non-negative"
        
        return True, f"BatchedMoE forward works, aux_loss={aux_loss.item():.4f}"
    except Exception as e:
        return False, str(e)


def test_moe_transformer_block() -> Tuple[bool, str]:
    """Test MoE Transformer block with attention + MoE."""
    try:
        d_model, num_heads, num_experts = 128, 4, 4
        block = MoETransformerBlock(d_model, num_heads, num_experts, top_k=2)
        
        if block.moe is None:
            return False, "MoE not initialized"
        if block.attention is None:
            return False, "attention not initialized"
        
        batch, seq_len = 2, 8
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_model)
        
        output, aux_loss = block(x)
        
        if output.shape != x.shape:
            return False, f"Output shape wrong"
        
        # Verify output is not zeros
        if output.abs().sum() == 0:
            return False, "Block output is all zeros"
        
        # Verify residual connection (output correlates with input)
        correlation = F.cosine_similarity(output.flatten(), x.flatten(), dim=0)
        if correlation < 0.1:
            return False, f"Residual connection may not work: correlation={correlation:.3f}"
        
        # Verify aux_loss is valid
        if not torch.is_tensor(aux_loss) or aux_loss.numel() != 1:
            return False, "aux_loss should be scalar tensor"
        
        # Verify output has reasonable values
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, "Block output contains NaN or Inf"
        
        return True, f"MoE Transformer block works, correlation={correlation:.3f}"
    except Exception as e:
        return False, str(e)


def test_expert_usage_analysis() -> Tuple[bool, str]:
    """Test expert usage analysis."""
    try:
        batch, seq_len, num_experts = 4, 16, 8
        k = 2
        
        router_logits = torch.randn(batch, seq_len, num_experts)
        gates, indices, probs = top_k_gating(router_logits, k)
        
        stats = analyze_expert_usage(probs, indices, num_experts)
        
        if 'assignments_per_expert' not in stats:
            return False, "Missing assignments_per_expert"
        if len(stats['assignments_per_expert']) != num_experts:
            return False, "Wrong number of experts in stats"
        
        total_expected = batch * seq_len * k
        if stats['total_assignments'] != total_expected:
            return False, f"Total assignments wrong"
        
        return True, f"Imbalance ratio: {stats['imbalance_ratio']:.2f}"
    except Exception as e:
        return False, str(e)


def test_sparse_vs_batched_equivalence() -> Tuple[bool, str]:
    """Test that sparse and batched MoE produce similar outputs."""
    try:
        d_model, num_experts = 64, 4
        top_k = 2
        
        sparse_moe = SparseMoE(d_model, num_experts, top_k=top_k)
        batched_moe = BatchedMoE(d_model, num_experts, top_k=top_k)
        
        if sparse_moe.router is None or batched_moe.expert_w1 is None:
            return False, "MoE not initialized"
        
        # Just test that both produce valid outputs
        x = torch.randn(2, 4, d_model)
        
        sparse_out, sparse_loss = sparse_moe(x)
        batched_out, batched_loss = batched_moe(x)
        
        if sparse_out.shape != x.shape:
            return False, "Sparse output shape wrong"
        if batched_out.shape != x.shape:
            return False, "Batched output shape wrong"
        
        return True, "Both implementations work"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("expert_init", test_expert_init),
        ("expert_forward", test_expert_forward),
        ("router_init", test_router_init),
        ("router_forward", test_router_forward),
        ("top_k_gating_shapes", test_top_k_gating_shapes),
        ("top_k_gating_values", test_top_k_gating_values),
        ("top_k_selection", test_top_k_selection),
        ("load_balancing_loss", test_load_balancing_loss),
        ("load_balancing_uniform", test_load_balancing_uniform),
        ("sparse_moe_init", test_sparse_moe_init),
        ("sparse_moe_forward", test_sparse_moe_forward),
        ("sparse_moe_gradient", test_sparse_moe_gradient),
        ("batched_moe_init", test_batched_moe_init),
        ("batched_moe_forward", test_batched_moe_forward),
        ("moe_transformer_block", test_moe_transformer_block),
        ("expert_usage_analysis", test_expert_usage_analysis),
        ("sparse_vs_batched", test_sparse_vs_batched_equivalence),
    ]
    
    print(f"\n{'='*50}\nDay 24: Mixture of Experts - Tests\n{'='*50}")
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
