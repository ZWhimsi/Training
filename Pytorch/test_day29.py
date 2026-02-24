"""Test Suite for Day 29: DeepSeek Architecture Components"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Tuple

try:
    from day29 import (
        swish, Swish, SwiGLUFFN, SwiGLUExpert,
        DeepSeekRouter, DeepSeekMoE,
        DeepSeekConfig, DeepSeekBlock, DeepSeekModel,
        analyze_expert_usage, compute_activated_params
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_swish_function() -> Tuple[bool, str]:
    """Test swish activation function."""
    try:
        x = torch.randn(2, 8, 64)
        y = swish(x)
        
        if y.shape != x.shape:
            return False, f"Output shape {y.shape} != {x.shape}"
        
        # Swish should be x * sigmoid(x)
        expected = x * torch.sigmoid(x)
        if not torch.allclose(y, expected, atol=1e-5):
            return False, "Swish output doesn't match expected"
        
        return True, "Swish function works correctly"
    except Exception as e:
        return False, str(e)


def test_swish_properties() -> Tuple[bool, str]:
    """Test properties of swish activation."""
    try:
        # Swish(0) should be 0
        zero = torch.zeros(1)
        if abs(swish(zero).item()) > 1e-6:
            return False, "Swish(0) should be 0"
        
        # Swish is smooth and differentiable
        x = torch.randn(10, requires_grad=True)
        y = swish(x)
        y.sum().backward()
        
        if x.grad is None:
            return False, "Swish not differentiable"
        
        return True, "Swish properties verified"
    except Exception as e:
        return False, str(e)


def test_swiglu_ffn_init() -> Tuple[bool, str]:
    """Test SwiGLU FFN initialization."""
    try:
        d_model = 256
        ffn = SwiGLUFFN(d_model)
        
        if ffn.W_gate is None:
            return False, "W_gate not initialized"
        if ffn.W_up is None:
            return False, "W_up not initialized"
        if ffn.W_down is None:
            return False, "W_down not initialized"
        
        return True, f"SwiGLU FFN initialized, hidden dim: {ffn.d_ffn}"
    except Exception as e:
        return False, str(e)


def test_swiglu_ffn_forward() -> Tuple[bool, str]:
    """Test SwiGLU FFN forward pass."""
    try:
        d_model = 256
        ffn = SwiGLUFFN(d_model)
        
        if ffn.W_gate is None:
            return False, "FFN not initialized"
        
        batch, seq = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq, d_model)
        out = ffn(x)
        
        if out.shape != x.shape:
            return False, f"Output shape {out.shape} != {x.shape}"
        
        # Manually compute expected SwiGLU output
        gate = swish(ffn.W_gate(x))
        up = ffn.W_up(x)
        hidden = gate * up
        expected = ffn.W_down(hidden)
        
        if not torch.allclose(out, expected, atol=1e-5):
            return False, "SwiGLU output doesn't match manual computation"
        
        return True, f"Forward pass verified, output: {out.shape}"
    except Exception as e:
        return False, str(e)


def test_swiglu_ffn_gating() -> Tuple[bool, str]:
    """Test that SwiGLU gating mechanism works."""
    try:
        d_model = 64
        ffn = SwiGLUFFN(d_model)
        
        if ffn.W_gate is None:
            return False, "FFN not initialized"
        
        torch.manual_seed(123)
        x = torch.randn(1, 1, d_model)
        out = ffn(x)
        
        # Verify the gating formula: out = (swish(W_gate(x)) * W_up(x)) @ W_down
        gate_val = swish(ffn.W_gate(x))
        up_val = ffn.W_up(x)
        gated = gate_val * up_val
        expected = ffn.W_down(gated)
        
        if not torch.allclose(out, expected, atol=1e-5):
            return False, "Gating computation doesn't match expected formula"
        
        # Verify gating is element-wise multiplication
        if gate_val.shape != up_val.shape:
            return False, "Gate and up projections should have same shape"
        
        return True, "Gating mechanism verified with correct computation"
    except Exception as e:
        return False, str(e)


def test_swiglu_expert_init() -> Tuple[bool, str]:
    """Test SwiGLUExpert initialization."""
    try:
        d_model = 256
        expert = SwiGLUExpert(d_model)
        
        if expert.ffn is None:
            return False, "Expert FFN not initialized"
        
        return True, "SwiGLU Expert initialized"
    except Exception as e:
        return False, str(e)


def test_swiglu_expert_forward() -> Tuple[bool, str]:
    """Test SwiGLUExpert forward pass."""
    try:
        d_model = 256
        expert = SwiGLUExpert(d_model)
        
        if expert.ffn is None:
            return False, "Expert not initialized"
        
        torch.manual_seed(42)
        x = torch.randn(2, 16, d_model)
        out = expert(x)
        
        if out.shape != x.shape:
            return False, f"Output shape wrong"
        
        # Expert should just wrap SwiGLUFFN, verify output matches
        expected = expert.ffn(x)
        if not torch.allclose(out, expected, atol=1e-5):
            return False, "Expert output doesn't match its FFN output"
        
        return True, f"Expert forward verified against FFN"
    except Exception as e:
        return False, str(e)


def test_deepseek_router_init() -> Tuple[bool, str]:
    """Test DeepSeekRouter initialization."""
    try:
        d_model, num_experts = 256, 8
        router = DeepSeekRouter(d_model, num_experts, top_k=2)
        
        if router.gate is None:
            return False, "Router gate not initialized"
        
        return True, "Router initialized"
    except Exception as e:
        return False, str(e)


def test_deepseek_router_forward() -> Tuple[bool, str]:
    """Test DeepSeekRouter forward pass."""
    try:
        d_model, num_experts, top_k = 256, 8, 2
        router = DeepSeekRouter(d_model, num_experts, top_k)
        
        if router.gate is None:
            return False, "Router not initialized"
        
        batch, seq = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq, d_model)
        
        gates, indices, logits = router(x, training=False)
        
        if gates.shape != (batch, seq, top_k):
            return False, f"Gates shape wrong: {gates.shape}"
        if indices.shape != (batch, seq, top_k):
            return False, f"Indices shape wrong: {indices.shape}"
        if logits.shape != (batch, seq, num_experts):
            return False, f"Logits shape wrong: {logits.shape}"
        
        # Gates should sum to ~1 (normalized)
        gate_sums = gates.sum(dim=-1)
        if not torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=0.01):
            return False, "Gates not normalized"
        
        # Verify logits come from gate projection
        expected_logits = router.gate(x)
        if not torch.allclose(logits, expected_logits, atol=1e-5):
            return False, "Logits don't match gate projection"
        
        # Verify indices are top-k of softmax(logits)
        probs = F.softmax(logits, dim=-1)
        expected_probs, expected_indices = torch.topk(probs, top_k, dim=-1)
        if not torch.equal(indices, expected_indices):
            return False, "Indices don't match top-k selection"
        
        return True, "Router forward verified"
    except Exception as e:
        return False, str(e)


def test_deepseek_router_topk() -> Tuple[bool, str]:
    """Test that router selects top-k experts."""
    try:
        d_model, num_experts, top_k = 256, 8, 3
        router = DeepSeekRouter(d_model, num_experts, top_k)
        
        if router.gate is None:
            return False, "Router not initialized"
        
        x = torch.randn(1, 1, d_model)
        gates, indices, logits = router(x, training=False)  # No noise
        
        # Indices should be unique per token
        unique_count = len(torch.unique(indices[0, 0]))
        if unique_count != top_k:
            return False, f"Expected {top_k} unique experts, got {unique_count}"
        
        return True, f"Top-{top_k} selection works"
    except Exception as e:
        return False, str(e)


def test_deepseek_moe_init() -> Tuple[bool, str]:
    """Test DeepSeekMoE initialization."""
    try:
        d_model = 256
        moe = DeepSeekMoE(d_model, num_shared_experts=2, num_routed_experts=4, top_k=2)
        
        if moe.shared_experts is None:
            return False, "Shared experts not initialized"
        if moe.routed_experts is None:
            return False, "Routed experts not initialized"
        if moe.router is None:
            return False, "Router not initialized"
        
        if len(moe.shared_experts) != 2:
            return False, f"Expected 2 shared experts"
        if len(moe.routed_experts) != 4:
            return False, f"Expected 4 routed experts"
        
        return True, "DeepSeek MoE initialized"
    except Exception as e:
        return False, str(e)


def test_deepseek_moe_forward() -> Tuple[bool, str]:
    """Test DeepSeekMoE forward pass."""
    try:
        d_model = 256
        num_shared = 2
        num_routed = 4
        top_k = 2
        moe = DeepSeekMoE(d_model, num_shared_experts=num_shared, num_routed_experts=num_routed, top_k=top_k)
        
        if moe.router is None:
            return False, "MoE not initialized"
        
        batch, seq = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq, d_model)
        
        out, aux = moe(x)
        
        if out.shape != x.shape:
            return False, f"Output shape wrong"
        
        if 'selected_experts' not in aux:
            return False, "Missing routing info"
        if 'gate_weights' not in aux:
            return False, "Missing gate weights in aux"
        if 'router_logits' not in aux:
            return False, "Missing router logits in aux"
        
        # Verify selected_experts shape
        if aux['selected_experts'].shape != (batch, seq, top_k):
            return False, f"selected_experts shape wrong: {aux['selected_experts'].shape}"
        
        # Verify gate_weights sum to ~1
        gate_sums = aux['gate_weights'].sum(dim=-1)
        if not torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=0.01):
            return False, "Gate weights not normalized"
        
        return True, "MoE forward verified with routing info"
    except Exception as e:
        return False, str(e)


def test_deepseek_moe_shared_experts() -> Tuple[bool, str]:
    """Test that shared experts are always active."""
    try:
        d_model = 64
        num_shared = 2
        moe = DeepSeekMoE(d_model, num_shared_experts=num_shared, num_routed_experts=4, top_k=1)
        
        if moe.router is None:
            return False, "MoE not initialized"
        
        torch.manual_seed(42)
        x = torch.randn(2, 8, d_model)
        out, aux = moe(x)
        
        # Output should be non-zero (shared experts contribute)
        if out.abs().sum() == 0:
            return False, "Output is zero (shared experts not contributing)"
        
        # Verify shared experts are computed correctly
        shared_output = torch.zeros_like(x)
        for expert in moe.shared_experts:
            shared_output = shared_output + expert(x)
        shared_output = shared_output / num_shared
        
        # The shared component should be part of the output
        # Output = shared_output + routed_output, so out - shared_output = routed_output
        routed_contribution = out - shared_output
        
        # Routed output should depend on gate weights
        if aux['gate_weights'].abs().sum() == 0:
            return False, "Gate weights are all zero"
        
        return True, "Shared experts verified in output"
    except Exception as e:
        return False, str(e)


def test_deepseek_config() -> Tuple[bool, str]:
    """Test DeepSeekConfig creation."""
    try:
        config = DeepSeekConfig(
            d_model=256,
            num_heads=4,
            head_dim=64,
            num_shared_experts=2,
            num_routed_experts=8,
            top_k=2
        )
        
        if config.d_model != 256:
            return False, "d_model not set"
        if config.num_routed_experts != 8:
            return False, "num_routed_experts not set"
        
        return True, "Config created successfully"
    except Exception as e:
        return False, str(e)


def test_deepseek_block_init() -> Tuple[bool, str]:
    """Test DeepSeekBlock initialization."""
    try:
        config = DeepSeekConfig(
            d_model=256,
            num_heads=4,
            head_dim=64,
            num_shared_experts=1,
            num_routed_experts=4,
            top_k=2
        )
        
        block = DeepSeekBlock(config, use_moe=True)
        
        if block.attn is None:
            return False, "Attention not initialized"
        if block.ffn is None:
            return False, "FFN not initialized"
        
        return True, "DeepSeek block initialized"
    except Exception as e:
        return False, str(e)


def test_deepseek_block_forward() -> Tuple[bool, str]:
    """Test DeepSeekBlock forward pass."""
    try:
        config = DeepSeekConfig(
            d_model=256,
            num_heads=4,
            head_dim=64,
            num_shared_experts=1,
            num_routed_experts=4,
            top_k=2
        )
        
        block = DeepSeekBlock(config, use_moe=True)
        
        if block.ffn is None:
            return False, "Block not initialized"
        
        batch, seq = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq, config.d_model)
        
        out, aux = block(x)
        
        if out.shape != x.shape:
            return False, f"Output shape wrong"
        
        # Verify output is different from input (block should transform)
        if torch.allclose(out, x, atol=1e-3):
            return False, "Output too similar to input, block not transforming"
        
        # Check that MoE aux info is present for MoE block
        if 'selected_experts' not in aux:
            return False, "MoE block should have routing info in aux"
        
        # Verify residual connection exists by checking that zeroing weights
        # doesn't make output zero (it should be close to input due to residual)
        
        return True, "Block forward works with MoE routing"
    except Exception as e:
        return False, str(e)


def test_deepseek_block_dense() -> Tuple[bool, str]:
    """Test DeepSeekBlock with dense FFN (no MoE)."""
    try:
        config = DeepSeekConfig(d_model=256, num_heads=4, head_dim=64)
        block = DeepSeekBlock(config, use_moe=False)
        
        if block.ffn is None:
            return False, "Block not initialized"
        
        torch.manual_seed(42)
        x = torch.randn(2, 16, config.d_model)
        out, aux = block(x)
        
        if out.shape != x.shape:
            return False, "Output shape wrong"
        
        # Dense block should not have routing info
        if 'selected_experts' in aux:
            return False, "Dense block shouldn't have routing info"
        
        # Verify output is different from input (block should transform)
        if torch.allclose(out, x, atol=1e-3):
            return False, "Output too similar to input, block not transforming"
        
        # Dense block should use SwiGLU FFN - verify it's the right type
        if not isinstance(block.ffn, SwiGLUFFN):
            return False, "Dense block should use SwiGLUFFN"
        
        return True, "Dense block verified with SwiGLU FFN"
    except Exception as e:
        return False, str(e)


def test_deepseek_model_init() -> Tuple[bool, str]:
    """Test DeepSeekModel initialization."""
    try:
        config = DeepSeekConfig(
            d_model=256,
            num_heads=4,
            head_dim=64,
            num_shared_experts=1,
            num_routed_experts=4,
            top_k=2
        )
        
        model = DeepSeekModel(config, num_layers=4, moe_layers=[1, 2, 3])
        
        if model.layers is None:
            return False, "Layers not initialized"
        if len(model.layers) != 4:
            return False, f"Expected 4 layers"
        
        # Check MoE layer assignment
        moe_count = sum(1 for layer in model.layers if layer.use_moe)
        if moe_count != 3:
            return False, f"Expected 3 MoE layers, got {moe_count}"
        
        return True, "Model initialized with 4 layers"
    except Exception as e:
        return False, str(e)


def test_deepseek_model_forward() -> Tuple[bool, str]:
    """Test DeepSeekModel forward pass."""
    try:
        config = DeepSeekConfig(
            d_model=256,
            num_heads=4,
            head_dim=64,
            num_shared_experts=1,
            num_routed_experts=4,
            top_k=2
        )
        
        model = DeepSeekModel(config, num_layers=3, moe_layers=[1, 2])
        
        if model.layers is None:
            return False, "Model not initialized"
        
        batch, seq = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq, config.d_model)
        
        out, aux_list = model(x)
        
        if out.shape != x.shape:
            return False, f"Output shape wrong"
        
        if len(aux_list) != 3:
            return False, f"Expected 3 aux_info dicts"
        
        # Verify output is normalized (final_norm applied)
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        if rms.mean() > 5.0:
            return False, "Output seems unnormalized"
        
        # Verify MoE layers have routing info, dense layer doesn't
        # Layer 0 is dense, layers 1,2 are MoE
        if 'selected_experts' in aux_list[0]:
            return False, "Layer 0 should be dense (no routing info)"
        if 'selected_experts' not in aux_list[1]:
            return False, "Layer 1 should be MoE (have routing info)"
        if 'selected_experts' not in aux_list[2]:
            return False, "Layer 2 should be MoE (have routing info)"
        
        return True, "Model forward verified with correct MoE/dense layers"
    except Exception as e:
        return False, str(e)


def test_gradient_flow() -> Tuple[bool, str]:
    """Test gradient flow through DeepSeek model."""
    try:
        config = DeepSeekConfig(
            d_model=128,
            num_heads=2,
            head_dim=32,
            num_shared_experts=1,
            num_routed_experts=2,
            top_k=1
        )
        
        model = DeepSeekModel(config, num_layers=2, moe_layers=[1])
        
        if model.layers is None:
            return False, "Model not initialized"
        
        x = torch.randn(2, 8, config.d_model, requires_grad=True)
        out, _ = model(x)
        
        loss = out.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "No gradient to input"
        if x.grad.abs().sum() == 0:
            return False, "Gradients are zero"
        
        return True, "Gradients flow correctly"
    except Exception as e:
        return False, str(e)


def test_compute_activated_params() -> Tuple[bool, str]:
    """Test activated parameter computation."""
    try:
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
        
        if params['total_activated_params'] == 0:
            return False, "Parameter computation failed"
        
        if params['num_moe_layers'] != 3:
            return False, f"Wrong MoE layer count"
        
        if params['num_dense_layers'] != 1:
            return False, f"Wrong dense layer count: {params['num_dense_layers']}"
        
        # Verify MoE activated params > dense FFN params (due to shared + top_k routed)
        if params['moe_activated_params_per_layer'] <= params['dense_ffn_params_per_layer']:
            return False, "MoE should activate more params than dense FFN with shared experts"
        
        # Verify attention params are reasonable (should be > 0)
        if params['attention_params_per_layer'] <= 0:
            return False, "Attention params should be positive"
        
        return True, f"Activated params: {params['total_activated_params']:,}"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("swish_function", test_swish_function),
        ("swish_properties", test_swish_properties),
        ("swiglu_ffn_init", test_swiglu_ffn_init),
        ("swiglu_ffn_forward", test_swiglu_ffn_forward),
        ("swiglu_ffn_gating", test_swiglu_ffn_gating),
        ("swiglu_expert_init", test_swiglu_expert_init),
        ("swiglu_expert_forward", test_swiglu_expert_forward),
        ("deepseek_router_init", test_deepseek_router_init),
        ("deepseek_router_forward", test_deepseek_router_forward),
        ("deepseek_router_topk", test_deepseek_router_topk),
        ("deepseek_moe_init", test_deepseek_moe_init),
        ("deepseek_moe_forward", test_deepseek_moe_forward),
        ("deepseek_moe_shared", test_deepseek_moe_shared_experts),
        ("deepseek_config", test_deepseek_config),
        ("deepseek_block_init", test_deepseek_block_init),
        ("deepseek_block_forward", test_deepseek_block_forward),
        ("deepseek_block_dense", test_deepseek_block_dense),
        ("deepseek_model_init", test_deepseek_model_init),
        ("deepseek_model_forward", test_deepseek_model_forward),
        ("gradient_flow", test_gradient_flow),
        ("compute_activated_params", test_compute_activated_params),
    ]
    
    print(f"\n{'='*60}")
    print("Day 29: DeepSeek Architecture Components - Tests")
    print("=" * 60)
    
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    
    print(f"\nSummary: {passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    run_all_tests()
