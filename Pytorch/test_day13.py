"""Test Suite for Day 13: Embedding Layers"""

import torch
import torch.nn as nn
from typing import Tuple

try:
    from day13 import (embedding_lookup, ManualEmbedding, create_embedding_from_pretrained,
                       extend_pretrained_embedding, get_padding_mask, masked_embedding_mean,
                       SimpleTextClassifier, embedding_bag_mean, TokenAndPositionEmbedding)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_embedding_lookup() -> Tuple[bool, str]:
    """Test manual embedding lookup."""
    try:
        torch.manual_seed(42)
        
        vocab_size, embed_dim = 100, 32
        embedding_table = torch.randn(vocab_size, embed_dim)
        indices = torch.tensor([[1, 5, 10], [20, 30, 40]])
        
        result = embedding_lookup(indices, embedding_table)
        
        # Check shape
        if result.shape != torch.Size([2, 3, 32]):
            return False, f"Expected shape [2, 3, 32], got {list(result.shape)}"
        
        # Check values
        expected = embedding_table[indices]
        if not torch.allclose(result, expected):
            return False, "Values don't match expected"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_manual_embedding() -> Tuple[bool, str]:
    """Test ManualEmbedding layer."""
    try:
        torch.manual_seed(42)
        
        vocab_size, embed_dim = 100, 32
        emb = ManualEmbedding(vocab_size, embed_dim, padding_idx=0)
        
        if emb.weight is None:
            return False, "Not implemented"
        
        # Check shape
        if emb.weight.shape != torch.Size([vocab_size, embed_dim]):
            return False, f"Weight shape: {list(emb.weight.shape)}"
        
        # Test forward
        indices = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = emb(indices)
        
        if result.shape != torch.Size([2, 3, 32]):
            return False, f"Output shape: {list(result.shape)}"
        
        # Check padding_idx is zero
        if emb.padding_idx is not None:
            if not torch.allclose(emb.weight[emb.padding_idx], torch.zeros(embed_dim)):
                return False, "Padding index not zeroed"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_embedding_against_pytorch() -> Tuple[bool, str]:
    """Test ManualEmbedding matches nn.Embedding."""
    try:
        torch.manual_seed(42)
        
        vocab_size, embed_dim = 50, 16
        
        manual_emb = ManualEmbedding(vocab_size, embed_dim)
        pytorch_emb = nn.Embedding(vocab_size, embed_dim)
        
        if manual_emb.weight is None:
            return False, "Not implemented"
        
        # Copy weights
        pytorch_emb.weight.data = manual_emb.weight.data.clone()
        
        indices = torch.randint(0, vocab_size, (8, 10))
        
        manual_out = manual_emb(indices)
        pytorch_out = pytorch_emb(indices)
        
        if not torch.allclose(manual_out, pytorch_out):
            return False, "Output doesn't match PyTorch"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_create_from_pretrained() -> Tuple[bool, str]:
    """Test creating embedding from pretrained vectors."""
    try:
        torch.manual_seed(42)
        
        pretrained = torch.randn(100, 64)
        
        # Non-frozen
        emb = create_embedding_from_pretrained(pretrained, freeze=False)
        
        if not torch.allclose(emb.weight.data, pretrained):
            return False, "Weights don't match pretrained"
        
        # Frozen
        emb_frozen = create_embedding_from_pretrained(pretrained, freeze=True)
        
        if emb_frozen.weight.requires_grad:
            return False, "Frozen embedding should not require grad"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_extend_pretrained() -> Tuple[bool, str]:
    """Test extending pretrained embeddings."""
    try:
        torch.manual_seed(42)
        
        pretrained = torch.randn(100, 64)
        additional = 5
        
        emb = extend_pretrained_embedding(pretrained, additional)
        
        # Check total size
        if emb.num_embeddings != 105:
            return False, f"Expected 105 embeddings, got {emb.num_embeddings}"
        
        # Check pretrained part is preserved
        if not torch.allclose(emb.weight[:100], pretrained):
            return False, "Pretrained vectors not preserved"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_padding_mask() -> Tuple[bool, str]:
    """Test padding mask creation."""
    try:
        indices = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
        
        mask = get_padding_mask(indices, padding_idx=0)
        
        expected = torch.tensor([[False, False, True, True],
                                  [False, False, False, True]])
        
        if not torch.equal(mask, expected):
            return False, f"Mask doesn't match expected"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_masked_embedding_mean() -> Tuple[bool, str]:
    """Test computing mean with masked positions."""
    try:
        # Create embeddings where we know the expected result
        embeddings = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],  # Last is padding
            [[2.0, 4.0], [0.0, 0.0], [0.0, 0.0]],  # Last two are padding
        ])
        mask = torch.tensor([
            [False, False, True],
            [False, True, True],
        ])
        
        result = masked_embedding_mean(embeddings, mask)
        
        # Sequence 1: mean of [1,2] and [3,4] = [2, 3]
        # Sequence 2: mean of [2,4] = [2, 4]
        expected = torch.tensor([[2.0, 3.0], [2.0, 4.0]])
        
        if not torch.allclose(result, expected):
            return False, f"Expected {expected.tolist()}, got {result.tolist()}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_simple_text_classifier() -> Tuple[bool, str]:
    """Test SimpleTextClassifier."""
    try:
        torch.manual_seed(42)
        model = SimpleTextClassifier(
            vocab_size=1000, embedding_dim=64, 
            num_classes=3, padding_idx=0
        )
        
        if model.embedding is None:
            return False, "Not implemented"
        
        # Test forward
        token_ids = torch.randint(1, 1000, (8, 20))  # Batch of 8, seq len 20
        token_ids[:, -5:] = 0  # Add some padding
        
        output = model(token_ids)
        
        if output.shape != torch.Size([8, 3]):
            return False, f"Expected shape [8, 3], got {list(output.shape)}"
        
        # Validate actual computation: embedding -> masked_mean -> fc
        with torch.no_grad():
            embeddings = model.embedding(token_ids)
            mask = token_ids == model.padding_idx
            mask_expanded = mask.unsqueeze(-1)
            embeddings_masked = embeddings.masked_fill(mask_expanded, 0)
            summed = embeddings_masked.sum(dim=1)
            lengths = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
            pooled = summed / lengths
            expected = model.fc(pooled)
        
        if not torch.allclose(output, expected, atol=1e-5):
            return False, f"Output doesn't match expected computation: max diff {(output - expected).abs().max():.6f}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_embedding_bag_mean() -> Tuple[bool, str]:
    """Test embedding bag mean computation."""
    try:
        torch.manual_seed(42)
        
        embedding_table = torch.tensor([
            [0.0, 0.0],  # Index 0
            [1.0, 2.0],  # Index 1
            [3.0, 4.0],  # Index 2
            [5.0, 6.0],  # Index 3
        ])
        
        indices = torch.tensor([1, 2, 3])  # Three tokens
        offsets = torch.tensor([0, 2])     # Seq1: [1,2], Seq2: [3]
        
        result = embedding_bag_mean(indices, offsets, embedding_table)
        
        if result.shape != torch.Size([2, 2]):
            return False, f"Expected shape [2, 2], got {list(result.shape)}"
        
        # Seq1: mean([1,2], [3,4]) = [2, 3]
        # Seq2: mean([5,6]) = [5, 6]
        expected = torch.tensor([[2.0, 3.0], [5.0, 6.0]])
        
        if not torch.allclose(result, expected):
            return False, f"Expected {expected.tolist()}, got {result.tolist()}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_token_and_position_embedding() -> Tuple[bool, str]:
    """Test TokenAndPositionEmbedding."""
    try:
        vocab_size, max_seq_len, embed_dim = 1000, 128, 64
        
        emb = TokenAndPositionEmbedding(vocab_size, max_seq_len, embed_dim)
        
        if emb.token_embedding is None:
            return False, "Not implemented"
        
        # Test forward
        token_ids = torch.randint(0, vocab_size, (4, 32))
        output = emb(token_ids)
        
        if output.shape != torch.Size([4, 32, 64]):
            return False, f"Expected shape [4, 32, 64], got {list(output.shape)}"
        
        # Verify position embeddings are added (same tokens at different positions should differ)
        token_emb_only = emb.token_embedding(token_ids)
        if torch.allclose(output, token_emb_only):
            return False, "Position embeddings not added"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_embedding_gradient_flow() -> Tuple[bool, str]:
    """Test that gradients flow through embeddings."""
    try:
        emb = ManualEmbedding(100, 32)
        
        if emb.weight is None:
            return False, "Not implemented"
        
        indices = torch.tensor([[1, 2, 3]])
        output = emb(indices)
        
        loss = output.sum()
        loss.backward()
        
        if emb.weight.grad is None:
            return False, "No gradient computed"
        
        # Only indexed rows should have non-zero gradient
        non_zero_grad_rows = (emb.weight.grad.abs().sum(dim=1) > 0).sum().item()
        if non_zero_grad_rows != 3:  # Rows 1, 2, 3
            return False, f"Expected 3 rows with gradient, got {non_zero_grad_rows}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("embedding_lookup", test_embedding_lookup),
        ("manual_embedding", test_manual_embedding),
        ("embedding_vs_pytorch", test_embedding_against_pytorch),
        ("create_from_pretrained", test_create_from_pretrained),
        ("extend_pretrained", test_extend_pretrained),
        ("padding_mask", test_padding_mask),
        ("masked_embedding_mean", test_masked_embedding_mean),
        ("simple_text_classifier", test_simple_text_classifier),
        ("embedding_bag_mean", test_embedding_bag_mean),
        ("token_position_embedding", test_token_and_position_embedding),
        ("embedding_gradient", test_embedding_gradient_flow),
    ]
    
    print(f"\n{'='*50}\nDay 13: Embedding Layers - Tests\n{'='*50}")
    
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        return
    
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    print(f"\nSummary: {passed}/{len(tests)}")


if __name__ == "__main__":
    run_all_tests()
