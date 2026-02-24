"""Test Suite for Day 13: Embedding Layers"""

import torch
import pytest
import torch.nn as nn
try:
    from day13 import (embedding_lookup, ManualEmbedding, create_embedding_from_pretrained,
                       extend_pretrained_embedding, get_padding_mask, masked_embedding_mean,
                       SimpleTextClassifier, embedding_bag_mean, TokenAndPositionEmbedding)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_embedding_lookup():
    """Test manual embedding lookup."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    
    vocab_size, embed_dim = 100, 32
    embedding_table = torch.randn(vocab_size, embed_dim)
    indices = torch.tensor([[1, 5, 10], [20, 30, 40]])
    
    result = embedding_lookup(indices, embedding_table)
    
    assert result.shape == torch.Size([2, 3, 32]), f"Expected shape [2, 3, 32], got {list(result.shape)}"
    
    expected = embedding_table[indices]
    assert torch.allclose(result, expected), "Values don't match expected"

def test_manual_embedding():
    """Test ManualEmbedding layer."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    
    vocab_size, embed_dim = 100, 32
    emb = ManualEmbedding(vocab_size, embed_dim, padding_idx=0)
    
    assert emb.weight is not None, "Not implemented"
    
    assert emb.weight.shape == torch.Size([vocab_size, embed_dim]), f"Weight shape: {list(emb.weight.shape)}"
    
    indices = torch.tensor([[1, 2, 3], [4, 5, 6]])
    result = emb(indices)
    
    assert result.shape == torch.Size([2, 3, 32]), f"Output shape: {list(result.shape)}"
    
    if emb.padding_idx is not None:
        assert torch.allclose(emb.weight[emb.padding_idx], torch.zeros(embed_dim)), "Padding index not zeroed"

def test_embedding_against_pytorch():
    """Test ManualEmbedding matches nn.Embedding."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    
    vocab_size, embed_dim = 50, 16
    
    manual_emb = ManualEmbedding(vocab_size, embed_dim)
    pytorch_emb = nn.Embedding(vocab_size, embed_dim)
    
    assert manual_emb.weight is not None, "Not implemented"
    
    pytorch_emb.weight.data = manual_emb.weight.data.clone()
    
    indices = torch.randint(0, vocab_size, (8, 10))
    
    manual_out = manual_emb(indices)
    pytorch_out = pytorch_emb(indices)
    
    assert torch.allclose(manual_out, pytorch_out), "Output doesn't match PyTorch"

def test_create_from_pretrained():
    """Test creating embedding from pretrained vectors."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    
    pretrained = torch.randn(100, 64)
    
    emb = create_embedding_from_pretrained(pretrained, freeze=False)
    
    assert torch.allclose(emb.weight.data, pretrained), "Weights don't match pretrained"
    
    emb_frozen = create_embedding_from_pretrained(pretrained, freeze=True)
    
    assert not emb_frozen.weight.requires_grad, "Frozen embedding should not require grad"

def test_extend_pretrained():
    """Test extending pretrained embeddings."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    
    pretrained = torch.randn(100, 64)
    additional = 5
    
    emb = extend_pretrained_embedding(pretrained, additional)
    
    assert emb.num_embeddings == 105, f"Expected 105 embeddings, got {emb.num_embeddings}"
    
    assert torch.allclose(emb.weight[:100], pretrained), "Pretrained vectors not preserved"

def test_padding_mask():
    """Test padding mask creation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    indices = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    
    mask = get_padding_mask(indices, padding_idx=0)
    
    expected = torch.tensor([[False, False, True, True],
                              [False, False, False, True]])
    
    assert torch.equal(mask, expected), "Mask doesn't match expected"

def test_masked_embedding_mean():
    """Test computing mean with masked positions."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    embeddings = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
        [[2.0, 4.0], [0.0, 0.0], [0.0, 0.0]],
    ])
    mask = torch.tensor([
        [False, False, True],
        [False, True, True],
    ])
    
    result = masked_embedding_mean(embeddings, mask)
    
    expected = torch.tensor([[2.0, 3.0], [2.0, 4.0]])
    
    assert torch.allclose(result, expected), f"Expected {expected.tolist()}, got {result.tolist()}"

def test_simple_text_classifier():
    """Test SimpleTextClassifier."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    model = SimpleTextClassifier(
        vocab_size=1000, embedding_dim=64, 
        num_classes=3, padding_idx=0
    )
    
    assert model.embedding is not None, "Not implemented"
    
    token_ids = torch.randint(1, 1000, (8, 20))
    token_ids[:, -5:] = 0
    
    output = model(token_ids)
    
    assert output.shape == torch.Size([8, 3]), f"Expected shape [8, 3], got {list(output.shape)}"
    
    with torch.no_grad():
        embeddings = model.embedding(token_ids)
        mask = token_ids == model.padding_idx
        mask_expanded = mask.unsqueeze(-1)
        embeddings_masked = embeddings.masked_fill(mask_expanded, 0)
        summed = embeddings_masked.sum(dim=1)
        lengths = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
        pooled = summed / lengths
        expected = model.fc(pooled)
    
    assert torch.allclose(output, expected, atol=1e-5), f"Output doesn't match expected computation: max diff {(output - expected).abs().max():.6f}"

def test_embedding_bag_mean():
    """Test embedding bag mean computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    
    embedding_table = torch.tensor([
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    
    indices = torch.tensor([1, 2, 3])
    offsets = torch.tensor([0, 2])
    
    result = embedding_bag_mean(indices, offsets, embedding_table)
    
    assert result.shape == torch.Size([2, 2]), f"Expected shape [2, 2], got {list(result.shape)}"
    
    expected = torch.tensor([[2.0, 3.0], [5.0, 6.0]])
    
    assert torch.allclose(result, expected), f"Expected {expected.tolist()}, got {result.tolist()}"

def test_token_and_position_embedding():
    """Test TokenAndPositionEmbedding."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    vocab_size, max_seq_len, embed_dim = 1000, 128, 64
    
    emb = TokenAndPositionEmbedding(vocab_size, max_seq_len, embed_dim)
    
    assert emb.token_embedding is not None, "Not implemented"
    
    token_ids = torch.randint(0, vocab_size, (4, 32))
    output = emb(token_ids)
    
    assert output.shape == torch.Size([4, 32, 64]), f"Expected shape [4, 32, 64], got {list(output.shape)}"
    
    token_emb_only = emb.token_embedding(token_ids)
    assert not torch.allclose(output, token_emb_only), "Position embeddings not added"

def test_embedding_gradient_flow():
    """Test that gradients flow through embeddings."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    emb = ManualEmbedding(100, 32)
    
    assert emb.weight is not None, "Not implemented"
    
    indices = torch.tensor([[1, 2, 3]])
    output = emb(indices)
    
    loss = output.sum()
    loss.backward()
    
    assert emb.weight.grad is not None, "No gradient computed"
    
    non_zero_grad_rows = (emb.weight.grad.abs().sum(dim=1) > 0).sum().item()
    assert non_zero_grad_rows == 3, f"Expected 3 rows with gradient, got {non_zero_grad_rows}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
