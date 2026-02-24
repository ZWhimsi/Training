"""
Day 13: Embedding Layers
========================
Estimated time: 1-2 hours
Prerequisites: Day 12 (dropout/regularization)

Learning objectives:
- Understand embeddings as learned lookup tables
- Implement embedding lookup manually
- Initialize embeddings from pretrained vectors
- Apply embeddings in NLP contexts
- Handle padding and special tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# CONCEPT: Embeddings
# ============================================================================
"""
An embedding layer is essentially a lookup table that maps integer indices
to dense vectors:

    embedding_table[index] -> vector of size embedding_dim

Key points:
- Input: integer indices (token IDs, word IDs, etc.)
- Output: dense vectors
- Parameters: embedding_table of shape (num_embeddings, embedding_dim)
- The embedding vectors are learned during training

Why embeddings?
- Convert discrete tokens to continuous representations
- Capture semantic relationships (similar words have similar embeddings)
- Much more efficient than one-hot encoding
"""


# ============================================================================
# Exercise 1: Manual Embedding Lookup
# ============================================================================

def embedding_lookup(indices: torch.Tensor, 
                     embedding_table: torch.Tensor) -> torch.Tensor:
    """
    Perform embedding lookup manually.
    
    Args:
        indices: Integer tensor of any shape containing indices
        embedding_table: Tensor of shape (num_embeddings, embedding_dim)
    
    Returns:
        Embedded vectors of shape (*indices.shape, embedding_dim)
    """
    # API hints:
    # - embedding_table[indices] -> advanced indexing returns embeddings at given indices
    # - Works with any shape of indices tensor
    return None


# ============================================================================
# Exercise 2: Manual Embedding Layer
# ============================================================================

class ManualEmbedding(nn.Module):
    """
    Manual implementation of nn.Embedding.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 padding_idx: Optional[int] = None):
        super().__init__()
        # API hints:
        # - nn.Parameter(tensor) -> register tensor as learnable parameter
        # - torch.randn(num_embeddings, embedding_dim) -> random initialization
        # - If padding_idx is set, zero out that row with torch.no_grad() context
        # - self.weight[padding_idx].fill_(0) -> zero out padding embedding
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = None
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Return embeddings for given indices using self.weight lookup."""
        # API hints:
        # - self.weight[indices] -> lookup embeddings by index
        return None


# ============================================================================
# Exercise 3: Initialize from Pretrained Vectors
# ============================================================================

def create_embedding_from_pretrained(pretrained_vectors: torch.Tensor,
                                      freeze: bool = False) -> nn.Embedding:
    """
    Create an embedding layer initialized with pretrained vectors.
    
    Args:
        pretrained_vectors: Tensor of shape (vocab_size, embedding_dim)
        freeze: If True, embeddings won't be updated during training
    
    Returns:
        nn.Embedding initialized with pretrained vectors
    """
    # API hints:
    # - nn.Embedding.from_pretrained(vectors, freeze=freeze) -> create from existing vectors
    # - freeze=True sets requires_grad=False
    return None


def extend_pretrained_embedding(pretrained_vectors: torch.Tensor,
                                 additional_tokens: int) -> nn.Embedding:
    """
    Create embedding with pretrained vectors + randomly initialized new tokens.
    Useful for adding special tokens like [PAD], [UNK], [CLS], [SEP].
    
    Args:
        pretrained_vectors: Tensor of shape (vocab_size, embedding_dim)
        additional_tokens: Number of new tokens to add
    
    Returns:
        nn.Embedding with extended vocabulary
    """
    # API hints:
    # - nn.Embedding(total_size, embed_dim) -> create new embedding
    # - torch.no_grad() context -> modify weights without tracking gradients
    # - embedding.weight[:vocab_size] = pretrained_vectors -> copy pretrained
    # - New tokens at end are randomly initialized by default
    return None


# ============================================================================
# Exercise 4: Embedding with Padding Mask
# ============================================================================

def get_padding_mask(indices: torch.Tensor, padding_idx: int) -> torch.Tensor:
    """
    Create a boolean mask where True indicates padding positions.
    
    Args:
        indices: Integer tensor of token indices
        padding_idx: The index used for padding
    
    Returns:
        Boolean tensor of same shape as indices
    """
    # API hints:
    # - (indices == padding_idx) -> boolean tensor, True where padding
    return None


def masked_embedding_mean(embeddings: torch.Tensor, 
                          mask: torch.Tensor) -> torch.Tensor:
    """
    Compute mean of embeddings, ignoring masked (padding) positions.
    
    Args:
        embeddings: Tensor of shape (batch, seq_len, embed_dim)
        mask: Boolean tensor of shape (batch, seq_len), True for padding
    
    Returns:
        Mean embeddings of shape (batch, embed_dim)
    """
    # API hints:
    # - mask.unsqueeze(-1) -> expand mask for broadcasting (batch, seq_len, 1)
    # - embeddings.masked_fill(mask, 0) -> zero out masked positions
    # - tensor.sum(dim=1) -> sum over sequence dimension
    # - (~mask).sum(dim=1) -> count non-padding positions
    # - tensor.clamp(min=1) -> avoid division by zero
    return None


# ============================================================================
# Exercise 5: Simple Word Embedding Model
# ============================================================================

class SimpleTextClassifier(nn.Module):
    """
    Simple text classifier using embeddings.
    
    Architecture: Embedding -> Mean Pooling -> Linear -> Output
    """
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 num_classes: int, padding_idx: int = 0):
        super().__init__()
        # API hints:
        # - nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # - nn.Linear(embedding_dim, num_classes) -> output layer
        self.embedding = None
        self.fc = None
        self.padding_idx = padding_idx
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Integer tensor of shape (batch, seq_len)
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        # API hints:
        # - self.embedding(token_ids) -> get embeddings (batch, seq_len, embed_dim)
        # - (token_ids == self.padding_idx) -> create padding mask
        # - masked_embedding_mean(embeddings, mask) -> mean pooling ignoring padding
        # - self.fc(pooled) -> classify
        return None


# ============================================================================
# Exercise 6: Embedding Bag (Efficient Sum/Mean)
# ============================================================================

def embedding_bag_mean(indices: torch.Tensor, offsets: torch.Tensor,
                       embedding_table: torch.Tensor) -> torch.Tensor:
    """
    Compute mean of embeddings for variable-length sequences.
    Similar to nn.EmbeddingBag but implemented manually.
    
    Args:
        indices: 1D tensor of all token indices concatenated
        offsets: 1D tensor of starting positions for each sequence
        embedding_table: Tensor of shape (num_embeddings, embedding_dim)
    
    Returns:
        Mean embeddings of shape (num_sequences, embedding_dim)
    
    Example:
        indices = [1, 2, 3, 4, 5]  # All tokens
        offsets = [0, 2]          # Sequence 1: tokens[0:2], Sequence 2: tokens[2:]
    """
    # API hints:
    # - torch.cat([offsets, torch.tensor([len(indices)])]) -> add end marker
    # - Loop through sequences using offset pairs (start, end)
    # - indices[start:end] -> get sequence indices
    # - embedding_table[seq_indices] -> lookup embeddings
    # - seq_embeds.mean(dim=0) -> mean over sequence
    return None


# ============================================================================
# Exercise 7: Token + Position Embedding
# ============================================================================

class TokenAndPositionEmbedding(nn.Module):
    """
    Combines token embeddings with learned position embeddings.
    
    Output = TokenEmbed(token_id) + PositionEmbed(position)
    """
    def __init__(self, vocab_size: int, max_seq_len: int, embedding_dim: int):
        super().__init__()
        # API hints:
        # - nn.Embedding(vocab_size, embedding_dim) -> token embedding
        # - nn.Embedding(max_seq_len, embedding_dim) -> position embedding
        self.token_embedding = None
        self.position_embedding = None
        self.embedding_dim = embedding_dim
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Integer tensor of shape (batch, seq_len)
        
        Returns:
            Combined embeddings of shape (batch, seq_len, embedding_dim)
        """
        # API hints:
        # - self.token_embedding(token_ids) -> (batch, seq_len, embed_dim)
        # - torch.arange(seq_len, device=token_ids.device) -> position indices [0, 1, ..., seq_len-1]
        # - positions.unsqueeze(0).expand(batch_size, -1) -> broadcast to batch
        # - self.position_embedding(positions) -> position embeddings
        # - Return: token_embeds + pos_embeds
        return None


if __name__ == "__main__":
    print("Day 13: Embedding Layers")
    print("=" * 50)
    
    # Test manual embedding lookup
    vocab_size, embed_dim = 1000, 64
    embedding_table = torch.randn(vocab_size, embed_dim)
    indices = torch.tensor([[1, 5, 10], [2, 3, 4]])
    
    embeddings = embedding_lookup(indices, embedding_table)
    print(f"\nEmbedding lookup: indices {indices.shape} -> embeddings {embeddings.shape}")
    
    # Test ManualEmbedding
    manual_emb = ManualEmbedding(vocab_size, embed_dim, padding_idx=0)
    if manual_emb.weight is not None:
        out = manual_emb(indices)
        print(f"ManualEmbedding output: {out.shape}")
    
    # Test padding mask
    padded_indices = torch.tensor([[1, 2, 3, 0, 0], [5, 6, 0, 0, 0]])
    mask = get_padding_mask(padded_indices, padding_idx=0)
    print(f"\nPadding mask:\n{mask}")
    
    # Test TokenAndPositionEmbedding
    tok_pos = TokenAndPositionEmbedding(vocab_size=5000, max_seq_len=512, embedding_dim=128)
    if tok_pos.token_embedding is not None:
        out = tok_pos(torch.randint(0, 5000, (4, 32)))
        print(f"\nToken+Position embedding: {out.shape}")
    
    print("\nRun test_day13.py to verify all implementations!")
