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
    
    TODO: Implement embedding lookup
    HINT:
        # Simple indexing does the job!
        return embedding_table[indices]
    """
    return torch.zeros(*indices.shape, embedding_table.shape[1])  # Replace


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
        """
        TODO: Initialize embedding table with random values
        HINT:
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.padding_idx = padding_idx
            
            # Initialize embedding table as a parameter
            self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
            
            # Zero out padding embedding if specified
            if padding_idx is not None:
                with torch.no_grad():
                    self.weight[padding_idx].fill_(0)
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = None  # Replace
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        TODO: Return embeddings for given indices
        HINT: return self.weight[indices]
        """
        return torch.zeros(*indices.shape, self.embedding_dim)  # Replace


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
    
    TODO: Create embedding from pretrained
    HINT:
        embedding = nn.Embedding.from_pretrained(pretrained_vectors, freeze=freeze)
        return embedding
    """
    vocab_size, embed_dim = pretrained_vectors.shape
    return nn.Embedding(vocab_size, embed_dim)  # Replace


def extend_pretrained_embedding(pretrained_vectors: torch.Tensor,
                                 additional_tokens: int) -> nn.Embedding:
    """
    Create embedding with pretrained vectors + randomly initialized new tokens.
    
    This is useful when you have pretrained embeddings but need to add
    special tokens like [PAD], [UNK], [CLS], [SEP], etc.
    
    Args:
        pretrained_vectors: Tensor of shape (vocab_size, embedding_dim)
        additional_tokens: Number of new tokens to add
    
    Returns:
        nn.Embedding with extended vocabulary
    
    TODO: Extend pretrained embeddings
    HINT:
        vocab_size, embed_dim = pretrained_vectors.shape
        total_size = vocab_size + additional_tokens
        
        embedding = nn.Embedding(total_size, embed_dim)
        
        # Copy pretrained vectors
        with torch.no_grad():
            embedding.weight[:vocab_size] = pretrained_vectors
            # Initialize new tokens randomly (already done by nn.Embedding)
        
        return embedding
    """
    vocab_size, embed_dim = pretrained_vectors.shape
    return nn.Embedding(vocab_size + additional_tokens, embed_dim)  # Replace


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
    
    TODO: Create padding mask
    HINT: return indices == padding_idx
    """
    return torch.zeros_like(indices, dtype=torch.bool)  # Replace


def masked_embedding_mean(embeddings: torch.Tensor, 
                          mask: torch.Tensor) -> torch.Tensor:
    """
    Compute mean of embeddings, ignoring masked (padding) positions.
    
    Args:
        embeddings: Tensor of shape (batch, seq_len, embed_dim)
        mask: Boolean tensor of shape (batch, seq_len), True for padding
    
    Returns:
        Mean embeddings of shape (batch, embed_dim)
    
    TODO: Compute masked mean
    HINT:
        # Expand mask for broadcasting
        mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Zero out padding embeddings
        embeddings_masked = embeddings.masked_fill(mask_expanded, 0)
        
        # Sum non-padding embeddings
        summed = embeddings_masked.sum(dim=1)
        
        # Count non-padding positions
        lengths = (~mask).sum(dim=1, keepdim=True).float()
        lengths = lengths.clamp(min=1)  # Avoid division by zero
        
        return summed / lengths
    """
    return embeddings.mean(dim=1)  # Replace


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
        """
        TODO: Create the model components
        HINT:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
            self.fc = nn.Linear(embedding_dim, num_classes)
            self.padding_idx = padding_idx
        """
        self.embedding = None  # Replace
        self.fc = None         # Replace
        self.padding_idx = padding_idx
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Integer tensor of shape (batch, seq_len)
        
        Returns:
            Logits of shape (batch, num_classes)
        
        TODO: Implement forward pass
        HINT:
            # Get embeddings
            embeddings = self.embedding(token_ids)  # (batch, seq_len, embed_dim)
            
            # Create padding mask
            mask = token_ids == self.padding_idx
            
            # Mean pooling (ignoring padding)
            pooled = masked_embedding_mean(embeddings, mask)
            
            # Classify
            return self.fc(pooled)
        """
        return torch.zeros(token_ids.shape[0], 2)  # Replace


# ============================================================================
# Exercise 6: Embedding Bag (Efficient Sum/Mean)
# ============================================================================

def embedding_bag_mean(indices: torch.Tensor, offsets: torch.Tensor,
                       embedding_table: torch.Tensor) -> torch.Tensor:
    """
    Compute mean of embeddings for variable-length sequences.
    
    This is similar to nn.EmbeddingBag but implemented manually.
    
    Args:
        indices: 1D tensor of all token indices concatenated
        offsets: 1D tensor of starting positions for each sequence
        embedding_table: Tensor of shape (num_embeddings, embedding_dim)
    
    Returns:
        Mean embeddings of shape (num_sequences, embedding_dim)
    
    Example:
        indices = [1, 2, 3, 4, 5]  # All tokens
        offsets = [0, 2]          # Sequence 1: tokens[0:2], Sequence 2: tokens[2:]
        -> Returns mean embeddings for each sequence
    
    TODO: Implement embedding bag mean
    HINT:
        num_seqs = offsets.shape[0]
        embed_dim = embedding_table.shape[1]
        result = torch.zeros(num_seqs, embed_dim)
        
        # Add end offset for easier iteration
        offsets_with_end = torch.cat([offsets, torch.tensor([len(indices)])])
        
        for i in range(num_seqs):
            start = offsets_with_end[i].item()
            end = offsets_with_end[i + 1].item()
            seq_indices = indices[start:end]
            seq_embeds = embedding_table[seq_indices]
            result[i] = seq_embeds.mean(dim=0)
        
        return result
    """
    num_seqs = offsets.shape[0]
    embed_dim = embedding_table.shape[1]
    return torch.zeros(num_seqs, embed_dim)  # Replace


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
        """
        TODO: Create token and position embedding layers
        HINT:
            self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
            self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        """
        self.token_embedding = None     # Replace
        self.position_embedding = None  # Replace
        self.embedding_dim = embedding_dim
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Integer tensor of shape (batch, seq_len)
        
        Returns:
            Combined embeddings of shape (batch, seq_len, embedding_dim)
        
        TODO: Combine token and position embeddings
        HINT:
            batch_size, seq_len = token_ids.shape
            
            # Get token embeddings
            token_embeds = self.token_embedding(token_ids)
            
            # Create position indices [0, 1, 2, ..., seq_len-1]
            positions = torch.arange(seq_len, device=token_ids.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
            
            # Get position embeddings
            pos_embeds = self.position_embedding(positions)
            
            # Combine
            return token_embeds + pos_embeds
        """
        return torch.zeros(token_ids.shape[0], token_ids.shape[1], 
                          self.embedding_dim)  # Replace


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
