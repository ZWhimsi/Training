"""
Day 18: Strided Memory Access
=============================
Estimated time: 1-2 hours
Prerequisites: Day 17 (outer products)

Learning objectives:
- Understand memory strides in multi-dimensional tensors
- Access non-contiguous memory patterns
- Handle transposed and permuted tensors
- Optimize memory access patterns
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# CONCEPT: Memory Strides
# ============================================================================
# A tensor's stride tells you how many elements to skip in memory for each dim.
# 
# For tensor of shape (A, B, C) stored row-major:
# - stride[0] = B * C (skip B*C elements to go to next A)
# - stride[1] = C (skip C elements to go to next B)
# - stride[2] = 1 (contiguous in last dimension)
#
# Transposed tensors have different strides but same underlying data!
# ============================================================================


# ============================================================================
# Exercise 1: Strided Load
# ============================================================================

@triton.jit
def strided_load_kernel(
    input_ptr, output_ptr,
    size, stride_in,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Load elements with a custom stride.
    output[i] = input[i * stride]
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < size
    
    # TODO: Compute strided offsets
    # HINT: strided_offs = offs * stride_in
    strided_offs = None  # Replace
    
    # TODO: Load with strided access
    # HINT: data = tl.load(input_ptr + strided_offs, mask=mask)
    data = None  # Replace
    
    # Store contiguously
    tl.store(output_ptr + offs, data, mask=mask)


def strided_load(x: torch.Tensor, stride: int, size: int) -> torch.Tensor:
    """Load elements with custom stride."""
    output = torch.empty(size, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    strided_load_kernel[grid](x, output, size, stride, BLOCK_SIZE=1024)
    
    return output


# ============================================================================
# Exercise 2: Column Major Access
# ============================================================================

@triton.jit
def column_access_kernel(
    input_ptr, output_ptr,
    M, N,  # Input shape (M rows, N cols)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Read a column from row-major matrix.
    For column c: elements are at [0*N+c, 1*N+c, 2*N+c, ...]
    """
    col_idx = tl.program_id(0)  # Which column
    
    row_offs = tl.arange(0, BLOCK_SIZE)
    mask = row_offs < M
    
    # TODO: Compute offsets for this column
    # Each row is N elements apart, add col_idx for the column
    # HINT: offs = row_offs * N + col_idx
    offs = None  # Replace
    
    # TODO: Load column
    col_data = None  # Replace: tl.load(input_ptr + offs, mask=mask)
    
    # Store to output (column col_idx of output)
    output_offs = row_offs * N + col_idx
    tl.store(output_ptr + output_offs, col_data, mask=mask)


def transpose_via_columns(x: torch.Tensor) -> torch.Tensor:
    """Transpose by reading columns."""
    M, N = x.shape
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = triton.next_power_of_2(M)
    column_access_kernel[(N,)](x, output, M, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return output


# ============================================================================
# Exercise 3: 2D Strided Access
# ============================================================================

@triton.jit
def strided_2d_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_m, stride_n,  # Input strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Access 2D tensor with arbitrary strides.
    
    Handles transposed/permuted tensors correctly.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # TODO: Compute strided offsets
    # HINT: offs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    offs = None  # Replace
    
    # TODO: Load with strided access
    data = None  # Replace: tl.load(input_ptr + offs, mask=mask)
    
    # Store contiguously (standard row-major)
    out_offs = offs_m[:, None] * N + offs_n[None, :]
    tl.store(output_ptr + out_offs, data, mask=mask)


def strided_2d_copy(x: torch.Tensor) -> torch.Tensor:
    """Copy 2D tensor handling arbitrary strides."""
    M, N = x.shape
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    strided_2d_kernel[grid](
        x, output, M, N,
        x.stride(0), x.stride(1),
        BLOCK_M, BLOCK_N
    )
    
    return output


# ============================================================================
# Exercise 4: Gather Operation
# ============================================================================

@triton.jit
def gather_kernel(
    input_ptr, indices_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Gather elements: output[i] = input[indices[i]]
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # TODO: Load indices
    indices = tl.load(indices_ptr + offs, mask=mask)
    
    # TODO: Gather from input using indices
    # HINT: data = tl.load(input_ptr + indices, mask=mask)
    data = None  # Replace
    
    # Store
    tl.store(output_ptr + offs, data, mask=mask)


def gather(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather elements by indices."""
    output = torch.empty_like(indices, dtype=x.dtype)
    n_elements = indices.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gather_kernel[grid](x, indices, output, n_elements, BLOCK_SIZE=1024)
    
    return output


# ============================================================================
# Exercise 5: Scatter Add Operation
# ============================================================================

@triton.jit  
def scatter_add_kernel(
    input_ptr, indices_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scatter add: output[indices[i]] += input[i]
    
    Note: Atomic operations needed for correctness with overlapping indices.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load values and indices
    values = tl.load(input_ptr + offs, mask=mask)
    indices = tl.load(indices_ptr + offs, mask=mask)
    
    # TODO: Atomic add to handle conflicts
    # HINT: tl.atomic_add(output_ptr + indices, values, mask=mask)
    pass  # Replace


def scatter_add(x: torch.Tensor, indices: torch.Tensor, size: int) -> torch.Tensor:
    """Scatter add values to indices."""
    output = torch.zeros(size, device=x.device, dtype=x.dtype)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    scatter_add_kernel[grid](x, indices, output, n_elements, BLOCK_SIZE=1024)
    
    return output


if __name__ == "__main__":
    print("Day 18: Strided Memory Access")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        # Test strided copy handles transposed tensors
        x = torch.randn(64, 128, device=device)
        x_t = x.T  # Transposed - different strides
        
        print("\nTesting strided 2D copy on transposed tensor:")
        result = strided_2d_copy(x_t.contiguous())
        expected = x_t.contiguous()
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
        
        # Test gather
        src = torch.randn(1000, device=device)
        idx = torch.randint(0, 1000, (500,), device=device)
        
        print("\nTesting gather:")
        result = gather(src, idx)
        expected = src[idx]
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day18.py to verify!")
