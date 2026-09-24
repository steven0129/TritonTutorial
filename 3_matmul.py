import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    dtype: tl.constexpr=tl.float32
):
    # [M, K] @ [K, N] -> [M, N]
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=dtype)

    a_rows = pid // tl.cdiv(N, BLOCK_SIZE_N) * BLOCK_SIZE_M \
        + tl.arange(0, BLOCK_SIZE_M)
    b_cols = (pid % tl.cdiv(N, BLOCK_SIZE_N)) * BLOCK_SIZE_N \
        + tl.arange(0, BLOCK_SIZE_N)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_cols = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        b_rows = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        a_block = a_ptr + a_rows[:, None] * K + a_cols[None, :]
        b_block = b_ptr + b_rows[:, None] * N + b_cols[None, :]

        mask_a = (a_rows[:, None] < M) & (a_cols[None, :] < K)
        mask_b = (b_rows[:, None] < K) & (b_cols[None, :] < N)

        a_matrix = tl.load(a_block, mask=mask_a, other=0)
        b_matrix = tl.load(b_block, mask=mask_b, other=0)
        acc += tl.dot(a_matrix, b_matrix)
    
    output_block = output_ptr + a_rows[:, None] * N + b_cols[None, :]
    mask_output = (a_rows[:, None] < M) & (b_cols[None, :] < N)
    tl.store(output_block, acc, mask=mask_output)


def matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0], "A 的欄數必須等於 B 的列數"
    assert a.is_cuda and b.is_cuda
    assert a.device == b.device
    assert a.is_contiguous() and b.is_contiguous()
    assert a.dtype == b.dtype

    M, K = a.shape
    _, N = b.shape

    output = torch.empty(
        (M, N),
        device=a.device
    )

    if M == 0 or N == 0:
        return output

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 32

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
    )

    matmul_kernel[grid](
        a,
        b,
        output,
        M,
        K,
        N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        dtype=tl.float32,
    )

    return output

a = torch.Tensor([[1, 2, 3], [4, 5, 6]]).cuda()
b = torch.Tensor([[1, 2], [3, 4], [5, 6]]).cuda()
print((a @ b) == matmul(a, b))

