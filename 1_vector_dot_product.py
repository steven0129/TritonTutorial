import torch
import triton
import triton.language as tl


@triton.jit
def dot_kernel(
    x_ptr,
    y_ptr,
    partial_sum_ptr,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    NUM_PROGRAMS = tl.num_programs(0)

    # Initialize the output buffer
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in range(pid * BLOCK_SIZE, N_ELEMENTS, NUM_PROGRAMS * BLOCK_SIZE):
        # Prepare offsets
        offsets = start + tl.arange(0, BLOCK_SIZE)

        # Remove out-of-boundary elements
        mask = offsets < N_ELEMENTS

        # Load from DRAM
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0)

        # Accumulate the result in the output buffer
        output += x * y

    # Write back to DRAM
    tl.store(partial_sum_ptr + pid, tl.sum(output))

@triton.jit
def product_kernel(
    partial_sum_ptr,
    output_ptr,
    NUM_PARTIAL_SUMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < NUM_PARTIAL_SUMS
    partial_sum = tl.load(partial_sum_ptr + idx, mask=mask)
    total = tl.sum(partial_sum)
    tl.store(output_ptr, total)


def dot_product(x: torch.Tensor, y: torch.Tensor):
    n_elements = x.numel()
    block_size = min(1024, n_elements)
    num_programs = min(20, triton.cdiv(n_elements, block_size))
    partial_sums = torch.empty((block_size), device=x.device)

    output = torch.empty((), device=x.device)

    dot_kernel[(num_programs,)](
        x,
        y,
        partial_sums,
        N_ELEMENTS=n_elements,
        BLOCK_SIZE=block_size,
    )

    product_kernel[(1,)](
        partial_sums,
        output,
        NUM_PARTIAL_SUMS=num_programs,
        BLOCK_SIZE=triton.next_power_of_2(num_programs),
    )

    return output


x = torch.Tensor([1, 2, 3, 4]).cuda()
y = torch.Tensor([4, 3, 2, 1]).cuda()

print(dot_product(x, y))
print(torch.dot(x, y))