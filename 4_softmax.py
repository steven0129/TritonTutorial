import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr = tl.float32
):
    pid = tl.program_id(0)

    running_max = float('-inf')
    running_sum = 0.0
    answer = 0.0
    
    # Find the final max and final sum
    for block_id in range(0, tl.cdiv(N, BLOCK_SIZE)):
        block = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = block < N
        input = tl.load(input_ptr + block, mask=mask, other=float('-inf'))
        new_max = tl.maximum(running_max, tl.max(input, axis=0))
        running_sum = tl.exp(running_max - new_max) * running_sum + \
            tl.sum(tl.exp(input - new_max))
        running_max = new_max

    inverse_running_sum = 1.0 / running_sum # Mult is faster than Div on GPU

    # Compute nominator
    for block_id in range(0, tl.cdiv(N, BLOCK_SIZE)):
        block = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = block < N
        input = tl.load(input_ptr + block, mask=mask, other=float('-inf'))
        numerator = tl.exp(input - running_max)
        output = numerator * inverse_running_sum
        tl.store(output_ptr + block, output, mask=mask)


def online_softmax(x: torch.Tensor):
    output = torch.empty_like(x)
    block_size = triton.next_power_of_2(x.shape[0])
    online_softmax_kernel[(1,)](x, output, x.shape[0], block_size)
    return output

if __name__ == '__main__':
    a = torch.Tensor([1, 2, 3, 4]).cuda()
    print(online_softmax(a))
    print(torch.softmax(a, dim=0))