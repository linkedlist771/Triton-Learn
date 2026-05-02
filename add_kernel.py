import torch
import triton

import triton.language as tl
from loguru import logger

DEVICE = torch.device("cuda", torch.cuda.current_device())


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elments, BLOCK_SIZE: tl.constexpr):
    # tl.constexpr 编译器常量， 让编译器知道
    pid = tl.program_id(axis=0)
    # 类似 cuda 里面的 blockIdx.x


    block_start = pid * BLOCK_SIZE
    # 类似 cuda里面的 idx = blockIdx.x * blockDim.x  
    # triton 里面隐藏掉了threadIdx.x
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elments
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)


    output = x + y
    tl.store(output_ptr+offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    # logger.debug(f"x.device:{x.device},y.device:{y.device}, DEVICE:{DEVICE}")
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # logger.debug(grid)
    add_kernel[grid](x,y,output, n_elements, BLOCK_SIZE=2048)
    return output
    


# x = torch.arange(2048)
# y = torch.arange(2048) + 5
# x = x.to(DEVICE)
# y = y.to(DEVICE)
# res = add(x, y)
# # print(res)
# logger.debug(f"x: {x}")    
# logger.debug(f"y: {y}")    
# logger.debug(f"res: {res}")    

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

from pathlib import Path
results = Path("results")
results.mkdir(exist_ok=True, parents=True)

benchmark.run(print_data=True, show_plots=True, save_path=results/ "add_kernel")
