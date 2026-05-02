import torch.nn as nn
from typing import Callable
import torch 
import triton.language as tl
import triton
import torch.nn.functional as F

def naive_silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1+torch.exp(-x))

def torch_silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)

@triton.jit
def triton_silu_kernel(x_ptr, y_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    
    # res = x / (1 + tl.exp(-x))
    res = x* tl.sigmoid(x)
    tl.store(y_ptr + offsets, res, mask=mask)



def triton_silu(x: torch.Tensor) -> torch.Tensor:
    block_size = 1024
    y = torch.empty_like(x)
    # y = torch.zeros_like(x)
    n_elements = x.numel()

    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    triton_silu_kernel[grid](x, y, n_elements, BLOCK_SIZE=block_size)
    return y


def silu_interface(method: str) -> Callable:
    match method:
        case "torch":
            return torch_silu
        
        case "triton":
            return triton_silu
        
        case "naive":
            return naive_silu
        case _:
            raise NameError("Not supported")
        
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 28)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch", "naive"],
        line_names=["Triton", "Torch", "Naive Torch"],
        styles=[
            ("blue", "-"),
            ("green", "-"),
            ("red", "-"),
        ],
        ylabel="GB/s",
        plot_name="silu-performance",
        args={},
    )
)

def benchmark_silu(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)

    if provider == "torch":
        fn = lambda: torch_silu(x)
    elif provider == "triton":
        fn = lambda: triton_silu(x)
    elif provider == "naive":
        fn = lambda: naive_silu(x)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms = triton.testing.do_bench(fn)

    # SiLU: read x once + write y once
    # float32: 4 bytes read + 4 bytes write = 8 bytes per element
    gbps = size * 8 / ms / 1e6

    return gbps


def correctness_check():
    seed = 42
    torch.manual_seed(seed)

    num = 10000
    input_tensor = torch.rand(num, device="cuda")

    silu_output_naive = naive_silu(input_tensor)
    silu_output_torch = torch_silu(input_tensor)
    silu_output_triton = triton_silu(input_tensor)

    assert torch.allclose(silu_output_naive, silu_output_torch)
    assert torch.allclose(silu_output_triton, silu_output_torch, rtol=1e-5, atol=1e-5)

    print("Correctness check passed!")


def main():
    correctness_check()
    output_dir = "results/silu_kernel_v2"
    import os
    os.makedirs(output_dir, exist_ok=True)
    benchmark_silu.run(
        print_data=True,
        show_plots=True,
        save_path=output_dir
    )


if __name__ == "__main__":
    main()
    
# def main():
#     seed = 42
#     torch.manual_seed(seed)
#     num = 1024
#     input_tensor = torch.rand(num, device=torch.device("cuda"))
#     silu_output_naive = naive_silu(input_tensor)
#     silu_output_torch = torch_silu(input_tensor)
#     silu_output_triton = triton_silu(input_tensor)
#     assert torch.allclose(silu_output_naive, silu_output_torch)
#     assert torch.allclose(silu_output_triton, silu_output_torch)

# if __name__ == "__main__":
#     main()