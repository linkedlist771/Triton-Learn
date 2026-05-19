"""torch.utils.cpp_extension 内联编译 C++/CUDA 扩展, 验证工具链可用性。

依赖: ninja, 以及目标后端编译器 (沐曦上通常是 mxcc / 对应 nvcc 兼容工具链)。
"""
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "tensors must be on cuda");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "only float32 supported");
    auto c = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}
"""

cpp_src = "torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);"

ext = load_inline(
    name="my_add_ext",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["add_cuda"],
    verbose=True,
)


def main():
    a = torch.randn(1 << 20, device="cuda")
    b = torch.randn(1 << 20, device="cuda")
    c = ext.add_cuda(a, b)
    err = (c - (a + b)).abs().max().item()
    print(f"cuda extension add err: {err:.3e}")


if __name__ == "__main__":
    main()
