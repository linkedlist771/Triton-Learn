"""torch.compile 与自定义 Triton kernel 互操作:
把 Triton kernel 包成 torch custom_op, inductor 编译图里能看到它, 不会 graph break。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@torch.library.custom_op("mylib::triton_add", mutates_args=())
def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out


@triton_add.register_fake
def _(x, y):
    return torch.empty_like(x)


def model(x, y):
    z = triton_add(x, y)
    return torch.relu(z) * 2.0


def main():
    device = "cuda"
    x = torch.randn(1 << 20, device=device)
    y = torch.randn(1 << 20, device=device)

    compiled = torch.compile(model, fullgraph=True)
    out = compiled(x, y)
    ref = torch.relu(x + y) * 2.0
    print("max abs err:", (out - ref).abs().max().item())


if __name__ == "__main__":
    main()
