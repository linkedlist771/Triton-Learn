"""torch.compile 多 mode 对比: eager / default / reduce-overhead / max-autotune。"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d=4096):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 4)
        self.fc2 = nn.Linear(d * 4, d)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


def bench(fn, x, iters=50, warmup=10):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(x)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    torch.set_float32_matmul_precision("high")
    device = "cuda"
    model = MLP().to(device).eval()
    x = torch.randn(32, 4096, device=device)

    variants = {
        "eager": model,
        "compile_default": torch.compile(model),
        "compile_reduce_overhead": torch.compile(model, mode="reduce-overhead"),
        "compile_max_autotune": torch.compile(model, mode="max-autotune"),
    }

    with torch.inference_mode():
        for name, m in variants.items():
            try:
                ms = bench(m, x)
                print(f"{name:>26s}: {ms:.3f} ms")
            except Exception as ex:
                print(f"{name:>26s}: FAILED {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
