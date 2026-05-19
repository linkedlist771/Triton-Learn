"""INT8 dynamic quantization 推理: 对 nn.Linear 做动态量化, 对比 fp32 / int8 时延与误差。

注: torch.ao.quantization.quantize_dynamic 主要走 CPU 路径。
GPU 上做 INT8 通常用 bitsandbytes / TensorRT / 自定义 kernel, 沐曦上请查对应支持。
"""
import time
import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic


class MLP(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d * 2, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def bench_cpu(fn, x, iters=20, warmup=3):
    for _ in range(warmup):
        fn(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    return (time.perf_counter() - t0) / iters * 1000


def main():
    torch.manual_seed(0)
    model = MLP().eval()
    x = torch.randn(64, 1024)

    qmodel = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    with torch.inference_mode():
        fp_ms = bench_cpu(model, x)
        q_ms = bench_cpu(qmodel, x)
        y_fp = model(x)
        y_q = qmodel(x)
        err = (y_fp - y_q).abs().mean().item()

    print(f"fp32 cpu: {fp_ms:.2f} ms/iter")
    print(f"int8 cpu: {q_ms:.2f} ms/iter   speedup={fp_ms/q_ms:.2f}x")
    print(f"mean abs diff fp32 vs int8: {err:.4e}")


if __name__ == "__main__":
    main()
