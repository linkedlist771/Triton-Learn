"""基于 torch.cuda.Event 的 microbenchmark 工具, 带 warmup / median 统计。"""
import torch


def benchmark(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median = times[len(times) // 2]
    return {"median_ms": median, "min_ms": times[0], "max_ms": times[-1]}


def main():
    device = "cuda"

    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    stats = benchmark(lambda: torch.mm(a, b))
    flops = 2 * 4096 ** 3
    tflops = flops / (stats["median_ms"] * 1e-3) / 1e12
    print(f"matmul 4096x4096: median={stats['median_ms']:.3f} ms, ~{tflops:.1f} TFLOPS")

    n = 1 << 22
    x = torch.randn(n, device=device)
    y = torch.randn(n, device=device)
    stats = benchmark(lambda: x + y)
    bw = 3 * x.element_size() * n / (stats["median_ms"] * 1e-3) / 1e9
    print(f"elementwise add {n} floats: median={stats['median_ms']:.3f} ms, ~{bw:.1f} GB/s")


if __name__ == "__main__":
    main()
