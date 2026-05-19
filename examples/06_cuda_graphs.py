"""CUDA Graph capture & replay: 固定 shape 下消除 kernel launch 开销。"""
import torch


def step(x, w1, w2):
    return torch.relu(x @ w1) @ w2


def bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    device = "cuda"
    x = torch.randn(64, 1024, device=device)
    w1 = torch.randn(1024, 1024, device=device)
    w2 = torch.randn(1024, 1024, device=device)

    # 必须先在 side stream 上预热, 再 capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step(x, w1, w2)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_static = step(x, w1, w2)

    base_ms = bench(lambda: step(x, w1, w2))
    graph_ms = bench(lambda: g.replay())
    print(f"eager:      {base_ms:.4f} ms / iter")
    print(f"cuda graph: {graph_ms:.4f} ms / iter   speedup={base_ms/graph_ms:.2f}x")
    print("output shape:", tuple(y_static.shape))


if __name__ == "__main__":
    main()
