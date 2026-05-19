"""torch.profiler 最小示例: profile 一个小 attention block, 导出 chrome trace。

运行:
    python 01_torch_profiler.py
然后用 chrome://tracing 或 perfetto.dev 打开 trace.json。
"""
import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity, schedule


def attention_block(q, k, v):
    with record_function("qk_matmul"):
        s = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)
    with record_function("softmax"):
        p = F.softmax(s, dim=-1)
    with record_function("pv_matmul"):
        return p @ v


def main():
    device = "cuda"
    B, H, N, D = 4, 8, 1024, 64
    q = torch.randn(B, H, N, D, device=device)
    k = torch.randn(B, H, N, D, device=device)
    v = torch.randn(B, H, N, D, device=device)

    for _ in range(5):
        attention_block(q, k, v)
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(6):
            attention_block(q, k, v)
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    prof.export_chrome_trace("trace.json")
    print("trace saved to trace.json")


if __name__ == "__main__":
    main()
