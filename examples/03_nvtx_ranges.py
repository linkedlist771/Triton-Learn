"""torch.cuda.nvtx 标记代码段, 配合外部 profiler (nsys / 沐曦 profiler) 抓段。

运行 (NVIDIA 示例):
    nsys profile -o myrun python 03_nvtx_ranges.py
沐曦上换成对应的 profiler 命令即可。
"""
import torch


def main():
    device = "cuda"
    x = torch.randn(2048, 2048, device=device)
    w = torch.randn(2048, 2048, device=device)

    for step in range(5):
        torch.cuda.nvtx.range_push(f"step_{step}")

        torch.cuda.nvtx.range_push("matmul")
        y = x @ w
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("relu")
        y = torch.relu(y)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("reduce")
        loss = y.sum()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    print("done, loss =", loss.item())


if __name__ == "__main__":
    main()
