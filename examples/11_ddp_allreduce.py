"""最小 DDP / NCCL all-reduce 示例。

单卡:
    python 11_ddp_allreduce.py            # 走单进程 fallback, 不真做 all_reduce
多卡 (单机 2 卡):
    torchrun --nproc_per_node=2 11_ddp_allreduce.py
"""
import os
import torch
import torch.distributed as dist


def main():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = "cuda"
        print("[single-process] RANK not set, skipping real all_reduce.")

    x = torch.full((4,), float(rank + 1), device=device)
    print(f"[rank {rank}] before all_reduce: {x.tolist()}")
    if dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}] after  all_reduce: {x.tolist()}  (expected sum=1+2+...+{world_size})")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
