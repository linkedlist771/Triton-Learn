"""AMP 训练一小步: fp16 (带 GradScaler) 与 bf16 (无 scaler) 对比。"""
import torch
import torch.nn as nn


def train_step(model, x, y, optim, scaler=None, dtype=torch.float16):
    optim.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=dtype):
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        optim.step()
    return loss.item()


def run(dtype, use_scaler):
    torch.manual_seed(0)
    device = "cuda"
    model = nn.Sequential(
        nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 512)
    ).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    x = torch.randn(64, 512, device=device)
    y = torch.randn(64, 512, device=device)
    losses = [train_step(model, x, y, optim, scaler, dtype) for _ in range(5)]
    return losses


def main():
    print("fp16 + GradScaler:", [f"{l:.4f}" for l in run(torch.float16, True)])
    print("bf16 (no scaler) :", [f"{l:.4f}" for l in run(torch.bfloat16, False)])


if __name__ == "__main__":
    main()
