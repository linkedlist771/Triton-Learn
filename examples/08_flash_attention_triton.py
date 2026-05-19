"""最小 Triton flash-attention forward (non-causal): tile 在 N 上, online softmax。

与 torch.nn.functional.scaled_dot_product_attention 对齐验证数值。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_fwd(
    Q, K, V, O,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B, H, N,
    D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_off = b * stride_qb + h * stride_qh
    k_off = b * stride_kb + h * stride_kh
    v_off = b * stride_vb + h * stride_vh
    o_off = b * stride_ob + h * stride_oh

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(
        Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        mask=offs_m[:, None] < N, other=0.0,
    )

    m_i = tl.full((BM,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BM,), dtype=tl.float32)
    acc = tl.zeros((BM, D), dtype=tl.float32)

    for n0 in range(0, N, BN):
        offs_n = n0 + tl.arange(0, BN)
        k = tl.load(
            K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
            mask=offs_n[:, None] < N, other=0.0,
        )
        v = tl.load(
            V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
            mask=offs_n[:, None] < N, other=0.0,
        )
        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(offs_n[None, :] < N, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        O + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
        acc.to(O.dtype.element_ty),
        mask=offs_m[:, None] < N,
    )


def flash_attention(q, k, v):
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    BM, BN = 64, 64
    grid = (triton.cdiv(N, BM), B * H)
    flash_attn_fwd[grid](
        q, k, v, o, 1.0 / (D ** 0.5),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, N, D=D, BM=BM, BN=BN,
    )
    return o


def main():
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 512, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    out = flash_attention(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"flash_attn vs sdpa max abs err: {err:.3e}")


if __name__ == "__main__":
    main()
