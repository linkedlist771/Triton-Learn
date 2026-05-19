"""TileLang GEMM 示例, 与现有 Triton GEMM 对照。

依赖:
    pip install tilelang
"""
import torch

try:
    import tilelang
    import tilelang.language as T
except ImportError:
    raise SystemExit("tilelang not installed. run: pip install tilelang")


def make_gemm(M, N, K, block_M=128, block_N=128, block_K=32,
              dtype="float16", accum_dtype="float"):
    @tilelang.jit(out_idx=[2])
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_l = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_l)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_s)
                T.copy(B[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_l)
            T.copy(C_l, C[by * block_M, bx * block_N])

    return gemm


def main():
    M, N, K = 1024, 1024, 1024
    gemm = make_gemm(M, N, K)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = gemm(a, b)
    ref = a @ b
    err = (c.float() - ref.float()).abs().max().item()
    print(f"tilelang gemm vs torch matmul max abs err: {err:.3e}")


if __name__ == "__main__":
    main()
