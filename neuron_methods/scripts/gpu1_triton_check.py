"""GPU job 1, part 1 (serving env): PyTorch sees the GPU, and Triton compiles and runs kernels on it,
including a tensor-core matmul. On a B200 it must be using the CUDA 12.9 ptxas that env.sh sets."""
import torch
import triton
import triton.language as tl
from triton import knobs

cap = torch.cuda.get_device_capability(0)
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| device", torch.cuda.get_device_name(0),
      "| capability", cap)
tool = knobs.nvidia.ptxas_blackwell if cap[0] >= 10 else knobs.nvidia.ptxas
print("ptxas Triton uses on this GPU:", tool.path, "| version", tool.version)
if cap[0] >= 10:
    assert tool.version == "12.9", "on a B200, Triton must use the CUDA 12.9 ptxas"


@triton.jit
def add_kernel(x, y, o, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o + offs, tl.load(x + offs, mask=m) + tl.load(y + offs, mask=m), mask=m)


@triton.jit
def matmul_kernel(a, b, c, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        acc += tl.dot(tl.load(a + rm[:, None] * K + (k0 + rk)[None, :]), tl.load(b + (k0 + rk)[:, None] * N + rn[None, :]))
    tl.store(c + rm[:, None] * N + rn[None, :], acc)


x = torch.randn(100_000, device="cuda")
y = torch.randn_like(x)
o = torch.empty_like(x)
add_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, o, x.numel(), BLOCK=1024)
torch.cuda.synchronize()
assert torch.equal(o, x + y), "Triton elementwise kernel gave wrong results"
print("Triton elementwise kernel: OK")

M = N = K = 512
a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
c = torch.empty(M, N, device="cuda", dtype=torch.float32)
matmul_kernel[(M // 128, N // 128)](a, b, c, N, K, BM=128, BN=128, BK=64)
torch.cuda.synchronize()
ref = a.float() @ b.float()
err = ((c - ref).norm() / ref.norm()).item()
print(f"Triton tensor-core matmul: relative error {err:.1e}")
assert err < 1e-2, "Triton matmul gave wrong results"
print("TRITON CHECK PASS")
