# -*- coding: utf-8 -*-
"""reduction sum：warp shuffle + 共享内存 + 两趟归约（block 内）。
对比 float64 参考；测有效带宽（只读 n*4 字节）。
运行：.\dev.bat reduction.py
"""
import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")
os.environ.setdefault("VSLANG", "1033")
import torch
from torch.utils.cpp_extension import load_inline

cpp = r"""
#include <torch/extension.h>
torch::Tensor sum_reduce(torch::Tensor x, int64_t blocks, int64_t threads);
"""
cuda = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;
}

__global__ void reduce_kernel(const float* __restrict__ x, float* __restrict__ out,
                              long long n) {
    float sum = 0.f;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        sum += x[i];

    __shared__ float smem[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    sum = warpReduceSum(sum);
    if (lane == 0) smem[wid] = sum;
    __syncthreads();
    sum = (threadIdx.x < (int)(blockDim.x / warpSize)) ? smem[threadIdx.x] : 0.f;
    if (wid == 0) sum = warpReduceSum(sum);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

torch::Tensor sum_reduce(torch::Tensor x, int64_t blocks, int64_t threads) {
    auto out = torch::zeros({}, x.options());
    reduce_kernel<<<(int)blocks, (int)threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), (long long)x.numel());
    return out;
}
"""
mod = load_inline(name="dsh_reduction", cpp_sources=cpp, cuda_sources=cuda,
                  functions=["sum_reduce"], verbose=False, extra_cuda_cflags=["-O3"])

def bench(fn, iters=30):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

if __name__ == "__main__":
    # 正确性
    for n in [0, 1, 1000, 1 << 20, (1 << 22) + 12345]:
        x = torch.randn(n, device="cuda")
        ref = x.to(torch.float64).sum().item()
        got = mod.sum_reduce(x, 512, 256).item()
        rel = abs(got - ref) / max(abs(ref), 1e-12)
        print(f"n={n:>10}  got={got:.6f}  ref={ref:.6f}  rel={rel:.2e}")

    # 性能：不同 grid/block
    print("\n-- bandwidth (read-only) --")
    for n in [1 << 24, 1 << 26, 1 << 27]:
        x = torch.randn(n, device="cuda")
        ref = x.to(torch.float64).sum().item()
        best = None
        for blocks in [128, 256, 512, 1024]:
            for threads in [128, 256, 512]:
                ms = bench(lambda: mod.sum_reduce(x, blocks, threads))
                gb = n * 4 / ms / 1e6
                if best is None or gb > best[0]:
                    best = (gb, blocks, threads, ms)
        got = mod.sum_reduce(x, best[1], best[2]).item()
        print(f"n={n:>10}  best={best[0]:7.1f} GB/s  blocks={best[1]:4d} threads={best[2]:4d} "
              f"ms={best[3]:.3f}  rel={abs(got-ref)/max(abs(ref),1e-12):.2e}")

    # torch 参考
    x = torch.randn(1 << 27, device="cuda")
    ms = bench(lambda: x.sum())
    print(f"torch.sum   n={1<<27}  {(1<<27)*4/ms/1e6:7.1f} GB/s  (ms={ms:.3f})")
