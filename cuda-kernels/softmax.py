# -*- coding: utf-8 -*-
"""row-wise softmax：减 max（数值稳定）+ 两次 block 归约（max / sum）。
覆盖 cols=1/7/4096/8192、非 2 的幂；与 torch.softmax 对齐；测有效带宽。
运行：.\dev.bat softmax.py
"""
import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")
os.environ.setdefault("VSLANG", "1033")
import torch
from torch.utils.cpp_extension import load_inline

cpp = r"""
#include <torch/extension.h>
torch::Tensor softmax_rows(torch::Tensor x);
"""
cuda = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__inline__ __device__ float warpMax(float v) {
    for (int o = warpSize / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}
__inline__ __device__ float warpSum(float v) {
    for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y,
                               int rows, int cols) {
    int row = blockIdx.x;
    const float* xr = x + (long long)row * cols;
    float* yr = y + (long long)row * cols;
    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    int nwarp = blockDim.x / warpSize;

    __shared__ float sm[32];

    float m = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) m = fmaxf(m, xr[i]);
    m = warpMax(m);
    if (lane == 0) sm[wid] = m;
    __syncthreads();
    m = (threadIdx.x < nwarp) ? sm[threadIdx.x] : -INFINITY;
    if (wid == 0) m = warpMax(m);
    __shared__ float row_max;
    if (threadIdx.x == 0) row_max = m;
    __syncthreads();
    m = row_max;

    float s = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) s += __expf(xr[i] - m);
    s = warpSum(s);
    if (lane == 0) sm[wid] = s;
    __syncthreads();
    s = (threadIdx.x < nwarp) ? sm[threadIdx.x] : 0.f;
    if (wid == 0) s = warpSum(s);
    __shared__ float row_sum;
    if (threadIdx.x == 0) row_sum = s;
    __syncthreads();
    s = row_sum;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) yr[i] = __expf(xr[i] - m) / s;
}

torch::Tensor softmax_rows(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 2);
    auto y = torch::empty_like(x);
    int rows = (int)x.size(0), cols = (int)x.size(1);
    softmax_kernel<<<rows, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
    return y;
}
"""
mod = load_inline(name="dsh_softmax", cpp_sources=cpp, cuda_sources=cuda,
                  functions=["softmax_rows"], verbose=False, extra_cuda_cflags=["-O3"])

def bench(fn, iters=30):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

if __name__ == "__main__":
    for rows, cols in [(64, 1), (64, 7), (256, 4096), (256, 8192), (1024, 1024)]:
        x = torch.randn(rows, cols, device="cuda") * 5.0
        ref = torch.softmax(x, dim=1)
        got = mod.softmax_rows(x)
        maxdiff = (got - ref).abs().max().item()
        denom = ref.abs().max().item()
        print(f"rows={rows:5d} cols={cols:5d}  max|d|={maxdiff:.3e}  rel={maxdiff/max(denom,1e-12):.3e}  "
              f"rowsum_err={ (got.sum(1)-1).abs().max().item():.2e}")

    print("\n-- performance (apparent bandwidth = 8 bytes/elem) --")
    for rows, cols in [(1024, 1024), (1024, 4096), (4096, 4096)]:
        x = torch.randn(rows, cols, device="cuda")
        n = rows * cols
        ms = bench(lambda: mod.softmax_rows(x))
        ms_t = bench(lambda: torch.softmax(x, dim=1))
        print(f"rows={rows:5d} cols={cols:5d}  custom {ms:7.3f} ms ({n*8/ms/1e6:6.1f} GB/s)   "
              f"torch {ms_t:7.3f} ms ({n*8/ms_t/1e6:6.1f} GB/s)   speedup {ms_t/ms:.2f}x")
