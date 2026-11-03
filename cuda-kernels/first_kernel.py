# -*- coding: utf-8 -*-
"""第一个 CUDA kernel：v1 = 朴素加法；v2 = float4 向量化 + grid-stride。
运行：.\dev.bat first_kernel.py
"""
import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
import torch
from torch.utils.cpp_extension import load_inline

cpp = r"""
#include <torch/extension.h>
torch::Tensor add_naive(torch::Tensor a, torch::Tensor b);
torch::Tensor add_vec4(torch::Tensor a, torch::Tensor b);
"""
cuda = r"""
#include <torch/extension.h>

__global__ void add_naive_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void add_vec4_kernel(const float4* __restrict__ a, const float4* __restrict__ b,
                                float4* __restrict__ c, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n4; i += stride) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}

torch::Tensor add_naive(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel();
    add_naive_kernel<<<(n + 255) / 256, 256>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}

torch::Tensor add_vec4(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel();
    int n4 = n / 4;
    int blocks = (n4 + 255) / 256;
    blocks = blocks > 4096 ? 4096 : blocks;
    add_vec4_kernel<<<blocks, 256>>>((const float4*)a.data_ptr<float>(), (const float4*)b.data_ptr<float>(),
                                     (float4*)c.data_ptr<float>(), n4);
    return c;
}
"""
mod = load_inline(name="dsh_first_kernel", cpp_sources=cpp, cuda_sources=cuda,
                  functions=["add_naive", "add_vec4"], verbose=True, extra_cuda_cflags=["-O3"])

def bench(fn, a, b, iters=20):
    for _ in range(3): fn(a, b)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn(a, b)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

if __name__ == "__main__":
    n = 1 << 26  # 64M floats = 256MB
    a = torch.randn(n, device="cuda"); b = torch.randn_like(a)
    ref = a + b
    print("naive allclose:", torch.allclose(mod.add_naive(a, b), ref, atol=1e-5))
    print("vec4  allclose:", torch.allclose(mod.add_vec4(a, b), ref, atol=1e-5))
    bytes_moved = 3 * n * 4
    for name, fn in [("torch  a+b", lambda x, y: x + y), ("naive  add", mod.add_naive), ("vec4   add", mod.add_vec4)]:
        ms = bench(fn, a, b)
        print(f"{name:12s} {ms:8.3f} ms   {bytes_moved / ms / 1e6:8.1f} GB/s")
