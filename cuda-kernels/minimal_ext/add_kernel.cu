// add_kernel.cu - device side (CUDA kernel + host launcher), compiled by nvcc
// NOTE: keep this file ASCII-only. nvcc on Windows can mis-parse non-ASCII
// source text (Chinese comments) and emit bogus "expected a declaration" errors.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// 1) device kernel: one thread per element
__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];          // bounds check: n may not be a multiple of blockDim
}

// 2) host launcher: checks -> allocate -> grid/block config -> launch -> return
torch::Tensor add_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "add_forward: inputs must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32,
                "add_forward: only float32 supported");
    TORCH_CHECK(a.sizes() == b.sizes(), "add_forward: shape mismatch");

    auto ac = a.contiguous();               // contiguous memory (otherwise data_ptr stride is wrong)
    auto bc = b.contiguous();
    auto c = torch::empty_like(ac);

    long long n = (long long)ac.numel();
    if (n == 0) return c;

    const int threads = 256;
    const long long blocks = (n + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();   // use PyTorch current stream

    add_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        ac.data_ptr<float>(), bc.data_ptr<float>(), c.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();          // throws if launch failed
    return c;
}
