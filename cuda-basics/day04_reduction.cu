// day04_reduction.cu - reduction: atomic vs warp-shuffle vs two-pass.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "common.cuh"

// ---------------- 1) naive atomic ----------------
__global__ void reduce_atomic(const float* x, float* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(out, x[i]);
}

// ---------------- 2) warp shuffle + shared ----------------
__global__ void reduce_warp(const float* x, float* out, int n) {
    float sum = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        sum += x[i];

    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);

    __shared__ float warps[32];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    if (lane == 0) warps[wid] = sum;
    __syncthreads();

    if (wid == 0) {
        sum = (threadIdx.x < blockDim.x / 32) ? warps[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (threadIdx.x == 0) atomicAdd(out, sum);
    }
}

// ---------------- 3) two-pass stage 1: global block_sums ----------------
__global__ void reduce_twopass_stage1(const float* x, float* block_sums, int n) {
    float sum = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        sum += x[i];

    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);

    __shared__ float warps[32];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    if (lane == 0) warps[wid] = sum;
    __syncthreads();

    if (wid == 0) {
        sum = (threadIdx.x < blockDim.x / 32) ? warps[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (threadIdx.x == 0) block_sums[blockIdx.x] = sum;   // no atomics
    }
}

// ---------------- 4) two-pass stage 2: reduce block_sums ----------------
__global__ void reduce_twopass_stage2(const float* block_sums, float* out, int nblocks) {
    float sum = 0.f;
    for (int i = threadIdx.x; i < nblocks; i += blockDim.x) sum += block_sums[i];

    __shared__ float warps[32];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) warps[wid] = sum;
    __syncthreads();

    if (wid == 0) {
        sum = (threadIdx.x < blockDim.x / 32) ? warps[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (threadIdx.x == 0) *out = sum;
    }
}

// ---------------- main (provided) ----------------
static void report(const char* name, double ref, float got, float ms, size_t bytes, double tol) {
    double rel = fabs((double)got - ref) / fmax(fabs(ref), 1e-12);
    printf("%-18s got=%12.4f ref=%12.4f rel=%.2e %s  %7.4f ms  %6.1f GB/s\n",
           name, got, ref, rel, rel < tol ? "PASS" : "FAIL", ms, bytes / (ms * 1e-3) / 1e9);
}

int main(int argc, char** argv) {
    const int n = (argc > 1) ? atoi(argv[1]) : (1 << 27);
    const size_t bytes = (size_t)n * sizeof(float);
    const int T = 256, B = 4096, TP_BLOCKS = 1024;

    float* h_x = (float*)malloc(bytes);
    srand(0);
    double ref = 0.0;
    for (int i = 0; i < n; ++i) {
        h_x[i] = (float)rand() / RAND_MAX - 0.5f;
        ref += (double)h_x[i];
    }
    printf("n=%d  cpu_ref=%.6f\n", n, ref);

    float* d_x = dalloc(n);
    float* d_out = dalloc(1);
    float* d_sums = dalloc(TP_BLOCKS);
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    float got = 0.f, ms = 0.f;

    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    reduce_atomic<<<B, T>>>(d_x, d_out, n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    ms = bench_ms([&] { CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float))); reduce_atomic<<<B, T>>>(d_x, d_out, n); }, 30);
    report("reduce_atomic", ref, got, ms, bytes, 5e-2);   // float atomics lose precision

    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    reduce_warp<<<B, T>>>(d_x, d_out, n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    ms = bench_ms([&] { CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float))); reduce_warp<<<B, T>>>(d_x, d_out, n); }, 30);
    report("reduce_warp", ref, got, ms, bytes, 1e-3);

    reduce_twopass_stage1<<<TP_BLOCKS, T>>>(d_x, d_sums, n);
    reduce_twopass_stage2<<<1, T>>>(d_sums, d_out, TP_BLOCKS);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    ms = bench_ms([&] {
        reduce_twopass_stage1<<<TP_BLOCKS, T>>>(d_x, d_sums, n);
        reduce_twopass_stage2<<<1, T>>>(d_sums, d_out, TP_BLOCKS);
    }, 30);
    report("reduce_twopass", ref, got, ms, bytes, 1e-3);

    cudaFree(d_x); cudaFree(d_out); cudaFree(d_sums);
    free(h_x);
    return 0;
}
