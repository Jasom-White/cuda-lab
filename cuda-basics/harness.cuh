// harness.cuh - LeetGPU-style local harness: you write only the kernel.
// Handles: host/device alloc, init, H2D, launch, sync, D2H, verify, timing, free.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include "common.cuh"

// 2-input / 1-output elementwise harness (a + b -> c reference)
inline void check_bench_add(
    const char* name, int n,
    const std::function<void(const float*, const float*, float*, int)>& kernel_launch,
    int iters = 100) {
    const size_t bytes = (size_t)n * sizeof(float);
    float *h_a = (float*)malloc(bytes), *h_b = (float*)malloc(bytes), *h_c = (float*)malloc(bytes);
    srand(0);
    for (int i = 0; i < n; ++i) { h_a[i] = (float)rand() / RAND_MAX; h_b[i] = (float)rand() / RAND_MAX; }

    float *d_a = dalloc(n), *d_b = dalloc(n), *d_c = dalloc(n);
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    kernel_launch(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double ref = (double)h_a[i] + (double)h_b[i];
        double err = fabs((double)h_c[i] - ref);
        if (err > max_err) max_err = err;
    }
    float ms = bench_ms([&] { kernel_launch(d_a, d_b, d_c, n); }, iters);
    printf("%-12s n=%-9d max_err=%.3e %s  %.4f ms  %.1f GB/s\n",
           name, n, max_err, max_err < 1e-5 ? "PASS" : "FAIL", ms, 3.0 * bytes / (ms * 1e-3) / 1e9);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
}
