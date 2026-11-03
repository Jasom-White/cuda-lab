// common.cuh - shared boilerplate for cuda-basics. ASCII only.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <functional>

inline void cuda_check(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), file, line);
        exit(1);
    }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

// time a kernel launch: warmup once, then average over iters, returns ms/iter
inline float bench_ms(const std::function<void()>& launch, int iters = 100) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return ms / iters;
}

// helper: allocate device array of n floats
inline float* dalloc(int n) {
    float* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, (size_t)n * sizeof(float)));
    return p;
}
