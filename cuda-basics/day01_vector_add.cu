// day01_vector_add.cu - grid-stride vs naive, with numbers. ASCII only.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "common.cuh"

// grid-stride: fixed block count, each thread loops with a stride
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        C[i] = A[i] + B[i];
    }
}

// naive: one thread per element, block count scales with N
__global__ void vector_add_naive(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char** argv) {
    const int n = (argc > 1) ? atoi(argv[1]) : (1 << 20);
    const size_t bytes = (size_t)n * sizeof(float);
    const int T = 256;
    const int B_naive = (n + T - 1) / T;   // grows with n
    const int B_grid  = 4096;              // fixed

    float *h_a = (float*)malloc(bytes), *h_b = (float*)malloc(bytes), *h_c = (float*)malloc(bytes);
    srand(0);
    for (int i = 0; i < n; ++i) { h_a[i] = (float)rand() / RAND_MAX; h_b[i] = (float)rand() / RAND_MAX; }

    float *d_a = dalloc(n), *d_b = dalloc(n), *d_c = dalloc(n);
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // correctness (grid-stride version)
    vector_add<<<B_grid, T>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double ref = (double)h_a[i] + (double)h_b[i];
        double err = fabs((double)h_c[i] - ref);
        if (err > max_err) max_err = err;
    }
    printf("check: n=%d max_err=%.3e %s\n", n, max_err, max_err < 1e-5 ? "PASS" : "FAIL");

    // timing: grid-stride (4096 blocks) vs naive (ceil(n/256) blocks)
    float ms_grid  = bench_ms([&] { vector_add<<<B_grid, T>>>(d_a, d_b, d_c, n); }, 100);
    float ms_naive = bench_ms([&] { vector_add_naive<<<B_naive, T>>>(d_a, d_b, d_c, n); }, 100);
    double gb = 3.0 * bytes / 1e9;
    printf("grid-stride: %8.4f ms  %7.1f GB/s  (blocks=%d)\n", ms_grid,  gb / (ms_grid  * 1e-3), B_grid);
    printf("naive      : %8.4f ms  %7.1f GB/s  (blocks=%d)\n", ms_naive, gb / (ms_naive * 1e-3), B_naive);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
