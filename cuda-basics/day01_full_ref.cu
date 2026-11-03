// day01_full_ref.cu - the "complete main" as given, verified end-to-end.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

#define CUDA_CHECK(x) do {                                                   \
    cudaError_t e = (x);                                                     \
    if (e != cudaSuccess) {                                                  \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1);                                                             \
    }                                                                        \
} while (0)

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 20);
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    srand(0);
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double ref = (double)h_a[i] + (double)h_b[i];
        double err = fabs((double)h_c[i] - ref);
        if (err > max_err) max_err = err;
    }
    printf("day01 vector_add: n=%d  max_err=%.3e  %s\n",
           n, max_err, (max_err < 1e-5 ? "PASS" : "FAIL"));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it)
        vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_total = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
    float ms = ms_total / iters;
    double gbs = 3.0 * n * sizeof(float) / (ms * 1e-3) / 1e9;
    printf("day01 vector_add: %.4f ms/iter  %.1f GB/s  (blocks=%d, threads=%d)\n",
           ms, gbs, blocks, threads);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
