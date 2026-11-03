// vector_add.cu - second step: 1D indexing, grid-stride loop, cudaMalloc, events.
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void vec_add_gridstride(const float* a, const float* b, float* c, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 20;              // 1M floats = 4MB per array
    size_t bytes = n * sizeof(float);
    float *h_a = (float*)malloc(bytes), *h_b = (float*)malloc(bytes), *h_c = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    double err = 0.0;
    for (int i = 0; i < n; ++i) err += fabs(h_c[i] - 3.0f);
    printf("vec_add: n=%d  mean|err|=%.3e  %s\n", n, err / n, (err / n < 1e-6 ? "PASS" : "FAIL"));

    // timing with cuda events (100 iters)
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    const int iters = 100;
    vec_add_gridstride<<<blocks, threads>>>(d_a, d_b, d_c, n);   // warmup
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) vec_add_gridstride<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.f; cudaEventElapsedTime(&ms, s, e);
    printf("vec_add_gridstride: %.3f ms/iter, %.1f GB/s\n", ms / iters, 3.0 * bytes * iters / ms / 1e6);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
