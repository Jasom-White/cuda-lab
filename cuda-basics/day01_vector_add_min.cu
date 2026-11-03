// day01_vector_add_min.cu - same thing as day01, but short.
// Build & run in VS Code: Ctrl+Shift+B -> "CUDA: build & run current .cu"
#include "common.cuh"

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = (size_t)n * sizeof(float);
    const int T = 256, B = (n + T - 1) / T;

    // host data: a=1, b=2 -> expected 3
    float *h_a = (float*)malloc(bytes), *h_b = (float*)malloc(bytes), *h_c = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    // device + H2D
    float *d_a = dalloc(n), *d_b = dalloc(n), *d_c = dalloc(n);
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // run + D2H
    add<<<B, T>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // verify + time
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += (h_c[i] != 3.0f);
    float ms = bench_ms([&] { add<<<B, T>>>(d_a, d_b, d_c, n); });
    printf("day01-min: n=%d  bad=%d  %s  %.4f ms  %.0f GB/s\n",
           n, bad, bad == 0 ? "PASS" : "FAIL", ms, 3.0 * bytes / (ms * 1e-3) / 1e9);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
