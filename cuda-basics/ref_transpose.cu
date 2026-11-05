// day04_transpose.cu - naive vs shared-memory tiled transpose (coalescing + bank conflict).
#include "common.cuh"

#define TILE 32

__global__ void transpose_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < n) out[x * n + y] = in[y * n + x];   // strided write
}

__global__ void transpose_tiled(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + 1];               // +1 padding: avoid bank conflict
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];   // coalesced read
    __syncthreads();
    int x2 = blockIdx.y * TILE + threadIdx.x;
    int y2 = blockIdx.x * TILE + threadIdx.y;
    if (x2 < n && y2 < n) out[y2 * n + x2] = tile[threadIdx.x][threadIdx.y];  // coalesced write
}

static void transpose_cpu(const float* in, float* out, int n) {
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) out[x * n + y] = in[y * n + x];
}

int main() {
    const int n = 4096;
    const size_t bytes = (size_t)n * n * sizeof(float);
    const int T = 32;
    dim3 block(T, T);
    dim3 grid((n + T - 1) / T, (n + T - 1) / T);

    float* h_in = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    for (int i = 0; i < n * n; ++i) h_in[i] = (float)(i % 1000) * 0.001f;
    transpose_cpu(h_in, h_ref, n);

    float* d_in = dalloc(n * n), * d_out = dalloc(n * n);
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    transpose_naive<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n * n; ++i) bad += (h_out[i] != h_ref[i]);
    printf("naive  bad=%d %s\n", bad, bad == 0 ? "PASS" : "FAIL");

    transpose_tiled<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    bad = 0;
    for (int i = 0; i < n * n; ++i) bad += (h_out[i] != h_ref[i]);
    printf("tiled  bad=%d %s\n", bad, bad == 0 ? "PASS" : "FAIL");

    float ms_naive = bench_ms([&] { transpose_naive<<<grid, block>>>(d_in, d_out, n); }, 50);
    float ms_tiled = bench_ms([&] { transpose_tiled<<<grid, block>>>(d_in, d_out, n); }, 50);
    double gb = 2.0 * n * n * sizeof(float) / 1e9;
    printf("n=%d  naive %.4f ms (%.1f GB/s)  tiled %.4f ms (%.1f GB/s)  speedup %.2fx\n",
           n, ms_naive, gb / (ms_naive * 1e-3), ms_tiled, gb / (ms_tiled * 1e-3), ms_naive / ms_tiled);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_ref); free(h_out);
    return 0;
}
