// day03_transpose.cu - YOUR TASK: fill the two kernels. main() is done.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "common.cuh"

// TODO 1: naive transpose: 1 thread per element, read coalesced / write strided
__global__ void transpose_naive(const float* in, float* out, int n) {
    // int x = blockIdx.x * blockDim.x + threadIdx.x;   // col
    // int y = blockIdx.y * blockDim.y + threadIdx.y;   // row
    // if (x < n && y < n) out[x * n + y] = in[y * n + x];
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // col
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // row
    if (x < n && y < n) {out[x*n+y]=in[y*n+x];}
}

// TODO 2: tiled transpose: shared memory + __syncthreads(), coalesced read AND write
__global__ void transpose_tiled(const float* in, float* out, int n) {
    // __shared__ float tile[32][33];
    // load: tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    // __syncthreads();
    // store: out[y2 * n + x2] = tile[threadIdx.x][threadIdx.y];  (x2/y2 swapped)
    __shared__ float tile[32][33];   // +1 col padding: avoid bank conflict

    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x*32 + tx;//input col
    int y = blockIdx.y*32 + ty;//input row
    if (x < n && y < n) {
        tile[ty][tx] = in[y*n+x];
    }
    __syncthreads();
    int x2 = blockIdx.y*32 + tx;//output col
    int y2 = blockIdx.x*32 + ty;//output row
    if (x < n && y < n) {
        out[y2*n+x2] = tile[tx][ty];
    }
}

// ---------------- main is provided, do NOT change it ----------------
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
    const double gb = 2.0 * n * n * sizeof(float) / 1e9;   // read + write

    float* h_in = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    for (int i = 0; i < n * n; ++i) h_in[i] = (float)(i % 1000) * 1e-3f;
    transpose_cpu(h_in, h_ref, n);

    float* d_in = dalloc(n * n), * d_out = dalloc(n * n);
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // --- verify naive ---
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    transpose_naive<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n * n; ++i) bad += (h_out[i] != h_ref[i]);
    printf("naive: bad=%d %s\n", bad, bad == 0 ? "PASS" : "FAIL");

    // --- verify tiled ---
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    transpose_tiled<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    bad = 0;
    for (int i = 0; i < n * n; ++i) bad += (h_out[i] != h_ref[i]);
    printf("tiled: bad=%d %s\n", bad, bad == 0 ? "PASS" : "FAIL");

    // --- timing ---
    float ms_naive = bench_ms([&] { transpose_naive<<<grid, block>>>(d_in, d_out, n); }, 50);
    float ms_tiled = bench_ms([&] { transpose_tiled<<<grid, block>>>(d_in, d_out, n); }, 50);
    printf("naive: %8.4f ms  %7.1f GB/s\n", ms_naive, gb / (ms_naive * 1e-3));
    printf("tiled: %8.4f ms  %7.1f GB/s  speedup %.2fx\n", ms_tiled, gb / (ms_tiled * 1e-3), ms_naive / ms_tiled);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_ref); free(h_out);
    return 0;
}
