// day05_stencil.cu - YOUR TASK: fill 2 kernels. main() is provided (do not change).
// 2D 5-point Laplacian: out[y][x] = up + down + left + right - 4*center, zero outside boundary.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "common.cuh"

// ---------------- TODO 1: naive, one thread per element, 5 global-memory reads ----------------
__global__ void stencil_naive(const float* in, float* out, int n) {
    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;
    // if (x >= n || y >= n) return;
    // float c  = in[y*n + x];
    // float up = (y > 0)     ? in[(y-1)*n + x] : 0.f;
    // float dn = (y < n - 1) ? in[(y+1)*n + x] : 0.f;
    // float lf = (x > 0)     ? in[y*n + (x-1)] : 0.f;
    // float rt = (x < n - 1) ? in[y*n + (x+1)] : 0.f;
    // out[y*n + x] = up + dn + lf + rt - 4.f * c;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= n || y >= n)
        return;
    float c = in[y*n + x];
    float up = (y > 0) ? in[(y-1)*n + x] : 0.f;
    float dn = (y < n - 1) ? in[(y+1)*n + x] : 0.f;
    float lf = (x > 0) ? in[y*n + (x-1)] : 0.f;
    float rt = (x < n - 1) ? in[y*n + (x+1)] : 0.f;
    out[y*n + x] = up + dn + lf + rt - 4.f * c;
}

// ---------------- TODO 2: shared-memory tiled, load 34x34 halo once ----------------
// block = 32x32 threads. tile[34][34] holds the 32x32 center + 1-cell halo.
__global__ void stencil_tiled(const float* in, float* out, int n) {
    // __shared__ float tile[34][34];
    // int tx = threadIdx.x, ty = threadIdx.y;
    // int bx = blockIdx.x * 32, by = blockIdx.y * 32;
    //
    // // load halo: k loops over 34*34 = 1156 entries with 1024 threads
    // for (int k = ty * 34 + tx; k < 34 * 34; k += 32 * 32) {
    //     int sy = k / 34, sx = k % 34;
    //     int gx = bx + sx - 1, gy = by + sy - 1;
    //     tile[sy][sx] = (gx >= 0 && gx < n && gy >= 0 && gy < n) ? in[gy*n + gx] : 0.f;
    // }
    // __syncthreads();
    //
    // int gx = bx + tx, gy = by + ty;
    // if (gx < n && gy < n) {
    //     out[gy*n + gx] = tile[ty][tx]       // up
    //                    + tile[ty+2][tx+1]   // down
    //                    + tile[ty+1][tx]     // left
    //                    + tile[ty+1][tx+2]   // right
    //                    - 4.f * tile[ty+1][tx+1];
    // }
    __shared__ float tile[34][34];   // 32x32 center + 1-cell halo
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * 32, by = blockIdx.y * 32;

    for (int k = ty * 32 + tx; k < 34 * 34; k += 32 * 32) {   // tid = ty*32+tx, stride 1024
        int sy = k / 34, sx = k % 34;
        int gx = bx + sx - 1, gy = by + sy - 1;
        tile[sy][sx] = (gx >= 0 && gx < n && gy >= 0 && gy < n) ? in[gy*n + gx] : 0.f;
    }
    __syncthreads();

    int gx = bx + tx, gy = by + ty;
    if (gx < n && gy < n) {
        out[gy*n + gx] = tile[ty][tx+1]
                       + tile[ty+2][tx+1]
                       + tile[ty+1][tx]
                       + tile[ty+1][tx+2]
                       - 4.f * tile[ty+1][tx+1];
    }
}

// ---------------- main (provided) ----------------
static void laplacian_cpu(const float* in, float* out, int n) {
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            float c  = in[y*n + x];
            float up = (y > 0)     ? in[(y-1)*n + x] : 0.f;
            float dn = (y < n - 1) ? in[(y+1)*n + x] : 0.f;
            float lf = (x > 0)     ? in[y*n + (x-1)] : 0.f;
            float rt = (x < n - 1) ? in[y*n + (x+1)] : 0.f;
            out[y*n + x] = up + dn + lf + rt - 4.f * c;
        }
}

static double max_abs_diff(const float* a, const float* b, int n) {
    double m = 0.0;
    for (int i = 0; i < n * n; ++i) m = fmax(m, fabs((double)a[i] - (double)b[i]));
    return m;
}

int main(int argc, char** argv) {
    const int n = (argc > 1) ? atoi(argv[1]) : 4096;
    const size_t bytes = (size_t)n * n * sizeof(float);
    const int T = 32;
    dim3 block(T, T);
    dim3 grid((n + T - 1) / T, (n + T - 1) / T);
    const double ideal_gb = 2.0 * n * n * sizeof(float) / 1e9;   // 1 read + 1 write of the grid

    float* h_in = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    srand(0);
    for (int i = 0; i < n * n; ++i) h_in[i] = (float)rand() / RAND_MAX - 0.5f;
    laplacian_cpu(h_in, h_ref, n);

    float* d_in = dalloc(n * n), * d_out = dalloc(n * n);
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    stencil_naive<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("naive: max_diff=%.3e %s\n", max_abs_diff(h_out, h_ref, n), max_abs_diff(h_out, h_ref, n) < 1e-4 ? "PASS" : "FAIL");

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    stencil_tiled<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("tiled: max_diff=%.3e %s\n", max_abs_diff(h_out, h_ref, n), max_abs_diff(h_out, h_ref, n) < 1e-4 ? "PASS" : "FAIL");

    float ms_naive = bench_ms([&] { stencil_naive<<<grid, block>>>(d_in, d_out, n); }, 50);
    float ms_tiled = bench_ms([&] { stencil_tiled<<<grid, block>>>(d_in, d_out, n); }, 50);
    printf("naive: %8.4f ms  %6.1f GB/s (ideal traffic 8 B/elem)\n", ms_naive, ideal_gb / (ms_naive * 1e-3));
    printf("tiled: %8.4f ms  %6.1f GB/s  speedup %.2fx\n", ms_tiled, ideal_gb / (ms_tiled * 1e-3), ms_naive / ms_tiled);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_ref); free(h_out);
    return 0;
}
