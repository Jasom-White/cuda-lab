// day05_stencil_opt.cu - experiment: 32x8 block + 10x34 halo tile (no div/mod).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "common.cuh"

#define BX 32
#define BY 8

__global__ void stencil_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * BX + threadIdx.x;
    int y = blockIdx.y * BY + threadIdx.y;
    if (x >= n || y >= n) return;
    float c  = in[y*n + x];
    float up = (y > 0)     ? in[(y-1)*n + x] : 0.f;
    float dn = (y < n - 1) ? in[(y+1)*n + x] : 0.f;
    float lf = (x > 0)     ? in[y*n + (x-1)] : 0.f;
    float rt = (x < n - 1) ? in[y*n + (x+1)] : 0.f;
    out[y*n + x] = up + dn + lf + rt - 4.f * c;
}

__global__ void stencil_tiled(const float* in, float* out, int n) {
    __shared__ float tile[BY + 2][BX + 2];
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * BX, by = blockIdx.y * BY;

    for (int row = ty; row < BY + 2; row += BY) {
        for (int col = tx; col < BX + 2; col += BX) {
            int gy = by + row - 1, gx = bx + col - 1;
            tile[row][col] = (gx >= 0 && gx < n && gy >= 0 && gy < n) ? in[gy*n + gx] : 0.f;
        }
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
static double mad(const float* a, const float* b, int n) { double m = 0; for (int i = 0; i < n*n; ++i) m = fmax(m, fabs((double)a[i]-(double)b[i])); return m; }

int main(int argc, char** argv) {
    const int n = (argc > 1) ? atoi(argv[1]) : 4096;
    const size_t bytes = (size_t)n * n * sizeof(float);
    dim3 block(BX, BY);
    dim3 grid((n + BX - 1) / BX, (n + BY - 1) / BY);
    const double gb = 2.0 * n * n * sizeof(float) / 1e9;

    float* h_in = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    srand(0);
    for (int i = 0; i < n*n; ++i) h_in[i] = (float)rand()/RAND_MAX - 0.5f;
    laplacian_cpu(h_in, h_ref, n);

    float* d_in = dalloc(n*n), * d_out = dalloc(n*n);
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_out, 0, bytes)); stencil_naive<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("naive: max_diff=%.3e\n", mad(h_out, h_ref, n));

    CUDA_CHECK(cudaMemset(d_out, 0, bytes)); stencil_tiled<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("tiled: max_diff=%.3e\n", mad(h_out, h_ref, n));

    float a = bench_ms([&]{ stencil_naive<<<grid, block>>>(d_in, d_out, n); }, 50);
    float b = bench_ms([&]{ stencil_tiled<<<grid, block>>>(d_in, d_out, n); }, 50);
    printf("naive %8.4f ms %6.1f GB/s   tiled %8.4f ms %6.1f GB/s   speedup %.2fx\n",
           a, gb/(a*1e-3), b, gb/(b*1e-3), a/b);
    cudaFree(d_in); cudaFree(d_out); free(h_in); free(h_ref); free(h_out);
    return 0;
}
