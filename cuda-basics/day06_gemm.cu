#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "common.cuh"

__global__ void gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N){
        float sum = 0.0f;
        for (int k=0; k<K; k++){
            float a = A[row * K + k];
            float b = B[k * N + col];
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

__global__ void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    const int TILE = 16;
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[threadIdx.y][threadIdx.x] = A[row * K + k0 + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = sum;
}

int main(){
    const int M =1024;
    const int N =1024;
    const int K =1024;

    const size_t byte_A = static_cast<size_t>(M) * K * sizeof(float);
    const size_t byte_B = static_cast<size_t>(K) * N * sizeof(float);
    const size_t byte_C = static_cast<size_t>(M) * N * sizeof(float);

    float* h_A = (float*)malloc(byte_A);
    float* h_B = (float*)malloc(byte_B);
    float* h_C = (float*)malloc(byte_C);

    for (int i=0; i<M*K; i++){
        h_A[i] = 1.0f;
    }

    for (int i=0; i<K*N; i++){
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, byte_A);
    cudaMalloc(&d_B, byte_B);
    cudaMalloc(&d_C, byte_C);

    cudaMemcpy(d_A, h_A, byte_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, byte_B, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, byte_C, cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    
    for (int i=0; i<M*N; i++){
        double err = fabs((double)h_C[i] - (double)K);
        if (err > max_err){
            max_err = err;
        }
    }

    printf("matrix_mul: max|err| =%.3e %s\n",
    max_err,
    max_err < 1e-6 ? "PASS" : "FAIL"
    );

    float ms_naive = bench_ms([&] { gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K); }, 20);
    printf("naive: %.4f ms  %.1f GFLOPS\n", ms_naive, 2.0 * M * N * K / (ms_naive * 1e-3) / 1e9);

    // ---- tiled: verify ----
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, byte_C, cudaMemcpyDeviceToHost));

    double max_err_t = 0.0;
    for (int i = 0; i < M * N; ++i)
        max_err_t = fmax(max_err_t, fabs((double)h_C[i] - (double)K));
    printf("tiled: max|err| =%.3e %s\n", max_err_t, max_err_t < 1e-6 ? "PASS" : "FAIL");

    // ---- tiled: time ----
    float ms_t = bench_ms([&] { gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K); }, 20);
    printf("tiled: %.4f ms  %.1f GFLOPS   speedup %.2fx\n", ms_t,
           2.0 * M * N * K / (ms_t * 1e-3) / 1e9, ms_naive / ms_t);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
