// leetgpu_style.cu - same shape as a LeetGPU submission, but runnable locally.
// LeetGPU gives you this skeleton: kernel + extern "C" solve(A, B, C, N).
// A, B, C are DEVICE pointers already allocated/filled by the platform.
// You must NOT cudaMalloc / cudaMemcpy / free here.
#include <cuda_runtime.h>
#include "harness.cuh"

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

// LeetGPU entry point: signature must stay exactly like this.
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

// ---- local main: replace LeetGPU platform harness ----
int main() {
    const int n = 1 << 24;   // 16M elements
    check_bench_add("solve", n, [&](const float* a, const float* b, float* c, int nn) {
        solve(a, b, c, nn);
    });
    return 0;
}
