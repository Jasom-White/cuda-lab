// day02_gridstride.cu - only the kernel changes; harness does the rest (LeetGPU style).
#include "harness.cuh"

__global__ void add_gridstride(const float* A, const float* B, float* C, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int n = 1 << 20;
    const int T = 256;
    const int B = (n + T - 1) / T;

    check_bench_add("gridstride", n,
        [&](const float* a, const float* b, float* c, int nn) {
            add_gridstride<<<B, T>>>(a, b, c, nn);
        });

    // HBM test: same kernel, bigger n (blocks recomputed inside)
    const int n2 = 1 << 26;
    const int B2 = (n2 + T - 1) / T;
    check_bench_add("gridstride", n2,
        [&](const float* a, const float* b, float* c, int nn) {
            add_gridstride<<<B2, T>>>(a, b, c, nn);
        });
    return 0;
}
