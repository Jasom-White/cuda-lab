// hello.cu - the canonical first CUDA program: no framework, just nvcc.
#include <cstdio>

__global__ void hello_kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    printf("CPU: launching 2 blocks x 4 threads = 8 threads\n");
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();      // must sync: printf runs on device
    printf("CPU: done\n");
    return 0;
}
