// day01_full_ref.cu —— 完整的 CUDA 向量加法示例，包含结果校验和性能测试。
#include <cstdio>        // printf
#include <cstdlib>       // malloc、free、rand、exit
#include <cmath>         // fabs
#include <cuda_runtime.h> // CUDA 运行时 API

// __global__ 表示该函数由主机调用、在设备上执行，每个线程计算一个元素。
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    // 根据线程块索引、块内线程数和线程索引计算全局元素索引。
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 最后一个线程块可能包含越界线程，因此需要检查索引范围。
    if (i < n) c[i] = a[i] + b[i];
}

// 检查 CUDA 调用的返回值，出错时打印错误信息和源码位置并退出。
// do { ... } while (0) 使宏能够像普通语句一样安全使用。
#define CUDA_CHECK(x) do {                                                   \
    cudaError_t e = (x);                                                     \
    if (e != cudaSuccess) {                                                  \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1);                                                             \
    }                                                                        \
} while (0)

int main(int argc, char** argv) {
    // 可通过第一个命令行参数指定元素个数，默认使用 2^20 个元素。
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 20);
    // 每个元素是 float，计算一块数组占用的总字节数。
    size_t bytes = (size_t)n * sizeof(float);

    // h_ 前缀表示主机（CPU）内存，分别保存两个输入和一个输出。
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    // 固定随机种子以便复现，输入数组元素取 [0, 1] 范围内的随机值。
    srand(0);
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    // d_ 前缀表示设备（GPU）内存，为输入和输出数组分配空间。
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    // 将两个输入数组从主机复制到设备，输出数组无需初始化。
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 每个线程块包含 256 个线程，线程块数量向上取整以覆盖全部 n 个元素。
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    // <<<blocks, threads>>> 指定网格和线程块配置，启动向量加法核函数。
    vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    // 核函数启动是异步的：先检查启动错误，再同步等待执行完成并检查运行错误。
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 将 GPU 计算结果复制回主机内存。
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    // 在 CPU 上计算双精度参考值，并统计所有元素中的最大绝对误差。
    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double ref = (double)h_a[i] + (double)h_b[i];
        double err = fabs((double)h_c[i] - ref);
        if (err > max_err) max_err = err;
    }
    // 最大误差小于阈值时判定结果通过。
    printf("day01 vector_add: n=%d  max_err=%.3e  %s\n",
           n, max_err, (max_err < 1e-5 ? "PASS" : "FAIL"));

    // 使用 CUDA 事件在 GPU 时间线上记录起止点，测量核函数执行时间。
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // 正式计时前先运行一次并同步，完成预热，避免首次运行开销影响结果。
    vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 连续执行 100 次，用总时间求平均，减小单次测量波动。
    const int iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it)
        vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaEventRecord(stop));
    // 等待 stop 事件完成，确保计时段内的核函数都已执行完毕。
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_total = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
    float ms = ms_total / iters;
    // 每次迭代读取 a、b 并写入 c，共访问 3*n 个 float，据此计算有效带宽。
    double gbs = 3.0 * n * sizeof(float) / (ms * 1e-3) / 1e9;
    printf("day01 vector_add: %.4f ms/iter  %.1f GB/s  (blocks=%d, threads=%d)\n",
           ms, gbs, blocks, threads);

    // 释放设备内存、主机内存以及 CUDA 事件，避免资源泄漏。
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
