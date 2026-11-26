// day02_gridstride.cu - 只写 kernel，其余交给 harness（LeetGPU 风格）。
#include "harness.cuh"

// grid-stride loop（网格跨步循环）版向量加法
// 核心思想：启动的线程总数可以少于元素个数 N，每个线程处理多个元素，
// 步长为整个网格的线程总数 gridDim.x * blockDim.x，从而"跨步"遍历数组。
__global__ void add_gridstride(const float* A, const float* B, float* C, int N) {
    // 初始线程全局编号：blockIdx.x * blockDim.x + threadIdx.x
    // 循环条件 i < N 保证尾部（N 不是线程数整数倍时）不会越界
    // 每次迭代 i += gridDim.x * blockDim.x，跳到"下一轮"同一个线程的位置
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        C[i] = A[i] + B[i];  // 每个元素只被一个线程处理一次，无需额外同步
    }
}

int main() {
    const int n = 1 << 20;              // 1M 元素，L2 缓存测试规模
    const int T = 256;                  // 每个 block 的线程数（8 个 warp）
    const int B = (n + T - 1) / T;      // 向上取整的 block 数，足够覆盖全部元素

    // 小规模（L2 命中）基线测试
    check_bench_add("gridstride", n,
        [&](const float* a, const float* b, float* c, int nn) {
            add_gridstride<<<B, T>>>(a, b, c, nn);  // 启动 B 个 block，T 个线程/block
        });

    // HBM 测试：同一个 kernel，换成更大的 n（block 数重新计算）
    const int n2 = 1 << 26;             // 64M 元素，超出 L2，主要受 HBM 带宽限制
    const int B2 = (n2 + T - 1) / T;    // 对应 n2 的 block 数
    check_bench_add("gridstride", n2,
        [&](const float* a, const float* b, float* c, int nn) {
            add_gridstride<<<B2, T>>>(a, b, c, nn);
        });
    return 0;
}
