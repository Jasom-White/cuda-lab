// day01_redo.cu - 默写练习（LeetGPU 风格：只给需求，不给答案）
//
// 题目：向量加法 C = A + B
// 输入：A、B 是长度 N 的 float 数组（device 指针），C 是输出数组
// 要求：
//   1) 用一维 grid/block，每个线程处理一个元素；
//   2) 越界线程必须跳过；
//   3) 结果写到 C。
// 验收：max_err < 1e-5（harness 会自动校验并打印 GB/s）
// 提示：没有。卡住就去看 day01_vector_add.cu / day01_vector_add_min.cu。

#include "harness.cuh"

__global__ void add(const float* A, const float* B, float* C, int N) {
    // TODO: 你的实现（约 3 行）
}

int main() {
    const int n = 1 << 20;
    const int T = 256;
    const int B = (n + T - 1) / T;
    check_bench_add("redo-add", n, [&](const float* a, const float* b, float* c, int nn) {
        add<<<B, T>>>(a, b, c, nn);
    });
    return 0;
}
