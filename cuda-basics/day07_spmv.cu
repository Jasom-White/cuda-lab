// day07_spmv.cu —— 稀疏矩阵乘向量（CSR SpMV）：y = A * x
// 你的任务：只填两个 kernel（TODO 1 / TODO 2），main 已写好，不要改。
// A 用 CSR 格式存储：2D 5 点 Laplacian（G×G 网格，每个节点一行）
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "common.cuh"

// ---------------- TODO 1：一行一个线程 ----------------
// 思路：线程 r 负责第 r 行，遍历 [row_ptr[r], row_ptr[r+1]) 里的非零元
__global__ void spmv_row_per_thread(const int* row_ptr, const int* col_idx,
                                    const float* values, const float* x,
                                    float* y, int M) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M) return;
    float sum = 0.f;
    for (int k = row_ptr[r]; k < row_ptr[r + 1]; ++k)
        sum += values[k] * x[col_idx[k]];
    y[r] = sum;
    // int r = blockIdx.x * blockDim.x + threadIdx.x;
    // if (r >= M) return;                       // 边界检查
    // float sum = 0.f;
    // for (int k = row_ptr[r]; k < row_ptr[r + 1]; ++k)   // 行内串行累加
    //     sum += values[k] * x[col_idx[k]];               // x[col_idx[k]] 是随机访存
    // y[r] = sum;
}

// ---------------- TODO 2：一行一个 warp ----------------
// 思路：32 个 lane 分摊同一行的非零元，warp 内用 shuffle 归约后由 lane 0 写出
__global__ void spmv_row_per_warp(const int* row_ptr, const int* col_idx,
                                  const float* values, const float* x,
                                  float* y, int M) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = gtid / 32, lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;      
    for(int r = warp_id; r < M; r += num_warps){
        float sum = 0.f;
        for (int k = row_ptr[r] + lane; k < row_ptr[r + 1]; k += 32)
            sum += values[k] * x[col_idx[k]];
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (lane == 0) y[r] = sum;
    }
    // int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    // int warp_id = gtid / 32, lane = threadIdx.x % 32;
    // int num_warps = (gridDim.x * blockDim.x) / 32;      // 总共多少个 warp
    // for (int r = warp_id; r < M; r += num_warps) {      // 每个 warp 处理多行
    //     float sum = 0.f;
    //     for (int k = row_ptr[r] + lane; k < row_ptr[r + 1]; k += 32)  // 行内元素按 lane 分摊
    //         sum += values[k] * x[col_idx[k]];
    //     for (int off = 16; off > 0; off >>= 1)                        // warp 内 32→1 归约
    //         sum += __shfl_down_sync(0xffffffffu, sum, off);
    //     if (lane == 0) y[r] = sum;                                    // 每行只写一次
    // }
}

// ---------------- 以下 main 已提供，不要修改 ----------------
// 申请 int 类型的显存（common.cuh 里的 dalloc 只支持 float）
static int* ialloc(size_t n) {
    int* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(int)));
    return p;
}

// 逐元素最大绝对误差（用于和 CPU 参考解对比）
static double max_abs_diff(const float* a, const float* b, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) m = fmax(m, fabs((double)a[i] - (double)b[i]));
    return m;
}

int main() {
    const int G = 1024;              // 网格边长
    const int M = G * G, N = G * G;  // 矩阵行/列数 = 节点数

    // ---- 1) 在 host 端生成 CSR：2D 5 点 Laplacian ----
    // 对角 = 4，上下左右邻居 = -1，越界方向没有这一项
    std::vector<int> row_ptr(M + 1);
    std::vector<int> col_idx;
    std::vector<float> values;
    col_idx.reserve((size_t)5 * M);
    values.reserve((size_t)5 * M);
    int nnz = 0;
    row_ptr[0] = 0;
    for (int r = 0; r < M; ++r) {
        int i = r / G, j = r % G;
        col_idx.push_back(r);       values.push_back(4.0f);   ++nnz;   // 对角
        if (i > 0)     { col_idx.push_back(r - G); values.push_back(-1.0f); ++nnz; }  // 上
        if (i < G - 1) { col_idx.push_back(r + G); values.push_back(-1.0f); ++nnz; }  // 下
        if (j > 0)     { col_idx.push_back(r - 1); values.push_back(-1.0f); ++nnz; }  // 左
        if (j < G - 1) { col_idx.push_back(r + 1); values.push_back(-1.0f); ++nnz; }  // 右
        row_ptr[r + 1] = nnz;
    }
    printf("CSR: M=%d N=%d nnz=%d (%.2f nnz/row)\n", M, N, nnz, (double)nnz / M);

    // ---- 2) host 数据 + CPU 参考解 ----
    std::vector<float> x(N), y_ref(M), y_out(M);
    srand(0);
    for (int i = 0; i < N; ++i) x[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int r = 0; r < M; ++r) {
        float s = 0.f;
        for (int k = row_ptr[r]; k < row_ptr[r + 1]; ++k) s += values[k] * x[col_idx[k]];
        y_ref[r] = s;
    }

    // ---- 3) 拷贝到显存 ----
    int* d_row_ptr = ialloc((size_t)M + 1);
    int* d_col_idx = ialloc((size_t)nnz);
    float* d_values = dalloc(nnz);
    float* d_x = dalloc(N);
    float* d_y = dalloc(M);
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), (size_t)(M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), (size_t)nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    // 有效访存字节数：values + col_idx（各 4B/nnz）+ x + y（各 4B/元素）
    const double bytes = (double)nnz * (4.0 + 4.0) + (double)(M + N) * 4.0;
    const int T = 256;   // 每个 block 的线程数

    // ---- 4) row per thread：校验 + 计时 ----
    {
        const int B = (M + T - 1) / T;
        CUDA_CHECK(cudaMemset(d_y, 0, (size_t)M * sizeof(float)));
        spmv_row_per_thread<<<B, T>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, M);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(y_out.data(), d_y, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost));
        double e = max_abs_diff(y_out.data(), y_ref.data(), M);
        float ms = bench_ms([&] { spmv_row_per_thread<<<B, T>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, M); }, 30);
        printf("row_per_thread: max_err=%.3e %s  %8.4f ms  %6.1f GB/s\n",
               e, e < 1e-3 ? "PASS" : "FAIL", ms, bytes / (ms * 1e-3) / 1e9);
    }

    // ---- 5) row per warp：校验 + 计时 ----
    {
        const int B = 512;   // 512 block × 256 线程 = 4096 个 warp
        CUDA_CHECK(cudaMemset(d_y, 0, (size_t)M * sizeof(float)));
        spmv_row_per_warp<<<B, T>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, M);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(y_out.data(), d_y, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost));
        double e = max_abs_diff(y_out.data(), y_ref.data(), M);
        float ms = bench_ms([&] { spmv_row_per_warp<<<B, T>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, M); }, 30);
        printf("row_per_warp:   max_err=%.3e %s  %8.4f ms  %6.1f GB/s\n",
               e, e < 1e-3 ? "PASS" : "FAIL", ms, bytes / (ms * 1e-3) / 1e9);
    }

    // ---- 6) 释放 ----
    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_values);
    cudaFree(d_x); cudaFree(d_y);
    return 0;
}
