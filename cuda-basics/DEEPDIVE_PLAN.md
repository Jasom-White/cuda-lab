# deepdive 目标下的 CUDA 准备计划

目标：浦江实验室 物理智能中心「科学仿真平台研究（deepdive）」实习。
JD 四个方向：① 仿真引擎(FEM/MPM/接触) ② 高性能计算(CUDA kernel/稀疏矩阵/访存) ③ 可微仿真与数字孪生(伴随/参数标定/传感→状态估计) ④ 仿真系统与 AI 接口。
你的主打组合：② 高性能计算 + ③ 可微仿真（用 HRG/PSINS 数据做验证）。

## 能力矩阵（deepdive 视角）

| 能力 | 权重 | 现在 | 训练方式 |
|---|---|---|---|
| CUDA 基本模式（vector add/transpose/reduction/stencil/SpMV/GEMM tiling） | 高 | 20% | day01–day07，每个 kernel naive vs 优化 + 数字 |
| 优化方法论（合并访存/bank conflict/占用率/roofline） | 高 | 10% | 每个 kernel 写“为什么快/慢”一段分析 |
| C++ 工程（RAII/模板/多线程/CMake） | 高 | 30% | DeviceBuffer 封装、多文件 launcher、CMake 版 |
| 可微仿真（RK4/伴随/参数标定） | 高 | 30% | DiffHRG：PyTorch 可微 → CUDA 批量 RK4 |
| 稀疏矩阵/求解器（CSR/CG/GMRES） | 中 | 0% | day08 SpMV；读 CG 并实现简化版 |
| 仿真引擎概念（FEM/MPM/接触） | 中 | 0% | 读概念 + 能画数据流，不要求实现 |
| 论文复现（可微仿真/GPU 优化） | 中 | 10% | 选 2 篇，写复现笔记 |
| 项目叙事（数字/瓶颈/trade-off） | 高 | 30% | 每个 kernel 记录到 RESULTS.md，面试可讲 |

## 6 周计划（与 DiffHRG 并行）

- W1：transpose（naive vs tiled + padding）、reduction 优化（占用率扫描）、写 RESULTS.md
- W2：stencil 1D/2D、SpMV(CSR)、GEMM shared tiling；每个都有 naive/优化两行数字
- W3：C++ 工程化：DeviceBuffer(RAII)、多文件 launcher、CMake；把 day01–07 的 kernel 组织成库
- W4：DiffHRG 的 CUDA 部分：批量 RK4 kernel（1024 条轨迹）+ 与 PyTorch 可微模型对齐
- W5：可微仿真：参数标定（伴随）、EKF/可微 Kalman、PSINS 数据接入；写性能/精度报告
- W6：读 2 篇论文（可微仿真/GPU 优化）做复现笔记；整理 repo、README、面试讲稿；模拟面试 2 次

## 面试清单（deepdive 版）

1. 你的 transpose/reduction 为什么快？naive 慢在哪（coalescing/bank conflict）？
2. 占用率怎么算？哪些资源限制它？怎么调？
3. RK4 批量并行为什么适合一个线程一条轨迹？寄存器压力怎么权衡？
4. 稀疏矩阵 CSR SpMV 的瓶颈是什么？怎么优化（row-per-warp/vectorized）？
5. 可微仿真的反向怎么实现（adjoint vs autograd）？计算图开销多大？
6. 如果给你一个 FEM/接触仿真的 kernel，你会怎么并行化（按单元/按节点/图着色）？
7. 你的 DiffHRG 里最慢的一步是什么？怎么定位的？优化了多少？
8. C++：RAII 怎么保证 cudaMalloc/cudaFree 配对？异常安全？
9. 单 GPU 到多 GPU，会怎么扩展（domain decomposition）？
10. 讲一篇你复现过的论文：方法、坑、结果。

## 交付物（面试时能打开的东西）

- cuda-basics/：每个 kernel 的源码 + RESULTS.md（naive vs 优化、GB/s、分析）
- cuda-kernels/DiffHRG：可微仿真 + CUDA 加速 + 参数标定 + 精度报告
- 2 篇论文复现笔记（Markdown）
- 一页英文 Project（deepdive 方向对齐）
