# cuda-lab

CUDA 练习与 DiffHRG 原型（RTX 5060 Laptop 8GB / sm_120 / CUDA 12.8 / torch 2.11 cu128）。

## 目录

- `cuda-basics/` — 原生 CUDA C 练习（nvcc 直接编译，不经过 PyTorch）
- `cuda-kernels/` — PyTorch `cpp_extension` 工程 + 后续 DiffHRG（可微仿真 + CUDA 批量 RK4）

## 已完成的 kernel 与实测数字

| kernel | 尺寸 | 结果 | 备注 |
|---|---|---|---|
| vector add（grid-stride，4096 blocks） | 1M | 1128 GB/s | 12 MB 全在 L2 |
| vector add（grid-stride） | 64M | 352 GB/s | HBM，≈ 理论峰值 448 的 79% |
| vector add（naive，ceil(n/256) blocks） | 64M | 355 GB/s | 与 grid-stride 基本持平 |
| transpose（naive） | 4096² | 125.5 GB/s | strided write，不合并 |
| transpose（tiled + padding） | 4096² | 241.6 GB/s（1.93×） | 读/写都合并，+1 防 bank conflict |
| reduction（warp shuffle + 两趟） | 2²⁷ | 375.6 GB/s | ≈ torch.sum 372.8 |
| reduction（day04 CUDA：atomic / warp / two-pass） | 2²⁷ | atomic 3.2 GB/s；warp 366.8；two-pass 362.6（约 113×） | warp shuffle + shared 归约 + 两趟无 atomic |
| stencil 5-point（32×32 block） | 4096² | naive 165.3 / tiled 116.1 GB/s | shared 版反而更慢（见下条） |
| stencil 5-point（32×8 block） | 4096² | naive 292.8 / tiled 170.9 GB/s | block 形状影响 occupancy；naive 靠 L1/L2 命中 |
| row softmax | 1024×4096 | 348 GB/s（1.30× torch） | 数值稳定 |
| LeetGPU 风格 vector add（solve 入口） | 16M | 353.5 GB/s | 与平台模板同构 |

## 环境（Windows）

- CUDA Toolkit 12.8，VS Build Tools 2022（MSVC 14.44 + Windows SDK 10.0.26100），ninja
- torch 2.11.0+cu128，RTX 5060 Laptop（sm_120）
- 入口脚本：`cuda-basics/build.bat`、`cuda-basics/buildrun.bat`、`cuda-kernels/dev.bat`
- Windows 踩坑记录：[cuda-kernels/NOTES.md](cuda-kernels/NOTES.md)

## 运行

```powershell
cd cuda-basics
.\buildrun.bat day01_vector_add.cu
.\day01_vector_add.exe 67108864
```

## 计划

- W1–W2：transpose / reduction / stencil / CSR SpMV / GEMM tiling，每个 kernel 记录 naive vs 优化数字
- W3：C++ 工程化（RAII、多文件、CMake）
- W4–W6：DiffHRG（PSINS HRG 仿真 → PyTorch 可微模型 → CUDA 批量 RK4 → 参数标定）