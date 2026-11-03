# cuda-kernels

CUDA 练习 + DiffHRG（PSINS HRG 仿真数据 → 可微仿真 → CUDA 批量加速）的代码仓库。

## 环境
- CUDA Toolkit 12.8（nvcc 12.8.61）
- torch 2.11.0+cu128，Python 3.11.9
- RTX 5060 Laptop 8GB（sm_120）
- VS Build Tools 2022（MSVC 14.44.35207 + Windows SDK 10.0.26100）
- ninja 1.13.2
- 编辑器：VS Code（`.vscode/tasks.json` 已配好，Ctrl+Shift+B）

> 本机两个 Windows 专属坑的修复方法见 **[NOTES.md](NOTES.md)**（cl 中文输出导致 torch 解码崩溃；nvcc+MSVC 14.44 的 C2872 头文件补丁）。

## 跑
```powershell
cd C:\Users\lenovo\Desktop\hrg\cuda-kernels
.\dev.bat check_env.py      # 全链路自检
.\dev.bat first_kernel.py   # naive vs float4
.\dev.bat bench_bw.py       # 带宽基线
.\dev.bat reduction.py      # warp shuffle + 共享内存归约
.\dev.bat softmax.py        # 行 softmax（数值稳定）
```

## 实测结果（2026-11-03）
| kernel | 结果 |
|---|---|
| D2D copy | 344 GB/s |
| torch add | 351 GB/s |
| reduction (n=2^27) | 375.6 GB/s（torch.sum 372.8） |
| softmax 1024×4096 | 348 GB/s，1.30× torch |
| softmax 4096×4096 | 256 GB/s，1.04× torch |

**memory-bound 达标线：≥300 GB/s（本机峰值 ≈350–375 GB/s）。**

## 路线
- W1: bandwidth → elementwise → reduction → softmax → **stencil** → **batched RK4**
- W2+: DiffHRG（PSINS HRG 数据 → PyTorch 可微模型 → CUDA 批量 RK4）
