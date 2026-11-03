# Windows CUDA 环境笔记（本机实测，2026-11-03）

## 环境
- GPU: NVIDIA GeForce RTX 5060 Laptop, 8GB, compute capability (12, 0) = sm_120
- 驱动 596.11（CUDA 13.2 capable），运行时 CUDA Toolkit **12.8**（nvcc 12.8.61，与 torch cu128 匹配）
- torch 2.11.0+cu128, Python 3.11.9
- VS Build Tools 2022: MSVC 14.44.35207 + Windows SDK 10.0.26100
- ninja 1.13.2（pip 安装）
- 编辑器：VS Code（cuda-kernels/.vscode 已配 tasks，Ctrl+Shift+B 运行）

## 两个 Windows 专属坑（已修复）
### 1) cl.exe 中文输出导致 torch OEM 解码崩溃
现象：`UnicodeDecodeError: 'cp1..' No mapping for the Unicode character exists`
位置：torch/utils/cpp_extension.py get_compiler_abi_compatibility_and_version（用 'oem' 解码 cl 输出）
修复：dev.bat 里 `set TORCH_DONT_CHECK_COMPILER_ABI=1`（跳过 ABI 版本检查；cl 14.44 与 torch cu128 本来就兼容）
备用：也设了 `VSLANG=1033`（本机 cl 仍是中文输出，未生效，留着无害）

### 2) nvcc + MSVC 14.44 编译 torch 头文件报 C2872 "std" 不明确
现象：`torch/csrc/dynamo/compiled_autograd.h(1143): error C2872: "std": 不明确的符号`
原因：torch 2.11 的 IValuePacker 里 `::std::is_same_v<T, ::std::string>` 分支在 nvcc/Windows 下与 MSVC 的 std 命名空间冲突（上游修复：pytorch/pytorch#188952）
修复：本地打补丁（backport）：`<site-packages>/torch/include/torch/csrc/dynamo/compiled_autograd.h`
  - 删除该 `::std::string` 分支
  - 改为 `template <> struct IValuePacker<std::string> { ... }` 全特化
备份：`compiled_autograd.h.bak_dsh`（同目录）。**升级/重装 torch 后需重新打补丁**（或升级到已包含该修复的版本）。

## 实测性能基线（RTX 5060 Laptop 8GB, 本机）
| 项目 | 数字 | 说明 |
|---|---|---|
| D2D copy（0.5GB×2） | **344 GB/s** | 基线 |
| torch add（0.5GB×3） | **351 GB/s** | 基线 |
| reduction sum（手写，n=2^27） | **375.6 GB/s**（=torch.sum 372.8） | 只读带宽 |
| first_kernel vec4 vs naive | 349 vs 352 GB/s | 该尺寸下已带宽饱和，向量化不涨 |
| softmax 1024×4096 | 0.096 ms, 348 GB/s（torch 0.126 ms，**1.30×**） | 行内两趟归约 |
| softmax 4096×4096 | 0.523 ms, 256 GB/s（torch 0.543 ms，1.04×） | 大行时带宽下降 |
| softmax 1024×1024 | 0.028 ms（torch 0.018 ms，0.64×） | 行太小、block 数不足、占用率低 |

**结论**：本机 memory-bound 实际可达带宽天花板 ≈ **350–375 GB/s**；后续 kernel 的达标线定为 **≥300 GB/s（80%）**。

## 快速命令
```powershell
cd C:\Users\lenovo\Desktop\hrg\cuda-kernels
.\dev.bat check_env.py
.\dev.bat first_kernel.py
.\dev.bat bench_bw.py
.\dev.bat reduction.py
.\dev.bat softmax.py
```
