# cuda-basics

原生 CUDA C 练习（不经过 PyTorch）。每天一个 .cu，编译用本目录 `build.bat`。

## VS Code 里怎么写/编译/运行
1. VS Code 打开 **这个目录**（File > Open Folder: `cuda-basics`）
2. 新建 `day01_vector_add.cu`（可从 `template.cu` 复制）
3. 按 **Ctrl+Shift+B** → 选 **"CUDA: build & run current .cu"**，结果直接打印在终端
4. 只编译：Ctrl+Shift+B → "CUDA: build current .cu"；查内存错误：选 "CUDA: memcheck"
5. 也可以在集成终端手动跑：
   ```powershell
   .\build.bat day01_vector_add.cu -o day01.exe
   .\day01.exe
   ```

> 不要点 VS Code 的 ▶ Run Code / C++ 运行按钮——那会用 MSVC 直接编译 C++，不认 `__global__`。必须走 nvcc（build.bat 已处理 vcvars 和 arch=sm_120）。

## 文件约定
- 每个练习一个文件：`dayXX_topic.cu`
- 顶部注释写：目标 / 正确性标准 / 预期带宽或加速比 / 实际结果
- 每天结束时把数字补进 `RESULTS.md`

## 已知坑
- `.cu` 里只写 ASCII 注释（nvcc 处理中文注释会出解析错误）
- nvcc 也需要 cl.exe 作为宿主编译器（build.bat 已 call vcvars64）
- 本机 L2 较大：小数组（几 MB）测出的带宽会高于显存带宽，做性能对比时用 ≥256MB 的工作集
