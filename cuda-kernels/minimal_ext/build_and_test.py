# -*- coding: utf-8 -*-
"""工程化写法：多文件（binding.cpp + add_kernel.cu）用 torch.utils.cpp_extension.load 编译。

与 load_inline 的区别：
  load_inline  —— 源码字符串写在 Python 里，适合快速验证/单文件 demo
  load         —— 多个真实源文件，适合正式项目（可加 setup.py 做 pip install）
运行：.\dev.bat minimal_ext/build_and_test.py
"""
import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")
os.environ.setdefault("VSLANG", "1033")
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
mod = load(
    name="minimal_add_ext",
    sources=[os.path.join(HERE, "binding.cpp"), os.path.join(HERE, "add_kernel.cu")],
    extra_cuda_cflags=["-O3", "-Xcompiler", "/utf-8"],
    verbose=True,
)

def bench(fn, iters=30):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

if __name__ == "__main__":
    # 正确性：各种形状 + 非连续输入 + 空张量
    for shape in [(1,), (257,), (1024, 1024), (3, 5, 7)]:
        a = torch.randn(*shape, device="cuda")
        b = torch.randn(*shape, device="cuda")
        out = mod.add(a, b)
        assert torch.allclose(out, a + b, atol=1e-6), shape
        print(f"shape={str(shape):>16}  allclose=True")

    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    assert torch.allclose(mod.add(a.t(), b.t()), a.t() + b.t(), atol=1e-6)
    print("non-contiguous (transpose) base: ok")

    # 性能
    n = 1 << 26
    a = torch.randn(n, device="cuda"); b = torch.randn(n, device="cuda")
    ms = bench(lambda: mod.add(a, b))
    print(f"\nadd n={n}: {ms:.3f} ms, {3 * n * 4 / ms / 1e6:.1f} GB/s")
