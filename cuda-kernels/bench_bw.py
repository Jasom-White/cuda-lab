# -*- coding: utf-8 -*-
"""本机显存带宽基线：D2D copy + triad，用于判断 memory-bound kernel 是否达标。"""
import torch

def bench(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

n = 1 << 27  # 512MB per array
a = torch.empty(n, dtype=torch.float32, device="cuda")
b = torch.empty_like(a)
c = torch.empty_like(a)

ms = bench(lambda: b.copy_(a))
print(f"D2D copy_ : {ms:8.3f} ms   {2 * n * 4 / ms / 1e6:8.1f} GB/s")

ms = bench(lambda: torch.add(a, b, out=c))
print(f"torch add : {ms:8.3f} ms   {3 * n * 4 / ms / 1e6:8.1f} GB/s")

# triad: c = a + 2*b （纯 PyTorch 参考，后面和手写 kernel 对比）
ms = bench(lambda: torch.add(a, b, alpha=2.0, out=c))
print(f"torch triad:{ms:8.3f} ms   {3 * n * 4 / ms / 1e6:8.1f} GB/s")
