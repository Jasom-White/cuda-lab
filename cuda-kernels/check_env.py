# -*- coding: utf-8 -*-
"""环境自检：确认 CUDA / torch / MSVC / ninja / cpp_extension 全链路可用。"""
import os, shutil, subprocess, sys

def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"ERR {e}"

print("=" * 60)
print("python      :", sys.version.split()[0])
print("TORCH_CUDA_ARCH_LIST:", os.environ.get("TORCH_CUDA_ARCH_LIST"))
print("cl          :", shutil.which("cl") or "NOT FOUND")
print("nvcc        :", sh("nvcc --version").splitlines()[0] if sh("nvcc --version") else "NOT FOUND")
print("ninja       :", shutil.which("ninja") or "NOT FOUND")
print("=" * 60)

import torch
print("torch       :", torch.__version__, "| cuda:", torch.version.cuda)
print("cuda avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      :", torch.cuda.get_device_name(0))
    print("capability  :", torch.cuda.get_device_capability(0))
    p = torch.cuda.get_device_properties(0)
    print("vram        :", round(p.total_memory / 1024**3, 2), "GB")
    x = torch.randn(1024, device="cuda")
    print("gpu matmul  :", float((x * x).sum()))
print("=" * 60)

if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
os.environ.setdefault("VSLANG", "1033")
os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")  # cl.exe 输出本地化，torch 用 OEM 解码会崩，跳过 ABI 检查

print("[1/2] building cpp_extension (load_inline) ...")
os.environ.setdefault("MAX_JOBS", "4")
from torch.utils.cpp_extension import load_inline
cpp = r"""
#include <torch/extension.h>
torch::Tensor add(torch::Tensor a, torch::Tensor b);
"""
cuda = r"""
#include <torch/extension.h>
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel();
    add_kernel<<<(n + 255) / 256, 256>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}
"""
try:
    mod = load_inline(name="dsh_add_ext", cpp_sources=cpp, cuda_sources=cuda,
                      functions=["add"], verbose=True, extra_cuda_cflags=["-O3"])
    a = torch.randn(1 << 20, device="cuda"); b = torch.randn_like(a)
    ok = torch.allclose(mod.add(a, b), a + b, atol=1e-6)
    print("[2/2] extension build OK, allclose =", ok)
    print("\n=== ENV READY ===")
except Exception as e:
    print("[2/2] extension build FAILED:", repr(e))
    print("\n检查：是否在 dev.bat / x64 Native Tools 环境中运行？是否装了 C++ 桌面开发工作负载？")
    raise
