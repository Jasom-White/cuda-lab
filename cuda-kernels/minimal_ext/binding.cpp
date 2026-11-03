// binding.cpp - host side (pybind11 binding), compiled by cl.exe
#include <torch/extension.h>

// declared here, implemented in add_kernel.cu (compiled by nvcc)
torch::Tensor add_forward(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add_forward, "elementwise add (CUDA)");
}
