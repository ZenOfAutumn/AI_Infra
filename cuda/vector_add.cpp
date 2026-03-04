#include <torch/extension.h>

// Declaration of the CUDA function
torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must have the same size");
  return vector_add_cuda(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &vector_add, "Vector Add (CUDA)");
}

