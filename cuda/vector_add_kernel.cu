#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void vector_add_cuda_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    size_t size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = index; i < size; i += stride) {
    c[i] = a[i] + b[i];
  }
}

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
  auto c = torch::empty_like(a);
  const int threads = 1024;
  const int blocks = (a.numel() + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "vector_add_cuda", ([&] {
    vector_add_cuda_kernel<scalar_t><<<blocks, threads>>>(
        a.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        c.data_ptr<scalar_t>(),
        a.numel());
  }));

  return c;
}

