import torch
import vector_add_cuda

# Ensure CUDA is available
if not torch.cuda.is_available():
    print("CUDA is not available. Cannot test the custom CUDA operator.")
    exit(0)

a = torch.randn(10000, device='cuda')
b = torch.randn(10000, device='cuda')

# Call our custom CUDA operator
c = vector_add_cuda.add(a, b)

# Verify the result
c_ref = a + b
assert torch.allclose(c, c_ref), "Results do not match!"
print("Success! Custom CUDA vector add matches PyTorch's add.")

