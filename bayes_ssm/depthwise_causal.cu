#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

__device__ __forceinline__ float load_bf16(const at::BFloat16* p) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(p));
}

__device__ __forceinline__ void store_bf16(at::BFloat16* p, float value) {
  *reinterpret_cast<__nv_bfloat16*>(p) = __float2bfloat16_rn(value);
}

__global__ void forward_kernel(
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ weight,
    at::BFloat16* __restrict__ y, int b, int t, int c) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = b * t * c;
  if (index >= total) return;
  const int bt = index / c;
  const int channel = index - bt * c;
  const int token = bt % t;
  const int batch = bt / t;
  float sum = 0.0f;
  for (int k = 0; k < 4; ++k) {
    const int source = token - 3 + k;
    if (source >= 0)
      sum += load_bf16(x + (batch * t + source) * c + channel) *
             load_bf16(weight + channel * 4 + k);
  }
  store_bf16(y + index, sum);
}

__global__ void input_backward_kernel(
    const at::BFloat16* __restrict__ grad_y,
    const at::BFloat16* __restrict__ weight,
    at::BFloat16* __restrict__ grad_x, int b, int t, int c) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = b * t * c;
  if (index >= total) return;
  const int bt = index / c;
  const int channel = index - bt * c;
  const int token = bt % t;
  const int batch = bt / t;
  float sum = 0.0f;
  for (int k = 0; k < 4; ++k) {
    const int output_token = token + 3 - k;
    if (output_token < t)
      sum += load_bf16(grad_y + (batch * t + output_token) * c + channel) *
             load_bf16(weight + channel * 4 + k);
  }
  store_bf16(grad_x + index, sum);
}

__global__ void weight_backward_kernel(
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ grad_y,
    at::BFloat16* __restrict__ grad_weight, int b, int t, int c) {
  const int index = blockIdx.x;
  const int total = c * 4;
  if (index >= total) return;
  const int channel = index / 4;
  const int k = index - channel * 4;
  const int lane = threadIdx.x;
  float sum = 0.0f;
  for (int bt = lane; bt < b * t; bt += blockDim.x) {
    const int token = bt % t;
    const int source = token - 3 + k;
    if (source >= 0)
      sum += load_bf16(x + (bt / t * t + source) * c + channel) *
             load_bf16(grad_y + bt * c + channel);
  }
  __shared__ float reduction[256];
  reduction[lane] = sum;
  __syncthreads();
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (lane < offset) reduction[lane] += reduction[lane + offset];
    __syncthreads();
  }
  if (lane == 0) store_bf16(grad_weight + index, reduction[0]);
}

}  // namespace

std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor weight) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.dim() == 3 && weight.dim() == 2 && weight.size(1) == 4);
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous());
  const int b = x.size(0), t = x.size(1), c = x.size(2);
  auto y = torch::empty_like(x);
  auto stream = at::cuda::getCurrentCUDAStream();
  forward_kernel<<<(b * t * c + 255) / 256, 256, 0, stream>>>(
      x.data_ptr<at::BFloat16>(), weight.data_ptr<at::BFloat16>(),
      y.data_ptr<at::BFloat16>(), b, t, c);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y};
}

std::vector<torch::Tensor> backward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor grad_y) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && grad_y.is_cuda());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
              grad_y.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && grad_y.is_contiguous());
  const int b = x.size(0), t = x.size(1), c = x.size(2);
  auto grad_x = torch::empty_like(x);
  auto grad_weight = torch::empty_like(weight);
  auto stream = at::cuda::getCurrentCUDAStream();
  input_backward_kernel<<<(b * t * c + 255) / 256, 256, 0, stream>>>(
      grad_y.data_ptr<at::BFloat16>(), weight.data_ptr<at::BFloat16>(),
      grad_x.data_ptr<at::BFloat16>(), b, t, c);
  weight_backward_kernel<<<c * 4, 256, 0, stream>>>(
      x.data_ptr<at::BFloat16>(), grad_y.data_ptr<at::BFloat16>(),
      grad_weight.data_ptr<at::BFloat16>(), b, t, c);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_x, grad_weight};
}

// The input gradient is a pointwise four tap reverse scan and is substantially
// cheaper in the dedicated kernel above.  Keep a split entry point so callers
// can pair it with PyTorch's optimized grouped-convolution weight reduction;
// the latter is faster than launching one reduction block per channel.
std::vector<torch::Tensor> backward_input(
    torch::Tensor x, torch::Tensor weight, torch::Tensor grad_y) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && grad_y.is_cuda());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
              weight.scalar_type() == at::kBFloat16 &&
              grad_y.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.dim() == 3 && weight.dim() == 2 && weight.size(1) == 4 &&
              grad_y.sizes() == x.sizes());
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && grad_y.is_contiguous());
  const int b = x.size(0), t = x.size(1), c = x.size(2);
  auto grad_x = torch::empty_like(x);
  auto stream = at::cuda::getCurrentCUDAStream();
  input_backward_kernel<<<(b * t * c + 255) / 256, 256, 0, stream>>>(
      grad_y.data_ptr<at::BFloat16>(), weight.data_ptr<at::BFloat16>(),
      grad_x.data_ptr<at::BFloat16>(), b, t, c);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_x};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward);
  m.def("backward", &backward);
  m.def("backward_input", &backward_input);
}
