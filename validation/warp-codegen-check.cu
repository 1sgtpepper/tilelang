#include <tl_templates/cuda/reduce.h>

// Compile this file to PTX for an ordinary CUDA target to inspect the partial
// tail fallback, full-warp XOR tree, and integral redux fast path. Also compile
// it for sm_100a and sm_100f to check the existing feature selection. These
// compile-only checks do not imply SM100 runtime coverage.

__global__ void warp_codegen_f32_sum(float const *input, float *output) {
  int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  output[tid] = tl::warp_reduce_sum(input[tid]);
}

__global__ void warp_codegen_f64_sum(double const *input, double *output) {
  int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  output[tid] = tl::warp_reduce_sum(input[tid]);
}

__global__ void warp_codegen_i32_sum(int32_t const *input, int32_t *output) {
  int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  output[tid] = tl::warp_reduce_sum(input[tid]);
}

__global__ void warp_codegen_f32_min(float const *input, float *output) {
  int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  output[tid] = tl::warp_reduce_min(input[tid]);
}

__global__ void warp_codegen_f32_max(float const *input, float *output) {
  int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  output[tid] = tl::warp_reduce_max(input[tid]);
}
