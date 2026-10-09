#include <tl_templates/cuda/reduce.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { const auto error = (call); if (error != cudaSuccess) { \
  std::fprintf(stderr, "CUDA_ERROR: %s\n", cudaGetErrorString(error)); std::exit(2); \
} } while (false)

__device__ __forceinline__ float ballot_sum(float value) {
  const unsigned mask = __ballot_sync(0xffffffff, 1);
  if (mask != 0xffffffff) {
    const int thread_idx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const int lane = thread_idx % 32;
    const int threads = __popc(mask);
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      const bool has_partner = lane + offset < threads;
      const float other = tl::shfl_down_sync(mask, value, has_partner ? offset : 0);
      if (has_partner) value += other;
    }
    return tl::shfl_sync(mask, value, 0);
  }
  value += tl::shfl_xor_sync(mask, value, 16);
  value += tl::shfl_xor_sync(mask, value, 8);
  value += tl::shfl_xor_sync(mask, value, 4);
  value += tl::shfl_xor_sync(mask, value, 2);
  value += tl::shfl_xor_sync(mask, value, 1);
  return value;
}

template <bool ballot>
__global__ void repeated_reduction(const float *input, float *output, int iterations) {
  const int rank = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int index = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) + rank;
  float value = input[index];
  const float increment = (rank % 2 + 1) * 0.015625f;
#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if constexpr (ballot) value = ballot_sum(value);
    else value = tl::warp_reduce_sum(value);
    value = value * 0.03125f + increment;
  }
  output[index] = value;
}

int main() {
  constexpr int blocks = 128;
  constexpr int trials = 9;
  const std::vector<dim3> shapes = {dim3(32), dim3(64), dim3(128), dim3(256),
      dim3(7), dim3(24), dim3(48), dim3(7,7), dim3(3,3,5)};
  cudaDeviceProp device;
  CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
  std::printf("GPU=%s SM=%d%d BLOCKS=%d TRIALS=%d\n", device.name, device.major, device.minor, blocks, trials);
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  std::printf("shape,iterations,geometry_ms,ballot_ms,ballot_over_geometry\n");
  for (const auto shape : shapes) {
    const size_t count = blocks * shape.x * shape.y * shape.z;
    float *input, *geometry, *ballot;
    CUDA_CHECK(cudaMalloc(&input, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&geometry, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ballot, count * sizeof(float)));
    CUDA_CHECK(cudaMemset(input, 0, count * sizeof(float)));
    for (const int iterations : {1, 1024}) {
      for (int warm = 0; warm < 10; ++warm) {
        repeated_reduction<false><<<blocks,shape>>>(input, geometry, iterations);
        repeated_reduction<true><<<blocks,shape>>>(input, ballot, iterations);
      }
      CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<float> geometry_times, ballot_times;
      for (int trial = 0; trial < trials; ++trial) {
        for (int order = 0; order < 2; ++order) {
          const bool use_ballot = ((trial + order) % 2) != 0;
          CUDA_CHECK(cudaEventRecord(start));
          if (use_ballot) repeated_reduction<true><<<blocks,shape>>>(input, ballot, iterations);
          else repeated_reduction<false><<<blocks,shape>>>(input, geometry, iterations);
          CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop));
          CUDA_CHECK(cudaEventSynchronize(stop));
          float elapsed;
          CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
          (use_ballot ? ballot_times : geometry_times).push_back(elapsed);
        }
      }
      std::vector<float> geometry_values(count), ballot_values(count);
      CUDA_CHECK(cudaMemcpy(geometry_values.data(), geometry, count * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(ballot_values.data(), ballot, count * sizeof(float), cudaMemcpyDeviceToHost));
      if (geometry_values != ballot_values) { std::fprintf(stderr, "OUTPUT_MISMATCH\n"); return 1; }
      std::sort(geometry_times.begin(), geometry_times.end());
      std::sort(ballot_times.begin(), ballot_times.end());
      const float g = geometry_times[trials/2], b = ballot_times[trials/2];
      std::printf("%ux%ux%u,%d,%.6f,%.6f,%.4f\n", shape.x,shape.y,shape.z,iterations,g,b,b/g);
    }
    CUDA_CHECK(cudaFree(input)); CUDA_CHECK(cudaFree(geometry)); CUDA_CHECK(cudaFree(ballot));
  }
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  std::puts("COMPARISON_OUTPUTS_EQUAL=PASS");
}
