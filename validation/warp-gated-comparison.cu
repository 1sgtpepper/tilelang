#include <tl_templates/cuda/reduce.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { const auto error = (call); if (error != cudaSuccess) { \
  std::fprintf(stderr, "CUDA_ERROR: %s\n", cudaGetErrorString(error)); std::exit(2); \
} } while (false)

#ifndef PROBE_DRIVER
#define JOIN_IMPL(a, b) a##b
#define JOIN(a, b) JOIN_IMPL(a, b)

#if defined(PROBE_HYBRID) || defined(PROBE_XGATE)
__device__ __forceinline__ float gated_sum(float value) {
  constexpr unsigned full = 0xffffffff;
#ifdef PROBE_HYBRID
  if ((blockDim.x * blockDim.y * blockDim.z) % 32 != 0) {
#else
  if (blockDim.x % 32 != 0) {
#endif
    const unsigned warp_mask = __ballot_sync(full, true);
    if (warp_mask != full) {
      const int rank = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
      const int lane = rank % 32;
      const int members = __popc(warp_mask);
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        const bool has_partner = lane + offset < members;
        const float other = tl::shfl_down_sync(warp_mask, value, has_partner ? offset : 0);
        if (has_partner) value += other;
      }
      return tl::shfl_sync(warp_mask, value, 0);
    }
  }
  value += tl::shfl_xor_sync(full, value, 16);
  value += tl::shfl_xor_sync(full, value, 8);
  value += tl::shfl_xor_sync(full, value, 4);
  value += tl::shfl_xor_sync(full, value, 2);
  value += tl::shfl_xor_sync(full, value, 1);
  return value;
}
#endif

__global__ void JOIN(repeated_reduction_, PROBE_VARIANT)(
    const float *input, float *output, int iterations) {
  const int rank = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int index = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) + rank;
  float value = input[index];
  const float increment = (rank % 2 + 1) * 0.015625f;
#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
#if defined(PROBE_HYBRID) || defined(PROBE_XGATE)
    value = gated_sum(value);
#else
    value = tl::warp_reduce_sum(value);
#endif
    value = value * 0.03125f + increment;
  }
  output[index] = value;
}

extern "C" void JOIN(launch_, PROBE_VARIANT)(
    const float *input, float *output, dim3 shape, int blocks, int iterations) {
  JOIN(repeated_reduction_, PROBE_VARIANT)<<<blocks, shape>>>(input, output, iterations);
}
#else
using Launch = void (*)(const float *, float *, dim3, int, int);
extern "C" void launch_baseline(const float *, float *, dim3, int, int);
extern "C" void launch_hybrid(const float *, float *, dim3, int, int);
extern "C" void launch_ballot(const float *, float *, dim3, int, int);
extern "C" void launch_xgate(const float *, float *, dim3, int, int);

int main() {
  constexpr int blocks = 128;
  constexpr int trials = 21;
  const Launch launches[] = {launch_baseline, launch_ballot, launch_hybrid, launch_xgate};
  const char *names[] = {"baseline", "ballot", "hybrid", "xgate"};
  const std::vector<dim3> shapes = {dim3(32), dim3(64), dim3(128), dim3(256),
      dim3(1024), dim3(8, 8), dim3(4, 8, 2), dim3(7), dim3(24), dim3(48),
      dim3(7, 7), dim3(3, 3, 5)};
  cudaDeviceProp device;
  CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
  std::printf("GPU=%s SM=%d%d BLOCKS=%d TRIALS=%d\n", device.name, device.major,
              device.minor, blocks, trials);
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::puts("sample,shape,iterations,trial,variant,milliseconds");
  for (const auto shape : shapes) {
    const int threads = shape.x * shape.y * shape.z;
    const size_t count = blocks * threads;
    const int first_variant = threads % 32 == 0 ? 0 : 1;
    const int versions = 4 - first_variant;
    std::vector<float> host_input(count);
    for (size_t index = 0; index < count; ++index) {
      const int rank = index % threads;
      host_input[index] = (rank % 7 + 1) * 0.125f + (rank / 32 % 3) * 0.0625f;
    }
    float *input, *outputs[4];
    CUDA_CHECK(cudaMalloc(&input, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input, host_input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    for (int variant = first_variant; variant < 4; ++variant)
      CUDA_CHECK(cudaMalloc(&outputs[variant], count * sizeof(float)));
    for (const int iterations : {1, 1024}) {
      for (int warm = 0; warm < 10; ++warm)
        for (int variant = first_variant; variant < 4; ++variant)
          launches[variant](input, outputs[variant], shape, blocks, iterations);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<float> times[4];
      for (int trial = 0; trial < trials; ++trial) {
        for (int position = 0; position < versions; ++position) {
          const int variant = first_variant + (trial + position) % versions;
          CUDA_CHECK(cudaEventRecord(start));
          launches[variant](input, outputs[variant], shape, blocks, iterations);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaEventRecord(stop));
          CUDA_CHECK(cudaEventSynchronize(stop));
          float elapsed;
          CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
          times[variant].push_back(elapsed);
          std::printf("sample,%ux%ux%u,%d,%d,%s,%.6f\n", shape.x, shape.y, shape.z,
                      iterations, trial, names[variant], elapsed);
        }
      }
      std::vector<float> reference(count), observed(count);
      CUDA_CHECK(cudaMemcpy(reference.data(), outputs[first_variant], count * sizeof(float), cudaMemcpyDeviceToHost));
      for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(reference[index])) {
          std::fprintf(stderr, "NONFINITE_OUTPUT\n");
          return 1;
        }
        if (iterations == 1) {
          const int rank = index % threads;
          const size_t begin = index - rank % 32;
          const size_t end = std::min(begin + 32, index - rank + threads);
          float sum = 0;
          for (size_t member = begin; member < end; ++member) sum += host_input[member];
          const float expected = sum * 0.03125f + (rank % 2 + 1) * 0.015625f;
          if (reference[index] != expected) {
            std::fprintf(stderr, "ORACLE_MISMATCH: index=%zu expected=%f actual=%f\n",
                         index, expected, reference[index]);
            return 1;
          }
        }
      }
      for (int variant = first_variant + 1; variant < 4; ++variant) {
        CUDA_CHECK(cudaMemcpy(observed.data(), outputs[variant], count * sizeof(float), cudaMemcpyDeviceToHost));
        if (observed != reference) {
          std::fprintf(stderr, "OUTPUT_MISMATCH: %s\n", names[variant]);
          return 1;
        }
      }
      for (int variant = first_variant; variant < 4; ++variant) {
        std::sort(times[variant].begin(), times[variant].end());
        std::printf("summary,%ux%ux%u,%d,%s,min=%.6f,q1=%.6f,median=%.6f,q3=%.6f,max=%.6f\n",
                    shape.x, shape.y, shape.z, iterations, names[variant],
                    times[variant][0], times[variant][trials / 4],
                    times[variant][trials / 2], times[variant][3 * trials / 4],
                    times[variant][trials - 1]);
      }
    }
    CUDA_CHECK(cudaFree(input));
    for (int variant = first_variant; variant < 4; ++variant) CUDA_CHECK(cudaFree(outputs[variant]));
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::puts("COMPARISON_OUTPUTS_AND_ONE_CALL_ORACLE=PASS");
}
#endif
