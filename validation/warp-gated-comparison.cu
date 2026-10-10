#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <tl_templates/cuda/reduce.h>
#include <type_traits>
#include <vector>

using ProbeType = PROBE_TYPE;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const auto error = (call);                                                 \
    if (error != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA_ERROR: %s\n", cudaGetErrorString(error));     \
      std::exit(2);                                                            \
    }                                                                          \
  } while (false)

#ifndef PROBE_DRIVER
#define JOIN_IMPL(a, b) a##b
#define JOIN(a, b) JOIN_IMPL(a, b)

__global__ void JOIN(repeated_reduction_, PROBE_VARIANT)(const ProbeType *input,
                                                         ProbeType *output,
                                                         int iterations) {
  const int rank =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int index = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) + rank;
  ProbeType value = input[index];
  if (iterations < 0) {
    __nanosleep((rank % 32) * 100);
    iterations = -iterations;
  }
  const ProbeType increment = ProbeType((rank % 2 + 1) * 0.015625f);
#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
    value = tl::warp_reduce_sum(value);
    value = ProbeType(value * ProbeType(0.03125f) + increment);
  }
  output[index] = value;
}

extern "C" void JOIN(launch_, PROBE_VARIANT)(const ProbeType *input,
                                             ProbeType *output, dim3 shape,
                                             int blocks, int iterations) {
  JOIN(repeated_reduction_, PROBE_VARIANT)<<<blocks, shape>>>(input, output,
                                                              iterations);
}

__global__ void JOIN(special_reduction_, PROBE_VARIANT)(const ProbeType *input,
                                                        ProbeType *output,
                                                        int op) {
  const ProbeType value = input[threadIdx.x];
  if (op == 0)
    output[threadIdx.x] = tl::warp_reduce(value, tl::SumOp{});
  else if (op == 1)
    output[threadIdx.x] = tl::warp_reduce(value, tl::MinOp{});
  else if (op == 2)
    output[threadIdx.x] = tl::warp_reduce(value, tl::MaxOp{});
  else if (op == 3)
    output[threadIdx.x] = tl::warp_reduce(value, tl::MinOpNan{});
  else
    output[threadIdx.x] = tl::warp_reduce(value, tl::MaxOpNan{});
}

extern "C" void JOIN(launch_special_, PROBE_VARIANT)(const ProbeType *input,
                                                     ProbeType *output,
                                                     int threads, int op) {
  JOIN(special_reduction_, PROBE_VARIANT)<<<1, threads>>>(input, output, op);
}
#else
using Launch = void (*)(const ProbeType *, ProbeType *, dim3, int, int);
extern "C" void launch_baseline(const ProbeType *, ProbeType *, dim3, int, int);
extern "C" void launch_ballot(const ProbeType *, ProbeType *, dim3, int, int);
extern "C" void launch_pruned(const ProbeType *, ProbeType *, dim3, int, int);
extern "C" void launch_special_ballot(const ProbeType *, ProbeType *, int, int);
extern "C" void launch_special_pruned(const ProbeType *, ProbeType *, int, int);

int main() {
  constexpr int blocks = 128;
  constexpr int trials = 21;
  const Launch launches[] = {launch_baseline, launch_ballot, launch_pruned};
  const char *names[] = {"baseline", "ballot", "pruned"};
  const std::vector<dim3> shapes = {
      dim3(32),     dim3(64),      dim3(128), dim3(256), dim3(1024),
      dim3(8, 8),   dim3(4, 8, 2), dim3(1),   dim3(2),   dim3(3),
      dim3(4),      dim3(7),       dim3(8),   dim3(13),  dim3(16),
      dim3(17),     dim3(24),      dim3(31),  dim3(48),  dim3(7, 7),
      dim3(3, 3, 5)};
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
    const int versions = 3 - first_variant;
    std::vector<ProbeType> host_input(count);
    for (size_t index = 0; index < count; ++index) {
      const int rank = index % threads;
      host_input[index] =
          ProbeType((rank % 7 + 1) * 0.125f + (rank / 32 % 3) * 0.125f);
    }
    ProbeType *input, *outputs[3];
    CUDA_CHECK(cudaMalloc(&input, count * sizeof(ProbeType)));
    CUDA_CHECK(cudaMemcpy(input, host_input.data(), count * sizeof(ProbeType),
                          cudaMemcpyHostToDevice));
    for (int variant = first_variant; variant < 3; ++variant)
      CUDA_CHECK(cudaMalloc(&outputs[variant], count * sizeof(ProbeType)));
    for (const int iterations : {1, 1024, -1}) {
      for (int warm = 0; warm < 10; ++warm)
        for (int variant = first_variant; variant < 3; ++variant)
          launches[variant](input, outputs[variant], shape, blocks, iterations);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      std::printf("telemetry,%ux%ux%u,%d,before\n", shape.x, shape.y, shape.z,
                  iterations);
      std::fflush(stdout);
      if (std::system("nvidia-smi "
                      "--query-gpu=clocks.sm,temperature.gpu,utilization.gpu "
                      "--format=csv,noheader") != 0)
        return 2;
      std::vector<float> times[3];
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
          std::printf("sample,%ux%ux%u,%d,%d,%s,%.6f\n", shape.x, shape.y,
                      shape.z, iterations, trial, names[variant], elapsed);
        }
      }
      std::printf("telemetry,%ux%ux%u,%d,after\n", shape.x, shape.y, shape.z,
                  iterations);
      std::fflush(stdout);
      if (std::system("nvidia-smi "
                      "--query-gpu=clocks.sm,temperature.gpu,utilization.gpu "
                      "--format=csv,noheader") != 0)
        return 2;
      std::vector<ProbeType> reference(count), observed(count);
      CUDA_CHECK(cudaMemcpy(reference.data(), outputs[first_variant],
                            count * sizeof(ProbeType), cudaMemcpyDeviceToHost));
      for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(double(reference[index]))) {
          std::fprintf(stderr, "NONFINITE_OUTPUT\n");
          return 1;
        }
        if (iterations == 1 || iterations == -1) {
          const int rank = index % threads;
          const size_t begin = index - rank % 32;
          const size_t end = std::min(begin + 32, index - rank + threads);
          ProbeType sum = ProbeType(0);
          for (size_t member = begin; member < end; ++member)
            sum += host_input[member];
          const ProbeType expected =
              ProbeType(sum * ProbeType(0.03125f) +
                        ProbeType((rank % 2 + 1) * 0.015625f));
          if (reference[index] != expected) {
            std::fprintf(stderr,
                         "ORACLE_MISMATCH: index=%zu expected=%f actual=%f\n",
                         index, double(expected), double(reference[index]));
            return 1;
          }
        }
      }
      for (int variant = first_variant + 1; variant < 3; ++variant) {
        CUDA_CHECK(cudaMemcpy(observed.data(), outputs[variant],
                              count * sizeof(ProbeType),
                              cudaMemcpyDeviceToHost));
        if (std::memcmp(observed.data(), reference.data(),
                        count * sizeof(ProbeType)) != 0) {
          std::fprintf(stderr, "OUTPUT_MISMATCH: %s\n", names[variant]);
          return 1;
        }
      }
      for (int variant = first_variant; variant < 3; ++variant) {
        std::sort(times[variant].begin(), times[variant].end());
        std::printf("summary,%ux%ux%u,%d,%s,min=%.6f,q1=%.6f,median=%.6f,q3=%."
                    "6f,max=%.6f\n",
                    shape.x, shape.y, shape.z, iterations, names[variant],
                    times[variant][0], times[variant][trials / 4],
                    times[variant][trials / 2], times[variant][3 * trials / 4],
                    times[variant][trials - 1]);
      }
    }
    CUDA_CHECK(cudaFree(input));
    for (int variant = first_variant; variant < 3; ++variant)
      CUDA_CHECK(cudaFree(outputs[variant]));
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  const double large = std::is_same_v<ProbeType, double>   ? 9007199254740992.0
                       : std::is_same_v<ProbeType, float>  ? 16777216.0
                       : std::is_same_v<ProbeType, half_t> ? 2048.0
                                                           : 256.0;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<std::vector<double>> patterns = {
      {large, 1.0, -large, 0.125, -0.5, 2.0, 0.25},
      {nan, 2.0, -3.0, nan, 1.0},
      {nan},
      {0.0, -0.0},
      {inf, -inf, 2.0, -3.0}};
  int differential_cases = 0;
  for (int threads = 1; threads <= 128; ++threads) {
    std::vector<ProbeType> input(threads), reference(threads),
        observed(threads);
    ProbeType *device_input, *device_output;
    CUDA_CHECK(cudaMalloc(&device_input, threads * sizeof(ProbeType)));
    CUDA_CHECK(cudaMalloc(&device_output, threads * sizeof(ProbeType)));
    for (const auto &pattern : patterns) {
      for (int lane = 0; lane < threads; ++lane)
        input[lane] = ProbeType(pattern[lane % pattern.size()]);
      CUDA_CHECK(cudaMemcpy(device_input, input.data(),
                            threads * sizeof(ProbeType),
                            cudaMemcpyHostToDevice));
      for (int op = 0; op < 5; ++op) {
        launch_special_ballot(device_input, device_output, threads, op);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(reference.data(), device_output,
                              threads * sizeof(ProbeType),
                              cudaMemcpyDeviceToHost));
        launch_special_pruned(device_input, device_output, threads, op);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(observed.data(), device_output,
                              threads * sizeof(ProbeType),
                              cudaMemcpyDeviceToHost));
        if (std::memcmp(reference.data(), observed.data(),
                        threads * sizeof(ProbeType)) != 0) {
          std::fprintf(stderr, "SPECIAL_BIT_MISMATCH: threads=%d op=%d\n",
                       threads, op);
          return 1;
        }
        ++differential_cases;
      }
    }
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
  }
  std::printf("SPECIAL_BIT_DIFFERENTIAL_CASES=%d\n", differential_cases);
  std::puts("COMPARISON_OUTPUTS_AND_ONE_CALL_ORACLE=PASS");
}
#endif
