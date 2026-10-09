#include <tl_templates/cuda/reduce.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { auto e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "CUDA_ERROR: %s\n", cudaGetErrorString(e)); \
  std::exit(2); } } while (false)

template <typename T, typename Op>
__global__ void reduce_kernel(const T *input, T *output, Op op) {
  unsigned tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  output[tid] = tl::warp_reduce(input[tid], op);
}

static unsigned failures = 0, cases = 0;

template <typename T, typename Op, typename HostOp>
void check(const char *type, const char *name, dim3 shape, Op op, HostOp host_op,
           bool negative = false, bool special = false) {
  unsigned n = shape.x * shape.y * shape.z;
  std::vector<T> input(n), output(n);
  for (unsigned i = 0; i < n; ++i) {
    if constexpr (std::is_integral_v<T>) {
      uint64_t value = (i % 13) + 1;
      if constexpr (sizeof(T) == 8) value += (1ULL << 40);
      input[i] = T(value);
    } else {
      input[i] = T(negative ? -float(i % 2 + 1) : float(i % 2 + 1));
      if (special && i % 17 == 0) input[i] = T(INFINITY);
    }
  }
  T *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  reduce_kernel<<<1, shape>>>(d_input, d_output, op);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost));
  for (unsigned first = 0; first < n; first += 32) {
    T expected = input[first];
    for (unsigned i = first + 1; i < std::min(first + 32, n); ++i)
      expected = host_op(expected, input[i]);
    for (unsigned i = first; i < std::min(first + 32, n); ++i) {
      bool equal = output[i] == expected;
      if (!equal) {
        if (failures < 20)
          std::printf("SEMANTIC_FAILURE %s %s shape=%u,%u,%u lane=%u got=%.17g expected=%.17g\n",
                      type, name, shape.x, shape.y, shape.z, i,
                      double(output[i]), double(expected));
        ++failures;
      }
    }
  }
  CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_output)); ++cases;
}

template <typename T> void check_type(const char *name, dim3 shape) {
  check<T>(name, "sum", shape, tl::SumOp{}, [](T a, T b) { return T(a+b); });
  check<T>(name, "min", shape, tl::MinOp{}, [](T a, T b) { return b < a ? b : a; });
  check<T>(name, "max", shape, tl::MaxOp{}, [](T a, T b) { return a < b ? b : a; }, true);
  if constexpr (std::is_integral_v<T>) {
    check<T>(name, "and", shape, tl::BitAndOp{}, [](T a, T b) { return T(a & b); });
    check<T>(name, "or", shape, tl::BitOrOp{}, [](T a, T b) { return T(a | b); });
    check<T>(name, "xor", shape, tl::BitXorOp{}, [](T a, T b) { return T(a ^ b); });
  } else {
    check<T>(name, "min-inf", shape, tl::MinOp{}, [](T a, T b) { return b < a ? b : a; }, false, true);
  }
}

int main() {
  cudaDeviceProp device;
  CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
  std::printf("GPU: %s sm%d%d\n", device.name, device.major, device.minor);
  std::vector<dim3> shapes;
  for (unsigned n = 1; n <= 128; ++n) shapes.emplace_back(n, 1, 1);
  for (auto shape : {dim3(7,7), dim3(5,9), dim3(9,5), dim3(3,3,5),
                     dim3(2,9,2), dim3(16,3), dim3(17,2), dim3(8,4,2), dim3(1024)})
    shapes.push_back(shape);
  for (auto shape : shapes) {
    check_type<float>("f32", shape); check_type<double>("f64", shape);
    check_type<half_t>("f16", shape); check_type<bfloat16_t>("bf16", shape);
    check_type<int32_t>("i32", shape); check_type<uint32_t>("u32", shape);
    check_type<int64_t>("i64", shape); check_type<uint64_t>("u64", shape);
  }
  std::printf("CASES=%u FAILURES=%u\n", cases, failures);
  return failures ? 1 : 0;
}
