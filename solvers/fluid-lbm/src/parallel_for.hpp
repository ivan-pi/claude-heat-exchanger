// parallel_for.hpp -- minimal execution-backend abstraction for the LBM kernels.
//
// Kernels are written once as lambdas, capturing raw pointers and scalars by value:
//
//     lbm::parallel_for(n, LBM_LAMBDA(int idx) { ... });
//
// Backends (selected at configure time via -DLBM_BACKEND=...):
//
//   STDPAR  std::for_each(std::execution::par_unseq, ...)
//           - nvc++ -stdpar=gpu -gpu=mem:managed : offloaded to the GPU, heap memory is CUDA-managed
//           - g++ + TBB                          : multithreaded on the host (testing)
//   CUDA    __global__ kernel launching the (extended) lambda; memory from cudaMallocManaged
//           - nvc++ -cuda -gpu=mem:managed, or nvcc --extended-lambda          [untested here]
//   SERIAL  plain loop (debugging)
//
// parallel_reduce_sum(n, f) returns sum_i f(i) (std::transform_reduce / CUDA block reduction).
//
// Rule for kernels: capture by value only ([=]), never `this` and never references to
// host-stack objects -- otherwise the lambda is not GPU-safe.
#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <new>
#include <vector>

#if defined(LBM_BACKEND_CUDA)
#include <cuda_runtime.h>
#define LBM_HD __host__ __device__
#define LBM_LAMBDA [=] __host__ __device__
#elif defined(LBM_BACKEND_STDPAR)
#include <algorithm>
#include <execution>
#include <functional>
#include <numeric>
#define LBM_HD
#define LBM_LAMBDA [=]
#else
#define LBM_HD
#define LBM_LAMBDA [=]
#endif

namespace lbm {

// ---------------------------------------------------------------------------------------
// Random-access counting iterator (portable replacement for std::views::iota, whose
// legacy iterator_category is only input_iterator_tag in libstdc++ and thus rejected by
// the parallel algorithms).
// ---------------------------------------------------------------------------------------
struct counting_iterator {
  using iterator_category = std::random_access_iterator_tag;
  using value_type        = int;
  using difference_type   = std::ptrdiff_t;
  using pointer           = const int *;
  using reference         = int;

  int i = 0;

  LBM_HD counting_iterator() = default;
  LBM_HD explicit counting_iterator(int v) : i(v) {}

  LBM_HD reference operator*() const { return i; }
  LBM_HD reference operator[](difference_type n) const { return i + static_cast<int>(n); }

  LBM_HD counting_iterator &operator++() { ++i; return *this; }
  LBM_HD counting_iterator operator++(int) { counting_iterator t = *this; ++i; return t; }
  LBM_HD counting_iterator &operator--() { --i; return *this; }
  LBM_HD counting_iterator operator--(int) { counting_iterator t = *this; --i; return t; }
  LBM_HD counting_iterator &operator+=(difference_type n) { i += static_cast<int>(n); return *this; }
  LBM_HD counting_iterator &operator-=(difference_type n) { i -= static_cast<int>(n); return *this; }

  LBM_HD friend counting_iterator operator+(counting_iterator a, difference_type n) { return counting_iterator(a.i + static_cast<int>(n)); }
  LBM_HD friend counting_iterator operator+(difference_type n, counting_iterator a) { return counting_iterator(a.i + static_cast<int>(n)); }
  LBM_HD friend counting_iterator operator-(counting_iterator a, difference_type n) { return counting_iterator(a.i - static_cast<int>(n)); }
  LBM_HD friend difference_type operator-(counting_iterator a, counting_iterator b) { return a.i - b.i; }

  LBM_HD friend bool operator==(counting_iterator a, counting_iterator b) { return a.i == b.i; }
  LBM_HD friend bool operator!=(counting_iterator a, counting_iterator b) { return a.i != b.i; }
  LBM_HD friend bool operator<(counting_iterator a, counting_iterator b) { return a.i < b.i; }
  LBM_HD friend bool operator>(counting_iterator a, counting_iterator b) { return a.i > b.i; }
  LBM_HD friend bool operator<=(counting_iterator a, counting_iterator b) { return a.i <= b.i; }
  LBM_HD friend bool operator>=(counting_iterator a, counting_iterator b) { return a.i >= b.i; }
};

// ---------------------------------------------------------------------------------------
// Memory: std::vector with an allocator that returns memory accessible from host and
// device. With nvc++ -gpu=mem:managed (or mem:unified on Grace Hopper) every heap
// allocation already is, so the default allocator suffices. The explicit managed
// allocator is only needed for the CUDA backend when built with nvcc.
// ---------------------------------------------------------------------------------------
#if defined(LBM_BACKEND_CUDA) && !defined(__NVCOMPILER)
template <class T>
struct managed_allocator {
  using value_type = T;
  managed_allocator() = default;
  template <class U>
  managed_allocator(const managed_allocator<U> &) {}
  T *allocate(std::size_t n)
  {
    void *p = nullptr;
    if (cudaMallocManaged(&p, n * sizeof(T)) != cudaSuccess) {
      throw std::bad_alloc();
    }
    return static_cast<T *>(p);
  }
  void deallocate(T *p, std::size_t) { cudaFree(p); }
  template <class U>
  bool operator==(const managed_allocator<U> &) const { return true; }
  template <class U>
  bool operator!=(const managed_allocator<U> &) const { return false; }
};
template <class T>
using device_vector = std::vector<T, managed_allocator<T>>;
#else
template <class T>
using device_vector = std::vector<T>;
#endif

// ---------------------------------------------------------------------------------------
// parallel_for / device_sync
// ---------------------------------------------------------------------------------------
#if defined(LBM_BACKEND_CUDA)

template <class F>
__global__ void parallel_for_kernel(int n, F f)
{
  const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (idx < n) {
    f(idx);
  }
}

template <class F>
inline void parallel_for(int n, F f)
{
  if (n <= 0) {
    return;
  }
  constexpr int block = 256;
  parallel_for_kernel<<<(n + block - 1) / block, block>>>(n, f);
}

template <class F>
__global__ void reduce_sum_kernel(int n, F f, double *out)
{
  __shared__ double buf[256];
  const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  buf[threadIdx.x] = idx < n ? f(idx) : 0.0;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      buf[threadIdx.x] += buf[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(out, buf[0]);
  }
}

// Sum of f(i), i in [0, n). Blocking (used for diagnostics only).
template <class F>
inline double parallel_reduce_sum(int n, F f)
{
  static double *acc = nullptr;
  if (acc == nullptr) {
    cudaMallocManaged(&acc, sizeof(double));
  }
  *acc = 0.0;
  if (n > 0) {
    reduce_sum_kernel<<<(n + 255) / 256, 256>>>(n, f, acc);
  }
  cudaDeviceSynchronize();
  return *acc;
}

// Kernels on the default stream are ordered; the host only has to wait before it
// touches managed memory (e.g. before handing interface buffers to preCICE).
inline void device_sync() { cudaDeviceSynchronize(); }

inline const char *backend_name() { return "CUDA"; }

#elif defined(LBM_BACKEND_STDPAR)

template <class F>
inline void parallel_for(int n, F f)
{
  std::for_each(std::execution::par_unseq, counting_iterator(0), counting_iterator(n), f);
}

template <class F>
inline double parallel_reduce_sum(int n, F f)
{
  return std::transform_reduce(std::execution::par_unseq, counting_iterator(0), counting_iterator(n), 0.0,
                               std::plus<double>(), f);
}

// The standard parallel algorithms are synchronous.
inline void device_sync() {}

inline const char *backend_name() { return "stdpar (std::for_each, par_unseq)"; }

#else

template <class F>
inline void parallel_for(int n, F f)
{
  for (int i = 0; i < n; ++i) {
    f(i);
  }
}

template <class F>
inline double parallel_reduce_sum(int n, F f)
{
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    s += f(i);
  }
  return s;
}

inline void device_sync() {}

inline const char *backend_name() { return "serial"; }

#endif

} // namespace lbm
