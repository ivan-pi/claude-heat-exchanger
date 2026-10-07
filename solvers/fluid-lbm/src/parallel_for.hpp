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
// parallel_reduce(n, f) returns sum_i f(i); the summand type is deduced from f and may be
// a plain struct with operator+ (value-initialised to zero), so several sums share one pass.
//
// Rule for kernels: capture by value only ([=]), never `this` and never references to
// host-stack objects -- otherwise the lambda is not GPU-safe.
#pragma once

#include <algorithm>
#include <cstddef>
#include <new>
#include <vector>

#if defined(LBM_BACKEND_CUDA)
#include <cuda_runtime.h>
#define LBM_HD __host__ __device__
#define LBM_LAMBDA [=] __host__ __device__
#else
#define LBM_HD
#define LBM_LAMBDA [=]
#endif

// Full unrolling of the short loops over the lattice directions, so that the direction
// tables and weights fold into constants.
#if defined(__NVCOMPILER) || defined(__CUDACC__)
#define LBM_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define LBM_UNROLL _Pragma("GCC unroll 9")
#else
#define LBM_UNROLL
#endif

#if defined(LBM_BACKEND_STDPAR)
#include <execution>
#include <functional>
#include <iterator>
#include <numeric>
#endif

namespace lbm {

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
// parallel_for / parallel_reduce / device_sync
// ---------------------------------------------------------------------------------------
#if defined(LBM_BACKEND_CUDA)

constexpr int kBlock = 256; // threads per block, also the shared-memory size of the reduction

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
  if (n > 0) {
    parallel_for_kernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(n, f);
  }
}

// Grid-stride partial sums, one per block; the host adds the partials (deterministic, no atomics).
template <class T, class F>
__global__ void reduce_kernel(int n, F f, T *partial)
{
  __shared__ T buf[kBlock];
  T s{};
  for (int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x); i < n;
       i += static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x)) {
    s = s + f(i);
  }
  buf[threadIdx.x] = s;
  __syncthreads();
  for (unsigned w = blockDim.x / 2; w > 0; w >>= 1) {
    if (threadIdx.x < w) {
      buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x + w];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = buf[0];
  }
}

// Sum of f(i), i in [0, n). Blocking (used for diagnostics only).
template <class F>
inline auto parallel_reduce(int n, F f) -> decltype(f(0))
{
  using T                   = decltype(f(0));
  constexpr int  kMaxBlocks = 1024;
  static T      *partial    = nullptr; // device buffer, one allocation per summand type
  static std::vector<T> host(kMaxBlocks);
  if (partial == nullptr) {
    cudaMalloc(&partial, kMaxBlocks * sizeof(T));
  }
  const int blocks = std::min(kMaxBlocks, (n + kBlock - 1) / kBlock);
  T         s{};
  if (blocks > 0) {
    reduce_kernel<<<blocks, kBlock>>>(n, f, partial);
    cudaMemcpy(host.data(), partial, blocks * sizeof(T), cudaMemcpyDeviceToHost); // synchronous
    for (int b = 0; b < blocks; ++b) {
      s = s + host[b];
    }
  }
  return s;
}

// Kernels on the default stream are ordered; the host only has to wait before it
// touches managed memory (e.g. before handing interface buffers to preCICE).
inline void device_sync() { cudaDeviceSynchronize(); }

inline const char *backend_name() { return "CUDA"; }

#elif defined(LBM_BACKEND_STDPAR)

// Random-access counting iterator (portable replacement for std::views::iota, whose
// legacy iterator_category is only input_iterator_tag in libstdc++ and thus rejected by
// the parallel algorithms). The full operator set is required by the PSTL.
struct counting_iterator {
  using iterator_category = std::random_access_iterator_tag;
  using value_type        = int;
  using difference_type   = std::ptrdiff_t;
  using pointer           = const int *;
  using reference         = int;

  int i = 0;

  counting_iterator() = default;
  explicit counting_iterator(int v) : i(v) {}

  reference operator*() const { return i; }
  reference operator[](difference_type n) const { return i + static_cast<int>(n); }

  counting_iterator &operator++() { ++i; return *this; }
  counting_iterator operator++(int) { counting_iterator t = *this; ++i; return t; }
  counting_iterator &operator--() { --i; return *this; }
  counting_iterator operator--(int) { counting_iterator t = *this; --i; return t; }
  counting_iterator &operator+=(difference_type n) { i += static_cast<int>(n); return *this; }
  counting_iterator &operator-=(difference_type n) { i -= static_cast<int>(n); return *this; }

  friend counting_iterator operator+(counting_iterator a, difference_type n) { return counting_iterator(a.i + static_cast<int>(n)); }
  friend counting_iterator operator+(difference_type n, counting_iterator a) { return counting_iterator(a.i + static_cast<int>(n)); }
  friend counting_iterator operator-(counting_iterator a, difference_type n) { return counting_iterator(a.i - static_cast<int>(n)); }
  friend difference_type operator-(counting_iterator a, counting_iterator b) { return a.i - b.i; }

  friend bool operator==(counting_iterator a, counting_iterator b) { return a.i == b.i; }
  friend bool operator!=(counting_iterator a, counting_iterator b) { return a.i != b.i; }
  friend bool operator<(counting_iterator a, counting_iterator b) { return a.i < b.i; }
  friend bool operator>(counting_iterator a, counting_iterator b) { return a.i > b.i; }
  friend bool operator<=(counting_iterator a, counting_iterator b) { return a.i <= b.i; }
  friend bool operator>=(counting_iterator a, counting_iterator b) { return a.i >= b.i; }
};

template <class F>
inline void parallel_for(int n, F f)
{
  std::for_each(std::execution::par_unseq, counting_iterator(0), counting_iterator(n), f);
}

template <class F>
inline auto parallel_reduce(int n, F f) -> decltype(f(0))
{
  using T = decltype(f(0));
  return std::transform_reduce(std::execution::par_unseq, counting_iterator(0), counting_iterator(n), T{},
                               std::plus<T>(), f);
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
inline auto parallel_reduce(int n, F f) -> decltype(f(0))
{
  decltype(f(0)) s{};
  for (int i = 0; i < n; ++i) {
    s = s + f(i);
  }
  return s;
}

inline void device_sync() {}

inline const char *backend_name() { return "serial"; }

#endif

} // namespace lbm
