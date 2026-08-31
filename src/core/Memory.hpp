#pragma once

#include "core/Macros.hpp"
#include "core/Types.hpp"

#include <cstdlib>
#include <new>
#include <utility>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace cascade {

// ---------------------------------------------------------------------------
// Memory spaces.
//
// These tags exist from day one even though only HostSpace is implemented, so
// that field types are already parameterised on where their storage lives.
// The CUDA backend (Week 2) adds a Memory<DeviceSpace> specialisation and
// nothing else in the codebase changes.
// ---------------------------------------------------------------------------
struct HostSpace {
  static constexpr const char* name() { return "host"; }
};
struct DeviceSpace {
  static constexpr const char* name() { return "device"; }
};

// Allocation alignment in bytes.  128 B is the CUDA L1/L2 cache line and the
// granularity of a fully coalesced 32-lane transaction; aligning the base of
// every field to it means lane 0 of a warp always starts a fresh line.
inline constexpr Size kAlignBytes = 128;

template <typename Space>
struct Memory;

template <>
struct Memory<HostSpace> {
  static void* allocate(Size bytes) {
    if (bytes == 0) return nullptr;
    const Size padded = ((bytes + kAlignBytes - 1) / kAlignBytes) * kAlignBytes;
#if defined(_MSC_VER)
    void* p = _aligned_malloc(padded, kAlignBytes);
#else
    void* p = std::aligned_alloc(kAlignBytes, padded);
#endif
    if (p == nullptr) throw std::bad_alloc();
    return p;
  }

  static void deallocate(void* p) {
    if (p == nullptr) return;
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
};

// ---------------------------------------------------------------------------
// Owning, move-only, uninitialised typed buffer.
//
// Deliberately not std::vector: vector value-initialises on resize (a hidden
// pass over gigabytes of solution data), cannot be given a memory space, and
// its data pointer cannot be handed to a CUDA kernel.
// ---------------------------------------------------------------------------
template <typename T, typename Space = HostSpace>
class Buffer {
 public:
  using value_type = T;
  using space_type = Space;

  Buffer() = default;

  explicit Buffer(Index count) { allocate(count); }

  ~Buffer() { release(); }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  Buffer(Buffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      release();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  void allocate(Index count) {
    release();
    if (count > 0) {
      ptr_ = static_cast<T*>(Memory<Space>::allocate(static_cast<Size>(count) * sizeof(T)));
      count_ = count;
    }
  }

  void release() {
    Memory<Space>::deallocate(ptr_);
    ptr_ = nullptr;
    count_ = 0;
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  Index size() const { return count_; }
  bool empty() const { return count_ == 0; }
  Size bytes() const { return static_cast<Size>(count_) * sizeof(T); }

 private:
  T* ptr_ = nullptr;
  Index count_ = 0;
};

}  // namespace cascade
