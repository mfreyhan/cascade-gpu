#pragma once

// ---------------------------------------------------------------------------
// Host/device decorators.
//
// Every numerical kernel in this project is written as a free function marked
// CASCADE_HDI so that the *same* source is compiled for the CPU backends and
// for CUDA (architecture decision K2 in ROADMAP.md).  When the translation
// unit is not compiled by nvcc the decorators expand to nothing, which is what
// lets the whole solver build and be tested on a machine without a GPU.
// ---------------------------------------------------------------------------

#if defined(__CUDACC__)
#define CASCADE_HD __host__ __device__
#define CASCADE_DEVICE __device__
#define CASCADE_HOST __host__
#define CASCADE_GLOBAL __global__
#else
#define CASCADE_HD
#define CASCADE_DEVICE
#define CASCADE_HOST
#define CASCADE_GLOBAL
#endif

#if defined(__CUDACC__)
#define CASCADE_FORCEINLINE __forceinline__
#elif defined(_MSC_VER)
#define CASCADE_FORCEINLINE __forceinline
#else
#define CASCADE_FORCEINLINE inline __attribute__((always_inline))
#endif

// Host+device, always inlined.  The default decoration for numerics.
#define CASCADE_HDI CASCADE_HD CASCADE_FORCEINLINE

// ---------------------------------------------------------------------------
// Pointer aliasing.  Field data pointers are marked CASCADE_RESTRICT so the
// compiler may keep reloaded values in registers across stores -- worth a
// measurable amount in flux kernels, where the same field is read many times.
// ---------------------------------------------------------------------------
#if defined(_MSC_VER) && !defined(__CUDACC__)
#define CASCADE_RESTRICT __restrict
#else
#define CASCADE_RESTRICT __restrict__
#endif

// ---------------------------------------------------------------------------
// Assertions.  <cassert> works in device code as well (device-side assert is
// compiled out in release builds, exactly like the host one).
// ---------------------------------------------------------------------------
#include <cassert>
#define CASCADE_ASSERT(cond) assert(cond)

#define CASCADE_UNUSED(x) (void)(x)
