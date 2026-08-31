#pragma once

#include "core/Macros.hpp"

#include <cstddef>
#include <cstdint>

namespace cascade {

// ---------------------------------------------------------------------------
// Solver floating point type.
//
// Set from CMake (-DCASCADE_REAL=float|double).  FP32 is the default: the GPUs
// this solver targets are consumer parts (RTX 3090/4090) whose FP64 throughput
// is 1/64 of their FP32 throughput.  Verification cases that measure order of
// accuracy are built separately in FP64.
// ---------------------------------------------------------------------------
#ifndef CASCADE_REAL_TYPE
#define CASCADE_REAL_TYPE float
#endif
using Real = CASCADE_REAL_TYPE;

// ---------------------------------------------------------------------------
// Index type.
//
// 32-bit on purpose: a 2D cascade mesh is O(10^5..10^6) cells, far inside the
// range, and 32-bit indices halve register and address-arithmetic cost on the
// GPU relative to 64-bit ones.  Any change here must be revisited for 3D.
// ---------------------------------------------------------------------------
using Index = std::int32_t;
using Size = std::size_t;

// ---------------------------------------------------------------------------
// Spatial dimension, fixed at compile time (architecture decision K1).
// Physics is written against NDIM, never against a literal 2.
// ---------------------------------------------------------------------------
#ifndef CASCADE_NDIM
#define CASCADE_NDIM 2
#endif
inline constexpr int NDIM = CASCADE_NDIM;
static_assert(NDIM == 2 || NDIM == 3, "NDIM must be 2 or 3");

// Number of mean-flow conserved variables: density + NDIM momenta + total energy.
inline constexpr int NVAR = NDIM + 2;

// ---------------------------------------------------------------------------
// Conserved variable ordering.  Turbulence variables are appended after ENE,
// starting at index NVAR (see turb/).
// ---------------------------------------------------------------------------
namespace vars {
inline constexpr int RHO = 0;         // rho
inline constexpr int MOM = 1;         // rho*u_i, i = 0..NDIM-1
inline constexpr int ENE = NDIM + 1;  // rho*E
}  // namespace vars

// Compile-time numeric limits usable from device code.
template <typename T>
struct Limits;

template <>
struct Limits<float> {
  static CASCADE_HDI constexpr float epsilon() { return 1.19209290e-07f; }
  static CASCADE_HDI constexpr float tiny() { return 1.0e-30f; }
  static CASCADE_HDI constexpr float huge() { return 3.0e+38f; }
};

template <>
struct Limits<double> {
  static CASCADE_HDI constexpr double epsilon() { return 2.220446049250313e-16; }
  static CASCADE_HDI constexpr double tiny() { return 1.0e-300; }
  static CASCADE_HDI constexpr double huge() { return 1.0e+300; }
};

}  // namespace cascade
