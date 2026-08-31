#pragma once

#include "core/Macros.hpp"

#include <cmath>

// ---------------------------------------------------------------------------
// Host/device math shims.
//
// Purpose: keep numerical code free of host-only <cmath> overload resolution.
// std::sqrt is not callable from device code, and calling ::sqrt on a float
// silently promotes to double -- which on a consumer GPU costs a factor of 64.
// Routing every call through cascade::math makes the precision explicit and
// keeps the same source compiling for both backends.
// ---------------------------------------------------------------------------

namespace cascade::math {

CASCADE_HDI float sqrt(float x) { return ::sqrtf(x); }
CASCADE_HDI double sqrt(double x) { return ::sqrt(x); }

CASCADE_HDI float abs(float x) { return ::fabsf(x); }
CASCADE_HDI double abs(double x) { return ::fabs(x); }

CASCADE_HDI float pow(float x, float y) { return ::powf(x, y); }
CASCADE_HDI double pow(double x, double y) { return ::pow(x, y); }

CASCADE_HDI float exp(float x) { return ::expf(x); }
CASCADE_HDI double exp(double x) { return ::exp(x); }

CASCADE_HDI float log(float x) { return ::logf(x); }
CASCADE_HDI double log(double x) { return ::log(x); }

CASCADE_HDI float sin(float x) { return ::sinf(x); }
CASCADE_HDI double sin(double x) { return ::sin(x); }

CASCADE_HDI float cos(float x) { return ::cosf(x); }
CASCADE_HDI double cos(double x) { return ::cos(x); }

CASCADE_HDI float atan2(float y, float x) { return ::atan2f(y, x); }
CASCADE_HDI double atan2(double y, double x) { return ::atan2(y, x); }

CASCADE_HDI float tanh(float x) { return ::tanhf(x); }
CASCADE_HDI double tanh(double x) { return ::tanh(x); }

// Branch-based min/max rather than std::min/std::max: those are host-only
// before C++14 constexpr relaxations in device compilation and take references,
// which prevents the compiler from keeping the operands in registers.
template <typename T>
CASCADE_HDI T min(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
CASCADE_HDI T max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
CASCADE_HDI T clamp(T x, T lo, T hi) {
  return min(max(x, lo), hi);
}

template <typename T>
CASCADE_HDI T sqr(T x) {
  return x * x;
}

template <typename T>
CASCADE_HDI T sign(T x) {
  return x > T(0) ? T(1) : (x < T(0) ? T(-1) : T(0));
}

}  // namespace cascade::math
