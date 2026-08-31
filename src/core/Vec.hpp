#pragma once

#include "core/Macros.hpp"
#include "core/Math.hpp"
#include "core/Types.hpp"

namespace cascade {

// ---------------------------------------------------------------------------
// Fixed-size vector, used both for geometry (Vec<Real,NDIM>) and for state
// vectors (Vec<Real,NVAR>).  Deliberately an aggregate over a raw array:
//
//  * trivially copyable, so it can be passed by value into CUDA kernels;
//  * no dynamic size, so every loop below unrolls and the whole object lives
//    in registers -- essential for the flux kernels;
//  * no constructors, so it can be a __shared__ variable.
//
// Aggregate initialisation needs the inner brace: Vec<Real,2> n{{nx, ny}}.
// The vec2()/vec3() helpers exist to avoid that noise in call sites.
// ---------------------------------------------------------------------------
template <typename T, int N>
struct Vec {
  T v[N];

  static constexpr int size() { return N; }

  CASCADE_HDI T& operator[](int i) { return v[i]; }
  CASCADE_HDI const T& operator[](int i) const { return v[i]; }

  CASCADE_HDI static Vec filled(T s) {
    Vec r{};
    for (int i = 0; i < N; ++i) r.v[i] = s;
    return r;
  }
  CASCADE_HDI static Vec zero() { return filled(T(0)); }

  CASCADE_HDI Vec& operator+=(const Vec& b) {
    for (int i = 0; i < N; ++i) v[i] += b.v[i];
    return *this;
  }
  CASCADE_HDI Vec& operator-=(const Vec& b) {
    for (int i = 0; i < N; ++i) v[i] -= b.v[i];
    return *this;
  }
  CASCADE_HDI Vec& operator*=(T s) {
    for (int i = 0; i < N; ++i) v[i] *= s;
    return *this;
  }
};

template <typename T>
CASCADE_HDI Vec<T, 2> vec2(T x, T y) {
  return Vec<T, 2>{{x, y}};
}

template <typename T>
CASCADE_HDI Vec<T, 3> vec3(T x, T y, T z) {
  return Vec<T, 3>{{x, y, z}};
}

template <typename T, int N>
CASCADE_HDI Vec<T, N> operator+(const Vec<T, N>& a, const Vec<T, N>& b) {
  Vec<T, N> r{};
  for (int i = 0; i < N; ++i) r.v[i] = a.v[i] + b.v[i];
  return r;
}

template <typename T, int N>
CASCADE_HDI Vec<T, N> operator-(const Vec<T, N>& a, const Vec<T, N>& b) {
  Vec<T, N> r{};
  for (int i = 0; i < N; ++i) r.v[i] = a.v[i] - b.v[i];
  return r;
}

template <typename T, int N>
CASCADE_HDI Vec<T, N> operator-(const Vec<T, N>& a) {
  Vec<T, N> r{};
  for (int i = 0; i < N; ++i) r.v[i] = -a.v[i];
  return r;
}

template <typename T, int N>
CASCADE_HDI Vec<T, N> operator*(const Vec<T, N>& a, T s) {
  Vec<T, N> r{};
  for (int i = 0; i < N; ++i) r.v[i] = a.v[i] * s;
  return r;
}

template <typename T, int N>
CASCADE_HDI Vec<T, N> operator*(T s, const Vec<T, N>& a) {
  return a * s;
}

template <typename T, int N>
CASCADE_HDI Vec<T, N> operator/(const Vec<T, N>& a, T s) {
  return a * (T(1) / s);
}

template <typename T, int N>
CASCADE_HDI T dot(const Vec<T, N>& a, const Vec<T, N>& b) {
  T r = T(0);
  for (int i = 0; i < N; ++i) r += a.v[i] * b.v[i];
  return r;
}

template <typename T, int N>
CASCADE_HDI T normSquared(const Vec<T, N>& a) {
  return dot(a, a);
}

template <typename T, int N>
CASCADE_HDI T norm(const Vec<T, N>& a) {
  return math::sqrt(normSquared(a));
}

// Safe normalisation: face normals of degenerate cells must not produce NaN,
// they must produce a zero vector that later checks can detect.
template <typename T, int N>
CASCADE_HDI Vec<T, N> normalized(const Vec<T, N>& a) {
  const T len = norm(a);
  return len > Limits<T>::tiny() ? a * (T(1) / len) : Vec<T, N>::zero();
}

// 2D cross product, returns the scalar z-component. Used for cell areas.
template <typename T>
CASCADE_HDI T cross(const Vec<T, 2>& a, const Vec<T, 2>& b) {
  return a.v[0] * b.v[1] - a.v[1] * b.v[0];
}

template <typename T>
CASCADE_HDI Vec<T, 3> cross(const Vec<T, 3>& a, const Vec<T, 3>& b) {
  return vec3(a.v[1] * b.v[2] - a.v[2] * b.v[1],
              a.v[2] * b.v[0] - a.v[0] * b.v[2],
              a.v[0] * b.v[1] - a.v[1] * b.v[0]);
}

}  // namespace cascade
