#pragma once

#include "core/Macros.hpp"
#include "core/Math.hpp"
#include "core/Types.hpp"

#include <vector>

#if defined(CASCADE_HAVE_OPENMP)
#include <omp.h>
#endif

namespace cascade {

// ===========================================================================
// Execution backends.
//
// A deliberately minimal parallel dispatch layer: one index-space loop and two
// reductions.  That is the entire set of parallel patterns a density-based
// explicit solver needs -- flux/residual/update loops are parallel_for, the
// time-step and residual norms are reductions.  Anything richer (Kokkos et al.)
// would be a dependency paid for and not used.
//
// Every kernel is written once against this interface and compiled for all
// available backends, which is what allows the solver to be developed and
// tested on a machine with no GPU.
// ===========================================================================

namespace exec {
struct Serial {
  static constexpr const char* name() { return "serial"; }
};
struct OpenMP {
  static constexpr const char* name() { return "openmp"; }
};
struct Cuda {
  static constexpr const char* name() { return "cuda"; }
};
}  // namespace exec

// ---------------------------------------------------------------------------
// Serial
// ---------------------------------------------------------------------------
template <typename Functor>
void parallelFor(exec::Serial, Index n, Functor f) {
  for (Index i = 0; i < n; ++i) f(i);
}

template <typename Functor>
auto reduceSum(exec::Serial, Index n, Functor f) -> decltype(f(Index(0))) {
  using R = decltype(f(Index(0)));
  R acc = R(0);
  for (Index i = 0; i < n; ++i) acc += f(i);
  return acc;
}

template <typename Functor>
auto reduceMax(exec::Serial, Index n, Functor f) -> decltype(f(Index(0))) {
  using R = decltype(f(Index(0)));
  R acc = -Limits<R>::huge();
  for (Index i = 0; i < n; ++i) acc = math::max(acc, f(i));
  return acc;
}

// ---------------------------------------------------------------------------
// OpenMP
//
// The reductions accumulate into a per-thread partial array and then sum the
// partials in thread order, instead of using `reduction(+:acc)`.  Two reasons:
//
//   1. Determinism.  Floating point addition is not associative, so a runtime
//      reduction order makes the residual history differ between runs and
//      makes regression tests flaky.  With schedule(static) and a fixed thread
//      count this version is bitwise reproducible.
//   2. Portability.  MSVC still implements OpenMP 2.0, which has no
//      `reduction(max:)` clause at all.
// ---------------------------------------------------------------------------
#if defined(CASCADE_HAVE_OPENMP)

template <typename Functor>
void parallelFor(exec::OpenMP, Index n, Functor f) {
#pragma omp parallel for schedule(static)
  for (Index i = 0; i < n; ++i) f(i);
}

template <typename Functor>
auto reduceSum(exec::OpenMP, Index n, Functor f) -> decltype(f(Index(0))) {
  using R = decltype(f(Index(0)));
  const int nThreads = omp_get_max_threads();
  std::vector<R> partial(static_cast<Size>(nThreads), R(0));
#pragma omp parallel
  {
    const int t = omp_get_thread_num();
    R local = R(0);
#pragma omp for schedule(static)
    for (Index i = 0; i < n; ++i) local += f(i);
    partial[static_cast<Size>(t)] = local;
  }
  R acc = R(0);
  for (int t = 0; t < nThreads; ++t) acc += partial[static_cast<Size>(t)];
  return acc;
}

template <typename Functor>
auto reduceMax(exec::OpenMP, Index n, Functor f) -> decltype(f(Index(0))) {
  using R = decltype(f(Index(0)));
  const int nThreads = omp_get_max_threads();
  std::vector<R> partial(static_cast<Size>(nThreads), -Limits<R>::huge());
#pragma omp parallel
  {
    const int t = omp_get_thread_num();
    R local = -Limits<R>::huge();
#pragma omp for schedule(static)
    for (Index i = 0; i < n; ++i) local = math::max(local, f(i));
    partial[static_cast<Size>(t)] = local;
  }
  R acc = -Limits<R>::huge();
  for (int t = 0; t < nThreads; ++t) acc = math::max(acc, partial[static_cast<Size>(t)]);
  return acc;
}

using DefaultExec = exec::OpenMP;

#else

using DefaultExec = exec::Serial;

#endif  // CASCADE_HAVE_OPENMP

// ---------------------------------------------------------------------------
// Default-backend shorthands used by solver code.
// ---------------------------------------------------------------------------
template <typename Functor>
void parallelFor(Index n, Functor f) {
  parallelFor(DefaultExec{}, n, f);
}

template <typename Functor>
auto reduceSum(Index n, Functor f) -> decltype(f(Index(0))) {
  return reduceSum(DefaultExec{}, n, f);
}

template <typename Functor>
auto reduceMax(Index n, Functor f) -> decltype(f(Index(0))) {
  return reduceMax(DefaultExec{}, n, f);
}

inline const char* defaultBackendName() { return DefaultExec::name(); }

inline int backendThreadCount() {
#if defined(CASCADE_HAVE_OPENMP)
  return omp_get_max_threads();
#else
  return 1;
#endif
}

}  // namespace cascade
