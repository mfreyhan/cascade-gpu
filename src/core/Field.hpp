#pragma once

#include "core/Macros.hpp"
#include "core/Memory.hpp"
#include "core/Types.hpp"
#include "core/Vec.hpp"

#include <utility>

namespace cascade {

// ===========================================================================
// Structure-of-Arrays field storage.
//
// A multi-component field (e.g. the NVAR conserved variables over all cells)
// is stored as NComp separate contiguous arrays inside one allocation:
//
//     [ comp 0 : n values | pad ][ comp 1 : n values | pad ] ...
//       <------- stride ------->
//
// Why SoA and not an array of NVAR-sized structs:
//
//   When 32 consecutive threads of a warp each read component c of their own
//   cell i, SoA gives addresses base + c*stride + i, i.e. 32 consecutive
//   4-byte words = one 128-byte transaction.  AoS gives base + i*NVAR + c,
//   a stride-NVAR gather that touches NVAR times as many cache lines and
//   wastes (NVAR-1)/NVAR of every transaction.  For a memory-bound solver
//   that ratio is the whole performance story.
//
// Why the per-component stride is padded:
//
//   Each component array must itself start on a 128-byte boundary, otherwise
//   only component 0 is aligned and the rest straddle cache lines.
// ===========================================================================

// Round n up so that a component array is a whole number of cache lines.
template <typename T>
CASCADE_HDI constexpr Index paddedStride(Index n) {
  constexpr Index kElemsPerLine = static_cast<Index>(kAlignBytes / sizeof(T));
  return ((n + kElemsPerLine - 1) / kElemsPerLine) * kElemsPerLine;
}

// ---------------------------------------------------------------------------
// Non-owning view. Trivially copyable, so it is passed by value into kernels;
// this is the only field type numerical code should ever see.
// ---------------------------------------------------------------------------
template <typename T, int NComp>
struct FieldView {
  T* CASCADE_RESTRICT data = nullptr;
  Index n = 0;       // number of live entries per component
  Index stride = 0;  // padded distance between components

  static constexpr int components() { return NComp; }

  CASCADE_HDI T& operator()(int comp, Index i) const {
    CASCADE_ASSERT(comp >= 0 && comp < NComp);
    CASCADE_ASSERT(i >= 0 && i < n);
    return data[static_cast<Index>(comp) * stride + i];
  }

  // Single-component convenience accessor (scalar fields).
  CASCADE_HDI T& operator()(Index i) const {
    static_assert(NComp == 1, "operator()(i) is only valid for scalar fields");
    CASCADE_ASSERT(i >= 0 && i < n);
    return data[i];
  }

  // Gather all components of one cell into a register-resident vector.
  CASCADE_HDI Vec<T, NComp> load(Index i) const {
    Vec<T, NComp> w{};
    for (int c = 0; c < NComp; ++c) w[c] = (*this)(c, i);
    return w;
  }

  CASCADE_HDI void store(Index i, const Vec<T, NComp>& w) const {
    for (int c = 0; c < NComp; ++c) (*this)(c, i) = w[c];
  }

  // Raw pointer to one component array; used by reductions and by I/O.
  CASCADE_HDI T* component(int comp) const {
    CASCADE_ASSERT(comp >= 0 && comp < NComp);
    return data + static_cast<Index>(comp) * stride;
  }
};

// ---------------------------------------------------------------------------
// Owning field.
// ---------------------------------------------------------------------------
template <typename T, int NComp, typename Space = HostSpace>
class Field {
 public:
  using value_type = T;
  using space_type = Space;
  static constexpr int kComponents = NComp;

  Field() = default;

  explicit Field(Index n) { resize(n); }

  Field(const Field&) = delete;
  Field& operator=(const Field&) = delete;

  // Explicit move: the implicitly generated one would move the buffer but copy
  // n_/stride_, leaving a moved-from field that reports a non-zero size over a
  // null pointer.
  Field(Field&& other) noexcept
      : storage_(std::move(other.storage_)), n_(other.n_), stride_(other.stride_) {
    other.n_ = 0;
    other.stride_ = 0;
  }

  Field& operator=(Field&& other) noexcept {
    if (this != &other) {
      storage_ = std::move(other.storage_);
      n_ = other.n_;
      stride_ = other.stride_;
      other.n_ = 0;
      other.stride_ = 0;
    }
    return *this;
  }

  void resize(Index n) {
    n_ = n;
    stride_ = paddedStride<T>(n);
    storage_.allocate(stride_ * NComp);
  }

  FieldView<T, NComp> view() { return FieldView<T, NComp>{storage_.data(), n_, stride_}; }

  FieldView<const T, NComp> view() const {
    return FieldView<const T, NComp>{storage_.data(), n_, stride_};
  }

  // Host-side element access. Intentionally absent from FieldView so that a
  // stray host access inside a kernel is a compile error, not a segfault.
  T& operator()(int comp, Index i) { return storage_.data()[comp * stride_ + i]; }
  const T& operator()(int comp, Index i) const { return storage_.data()[comp * stride_ + i]; }

  Index size() const { return n_; }
  Index stride() const { return stride_; }
  Size bytes() const { return storage_.bytes(); }
  T* data() { return storage_.data(); }
  const T* data() const { return storage_.data(); }

  void fill(T value) {
    T* p = storage_.data();
    const Index total = stride_ * NComp;
    for (Index i = 0; i < total; ++i) p[i] = value;
  }

 private:
  Buffer<T, Space> storage_;
  Index n_ = 0;
  Index stride_ = 0;
};

// Project-level aliases.
using ConsField = Field<Real, NVAR>;
using ScalarField = Field<Real, 1>;
using VectorField = Field<Real, NDIM>;

}  // namespace cascade
