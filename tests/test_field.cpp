#include "core/Field.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

using namespace cascade;
using Catch::Approx;

TEST_CASE("padded stride rounds up to a full cache line", "[field]") {
  constexpr Index perLine = static_cast<Index>(kAlignBytes / sizeof(Real));

  REQUIRE(paddedStride<Real>(1) == perLine);
  REQUIRE(paddedStride<Real>(perLine) == perLine);
  REQUIRE(paddedStride<Real>(perLine + 1) == 2 * perLine);
  REQUIRE(paddedStride<Real>(0) == 0);
}

TEST_CASE("every component array starts on the alignment boundary", "[field]") {
  // This is the whole point of the padding: if only component 0 is aligned,
  // warps reading the other components straddle cache lines.
  Field<Real, NVAR> f(1000);
  auto view = f.view();

  for (int c = 0; c < NVAR; ++c) {
    const auto address = reinterpret_cast<std::uintptr_t>(view.component(c));
    REQUIRE(address % kAlignBytes == 0);
  }
}

TEST_CASE("SoA layout places components in separate contiguous blocks", "[field]") {
  const Index n = 7;
  Field<Real, 3> f(n);
  auto view = f.view();

  for (int c = 0; c < 3; ++c)
    for (Index i = 0; i < n; ++i) view(c, i) = Real(100 * c + i);

  // Component c must be laid out as one run of n consecutive values.
  for (int c = 0; c < 3; ++c) {
    const Real* base = view.component(c);
    for (Index i = 0; i < n; ++i) REQUIRE(base[i] == Approx(Real(100 * c + i)));
  }

  REQUIRE(view.stride == paddedStride<Real>(n));
}

TEST_CASE("load/store move a whole cell state through registers", "[field]") {
  Field<Real, NVAR> f(4);
  auto view = f.view();

  Vec<Real, NVAR> w{};
  for (int c = 0; c < NVAR; ++c) w[c] = Real(c + 1);

  view.store(2, w);
  const auto readBack = view.load(2);

  for (int c = 0; c < NVAR; ++c) REQUIRE(readBack[c] == Approx(w[c]));
}

TEST_CASE("fill covers padding as well as live entries", "[field]") {
  // Padding must be initialised too: reductions and I/O may sweep the padded
  // range, and uninitialised floats there show up as NaN in residual norms.
  Field<Real, 2> f(5);
  f.fill(Real(0));

  const Real* raw = f.data();
  for (Index i = 0; i < f.stride() * 2; ++i) REQUIRE(raw[i] == Approx(0.0));
}

TEST_CASE("Field is move-only and FieldView is trivially copyable", "[field]") {
  STATIC_REQUIRE_FALSE(std::is_copy_constructible<Field<Real, NVAR>>::value);
  STATIC_REQUIRE(std::is_move_constructible<Field<Real, NVAR>>::value);
  // FieldView is passed by value into kernels, so it must survive a memcpy.
  STATIC_REQUIRE(std::is_trivially_copyable<FieldView<Real, NVAR>>::value);
}

TEST_CASE("moved-from field releases its storage", "[field]") {
  Field<Real, 1> a(16);
  a.fill(Real(3));
  Field<Real, 1> b = std::move(a);

  REQUIRE(b.size() == 16);
  REQUIRE(b(0, 0) == Approx(3.0));
  REQUIRE(a.size() == 0);
  REQUIRE(a.data() == nullptr);
}
