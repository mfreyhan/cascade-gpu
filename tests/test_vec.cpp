#include "core/Vec.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

using cascade::Real;
using cascade::Vec;
using Catch::Approx;

TEST_CASE("Vec arithmetic", "[vec]") {
  const auto a = cascade::vec2<Real>(3, 4);
  const auto b = cascade::vec2<Real>(1, 2);

  SECTION("addition and subtraction") {
    const auto sum = a + b;
    REQUIRE(sum[0] == Approx(4.0));
    REQUIRE(sum[1] == Approx(6.0));

    const auto diff = a - b;
    REQUIRE(diff[0] == Approx(2.0));
    REQUIRE(diff[1] == Approx(2.0));
  }

  SECTION("scalar multiplication is commutative") {
    const auto left = a * Real(2);
    const auto right = Real(2) * a;
    REQUIRE(left[0] == Approx(right[0]));
    REQUIRE(left[1] == Approx(right[1]));
  }

  SECTION("dot product and norm") {
    REQUIRE(dot(a, b) == Approx(11.0));
    REQUIRE(norm(a) == Approx(5.0));
    REQUIRE(normSquared(a) == Approx(25.0));
  }

  SECTION("compound assignment") {
    auto c = a;
    c += b;
    c -= b;
    c *= Real(2);
    REQUIRE(c[0] == Approx(6.0));
    REQUIRE(c[1] == Approx(8.0));
  }
}

TEST_CASE("normalized is unit length and safe at zero", "[vec]") {
  const auto n = normalized(cascade::vec2<Real>(3, 4));
  REQUIRE(norm(n) == Approx(1.0));
  REQUIRE(n[0] == Approx(0.6));
  REQUIRE(n[1] == Approx(0.8));

  // A degenerate face must not produce NaN: downstream checks look for a zero
  // normal, and NaN would propagate silently through the whole residual.
  const auto z = normalized(Vec<Real, 2>::zero());
  REQUIRE(z[0] == Approx(0.0));
  REQUIRE(z[1] == Approx(0.0));
}

TEST_CASE("cross products", "[vec]") {
  // 2D cross gives twice the triangle area spanned by the two edges.
  REQUIRE(cross(cascade::vec2<Real>(1, 0), cascade::vec2<Real>(0, 1)) == Approx(1.0));
  REQUIRE(cross(cascade::vec2<Real>(0, 1), cascade::vec2<Real>(1, 0)) == Approx(-1.0));

  const auto c = cross(cascade::vec3<Real>(1, 0, 0), cascade::vec3<Real>(0, 1, 0));
  REQUIRE(c[2] == Approx(1.0));
}

TEST_CASE("Vec is trivially copyable so it can cross to the device", "[vec]") {
  STATIC_REQUIRE(std::is_trivially_copyable<Vec<Real, 4>>::value);
  STATIC_REQUIRE(sizeof(Vec<Real, 4>) == 4 * sizeof(Real));
}
