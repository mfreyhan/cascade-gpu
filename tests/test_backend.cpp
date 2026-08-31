#include "core/Backend.hpp"
#include "core/Field.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace cascade;
using Catch::Approx;

TEST_CASE("parallelFor visits every index exactly once", "[backend]") {
  const Index n = 10000;
  std::vector<int> hits(static_cast<Size>(n), 0);

  parallelFor(n, [&](Index i) { hits[static_cast<Size>(i)] += 1; });

  for (Index i = 0; i < n; ++i) REQUIRE(hits[static_cast<Size>(i)] == 1);
}

TEST_CASE("reductions match their closed forms", "[backend]") {
  const Index n = 1000;

  // sum of i = n(n-1)/2
  const Real sum = reduceSum(n, [](Index i) { return Real(i); });
  REQUIRE(sum == Approx(Real(n) * Real(n - 1) / Real(2)).epsilon(1e-5));

  const Real maxValue = reduceMax(n, [](Index i) { return Real(i) * Real(0.5); });
  REQUIRE(maxValue == Approx(Real(n - 1) * Real(0.5)).epsilon(1e-6));
}

TEST_CASE("reduceMax handles all-negative data", "[backend]") {
  // The identity element is -huge, not zero: a max over negative values (a
  // residual expressed in log units, for instance) must not return 0.
  const Real maxValue = reduceMax(Index(16), [](Index i) { return Real(-1) - Real(i); });
  REQUIRE(maxValue == Approx(-1.0));
}

TEST_CASE("parallel backend agrees with serial", "[backend]") {
  const Index n = 4096;
  Field<Real, 1> f(n);
  auto view = f.view();
  for (Index i = 0; i < n; ++i) view(0, i) = Real(1) / Real(i + 1);

  const auto kernel = [view](Index i) { return view(0, i); };

  const Real serial = reduceSum(exec::Serial{}, n, kernel);
  const Real dflt = reduceSum(n, kernel);

  // Not bitwise equal in general (different summation order), but the
  // deterministic partial-array reduction keeps them within rounding.
  REQUIRE(dflt == Approx(serial).epsilon(1e-5));
}

TEST_CASE("reductions are reproducible run to run", "[backend]") {
  // Regression tests compare residual histories, so a reduction that changes
  // its answer between identical runs would make them flaky.
  const Index n = 8192;
  const auto kernel = [](Index i) { return Real(1) / Real(i * i + 1); };

  const Real first = reduceSum(n, kernel);
  for (int trial = 0; trial < 5; ++trial) REQUIRE(reduceSum(n, kernel) == first);
}

TEST_CASE("parallelFor writes into a field correctly", "[backend]") {
  const Index n = 2048;
  Field<Real, NVAR> f(n);
  auto view = f.view();

  parallelFor(n, [view](Index i) {
    Vec<Real, NVAR> w{};
    for (int c = 0; c < NVAR; ++c) w[c] = Real(c) + Real(i);
    view.store(i, w);
  });

  for (Index i = 0; i < n; i += 257)
    for (int c = 0; c < NVAR; ++c) REQUIRE(view(c, i) == Approx(Real(c) + Real(i)));
}
