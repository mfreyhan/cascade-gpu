#include "core/State.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace cascade;
using Catch::Approx;

namespace {
// Tolerance scaled to the working precision: FP32 carries ~7 decimal digits,
// so a fixed 1e-12 tolerance would make these tests fail for the default build
// rather than catch real errors.
constexpr double kTol = sizeof(Real) == 4 ? 1e-5 : 1e-12;
}  // namespace

TEST_CASE("primitive <-> conserved round trip", "[state]") {
  const Gas gas = Gas::air();

  Prim q{};
  q.rho = Real(1.2);
  q.u[0] = Real(0.35);
  q.u[1] = Real(-0.12);
  q.p = Real(0.9);

  const Cons w = toConserved(q, gas);
  const Prim back = toPrimitive(w, gas);

  REQUIRE(back.rho == Approx(q.rho).epsilon(kTol));
  REQUIRE(back.u[0] == Approx(q.u[0]).epsilon(kTol));
  REQUIRE(back.u[1] == Approx(q.u[1]).epsilon(kTol));
  REQUIRE(back.p == Approx(q.p).epsilon(kTol));
}

TEST_CASE("conserved variable ordering matches cascade::vars", "[state]") {
  const Gas gas = Gas::air();
  Prim q{};
  q.rho = Real(2);
  q.u[0] = Real(3);
  q.u[1] = Real(5);
  q.p = Real(1);

  const Cons w = toConserved(q, gas);
  REQUIRE(w[vars::RHO] == Approx(2.0));
  REQUIRE(w[vars::MOM + 0] == Approx(6.0));
  REQUIRE(w[vars::MOM + 1] == Approx(10.0));
  // rho*E = p/(g-1) + 0.5*rho*|u|^2 = 1/0.4 + 0.5*2*34 = 2.5 + 34
  REQUIRE(w[vars::ENE] == Approx(36.5));
}

TEST_CASE("pressure from conserved state agrees with the conversion", "[state]") {
  const Gas gas = Gas::air();
  Prim q{};
  q.rho = Real(0.8);
  q.u[0] = Real(0.6);
  q.u[1] = Real(0.25);
  q.p = Real(0.71);

  const Cons w = toConserved(q, gas);
  REQUIRE(pressure(w, gas) == Approx(q.p).epsilon(kTol));
}

TEST_CASE("non-dimensional inlet stagnation state is self-consistent", "[state][nondim]") {
  // The scaling defined in State.hpp fixes rho01 = 1, T01 = 1, p01 = 1/gamma
  // and hence a01 = 1. If any of these drift, the non-dimensionalisation and
  // the gas model have gone out of sync.
  const Gas gas = Gas::air();
  const Real rho01 = Real(1);
  const Real p01 = Real(1) / gas.gamma;

  REQUIRE(temperature(rho01, p01, gas) == Approx(1.0).epsilon(kTol));
  REQUIRE(soundSpeed(rho01, p01, gas) == Approx(1.0).epsilon(kTol));
}

TEST_CASE("isentropic relations invert each other", "[state]") {
  const Gas gas = Gas::air();
  const Real mach = Real(0.85);

  const Real p0OverP = stagnationPressureRatio(mach, gas);
  // Feed the ratio back through the cascade post-processing formula.
  const Real recovered = isentropicMach(Real(1), p0OverP, gas);
  REQUIRE(recovered == Approx(mach).epsilon(sizeof(Real) == 4 ? 1e-4 : 1e-10));
}

TEST_CASE("total enthalpy of a stagnant flow equals cp*T0", "[state]") {
  const Gas gas = Gas::air();
  Prim q{};
  q.rho = Real(1);
  q.u = Vec<Real, NDIM>::zero();
  q.p = Real(1) / gas.gamma;

  const Cons w = toConserved(q, gas);
  // H = a^2/(gamma-1) for a stagnant state; with a01 = 1 that is 1/(gamma-1).
  REQUIRE(totalEnthalpy(w, q.p) == Approx(1.0 / (1.4 - 1.0)).epsilon(kTol));
}

TEST_CASE("isPhysical rejects negative density and pressure", "[state]") {
  const Gas gas = Gas::air();
  Cons good{};
  good[vars::RHO] = Real(1);
  good[vars::ENE] = Real(2);
  REQUIRE(isPhysical(good, gas));

  Cons negativeDensity = good;
  negativeDensity[vars::RHO] = Real(-0.1);
  REQUIRE_FALSE(isPhysical(negativeDensity, gas));

  Cons negativePressure = good;
  negativePressure[vars::ENE] = Real(-1);
  REQUIRE_FALSE(isPhysical(negativePressure, gas));
}
