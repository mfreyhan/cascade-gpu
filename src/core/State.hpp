#pragma once

#include "core/Macros.hpp"
#include "core/Math.hpp"
#include "core/Types.hpp"
#include "core/Vec.hpp"

namespace cascade {

// ===========================================================================
// Thermodynamic state, in NON-DIMENSIONAL form.
//
// Reference quantities (inlet stagnation conditions and blade chord):
//
//     rho_ref = p01 / (R T01)          a_ref = sqrt(gamma R T01)      L_ref = c
//     p_ref   = rho_ref * a_ref^2      t_ref = L_ref / a_ref
//
// With this scaling the perfect gas law becomes
//
//     p* = rho* T* / gamma      i.e.   R* = 1 / gamma
//
// and every other relation keeps its dimensional form:
//
//     e*  = p* / ((gamma-1) rho*)
//     a*  = sqrt(gamma p* / rho*)
//     E*  = e* + |u*|^2 / 2
//
// Inlet stagnation values are therefore rho01* = 1, T01* = 1, p01* = 1/gamma,
// which is a convenient sanity check when reading solver output.
//
// Only I/O converts to dimensional units.  Nothing inside the solver does.
// ===========================================================================

template <typename T>
struct GasModel {
  T gamma;  // ratio of specific heats: 1.4 for air, ~1.33 for turbine gas

  CASCADE_HDI T gammaMinusOne() const { return gamma - T(1); }
  // Non-dimensional gas constant implied by the scaling above.
  CASCADE_HDI T gasConstant() const { return T(1) / gamma; }

  CASCADE_HDI static GasModel air() { return GasModel{T(1.4)}; }
};

// Primitive variables. Kept as a named struct (not a flat array) because the
// numerics read them by name; conserved variables are a flat Vec because the
// residual loop treats them uniformly.
template <typename T, int Dim>
struct PrimitiveState {
  T rho;
  Vec<T, Dim> u;
  T p;
};

// Conserved variables, ordered as in cascade::vars: [rho, rho*u_i, rho*E].
template <typename T, int Dim>
using ConservedState = Vec<T, Dim + 2>;

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

template <typename T, int Dim>
CASCADE_HDI ConservedState<T, Dim> toConserved(const PrimitiveState<T, Dim>& q,
                                               const GasModel<T>& gas) {
  ConservedState<T, Dim> w{};
  w[vars::RHO] = q.rho;
  for (int d = 0; d < Dim; ++d) w[vars::MOM + d] = q.rho * q.u[d];
  // rho*E = p/(gamma-1) + 0.5 rho |u|^2
  w[Dim + 1] = q.p / gas.gammaMinusOne() + T(0.5) * q.rho * normSquared(q.u);
  return w;
}

// Note on the signatures below: ConservedState<T,Dim> is an alias for
// Vec<T,Dim+2>, and Dim is a non-deduced context inside it. Functions that take
// a conserved state therefore template on the vector length NVars and recover
// Dim = NVars - 2, otherwise every call would need explicit template arguments.
template <typename T, int NVars>
CASCADE_HDI PrimitiveState<T, NVars - 2> toPrimitive(const Vec<T, NVars>& w,
                                                     const GasModel<T>& gas) {
  constexpr int Dim = NVars - 2;
  PrimitiveState<T, Dim> q{};
  q.rho = w[vars::RHO];
  const T invRho = T(1) / q.rho;
  T kinetic = T(0);
  for (int d = 0; d < Dim; ++d) {
    q.u[d] = w[vars::MOM + d] * invRho;
    kinetic += w[vars::MOM + d] * w[vars::MOM + d];
  }
  kinetic *= T(0.5) * invRho;
  q.p = gas.gammaMinusOne() * (w[Dim + 1] - kinetic);
  return q;
}

// ---------------------------------------------------------------------------
// Thermodynamic relations
// ---------------------------------------------------------------------------

template <typename T, int NVars>
CASCADE_HDI T pressure(const Vec<T, NVars>& w, const GasModel<T>& gas) {
  constexpr int Dim = NVars - 2;
  T momSq = T(0);
  for (int d = 0; d < Dim; ++d) momSq += w[vars::MOM + d] * w[vars::MOM + d];
  return gas.gammaMinusOne() * (w[Dim + 1] - T(0.5) * momSq / w[vars::RHO]);
}

// a = sqrt(gamma p / rho)
template <typename T>
CASCADE_HDI T soundSpeed(T rho, T p, const GasModel<T>& gas) {
  return math::sqrt(gas.gamma * p / rho);
}

// T* = gamma p* / rho*   (from p* = rho* T* / gamma)
template <typename T>
CASCADE_HDI T temperature(T rho, T p, const GasModel<T>& gas) {
  return gas.gamma * p / rho;
}

// H = (rho*E + p) / rho, the quantity that is constant across a Roe average.
template <typename T, int NVars>
CASCADE_HDI T totalEnthalpy(const Vec<T, NVars>& w, T p) {
  return (w[NVars - 1] + p) / w[vars::RHO];
}

template <typename T, int Dim>
CASCADE_HDI T machNumber(const PrimitiveState<T, Dim>& q, const GasModel<T>& gas) {
  return norm(q.u) / soundSpeed(q.rho, q.p, gas);
}

// Isentropic stagnation-to-static pressure ratio: p0/p = (1 + (g-1)/2 M^2)^(g/(g-1))
template <typename T>
CASCADE_HDI T stagnationPressureRatio(T mach, const GasModel<T>& gas) {
  const T gm1 = gas.gammaMinusOne();
  return math::pow(T(1) + T(0.5) * gm1 * mach * mach, gas.gamma / gm1);
}

// Isentropic Mach number from a measured surface pressure and the inlet
// stagnation pressure. This is the primary quantity compared against cascade
// experiments, so it lives here rather than in post-processing.
template <typename T>
CASCADE_HDI T isentropicMach(T p, T p0, const GasModel<T>& gas) {
  const T gm1 = gas.gammaMinusOne();
  const T ratio = math::pow(p0 / p, gm1 / gas.gamma) - T(1);
  return math::sqrt(math::max(T(2) * ratio / gm1, T(0)));
}

// ---------------------------------------------------------------------------
// Physical admissibility. Called from debug builds and from the solver's
// divergence detector: a negative density or pressure means the run is dead,
// and catching it at the cell that produced it saves hours of bisecting.
// ---------------------------------------------------------------------------
template <typename T, int NVars>
CASCADE_HDI bool isPhysical(const Vec<T, NVars>& w, const GasModel<T>& gas) {
  return w[vars::RHO] > T(0) && pressure(w, gas) > T(0);
}

// Convenience aliases at the project's compile-time dimension and precision.
using Prim = PrimitiveState<Real, NDIM>;
using Cons = ConservedState<Real, NDIM>;
using Gas = GasModel<Real>;

}  // namespace cascade
