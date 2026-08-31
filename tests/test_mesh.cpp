#include "mesh/Block.hpp"
#include "mesh/Indexing.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <set>

using namespace cascade;
using namespace cascade::mesh;
using Catch::Approx;

namespace {
constexpr double kTol = sizeof(Real) == 4 ? 1e-5 : 1e-12;

// Geometric tolerance for identities that are exact in real arithmetic
// (metric closure, the divergence-theorem volume) but not in floating point.
constexpr Real kGeomTol = sizeof(Real) == 4 ? Real(1e-5) : Real(1e-12);
}  // namespace

// ===========================================================================
// Indexing
// ===========================================================================

TEST_CASE("index spaces have the right extents", "[mesh][index]") {
  const Index n0 = 7, n1 = 5, g = 2;

  REQUIRE(cellIndexer(n0, n1, g).count() == (n0 + 2 * g) * (n1 + 2 * g));
  REQUIRE(nodeIndexer(n0, n1, g).count() == (n0 + 2 * g + 1) * (n1 + 2 * g + 1));
  REQUIRE(iFaceIndexer(n0, n1, g).count() == (n0 + 2 * g + 1) * (n1 + 2 * g));
  REQUIRE(jFaceIndexer(n0, n1, g).count() == (n0 + 2 * g) * (n1 + 2 * g + 1));
}

TEST_CASE("the cell indexer is a bijection over the ghosted index space", "[mesh][index]") {
  const auto idx = cellIndexer(6, 4, 2);
  std::set<Index> seen;

  for (Index j = idx.beginJ(); j < idx.endJ(); ++j)
    for (Index i = idx.beginI(); i < idx.endI(); ++i) {
      const Index k = idx(i, j);
      REQUIRE(k >= 0);
      REQUIRE(k < idx.count());
      REQUIRE(seen.insert(k).second);  // no collisions
    }

  REQUIRE(static_cast<Index>(seen.size()) == idx.count());
}

TEST_CASE("i is the fastest-varying index", "[mesh][index]") {
  // Streamwise contiguity: consecutive i must be consecutive in memory, which
  // is what lets a warp read a streamwise run coalesced.
  const auto idx = cellIndexer(8, 8, 2);
  REQUIRE(idx(1, 0) - idx(0, 0) == 1);
  REQUIRE(idx(0, 1) - idx(0, 0) == idx.dim0);
}

TEST_CASE("side helpers", "[mesh][index]") {
  REQUIRE(sideAxis(Side::IMin) == 0);
  REQUIRE(sideAxis(Side::JMax) == 1);
  REQUIRE(sideIsMax(Side::IMax));
  REQUIRE_FALSE(sideIsMax(Side::JMin));
  REQUIRE(oppositeSide(Side::IMin) == Side::IMax);
  REQUIRE(oppositeSide(Side::JMax) == Side::JMin);
}

// ===========================================================================
// Metrics
// ===========================================================================

TEST_CASE("Cartesian metrics are exact", "[mesh][metrics]") {
  const Real dx = Real(0.25), dy = Real(0.1);
  const Block b = makeCartesianBlock(4, 3, Real(0), Real(0), dx, dy);

  SECTION("volumes") {
    for (Index j = 0; j < b.n1(); ++j)
      for (Index i = 0; i < b.n0(); ++i) REQUIRE(b.volume(i, j) == Approx(dx * dy).epsilon(kTol));
  }

  SECTION("i-face normals point along +x with magnitude dy") {
    const auto s = b.faceI(2, 1);
    REQUIRE(s[0] == Approx(dy).epsilon(kTol));
    REQUIRE(s[1] == Approx(0.0).margin(1e-6));
  }

  SECTION("j-face normals point along +y with magnitude dx") {
    const auto s = b.faceJ(2, 1);
    REQUIRE(s[0] == Approx(0.0).margin(1e-6));
    REQUIRE(s[1] == Approx(dx).epsilon(kTol));
  }

  SECTION("metrics are computed in the ghost layers too") {
    REQUIRE(b.volume(-1, -1) == Approx(dx * dy).epsilon(kTol));
    REQUIRE(b.volume(b.n0() + 1, b.n1() + 1) == Approx(dx * dy).epsilon(kTol));
  }
}

namespace {
// A deliberately distorted but valid grid: sinusoidally perturbed Cartesian.
Block makeWavyBlock(Index n0, Index n1) {
  Block b(n0, n1);
  const auto nodes = b.nodes();
  for (Index j = nodes.beginJ(); j < nodes.endJ(); ++j)
    for (Index i = nodes.beginI(); i < nodes.endI(); ++i) {
      const Real x = Real(i) / Real(n0);
      const Real y = Real(j) / Real(n1);
      b.setNode(i, j,
                vec2<Real>(x + Real(0.05) * std::sin(Real(3) * y),
                           y + Real(0.07) * std::sin(Real(2) * x)));
    }
  b.computeMetrics();
  return b;
}
}  // namespace

TEST_CASE("the outward face vectors of every cell sum to zero", "[mesh][metrics]") {
  // Discrete form of the closed-surface identity  \oint n dA = 0.  It holds
  // exactly for the rotate-the-edge construction, so any violation means the
  // face normals and the cell they belong to have gone out of step -- the
  // single most damaging class of metric bug, because the scheme silently
  // stops being conservative.
  const Block b = makeWavyBlock(12, 9);

  for (Index j = 0; j < b.n1(); ++j)
    for (Index i = 0; i < b.n0(); ++i) {
      const Point sum = b.faceI(i + 1, j) - b.faceI(i, j) + b.faceJ(i, j + 1) - b.faceJ(i, j);
      REQUIRE(norm(sum) < kGeomTol);
    }
}

TEST_CASE("volume agrees with the divergence theorem", "[mesh][metrics]") {
  // V = (1/d) * sum_f  x_f . S_f   with d = 2 in two dimensions.
  // An independent route to the same number: it uses the face vectors, while
  // computeMetrics() uses the diagonals. Agreement ties the two together.
  const Block b = makeWavyBlock(10, 8);

  for (Index j = 0; j < b.n1(); ++j)
    for (Index i = 0; i < b.n0(); ++i) {
      const Point p00 = b.node(i, j);
      const Point p10 = b.node(i + 1, j);
      const Point p11 = b.node(i + 1, j + 1);
      const Point p01 = b.node(i, j + 1);

      const Point xHighI = (p10 + p11) * Real(0.5);
      const Point xLowI = (p00 + p01) * Real(0.5);
      const Point xHighJ = (p01 + p11) * Real(0.5);
      const Point xLowJ = (p00 + p10) * Real(0.5);

      const Real flux = dot(xHighI, b.faceI(i + 1, j)) - dot(xLowI, b.faceI(i, j)) +
                        dot(xHighJ, b.faceJ(i, j + 1)) - dot(xLowJ, b.faceJ(i, j));

      REQUIRE(Real(0.5) * flux == Approx(b.volume(i, j)).epsilon(kTol));
    }
}

namespace {
// Annular sector. theta DECREASES with i so that (e_i, e_j) is right-handed;
// see the handedness test below.
Block makeAnnularBlock(Index nTheta, Index nR, Real r1, Real r2, Real thetaMax,
                       bool rightHanded = true) {
  Block b(nTheta, nR);
  const auto nodes = b.nodes();
  const Real dTheta = thetaMax / Real(nTheta);
  const Real dR = (r2 - r1) / Real(nR);
  for (Index j = nodes.beginJ(); j < nodes.endJ(); ++j)
    for (Index i = nodes.beginI(); i < nodes.endI(); ++i) {
      const Real r = r1 + Real(j) * dR;
      const Real th = rightHanded ? (thetaMax - Real(i) * dTheta) : (Real(i) * dTheta);
      b.setNode(i, j, vec2<Real>(r * std::cos(th), r * std::sin(th)));
    }
  b.computeMetrics();
  return b;
}
}  // namespace

TEST_CASE("curvilinear volumes match the closed form for an annular sector",
          "[mesh][metrics]") {
  // For a polar grid every cell is a quadrilateral of area
  //     A = 1/2 (r_{j+1}^2 - r_j^2) sin(dTheta)
  // so the exact discrete total is
  //     A_total = 1/2 (r2^2 - r1^2) * nTheta * sin(dTheta)
  // which tends to the analytic sector area as dTheta -> 0. Comparing against
  // the discrete value rather than the analytic one makes this a tight test
  // instead of a "close enough" one.
  const Index nTheta = 32, nR = 6;
  const Real r1 = Real(1), r2 = Real(2), thetaMax = Real(1.2);
  const Real dTheta = thetaMax / Real(nTheta);

  const Block b = makeAnnularBlock(nTheta, nR, r1, r2, thetaMax);
  const QualityReport q = b.quality();

  const Real expected =
      Real(0.5) * (r2 * r2 - r1 * r1) * Real(nTheta) * Real(std::sin(double(dTheta)));

  REQUIRE(q.valid());
  REQUIRE(q.totalVolume == Approx(expected).epsilon(kTol));

  // and it is within 0.1% of the analytic sector area at this resolution
  const Real analytic = Real(0.5) * (r2 * r2 - r1 * r1) * thetaMax;
  REQUIRE(q.totalVolume == Approx(analytic).epsilon(1e-3));
}

TEST_CASE("a left-handed (i,j) grid is reported as negative volume", "[mesh][metrics]") {
  // Handedness convention: cross(e_i, e_j) must be positive, i.e. j (outward
  // wall-normal) must be 90 degrees counter-clockwise from i (streamwise).
  // Around a blade with j pointing away from the surface this makes i run
  // clockwise. If the mesh generator gets it backwards every cell reports a
  // negative volume, which is exactly the diagnostic we want.
  const Block b = makeAnnularBlock(16, 4, Real(1), Real(2), Real(1.0), /*rightHanded=*/false);
  const QualityReport q = b.quality();

  REQUIRE_FALSE(q.valid());
  REQUIRE(q.negativeVolumeCells == b.interiorCellCount());
}

TEST_CASE("quality report on a stretched Cartesian block", "[mesh][metrics]") {
  const Real dx = Real(0.4), dy = Real(0.05);
  const Block b = makeCartesianBlock(5, 5, Real(0), Real(0), dx, dy);
  const QualityReport q = b.quality();

  REQUIRE(q.valid());
  REQUIRE(q.minVolume == Approx(dx * dy).epsilon(kTol));
  REQUIRE(q.maxVolume == Approx(dx * dy).epsilon(kTol));
  REQUIRE(q.negativeVolumeCells == 0);
  REQUIRE(q.maxAspectRatio == Approx(dx / dy).epsilon(kTol));
  REQUIRE(q.minFaceAngleDeg == Approx(90.0).epsilon(1e-4));
  REQUIRE(q.totalVolume == Approx(25 * dx * dy).epsilon(kTol));
}

TEST_CASE("ghost node extrapolation mirrors the interior", "[mesh][metrics]") {
  Block b(4, 4, 2);
  // Fill only the interior nodes; leave the ghosts at zero.
  for (Index j = 0; j <= b.n1(); ++j)
    for (Index i = 0; i <= b.n0(); ++i)
      b.setNode(i, j, vec2<Real>(Real(i) * Real(0.3), Real(j) * Real(0.2)));

  b.extrapolateGhostNodes();
  b.computeMetrics();

  // node(-1, j) should be the mirror image of node(1, j) about node(0, j).
  REQUIRE(b.node(-1, 2)[0] == Approx(-0.3).epsilon(kTol));
  REQUIRE(b.node(-2, 2)[0] == Approx(-0.6).epsilon(kTol));

  // Ghost cells therefore have the same volume as their interior mirror.
  REQUIRE(b.volume(-1, 2) == Approx(0.3 * 0.2).epsilon(kTol));
  REQUIRE(b.quality().valid());
}
