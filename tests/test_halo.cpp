#include "core/Field.hpp"
#include "mesh/Block.hpp"
#include "mesh/Connectivity.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

using namespace cascade;
using namespace cascade::mesh;
using Catch::Approx;

// ===========================================================================
// Halo exchange.
//
// ROADMAP.md marks periodic/ghost indexing as the most common source of bugs
// in this kind of solver, so this layer is tested on its own rather than only
// through a running case: an analytic field is written into the interior,
// exchanged, and read back out of the ghost cells at the position the mapping
// claims it came from.
// ===========================================================================

namespace {

constexpr double kTol = sizeof(Real) == 4 ? 1e-5 : 1e-12;

// A field with no symmetry, so a wrong index mapping cannot accidentally pass.
Real analytic(const Point& p) {
  return Real(1) + Real(3.1) * p[0] - Real(1.7) * p[1] + Real(0.9) * p[0] * p[1];
}

void fillInterior(const Block& b, Field<Real, 1>& f) {
  f.fill(Real(-1000));  // poison: anything untouched by the exchange shows up
  const auto cells = b.cells();
  auto view = f.view();
  for (Index j = 0; j < b.n1(); ++j)
    for (Index i = 0; i < b.n0(); ++i) view(0, cells(i, j)) = analytic(b.cellCenter(i, j));
}

}  // namespace

TEST_CASE("pitchwise periodicity transports the field and the geometry", "[mesh][halo]") {
  const Index n0 = 8, n1 = 6;
  const Real pitch = Real(0.75);
  const Real dx = Real(0.2);
  const Real dy = pitch / Real(n1);  // the block spans exactly one pitch in y

  std::vector<Block> blocks;
  blocks.push_back(makeCartesianBlock(n0, n1, Real(0), Real(0), dx, dy));

  const auto conns = makePeriodicJ(0, n0, pitch);
  REQUIRE(conns.size() == 2);

  // Geometry first: ghost nodes come from the opposite side, shifted by a pitch.
  exchangeNodes(blocks, conns);
  blocks[0].computeMetrics();

  const Block& b = blocks[0];

  SECTION("ghost cell centres land one pitch away from their donor") {
    for (Index i = 0; i < n0; ++i) {
      const Point ghost = b.cellCenter(i, -1);
      const Point donor = b.cellCenter(i, n1 - 1);
      REQUIRE(ghost[0] == Approx(donor[0]).epsilon(kTol));
      REQUIRE(ghost[1] == Approx(donor[1] - pitch).epsilon(kTol));

      const Point ghostTop = b.cellCenter(i, n1);
      const Point donorTop = b.cellCenter(i, 0);
      REQUIRE(ghostTop[1] == Approx(donorTop[1] + pitch).epsilon(kTol));
    }
  }

  SECTION("ghost volumes match their donors") {
    for (Index i = 0; i < n0; ++i) {
      REQUIRE(b.volume(i, -1) == Approx(b.volume(i, n1 - 1)).epsilon(kTol));
      REQUIRE(b.volume(i, -2) == Approx(b.volume(i, n1 - 2)).epsilon(kTol));
    }
  }

  SECTION("the field is copied into both ghost layers") {
    Field<Real, 1> f(b.cellCount());
    fillInterior(b, f);

    std::vector<FieldView<Real, 1>> views{f.view()};
    exchangeHalos(blocks, conns, views);

    const auto cells = b.cells();
    const auto view = f.view();

    for (Index i = 0; i < n0; ++i) {
      // Layer 1 below <- top interior row; layer 2 below <- next one down.
      REQUIRE(view(0, cells(i, -1)) == Approx(analytic(b.cellCenter(i, n1 - 1))).epsilon(kTol));
      REQUIRE(view(0, cells(i, -2)) == Approx(analytic(b.cellCenter(i, n1 - 2))).epsilon(kTol));
      // and symmetrically above
      REQUIRE(view(0, cells(i, n1)) == Approx(analytic(b.cellCenter(i, 0))).epsilon(kTol));
      REQUIRE(view(0, cells(i, n1 + 1)) == Approx(analytic(b.cellCenter(i, 1))).epsilon(kTol));
    }
  }
}

TEST_CASE("a periodic field is invariant under the exchange", "[mesh][halo]") {
  // The strongest statement: if the interior data really is periodic with the
  // pitch, then after the exchange the ghost value must equal the analytic
  // function evaluated at the GHOST cell's own centre. This catches a sign
  // error in the translation that the donor-value test above would not.
  const Index n0 = 6, n1 = 5;
  const Real pitch = Real(1.0);
  const Real dy = pitch / Real(n1);

  std::vector<Block> blocks;
  blocks.push_back(makeCartesianBlock(n0, n1, Real(0), Real(0), Real(0.3), dy));
  const auto conns = makePeriodicJ(0, n0, pitch);
  exchangeNodes(blocks, conns);
  blocks[0].computeMetrics();
  const Block& b = blocks[0];

  // f is exactly periodic in y with period `pitch`, and not symmetric in x.
  const auto periodicF = [pitch](const Point& p) {
    const Real k = Real(6.283185307179586) / pitch;
    return Real(2) + Real(1.3) * p[0] + Real(0.5) * std::sin(k * p[1]);
  };

  Field<Real, 1> f(b.cellCount());
  f.fill(Real(-1000));
  const auto cells = b.cells();
  auto view = f.view();
  for (Index j = 0; j < n1; ++j)
    for (Index i = 0; i < n0; ++i) view(0, cells(i, j)) = periodicF(b.cellCenter(i, j));

  std::vector<FieldView<Real, 1>> views{f.view()};
  exchangeHalos(blocks, conns, views);

  for (Index i = 0; i < n0; ++i) {
    REQUIRE(view(0, cells(i, -1)) == Approx(periodicF(b.cellCenter(i, -1))).epsilon(kTol));
    REQUIRE(view(0, cells(i, n1)) == Approx(periodicF(b.cellCenter(i, n1))).epsilon(kTol));
  }
}

TEST_CASE("the wake cut reverses the tangential index", "[mesh][halo]") {
  // C-grid wake cut: the first `cut` cells of the jmin side face the last
  // `cut` cells of the same side, running the other way.
  const Index n0 = 10, n1 = 4, cut = 3;

  std::vector<Block> blocks;
  blocks.push_back(makeCartesianBlock(n0, n1, Real(0), Real(0), Real(0.1), Real(0.1)));
  const Block& b = blocks[0];

  Field<Real, 1> f(b.cellCount());
  f.fill(Real(-1000));
  const auto cells = b.cells();
  auto view = f.view();
  // Tag each interior cell with its own i index so the mapping is readable.
  for (Index j = 0; j < n1; ++j)
    for (Index i = 0; i < n0; ++i) view(0, cells(i, j)) = Real(i) + Real(100) * Real(j);

  const auto conns = makeWakeCut(0, n0, cut);
  std::vector<FieldView<Real, 1>> views{f.view()};
  exchangeHalos(blocks, conns, views);

  SECTION("lower cut line reads the upper one backwards") {
    for (Index t = 0; t < cut; ++t) {
      const Index donorI = n0 - 1 - t;
      REQUIRE(view(0, cells(t, -1)) == Approx(Real(donorI)));            // layer 1 <- j = 0
      REQUIRE(view(0, cells(t, -2)) == Approx(Real(donorI) + Real(100)));  // layer 2 <- j = 1
    }
  }

  SECTION("and the mapping is symmetric") {
    for (Index t = n0 - cut; t < n0; ++t) {
      const Index donorI = n0 - 1 - t;
      REQUIRE(view(0, cells(t, -1)) == Approx(Real(donorI)));
    }
  }

  SECTION("cells outside the cut are left untouched") {
    for (Index t = cut; t < n0 - cut; ++t) REQUIRE(view(0, cells(t, -1)) == Approx(-1000.0));
  }
}

TEST_CASE("mapTangential matches the hand-written mapping", "[mesh][halo]") {
  Connection c{};
  c.self = PatchRange{0, Side::JMin, 0, 4};
  c.donor = PatchRange{0, Side::JMin, 6, 10};

  c.reverse = false;
  REQUIRE(mapTangential(c, 0) == 6);
  REQUIRE(mapTangential(c, 3) == 9);

  c.reverse = true;
  REQUIRE(mapTangential(c, 0) == 9);
  REQUIRE(mapTangential(c, 3) == 6);
}

TEST_CASE("an i-direction interface between two blocks", "[mesh][halo]") {
  // Two Cartesian blocks laid side by side: block 0 occupies x in [0,1),
  // block 1 occupies x in [1,2). Block 0's imax ghosts must read block 1's
  // first interior columns and vice versa.
  const Index n0 = 5, n1 = 4;
  const Real dx = Real(0.2), dy = Real(0.25);

  std::vector<Block> blocks;
  blocks.push_back(makeCartesianBlock(n0, n1, Real(0), Real(0), dx, dy));
  blocks.push_back(makeCartesianBlock(n0, n1, Real(n0) * dx, Real(0), dx, dy));

  const auto conns = makeInterface(PatchRange{0, Side::IMax, 0, n1},
                                   PatchRange{1, Side::IMin, 0, n1},
                                   /*reverse=*/false, Vec<Real, NDIM>::zero());

  std::vector<Field<Real, 1>> fields;
  fields.emplace_back(blocks[0].cellCount());
  fields.emplace_back(blocks[1].cellCount());
  for (Size b = 0; b < 2; ++b) fillInterior(blocks[b], fields[b]);

  std::vector<FieldView<Real, 1>> views{fields[0].view(), fields[1].view()};
  exchangeHalos(blocks, conns, views);

  const auto c0 = blocks[0].cells();
  const auto c1 = blocks[1].cells();

  for (Index j = 0; j < n1; ++j) {
    // Block 0's ghosts hold the analytic field evaluated in block 1 -- and
    // because the blocks are geometrically contiguous, that is the same as
    // evaluating it at block 0's own ghost cell centre.
    REQUIRE(views[0](0, c0(n0, j)) == Approx(analytic(blocks[1].cellCenter(0, j))).epsilon(kTol));
    REQUIRE(views[0](0, c0(n0, j)) ==
            Approx(analytic(blocks[0].cellCenter(n0, j))).epsilon(kTol));
    REQUIRE(views[0](0, c0(n0 + 1, j)) ==
            Approx(analytic(blocks[1].cellCenter(1, j))).epsilon(kTol));

    REQUIRE(views[1](0, c1(-1, j)) == Approx(analytic(blocks[0].cellCenter(n0 - 1, j))).epsilon(kTol));
    REQUIRE(views[1](0, c1(-2, j)) == Approx(analytic(blocks[0].cellCenter(n0 - 2, j))).epsilon(kTol));
  }
}

TEST_CASE("multi-component fields exchange every component", "[mesh][halo]") {
  const Index n0 = 4, n1 = 4;
  std::vector<Block> blocks;
  blocks.push_back(makeCartesianBlock(n0, n1, Real(0), Real(0), Real(0.1), Real(0.1)));
  const Block& b = blocks[0];

  Field<Real, NVAR> f(b.cellCount());
  f.fill(Real(-1000));
  const auto cells = b.cells();
  auto view = f.view();
  for (Index j = 0; j < n1; ++j)
    for (Index i = 0; i < n0; ++i)
      for (int comp = 0; comp < NVAR; ++comp)
        view(comp, cells(i, j)) = Real(comp) * Real(1000) + Real(i) + Real(10) * Real(j);

  const auto conns = makePeriodicJ(0, n0, Real(n1) * Real(0.1));
  std::vector<FieldView<Real, NVAR>> views{f.view()};
  exchangeHalos(blocks, conns, views);

  for (Index i = 0; i < n0; ++i)
    for (int comp = 0; comp < NVAR; ++comp)
      REQUIRE(view(comp, cells(i, -1)) ==
              Approx(view(comp, cells(i, n1 - 1))));
}

TEST_CASE("malformed connections are rejected", "[mesh][halo]") {
  std::vector<Block> blocks;
  blocks.push_back(makeCartesianBlock(4, 4, Real(0), Real(0), Real(0.1), Real(0.1)));

  Connection bad{};
  bad.self = PatchRange{0, Side::JMin, 0, 4};
  bad.donor = PatchRange{0, Side::JMax, 0, 3};  // length mismatch
  REQUIRE_THROWS_AS(validateConnections(blocks, {bad}), std::runtime_error);

  Connection missing{};
  missing.self = PatchRange{0, Side::JMin, 0, 4};
  missing.donor = PatchRange{7, Side::JMax, 0, 4};  // no such block
  REQUIRE_THROWS_AS(validateConnections(blocks, {missing}), std::runtime_error);

  Connection empty{};
  empty.self = PatchRange{0, Side::JMin, 2, 2};
  empty.donor = PatchRange{0, Side::JMax, 2, 2};
  REQUIRE_THROWS_AS(validateConnections(blocks, {empty}), std::runtime_error);
}
