#pragma once

#include "core/Field.hpp"
#include "core/Types.hpp"
#include "core/Vec.hpp"
#include "mesh/Block.hpp"
#include "mesh/Indexing.hpp"

#include <vector>

namespace cascade::mesh {

// ===========================================================================
// Block connectivity and halo exchange.
//
// This layer exists in Phase 1 even though the C-grid is a single block,
// because the two things it has to do are already connectivity problems:
//
//   * the WAKE CUT, where the two coincident lines of the C-grid downstream
//     of the trailing edge are neighbours of each other, with the tangential
//     index running in opposite directions;
//   * PITCHWISE PERIODICITY, where the block's jmin and jmax boundaries are
//     neighbours separated by a rigid translation of one blade pitch.
//
// Both are the same operation: copy G layers of a donor's interior cells into
// this patch's ghost cells. In Phase 2 the identical description drives
// multi-GPU halo exchange, with the copy going over P2P instead of memcpy.
//
// Treating them this way rather than as special-cased boundary code is
// architecture decision K3 in ROADMAP.md.
// ===========================================================================

// A contiguous run of cells along one side of one block.
//
// `begin`/`end` are indices in the TANGENTIAL direction of that side: for an
// IMin/IMax side the tangential index is j, for a JMin/JMax side it is i.
struct PatchRange {
  int block = 0;
  Side side = Side::IMin;
  Index begin = 0;  // inclusive
  Index end = 0;    // exclusive

  Index length() const { return end - begin; }
};

// One directed connection: fill `self`'s ghost cells from `donor`'s interior.
// A conforming interface needs two of these, one in each direction.
struct Connection {
  PatchRange self;
  PatchRange donor;

  // true when the two patches run in opposite tangential directions. The wake
  // cut of a C-grid is the canonical case: cell i of the lower cut line faces
  // cell (imax-1-i) of the upper one.
  bool reverse = false;

  // Rigid translation added to donor NODE coordinates to bring them into this
  // patch's frame; one blade pitch for the periodic boundaries, zero for the
  // wake cut.
  //
  // Deliberately NOT applied to state vectors: under a pure translation the
  // momentum components are unchanged. Rotational (annular) periodicity in
  // Phase 2 will need a rotation applied to the momentum, and that belongs
  // here as an explicit transform rather than being smuggled in.
  Vec<Real, NDIM> translation{};
};

// Build the two directed connections of a conforming interface.
std::vector<Connection> makeInterface(const PatchRange& a, const PatchRange& b, bool reverse,
                                      const Vec<Real, NDIM>& translationAtoB);

// Pitchwise periodicity of a single block: jmin <-> jmax, offset by `pitch`
// along y, over the full i range.
std::vector<Connection> makePeriodicJ(int block, Index n0, Real pitch);

// C-grid wake cut: the first `cutLength` cells of the jmin side face the last
// `cutLength` cells of the same side, in reverse order.
std::vector<Connection> makeWakeCut(int block, Index n0, Index cutLength);

// ---------------------------------------------------------------------------
// Index mapping helpers, exposed because the halo tests exercise them
// directly and because the boundary-condition kernels will reuse them.
// ---------------------------------------------------------------------------

// The ghost cell of `block` on `side`, at ghost layer `layer` (1..G) and
// tangential position `t`.
void ghostCellIndex(const Block& block, Side side, Index layer, Index t, Index& i, Index& j);

// The interior cell of `block` on `side` that sits `layer` cells in from the
// boundary (layer 1 is the first interior cell).
void donorCellIndex(const Block& block, Side side, Index layer, Index t, Index& i, Index& j);

// The tangential index in the donor patch that corresponds to tangential index
// `t` in the self patch.
Index mapTangential(const Connection& c, Index t);

// Throws std::runtime_error if any connection refers to a missing block, is
// empty, or pairs patches of different lengths. Call once at setup; the
// exchange itself does not re-check, because it runs every iteration.
void validateConnections(const std::vector<Block>& blocks,
                         const std::vector<Connection>& connections);

// ---------------------------------------------------------------------------
// Halo exchange
// ---------------------------------------------------------------------------

// Copy every component of the given cell fields across every connection.
// `views[b]` is the field of block `b`.
template <int NComp>
void exchangeHalos(const std::vector<Block>& blocks, const std::vector<Connection>& connections,
                   const std::vector<FieldView<Real, NComp>>& views) {
  for (const Connection& c : connections) {
    const Block& self = blocks[static_cast<Size>(c.self.block)];
    const Block& donor = blocks[static_cast<Size>(c.donor.block)];
    const auto selfCells = self.cells();
    const auto donorCells = donor.cells();
    const FieldView<Real, NComp>& selfView = views[static_cast<Size>(c.self.block)];
    const FieldView<Real, NComp>& donorView = views[static_cast<Size>(c.donor.block)];

    for (Index layer = 1; layer <= self.ghost(); ++layer) {
      for (Index t = c.self.begin; t < c.self.end; ++t) {
        Index gi = 0, gj = 0, di = 0, dj = 0;
        ghostCellIndex(self, c.self.side, layer, t, gi, gj);
        donorCellIndex(donor, c.donor.side, layer, mapTangential(c, t), di, dj);

        const Index gk = selfCells(gi, gj);
        const Index dk = donorCells(di, dj);
        for (int comp = 0; comp < NComp; ++comp) selfView(comp, gk) = donorView(comp, dk);
      }
    }
  }
}

// Fill ghost NODE coordinates across every connection, applying each
// connection's translation. Must run before Block::computeMetrics(), otherwise
// ghost cells on a periodic or cut boundary get garbage volumes.
void exchangeNodes(std::vector<Block>& blocks, const std::vector<Connection>& connections);

}  // namespace cascade::mesh
