#pragma once

#include "core/Macros.hpp"
#include "core/Types.hpp"

namespace cascade::mesh {

// ===========================================================================
// Index conventions for a structured block.
//
//   i runs streamwise  (around the C-grid, starting from the wake cut)
//   j runs wall-normal (outward from the blade surface)
//
// Interior cells are i in [0, n0), j in [0, n1).  Ghost cells extend that by
// G layers on every side, so the full cell index space is
//
//   i in [-G, n0+G),   j in [-G, n1+G)
//
// G = 2 because 2nd-order MUSCL reconstruction of the face between cells 0
// and 1 needs cell -1, and the face between -1 and 0 needs cell -2.
//
// Node (i,j) is the LOWER-LEFT corner of cell (i,j), so a block with n0 x n1
// cells has (n0+1) x (n1+1) interior nodes; the ghost layers add G more on
// each side.
//
// Faces are named by the direction of their normal:
//
//   i-face (i,j) separates cell (i-1,j) from cell (i,j); its stored normal
//   points towards increasing i.  Cell (i,j) therefore sees -S_i(i,j) on its
//   low side and +S_i(i+1,j) on its high side.
// ===========================================================================

inline constexpr Index kDefaultGhost = 2;

enum class Side : int {
  IMin = 0,
  IMax = 1,
  JMin = 2,
  JMax = 3,
};

inline constexpr int kNumSides = 4;

// 0 for the i-sides, 1 for the j-sides.
CASCADE_HDI constexpr int sideAxis(Side s) { return static_cast<int>(s) / 2; }

// true for IMax / JMax.
CASCADE_HDI constexpr bool sideIsMax(Side s) { return (static_cast<int>(s) % 2) == 1; }

CASCADE_HDI constexpr Side oppositeSide(Side s) {
  return static_cast<Side>(static_cast<int>(s) ^ 1);
}

inline const char* sideName(Side s) {
  switch (s) {
    case Side::IMin: return "imin";
    case Side::IMax: return "imax";
    case Side::JMin: return "jmin";
    case Side::JMax: return "jmax";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// One indexer type serves cells, nodes and both face families; only the
// extents differ.  i is the fastest-varying index, which keeps the streamwise
// direction contiguous in memory -- the direction along which a warp will read
// when the flux kernels are written.
// ---------------------------------------------------------------------------
struct Indexer2D {
  Index dim0 = 0;  // total extent in i, ghosts included
  Index dim1 = 0;  // total extent in j, ghosts included
  Index ghost = 0;

  CASCADE_HDI Index operator()(Index i, Index j) const {
    CASCADE_ASSERT(contains(i, j));
    return (i + ghost) + (j + ghost) * dim0;
  }

  CASCADE_HDI Index count() const { return dim0 * dim1; }

  CASCADE_HDI bool contains(Index i, Index j) const {
    return i >= -ghost && i < dim0 - ghost && j >= -ghost && j < dim1 - ghost;
  }

  // Inclusive index bounds, useful for writing loops without off-by-one noise.
  CASCADE_HDI Index beginI() const { return -ghost; }
  CASCADE_HDI Index endI() const { return dim0 - ghost; }
  CASCADE_HDI Index beginJ() const { return -ghost; }
  CASCADE_HDI Index endJ() const { return dim1 - ghost; }
};

// n0 x n1 interior cells + g ghost layers on each side.
CASCADE_HDI Indexer2D cellIndexer(Index n0, Index n1, Index g) {
  return Indexer2D{n0 + 2 * g, n1 + 2 * g, g};
}

// One more node than cell in each direction.
CASCADE_HDI Indexer2D nodeIndexer(Index n0, Index n1, Index g) {
  return Indexer2D{n0 + 2 * g + 1, n1 + 2 * g + 1, g};
}

// i-faces: one more than cells along i, same as cells along j.
CASCADE_HDI Indexer2D iFaceIndexer(Index n0, Index n1, Index g) {
  return Indexer2D{n0 + 2 * g + 1, n1 + 2 * g, g};
}

// j-faces: same as cells along i, one more along j.
CASCADE_HDI Indexer2D jFaceIndexer(Index n0, Index n1, Index g) {
  return Indexer2D{n0 + 2 * g, n1 + 2 * g + 1, g};
}

}  // namespace cascade::mesh
