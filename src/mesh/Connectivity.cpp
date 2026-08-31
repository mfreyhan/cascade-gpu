#include "mesh/Connectivity.hpp"

#include <stdexcept>

namespace cascade::mesh {

void ghostCellIndex(const Block& block, Side side, Index layer, Index t, Index& i, Index& j) {
  switch (side) {
    case Side::IMin: i = -layer;                 j = t;                        break;
    case Side::IMax: i = block.n0() + layer - 1; j = t;                        break;
    case Side::JMin: i = t;                      j = -layer;                   break;
    case Side::JMax: i = t;                      j = block.n1() + layer - 1;   break;
  }
}

void donorCellIndex(const Block& block, Side side, Index layer, Index t, Index& i, Index& j) {
  switch (side) {
    case Side::IMin: i = layer - 1;              j = t;                        break;
    case Side::IMax: i = block.n0() - layer;     j = t;                        break;
    case Side::JMin: i = t;                      j = layer - 1;                break;
    case Side::JMax: i = t;                      j = block.n1() - layer;       break;
  }
}

Index mapTangential(const Connection& c, Index t) {
  const Index offset = t - c.self.begin;
  return c.reverse ? (c.donor.end - 1 - offset) : (c.donor.begin + offset);
}

namespace {

// Node counterpart of mapTangential.
//
// Cells and nodes reverse differently, and getting this wrong is the classic
// off-by-one of a C-grid wake cut. Under reversal, cell `self.begin` maps to
// cell `donor.end-1`; the LEFT node of the first cell is therefore the RIGHT
// node of the last donor cell, i.e. node `donor.end`, not `donor.end-1`.
Index mapTangentialNode(const Connection& c, Index t) {
  const Index offset = t - c.self.begin;
  return c.reverse ? (c.donor.end - offset) : (c.donor.begin + offset);
}

void ghostNodeIndex(const Block& block, Side side, Index layer, Index t, Index& i, Index& j) {
  switch (side) {
    case Side::IMin: i = -layer;             j = t;                    break;
    case Side::IMax: i = block.n0() + layer; j = t;                    break;
    case Side::JMin: i = t;                  j = -layer;               break;
    case Side::JMax: i = t;                  j = block.n1() + layer;   break;
  }
}

void donorNodeIndex(const Block& block, Side side, Index layer, Index t, Index& i, Index& j) {
  switch (side) {
    case Side::IMin: i = layer;                  j = t;                    break;
    case Side::IMax: i = block.n0() - layer;     j = t;                    break;
    case Side::JMin: i = t;                      j = layer;                break;
    case Side::JMax: i = t;                      j = block.n1() - layer;   break;
  }
}

}  // namespace

void validateConnections(const std::vector<Block>& blocks,
                         const std::vector<Connection>& connections) {
  for (const Connection& c : connections) {
    if (c.self.block < 0 || static_cast<Size>(c.self.block) >= blocks.size() ||
        c.donor.block < 0 || static_cast<Size>(c.donor.block) >= blocks.size()) {
      throw std::runtime_error("connection refers to a block index that does not exist");
    }
    if (c.self.length() != c.donor.length()) {
      throw std::runtime_error(
          "connection patches have different lengths (non-conforming interfaces are Phase 2)");
    }
    if (c.self.length() <= 0) throw std::runtime_error("connection patch is empty");
  }
}

std::vector<Connection> makeInterface(const PatchRange& a, const PatchRange& b, bool reverse,
                                      const Vec<Real, NDIM>& translationAtoB) {
  Connection forward{};
  forward.self = a;
  forward.donor = b;
  forward.reverse = reverse;
  forward.translation = translationAtoB;

  Connection backward{};
  backward.self = b;
  backward.donor = a;
  backward.reverse = reverse;
  backward.translation = -translationAtoB;

  return {forward, backward};
}

std::vector<Connection> makePeriodicJ(int block, Index n0, Real pitch) {
  const PatchRange low{block, Side::JMin, 0, n0};
  const PatchRange high{block, Side::JMax, 0, n0};
  // Donor nodes on the jmax side must be shifted DOWN by one pitch to sit
  // below the jmin boundary.
  return makeInterface(low, high, /*reverse=*/false, vec2<Real>(Real(0), -pitch));
}

std::vector<Connection> makeWakeCut(int block, Index n0, Index cutLength) {
  const PatchRange lower{block, Side::JMin, 0, cutLength};
  const PatchRange upper{block, Side::JMin, n0 - cutLength, n0};
  // The two cut lines are geometrically coincident: no translation, but the
  // tangential index runs the other way.
  return makeInterface(lower, upper, /*reverse=*/true, Vec<Real, NDIM>::zero());
}

void exchangeNodes(std::vector<Block>& blocks, const std::vector<Connection>& connections) {
  validateConnections(blocks, connections);

  for (const Connection& c : connections) {
    const Index selfIdx = c.self.block;
    const Index donorIdx = c.donor.block;
    const Index ghost = blocks[static_cast<Size>(selfIdx)].ghost();

    for (Index layer = 1; layer <= ghost; ++layer) {
      // Node ranges are inclusive on both ends: a patch spanning cells
      // [begin, end) touches nodes [begin, end].
      for (Index t = c.self.begin; t <= c.self.end; ++t) {
        Index gi = 0, gj = 0, di = 0, dj = 0;
        ghostNodeIndex(blocks[static_cast<Size>(selfIdx)], c.self.side, layer, t, gi, gj);
        donorNodeIndex(blocks[static_cast<Size>(donorIdx)], c.donor.side, layer,
                       mapTangentialNode(c, t), di, dj);

        const Point donated = blocks[static_cast<Size>(donorIdx)].node(di, dj) + c.translation;
        blocks[static_cast<Size>(selfIdx)].setNode(gi, gj, donated);
      }
    }
  }
}

}  // namespace cascade::mesh
