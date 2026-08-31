#include "mesh/Block.hpp"

#include "core/Math.hpp"

#include <sstream>

namespace cascade::mesh {

static_assert(NDIM == 2, "Block metrics are 2D for Phase 1; 3D metrics are Phase 2 work");

namespace {

// Rotate a tangent vector by -90 degrees: t = (tx,ty) -> (ty,-tx).
// Applied to the edge running from node (i,j) to node (i,j+1) this yields a
// vector pointing towards increasing i, which is the stored i-face convention.
CASCADE_HDI Point rotateMinus90(const Point& t) { return vec2<Real>(t[1], -t[0]); }

// Rotate by +90: (tx,ty) -> (-ty,tx).  Applied to the edge from node (i,j) to
// node (i+1,j) this points towards increasing j.
CASCADE_HDI Point rotatePlus90(const Point& t) { return vec2<Real>(-t[1], t[0]); }

}  // namespace

Block::Block(Index n0, Index n1, Index ghost) : n0_(n0), n1_(n1), ghost_(ghost) {
  cells_ = cellIndexer(n0_, n1_, ghost_);
  nodes_ = nodeIndexer(n0_, n1_, ghost_);
  iFaces_ = iFaceIndexer(n0_, n1_, ghost_);
  jFaces_ = jFaceIndexer(n0_, n1_, ghost_);

  nodeXY_.resize(nodes_.count());
  volume_.resize(cells_.count());
  faceI_.resize(iFaces_.count());
  faceJ_.resize(jFaces_.count());

  nodeXY_.fill(Real(0));
  volume_.fill(Real(0));
  faceI_.fill(Real(0));
  faceJ_.fill(Real(0));
}

void Block::setNode(Index i, Index j, const Point& p) {
  const Index k = nodes_(i, j);
  for (int d = 0; d < NDIM; ++d) nodeXY_(d, k) = p[d];
}

Point Block::node(Index i, Index j) const {
  const Index k = nodes_(i, j);
  Point p{};
  for (int d = 0; d < NDIM; ++d) p[d] = nodeXY_(d, k);
  return p;
}

Point Block::faceI(Index i, Index j) const {
  const Index k = iFaces_(i, j);
  Point s{};
  for (int d = 0; d < NDIM; ++d) s[d] = faceI_(d, k);
  return s;
}

Point Block::faceJ(Index i, Index j) const {
  const Index k = jFaces_(i, j);
  Point s{};
  for (int d = 0; d < NDIM; ++d) s[d] = faceJ_(d, k);
  return s;
}

Point Block::cellCenter(Index i, Index j) const {
  const Point p00 = node(i, j);
  const Point p10 = node(i + 1, j);
  const Point p11 = node(i + 1, j + 1);
  const Point p01 = node(i, j + 1);
  return (p00 + p10 + p11 + p01) * Real(0.25);
}

void Block::extrapolateGhostNodes() {
  // Mirror the interior about the boundary node line: node(-L) = 2*node(0) -
  // node(L). This gives ghost cells the same size and shape as their interior
  // mirror image, which keeps ghost volumes positive and finite.
  for (Index j = -ghost_; j <= n1_ + ghost_; ++j) {
    for (Index L = 1; L <= ghost_; ++L) {
      const Index jSrc = math::clamp<Index>(j, 0, n1_);
      const Point lo = node(0, jSrc) * Real(2) - node(L, jSrc);
      const Point hi = node(n0_, jSrc) * Real(2) - node(n0_ - L, jSrc);
      if (nodes_.contains(-L, j)) setNode(-L, j, lo);
      if (nodes_.contains(n0_ + L, j)) setNode(n0_ + L, j, hi);
    }
  }
  for (Index i = -ghost_; i <= n0_ + ghost_; ++i) {
    for (Index L = 1; L <= ghost_; ++L) {
      const Index iSrc = math::clamp<Index>(i, 0, n0_);
      const Point lo = node(iSrc, 0) * Real(2) - node(iSrc, L);
      const Point hi = node(iSrc, n1_) * Real(2) - node(iSrc, n1_ - L);
      if (nodes_.contains(i, -L)) setNode(i, -L, lo);
      if (nodes_.contains(i, n1_ + L)) setNode(i, n1_ + L, hi);
    }
  }
}

void Block::computeMetrics() {
  // i-faces: edge from node (i,j) to node (i,j+1), normal towards +i.
  for (Index j = iFaces_.beginJ(); j < iFaces_.endJ(); ++j) {
    for (Index i = iFaces_.beginI(); i < iFaces_.endI(); ++i) {
      const Point s = rotateMinus90(node(i, j + 1) - node(i, j));
      const Index k = iFaces_(i, j);
      for (int d = 0; d < NDIM; ++d) faceI_(d, k) = s[d];
    }
  }

  // j-faces: edge from node (i,j) to node (i+1,j), normal towards +j.
  for (Index j = jFaces_.beginJ(); j < jFaces_.endJ(); ++j) {
    for (Index i = jFaces_.beginI(); i < jFaces_.endI(); ++i) {
      const Point s = rotatePlus90(node(i + 1, j) - node(i, j));
      const Index k = jFaces_(i, j);
      for (int d = 0; d < NDIM; ++d) faceJ_(d, k) = s[d];
    }
  }

  // Cell volume: half the cross product of the quadrilateral's diagonals,
  //     V = 1/2 * (P11 - P00) x (P01 - P10)
  // which is exact for any planar quadrilateral (convex or not) and keeps its
  // sign, so a folded cell reports a negative volume rather than a plausible
  // positive one.
  for (Index j = cells_.beginJ(); j < cells_.endJ(); ++j) {
    for (Index i = cells_.beginI(); i < cells_.endI(); ++i) {
      const Point d1 = node(i + 1, j + 1) - node(i, j);
      const Point d2 = node(i, j + 1) - node(i + 1, j);
      volume_.data()[cells_(i, j)] = Real(0.5) * cross(d1, d2);
    }
  }
}

QualityReport Block::quality() const {
  QualityReport r{};
  r.minVolume = Limits<Real>::huge();
  r.maxVolume = -Limits<Real>::huge();
  r.minFaceAngleDeg = Real(180);

  constexpr Real kRadToDeg = Real(57.29577951308232);

  for (Index j = 0; j < n1_; ++j) {
    for (Index i = 0; i < n0_; ++i) {
      const Real v = volume(i, j);
      r.totalVolume += v;
      r.minVolume = math::min(r.minVolume, v);
      r.maxVolume = math::max(r.maxVolume, v);
      if (v <= Real(0)) ++r.negativeVolumeCells;

      // Mean face lengths in each direction; the cell extent along i is the
      // volume divided by the mean i-face length, and vice versa.
      const Real li = Real(0.5) * (norm(faceI(i, j)) + norm(faceI(i + 1, j)));
      const Real lj = Real(0.5) * (norm(faceJ(i, j)) + norm(faceJ(i, j + 1)));
      if (li > Limits<Real>::tiny() && lj > Limits<Real>::tiny() && v > Real(0)) {
        const Real hi = v / li;
        const Real hj = v / lj;
        const Real ar = math::max(hi, hj) / math::max(math::min(hi, hj), Limits<Real>::tiny());
        r.maxAspectRatio = math::max(r.maxAspectRatio, ar);
      }

      // Angle between the i- and j-face normals. 90 degrees is orthogonal; a
      // value approaching 0 or 180 means a collapsing cell.
      const Point ni = normalized(faceI(i, j) + faceI(i + 1, j));
      const Point nj = normalized(faceJ(i, j) + faceJ(i, j + 1));
      if (norm(ni) > Real(0.5) && norm(nj) > Real(0.5)) {
        const Real c = math::clamp(dot(ni, nj), Real(-1), Real(1));
        const Real angle = kRadToDeg * math::atan2(math::abs(cross(ni, nj)), c);
        r.minFaceAngleDeg = math::min(r.minFaceAngleDeg, angle);
      }
    }
  }

  if (r.minVolume > r.maxVolume) {  // empty block
    r.minVolume = 0;
    r.maxVolume = 0;
  }
  return r;
}

std::string QualityReport::summary() const {
  std::ostringstream out;
  out << "volume [" << minVolume << ", " << maxVolume << "]  total " << totalVolume
      << "\n  negative cells   : " << negativeVolumeCells
      << "\n  max aspect ratio : " << maxAspectRatio
      << "\n  min face angle   : " << minFaceAngleDeg << " deg";
  return out.str();
}

Block makeCartesianBlock(Index n0, Index n1, Real x0, Real y0, Real dx, Real dy, Index ghost) {
  Block b(n0, n1, ghost);
  const auto nodes = b.nodes();
  for (Index j = nodes.beginJ(); j < nodes.endJ(); ++j)
    for (Index i = nodes.beginI(); i < nodes.endI(); ++i)
      b.setNode(i, j, vec2<Real>(x0 + Real(i) * dx, y0 + Real(j) * dy));
  b.computeMetrics();
  return b;
}

}  // namespace cascade::mesh
