#pragma once

#include "core/Field.hpp"
#include "core/Types.hpp"
#include "core/Vec.hpp"
#include "mesh/Indexing.hpp"

#include <string>

namespace cascade::mesh {

using Point = Vec<Real, NDIM>;

// ---------------------------------------------------------------------------
// Grid quality summary, produced by Block::quality().
//
// The mesh generator (Week 1, D3-5) reports this after elliptic smoothing:
// a negative volume means the smoother folded the grid, and a collapsed face
// angle means the Roe flux will be evaluated on an almost-degenerate face.
// Catching both here is cheaper than debugging a diverged run.
// ---------------------------------------------------------------------------
struct QualityReport {
  Real minVolume = 0;
  Real maxVolume = 0;
  Index negativeVolumeCells = 0;
  Real maxAspectRatio = 0;
  Real minFaceAngleDeg = 0;  // angle between the i- and j-face normals
  Real totalVolume = 0;

  bool valid() const { return negativeVolumeCells == 0 && minVolume > Real(0); }
  std::string summary() const;
};

// ===========================================================================
// A structured block: node coordinates plus the metric quantities derived
// from them.
//
// Stored metrics (the conventions are fixed and must not be changed silently):
//
//   * face normals are AREA-WEIGHTED outward vectors S = n*A, never unit
//     normals.  The finite volume update needs the product, and storing it
//     avoids a normalise-then-multiply round trip in every flux evaluation.
//     In 2D, |S| is the edge length times unit span.
//   * cell volumes are stored, not recomputed.
//
// Metrics are computed over the ghost layers as well, so reconstruction into
// ghost cells has geometry to work with.  That requires ghost NODE coordinates
// to be filled first -- for a boundary that is a connection or a periodic
// interface, exchangeNodes() in Connectivity.hpp does it; for a physical
// boundary, extrapolateGhostNodes() mirrors them.
// ===========================================================================
class Block {
 public:
  Block() = default;
  Block(Index n0, Index n1, Index ghost = kDefaultGhost);

  // ---- extents -----------------------------------------------------------
  Index n0() const { return n0_; }
  Index n1() const { return n1_; }
  Index ghost() const { return ghost_; }
  Index cellCount() const { return cells_.count(); }
  Index interiorCellCount() const { return n0_ * n1_; }

  Indexer2D cells() const { return cells_; }
  Indexer2D nodes() const { return nodes_; }
  Indexer2D iFaces() const { return iFaces_; }
  Indexer2D jFaces() const { return jFaces_; }

  // ---- node coordinates --------------------------------------------------
  void setNode(Index i, Index j, const Point& p);
  Point node(Index i, Index j) const;

  // Reflect interior nodes about the boundary node line to give ghost cells a
  // plausible (mirror-image) geometry. Used on physical boundaries, where
  // there is no donor block to copy from.
  void extrapolateGhostNodes();

  // ---- metrics -----------------------------------------------------------
  // Fills volume, faceI and faceJ over the full ghosted index space from the
  // current node coordinates. Must be called after any node change.
  void computeMetrics();

  Real volume(Index i, Index j) const { return volume_.data()[cells_(i, j)]; }

  // Area-weighted normal of the i-face at (i,j), pointing towards +i.
  Point faceI(Index i, Index j) const;
  // Area-weighted normal of the j-face at (i,j), pointing towards +j.
  Point faceJ(Index i, Index j) const;

  Point cellCenter(Index i, Index j) const;

  QualityReport quality() const;

  // ---- raw field access, for the solver's kernels -------------------------
  FieldView<const Real, 1> volumeView() const { return volume_.view(); }
  FieldView<const Real, NDIM> faceIView() const { return faceI_.view(); }
  FieldView<const Real, NDIM> faceJView() const { return faceJ_.view(); }
  FieldView<const Real, NDIM> nodeView() const { return nodeXY_.view(); }
  FieldView<Real, NDIM> nodeView() { return nodeXY_.view(); }

 private:
  Index n0_ = 0;
  Index n1_ = 0;
  Index ghost_ = kDefaultGhost;

  Indexer2D cells_{};
  Indexer2D nodes_{};
  Indexer2D iFaces_{};
  Indexer2D jFaces_{};

  Field<Real, NDIM> nodeXY_;  // node coordinates, SoA
  Field<Real, 1> volume_;     // cell volumes
  Field<Real, NDIM> faceI_;   // area-weighted normals of the i-faces
  Field<Real, NDIM> faceJ_;   // area-weighted normals of the j-faces
};

// Convenience generator used by tests and by the first verification cases.
Block makeCartesianBlock(Index n0, Index n1, Real x0, Real y0, Real dx, Real dy,
                         Index ghost = kDefaultGhost);

}  // namespace cascade::mesh
