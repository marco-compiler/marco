//===- DataRecomputationIndexing.h - Minimal point-set type ---------------===//
//
//===----------------------------------------------------------------------===//
//
// Defines PointSet: a minimal set-of-integer-coordinate-tuples type used by
// DataRecomputation.cpp for constant-index coverage tracking.
//
//===----------------------------------------------------------------------===//

#ifndef MARCO_TRANSFORMS_DATARECOMPUTATIONINDEXING_H
#define MARCO_TRANSFORMS_DATARECOMPUTATIONINDEXING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

/// A single integer-coordinate tuple (one point in N-dimensional index space).
using IndexCoords = llvm::SmallVector<int64_t, 4>;

/// A finite set of integer-coordinate tuples.
///
/// Supports the operations needed by DataRecomputation.cpp:
///   empty(), operator+=, operator-=, overlaps().
struct PointSet {
  llvm::SmallVector<IndexCoords, 4> points;

  bool empty() const { return points.empty(); }

  /// Union: add all points from \p rhs not already present.
  PointSet &operator+=(const PointSet &rhs) {
    for (const auto &pt : rhs.points)
      if (!llvm::is_contained(points, pt))
        points.push_back(pt);
    return *this;
  }

  /// Difference: remove all points that appear in \p rhs.
  PointSet &operator-=(const PointSet &rhs) {
    llvm::erase_if(points, [&](const IndexCoords &pt) {
      return llvm::is_contained(rhs.points, pt);
    });
    return *this;
  }

  /// Returns true if any point in this set also appears in \p other.
  bool overlaps(const PointSet &other) const {
    return llvm::any_of(points, [&](const IndexCoords &pt) {
      return llvm::is_contained(other.points, pt);
    });
  }

  /// Construct a single-point PointSet from a coordinate array.
  static PointSet fromCoords(llvm::ArrayRef<int64_t> coords) {
    PointSet ps;
    ps.points.push_back(IndexCoords(coords));
    return ps;
  }
};

#endif // MARCO_TRANSFORMS_DATARECOMPUTATIONINDEXING_H
