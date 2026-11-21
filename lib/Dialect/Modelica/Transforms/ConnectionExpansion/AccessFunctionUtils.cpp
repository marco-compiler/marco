#include "marco/Dialect/Modelica/Transforms/ConnectionExpansion/AccessFunctionUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include <limits>

using namespace ::marco::modeling;

namespace mlir::modelica::connection {
std::optional<SimpleResults>
getSimpleResults(const AccessFunction &accessFunction) {
  mlir::AffineMap map = mlir::simplifyAffineMap(accessFunction.getAffineMap());
  SimpleResults results;

  for (mlir::AffineExpr expression : map.getResults()) {
    if (auto constantExpr =
            mlir::dyn_cast<mlir::AffineConstantExpr>(expression)) {
      results.push_back({std::nullopt, constantExpr.getValue()});
      continue;
    }

    if (auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
      results.push_back({dimExpr.getPosition(), 0});
      continue;
    }

    auto binaryExpr = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expression);

    if (!binaryExpr || binaryExpr.getKind() != mlir::AffineExprKind::Add) {
      return std::nullopt;
    }

    auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(binaryExpr.getLHS());

    auto constantExpr =
        mlir::dyn_cast<mlir::AffineConstantExpr>(binaryExpr.getRHS());

    if (!dimExpr || !constantExpr) {
      return std::nullopt;
    }

    results.push_back({dimExpr.getPosition(), constantExpr.getValue()});
  }

  return results;
}

Point evaluate(const AccessFunction &accessFunction, const Point &point) {
  llvm::SmallVector<int64_t, 3> coordinates(point.begin(), point.end());

  llvm::SmallVector<int64_t, 3> result =
      accessFunction.getAffineMap().compose(coordinates);

  return {result};
}

bool hasRepeatedDimensions(const SimpleResults &results,
                           const MultidimensionalRange &range) {
  llvm::SmallVector<unsigned, 3> uses(range.rank(), 0);

  for (const SimpleResult &result : results) {
    if (result.dimension && range[*result.dimension].size() > 1 &&
        ++uses[*result.dimension] > 1) {
      return true;
    }
  }

  return false;
}

IndexSet getImage(const AccessFunction &accessFunction,
                  const IndexSet &indices) {
  IndexSet result;
  std::optional<SimpleResults> simpleResults = getSimpleResults(accessFunction);

  for (const MultidimensionalRange &range :
       llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
    if (!simpleResults || hasRepeatedDimensions(*simpleResults, range)) {
      // The image is not a range.
      // TODO: keep compact once ranges with a step are supported.
      for (Point point : range) {
        result += evaluate(accessFunction, point);
      }

      continue;
    }

    llvm::SmallVector<Range, 3> ranges;

    for (const SimpleResult &simpleResult : *simpleResults) {
      if (simpleResult.dimension) {
        const Range &dimension = range[*simpleResult.dimension];

        ranges.push_back(Range(dimension.getBegin() + simpleResult.offset,
                               dimension.getEnd() + simpleResult.offset));
      } else {
        ranges.push_back(Range(simpleResult.offset, simpleResult.offset + 1));
      }
    }

    result += MultidimensionalRange(ranges);
  }

  return result;
}

MultidimensionalRange getImageBounds(const AccessFunction &accessFunction,
                                     const MultidimensionalRange &range) {
  // The extremes of affine functions on a range are reached on its corners.
  size_t rank = range.rank();
  size_t numOfResults = accessFunction.getNumOfResults();
  llvm::SmallVector<int64_t, 3> lowerBounds(
      numOfResults, std::numeric_limits<int64_t>::max());
  llvm::SmallVector<int64_t, 3> upperBounds(
      numOfResults, std::numeric_limits<int64_t>::min());

  for (size_t corner = 0; corner < (size_t(1) << rank); ++corner) {
    llvm::SmallVector<int64_t, 3> coordinates;

    for (size_t i = 0; i < rank; ++i) {
      coordinates.push_back((corner >> i) & 1 ? range[i].getEnd() - 1
                                              : range[i].getBegin());
    }

    Point image = evaluate(accessFunction, Point(coordinates));

    for (size_t i = 0; i < numOfResults; ++i) {
      lowerBounds[i] = std::min(lowerBounds[i], image[i]);
      upperBounds[i] = std::max(upperBounds[i], image[i]);
    }
  }

  llvm::SmallVector<Range, 3> ranges;

  for (size_t i = 0; i < numOfResults; ++i) {
    ranges.push_back(Range(lowerBounds[i], upperBounds[i] + 1));
  }

  return {ranges};
}

bool isAccessWithin(const AccessFunction &accessFunction,
                    const IndexSet &indices, const IndexSet &allowedIndices) {
  // The bounds of the accessed indices are checked first, so that the
  // accessed indices are computed only if needed.
  bool isWithinBounds = llvm::all_of(
      llvm::make_range(indices.rangesBegin(), indices.rangesEnd()),
      [&](const MultidimensionalRange &range) {
        return allowedIndices.contains(getImageBounds(accessFunction, range));
      });

  return isWithinBounds ||
         allowedIndices.contains(getImage(accessFunction, indices));
}

IndexSet getPreimage(const AccessFunction &accessFunction,
                     const IndexSet &accessedIndices, const IndexSet &indices) {
  IndexSet result;

  if (accessedIndices.empty() || indices.empty()) {
    return result;
  }

  std::optional<SimpleResults> simpleResults = getSimpleResults(accessFunction);

  if (!simpleResults) {
    // TODO: keep compact once ranges with a step are supported.
    for (Point point : indices) {
      if (accessedIndices.contains(evaluate(accessFunction, point))) {
        result += point;
      }
    }

    return result;
  }

  for (const MultidimensionalRange &range :
       llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
    for (const MultidimensionalRange &accessedRange : llvm::make_range(
             accessedIndices.rangesBegin(), accessedIndices.rangesEnd())) {
      // Restrict the range dimension by dimension.
      llvm::SmallVector<int64_t, 3> begins;
      llvm::SmallVector<int64_t, 3> ends;

      for (size_t i = 0, e = range.rank(); i < e; ++i) {
        begins.push_back(range[i].getBegin());
        ends.push_back(range[i].getEnd());
      }

      bool isEmpty = false;

      for (size_t i = 0, e = simpleResults->size(); i < e; ++i) {
        const SimpleResult &simpleResult = (*simpleResults)[i];
        const Range &accessed = accessedRange[i];

        if (!simpleResult.dimension) {
          if (!accessed.contains(simpleResult.offset)) {
            isEmpty = true;
          }

          continue;
        }

        unsigned dimension = *simpleResult.dimension;

        begins[dimension] = std::max(begins[dimension],
                                     accessed.getBegin() - simpleResult.offset);

        ends[dimension] =
            std::min(ends[dimension], accessed.getEnd() - simpleResult.offset);
      }

      llvm::SmallVector<Range, 3> ranges;

      for (size_t i = 0, e = range.rank(); i < e && !isEmpty; ++i) {
        if (begins[i] >= ends[i]) {
          isEmpty = true;
        } else {
          ranges.push_back(Range(begins[i], ends[i]));
        }
      }

      if (!isEmpty) {
        result += MultidimensionalRange(ranges);
      }
    }
  }

  return result;
}

bool agreeOn(const AccessFunction &lhs, const AccessFunction &rhs,
             const IndexSet &indices) {
  if (lhs == rhs) {
    return true;
  }

  // Affine functions are determined by their values on the corner of a
  // range and on its neighbours along each non-degenerate dimension.
  for (const MultidimensionalRange &range :
       llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
    llvm::SmallVector<int64_t, 3> corner;

    for (size_t i = 0, e = range.rank(); i < e; ++i) {
      corner.push_back(range[i].getBegin());
    }

    if (evaluate(lhs, Point(corner)) != evaluate(rhs, Point(corner))) {
      return false;
    }

    for (size_t i = 0, e = range.rank(); i < e; ++i) {
      if (range[i].size() == 1) {
        continue;
      }

      llvm::SmallVector<int64_t, 3> neighbour(corner);
      ++neighbour[i];

      if (evaluate(lhs, Point(neighbour)) != evaluate(rhs, Point(neighbour))) {
        return false;
      }
    }
  }

  return true;
}

bool isInvertibleOn(const AccessFunction &accessFunction,
                    const IndexSet &indices) {
  if (indices.size() == 1) {
    return true;
  }

  if (!getSimpleResults(accessFunction)) {
    // TODO: invert the other affine functions that are injective.
    return false;
  }

  return getImage(accessFunction, indices).size() == indices.size();
}

std::unique_ptr<AccessFunction>
getInverseOn(const AccessFunction &accessFunction,
             const MultidimensionalRange &range) {
  mlir::MLIRContext *context = accessFunction.getContext();
  llvm::SmallVector<mlir::AffineExpr, 3> expressions;

  if (range.size() == 1) {
    // The inverse maps the only accessed indices to the only point.
    for (size_t i = 0, e = range.rank(); i < e; ++i) {
      expressions.push_back(
          mlir::getAffineConstantExpr(range[i].getBegin(), context));
    }

    return AccessFunction::build(mlir::AffineMap::get(
        accessFunction.getNumOfResults(), 0, expressions, context));
  }

  std::optional<SimpleResults> simpleResults = getSimpleResults(accessFunction);
  assert(simpleResults && "Access function not invertible");

  for (size_t i = 0, e = range.rank(); i < e; ++i) {
    auto it = llvm::find_if(*simpleResults, [&](const SimpleResult &result) {
      return result.dimension == i;
    });

    if (it == simpleResults->end()) {
      // The dimension isn't used, and is therefore degenerate.
      assert(range[i].size() == 1 && "Access function not invertible");

      expressions.push_back(
          mlir::getAffineConstantExpr(range[i].getBegin(), context));
    } else {
      size_t resultIndex = std::distance(simpleResults->begin(), it);

      expressions.push_back(mlir::getAffineDimExpr(resultIndex, context) -
                            it->offset);
    }
  }

  return AccessFunction::build(mlir::AffineMap::get(
      accessFunction.getNumOfResults(), 0, expressions, context));
}
std::unique_ptr<AccessFunction> getIdentity(mlir::MLIRContext *context,
                                            size_t rank) {
  return AccessFunction::build(
      mlir::AffineMap::getMultiDimIdentityMap(rank, context));
}
} // namespace mlir::modelica::connection
