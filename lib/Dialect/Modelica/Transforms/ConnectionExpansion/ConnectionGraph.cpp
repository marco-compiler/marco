#include "marco/Dialect/Modelica/Transforms/ConnectionExpansion/ConnectionGraph.h"
#include "marco/Dialect/Modelica/Transforms/ConnectionExpansion/AccessFunctionUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include <map>
#include <optional>
#include <tuple>

using namespace ::marco::modeling;
using namespace ::mlir::modelica;
using namespace ::mlir::modelica::connection;

//===---------------------------------------------------------------------===//
// Utilities on edges and ranges
//===---------------------------------------------------------------------===//

/// Drop the dimensions of the equation indices of an edge that are not used
/// by any of its access functions. If none is used, the equation indices
/// become the single point {0}.
static void dropUnusedDimensions(Edge &edge) {
  mlir::AffineMap lhsMap = edge.lhsAccess->getAffineMap();
  mlir::AffineMap rhsMap = edge.rhsAccess->getAffineMap();
  unsigned numOfDims = lhsMap.getNumDims();
  assert(rhsMap.getNumDims() == numOfDims);

  llvm::SmallVector<unsigned, 3> usedDimensions;

  for (unsigned i = 0; i < numOfDims; ++i) {
    if (lhsMap.isFunctionOfDim(i) || rhsMap.isFunctionOfDim(i)) {
      usedDimensions.push_back(i);
    }
  }

  if (usedDimensions.size() == numOfDims) {
    return;
  }

  mlir::MLIRContext *context = lhsMap.getContext();
  IndexSet equationIndices;

  if (usedDimensions.empty()) {
    equationIndices = IndexSet(MultidimensionalRange(Range(0, 1)));
  } else {
    for (const MultidimensionalRange &range :
         llvm::make_range(edge.equationIndices.rangesBegin(),
                          edge.equationIndices.rangesEnd())) {
      llvm::SmallVector<Range, 3> ranges;

      for (unsigned dimension : usedDimensions) {
        ranges.push_back(range[dimension]);
      }

      equationIndices += MultidimensionalRange(ranges);
    }
  }

  // The unused dimensions are replaced by a constant, which has no effect.
  llvm::SmallVector<mlir::AffineExpr, 3> replacements(
      numOfDims, mlir::getAffineConstantExpr(0, context));

  for (const auto &usedDimension : llvm::enumerate(usedDimensions)) {
    replacements[usedDimension.value()] =
        mlir::getAffineDimExpr(usedDimension.index(), context);
  }

  unsigned newNumOfDims = std::max<size_t>(usedDimensions.size(), 1);

  auto project = [&](mlir::AffineMap map) {
    return AccessFunction::build(mlir::simplifyAffineMap(
        map.replaceDimsAndSymbols(replacements, {}, newNumOfDims, 0)));
  };

  edge.equationIndices = std::move(equationIndices);
  edge.lhsAccess = project(lhsMap);
  edge.rhsAccess = project(rhsMap);
}

/// Get the dimensions a simple access function depends on.
static llvm::SmallBitVector
getUsedDimensions(const AccessFunction &accessFunction, size_t rank) {
  llvm::SmallBitVector result(rank);
  std::optional<SimpleResults> simpleResults = getSimpleResults(accessFunction);
  assert(simpleResults && "Not a simple access function");

  for (const SimpleResult &simpleResult : *simpleResults) {
    if (simpleResult.dimension) {
      result.set(*simpleResult.dimension);
    }
  }

  return result;
}

/// Restrict a range to the first index of the given dimensions.
static MultidimensionalRange
fixDimensions(const MultidimensionalRange &range,
              const llvm::SmallBitVector &dimensions) {
  llvm::SmallVector<Range, 3> ranges;

  for (size_t i = 0, e = range.rank(); i < e; ++i) {
    if (dimensions[i]) {
      ranges.push_back(Range(range[i].getBegin(), range[i].getBegin() + 1));
    } else {
      ranges.push_back(range[i]);
    }
  }

  return {ranges};
}

/// Call a function on the points of a lexicographically sorted list that
/// belong to a range. The points between the corners of the range are
/// visited either one by one, or row by row (a row fixing all the dimensions
/// but the last one), whichever is cheaper.
static void forEachPointIn(llvm::ArrayRef<Point> sortedPoints,
                           const MultidimensionalRange &range,
                           llvm::function_ref<void(const Point &)> callback) {
  size_t rank = range.rank();
  llvm::SmallVector<int64_t, 3> first;
  llvm::SmallVector<int64_t, 3> last;

  for (size_t i = 0; i < rank; ++i) {
    first.push_back(range[i].getBegin());
    last.push_back(range[i].getEnd() - 1);
  }

  const Point *begin = llvm::lower_bound(sortedPoints, Point(first));
  const Point *end = std::upper_bound(begin, sortedPoints.end(), Point(last));
  size_t numOfRows = range.size() / range[rank - 1].size();

  if (rank == 1 || static_cast<size_t>(end - begin) <= numOfRows) {
    for (const Point *it = begin; it != end; ++it) {
      if (range.contains(*it)) {
        callback(*it);
      }
    }

    return;
  }

  llvm::SmallVector<Range, 3> rowRanges;

  for (size_t i = 0; i + 1 < rank; ++i) {
    rowRanges.push_back(range[i]);
  }

  for (Point row : MultidimensionalRange(rowRanges)) {
    llvm::SmallVector<int64_t, 3> rowFirst(row.begin(), row.end());
    llvm::SmallVector<int64_t, 3> rowLast(row.begin(), row.end());
    rowFirst.push_back(range[rank - 1].getBegin());
    rowLast.push_back(range[rank - 1].getEnd() - 1);

    const Point *rowBegin = std::lower_bound(begin, end, Point(rowFirst));
    const Point *rowEnd = std::upper_bound(rowBegin, end, Point(rowLast));

    for (const Point *it = rowBegin; it != rowEnd; ++it) {
      callback(*it);
    }
  }
}

/// Get the access function mapping all the indices of the given rank to a
/// point.
static std::unique_ptr<AccessFunction>
getConstantAccess(mlir::MLIRContext *context, size_t rank, const Point &point) {
  llvm::SmallVector<mlir::AffineExpr, 3> expressions;

  for (int64_t index : point) {
    expressions.push_back(mlir::getAffineConstantExpr(index, context));
  }

  return AccessFunction::build(
      mlir::AffineMap::get(rank, 0, expressions, context));
}

//===---------------------------------------------------------------------===//
// ConnectionGraph
//===---------------------------------------------------------------------===//

namespace mlir::modelica::connection {
ConnectionGraph::ConnectionGraph(mlir::MLIRContext *context)
    : context(context) {}

VertexId ConnectionGraph::addVertex(Vertex vertex) {
  assert(!vertex.indices.empty() && "Vertex without elements");
  auto id = static_cast<VertexId>(vertices.size());

  Piece piece;
  piece.domain = vertex.indices;
  piece.root = id;
  piece.toRoot = getIdentity(context, vertex.indices.rank());
  piece.isRoot = true;

  pieces.emplace_back();
  pieces.back().push_back(std::move(piece));

  dependents.emplace_back();
  dependents.back().insert(id);

  vertices.push_back(std::move(vertex));
  return id;
}

void ConnectionGraph::addEdge(Edge edge) {
  assert(edge.lhs < vertices.size() && edge.rhs < vertices.size());
  assert(edge.lhsAccess->isAffine() && edge.rhsAccess->isAffine());

  assert(!edge.equationIndices.empty() && "Edge without equations");

  assert(isAccessWithin(*edge.lhsAccess, edge.equationIndices,
                        vertices[edge.lhs].indices) &&
         "Left-hand side access out of bounds");

  assert(isAccessWithin(*edge.rhsAccess, edge.equationIndices,
                        vertices[edge.rhs].indices) &&
         "Right-hand side access out of bounds");

  dropUnusedDimensions(edge);
  edges.push_back(std::move(edge));
}

size_t ConnectionGraph::getNumVertices() const { return vertices.size(); }

const Vertex &ConnectionGraph::getVertex(VertexId id) const {
  assert(id < vertices.size());
  return vertices[id];
}

void ConnectionGraph::computeRepresentatives() {
  numOfExaminedPairs = 0;
  numOfVisitedPieces = 0;

  for (const Edge &edge : edges) {
    unite(edge);
  }
}

llvm::ArrayRef<Piece> ConnectionGraph::getPieces(VertexId id) const {
  assert(id < pieces.size());
  return pieces[id];
}

size_t ConnectionGraph::getNumOfExaminedPairs() const {
  return numOfExaminedPairs;
}

size_t ConnectionGraph::getNumOfVisitedPieces() const {
  return numOfVisitedPieces;
}

void ConnectionGraph::unite(const Edge &edge) {
  uniteEquations(edge, edge.equationIndices);

  for (VertexId vertex : modifiedVertices) {
    mergePieces(vertex);
  }

  modifiedVertices.clear();
}

void ConnectionGraph::uniteEquations(const Edge &edge,
                                     IndexSet equationIndices) {
  // Each processed pair modifies the pieces, so the search restarts.
  while (!equationIndices.empty()) {
    std::optional<IndexSet> processed = uniteFirstPair(edge, equationIndices);

    if (!processed) {
      return;
    }

    equationIndices -= *processed;
  }
}

std::optional<IndexSet>
ConnectionGraph::uniteFirstPair(const Edge &edge,
                                const IndexSet &edgeEquationIndices) {
  for (size_t lhsPiece = 0; lhsPiece < pieces[edge.lhs].size(); ++lhsPiece) {
    for (size_t rhsPiece = 0; rhsPiece < pieces[edge.rhs].size(); ++rhsPiece) {
      const Piece &lhs = pieces[edge.lhs][lhsPiece];
      const Piece &rhs = pieces[edge.rhs][rhsPiece];
      ++numOfVisitedPieces;

      IndexSet equationIndices =
          getPreimage(*edge.lhsAccess, lhs.domain, edgeEquationIndices)
              .intersect(getPreimage(*edge.rhsAccess, rhs.domain,
                                     edgeEquationIndices));

      if (equationIndices.empty()) {
        continue;
      }

      ++numOfExaminedPairs;

      // Maps from the equation indices to the indices of the roots.
      auto lhsToRoot = edge.lhsAccess->combine(*lhs.toRoot);
      auto rhsToRoot = edge.rhsAccess->combine(*rhs.toRoot);
      VertexId lhsRoot = lhs.root;
      VertexId rhsRoot = rhs.root;

      if (lhsRoot == rhsRoot) {
        if (agreeOn(*lhsToRoot, *rhsToRoot, equationIndices)) {
          continue;
        }

        if (!constantSelfUnion(lhsRoot, *lhsToRoot, *rhsToRoot,
                               equationIndices) &&
            !selfUnion(lhsRoot, *lhsToRoot, *rhsToRoot, equationIndices)) {
          // Translations by more than one element, whose sets are residue
          // classes, and the other self-relations: permutations (such as
          // transposes) and translations along multiple dimensions (such as
          // diagonals), whose sets are not boxes but pairs of elements or
          // diagonal runs.
          // TODO: keep compact the translations by more than one element
          // once ranges with a step are supported.
          // TODO: keep compact the permutations and the translations along
          // multiple dimensions, by representing their sets directly.
          pointwiseUnion(equationIndices, edge);
        }

        return equationIndices;
      }

      bool isLhsInvertible = isInvertibleOn(*lhsToRoot, equationIndices);
      bool isRhsInvertible = isInvertibleOn(*rhsToRoot, equationIndices);

      if (!isLhsInvertible && !isRhsInvertible) {
        if (getSimpleResults(*lhsToRoot) && getSimpleResults(*rhsToRoot)) {
          sliceUnion(*lhsToRoot, *rhsToRoot, equationIndices, edge);
        } else {
          pointwiseUnion(equationIndices, edge);
        }

        return equationIndices;
      }

      // The representatives of the root vertex with the smaller id are kept,
      // unless only the other side can be inverted.
      bool isLhsChild = lhsRoot > rhsRoot;

      if (isLhsChild && !isLhsInvertible) {
        isLhsChild = false;
      } else if (!isLhsChild && !isRhsInvertible) {
        isLhsChild = true;
      }

      VertexId childRoot = isLhsChild ? lhsRoot : rhsRoot;
      VertexId parentRoot = isLhsChild ? rhsRoot : lhsRoot;
      auto childMap = isLhsChild ? std::move(lhsToRoot) : std::move(rhsToRoot);
      auto parentMap = isLhsChild ? std::move(rhsToRoot) : std::move(lhsToRoot);

      // The inverse of the child map is computed range by range.
      for (const MultidimensionalRange &range : llvm::make_range(
               equationIndices.rangesBegin(), equationIndices.rangesEnd())) {
        IndexSet childRoots = getImage(*childMap, IndexSet(range));
        auto childToParent =
            getInverseOn(*childMap, range)->combine(*parentMap);
        rePoint(childRoot, childRoots, parentRoot, *childToParent);
      }

      return equationIndices;
    }
  }

  return std::nullopt;
}

bool ConnectionGraph::constantSelfUnion(VertexId root,
                                        const AccessFunction &lhsToRoot,
                                        const AccessFunction &rhsToRoot,
                                        const IndexSet &equationIndices) {
  for (bool isLhsConstant : {true, false}) {
    const AccessFunction &constantMap = isLhsConstant ? lhsToRoot : rhsToRoot;
    const AccessFunction &otherMap = isLhsConstant ? rhsToRoot : lhsToRoot;
    IndexSet representative = getImage(constantMap, equationIndices);

    if (representative.size() != 1) {
      continue;
    }

    IndexSet childRoots = getImage(otherMap, equationIndices) - representative;

    llvm::SmallVector<mlir::AffineExpr, 3> expressions;

    for (int64_t index : *representative.begin()) {
      expressions.push_back(mlir::getAffineConstantExpr(index, context));
    }

    auto childToParent = AccessFunction::build(
        mlir::AffineMap::get(representative.rank(), 0, expressions, context));

    rePoint(root, childRoots, root, *childToParent);
    return true;
  }

  return false;
}

bool ConnectionGraph::selfUnion(VertexId root, const AccessFunction &lhsToRoot,
                                const AccessFunction &rhsToRoot,
                                const IndexSet &equationIndices) {
  // The right-hand side map must be the left-hand side one followed by a
  // translation, whose vector is computed on the first equation.
  Point firstEquation = *equationIndices.begin();
  Point lhsFirst = evaluate(lhsToRoot, firstEquation);
  Point rhsFirst = evaluate(rhsToRoot, firstEquation);
  size_t rank = lhsFirst.rank();

  mlir::AffineMap lhsMap = lhsToRoot.getAffineMap();
  llvm::SmallVector<mlir::AffineExpr, 3> translatedResults;
  std::optional<size_t> dimension;

  for (size_t i = 0; i < rank; ++i) {
    int64_t translation = rhsFirst[i] - lhsFirst[i];
    translatedResults.push_back(lhsMap.getResult(i) + translation);

    if (translation == 0) {
      continue;
    }

    if (dimension || std::abs(translation) != 1) {
      // Not a translation by one along a single dimension.
      return false;
    }

    dimension = i;
  }

  if (!dimension) {
    // The maps agree on the first equation, but not on all of them.
    return false;
  }

  auto translated = AccessFunction::build(
      mlir::AffineMap::get(lhsMap.getNumDims(), 0, translatedResults, context));

  if (!agreeOn(*translated, rhsToRoot, equationIndices)) {
    return false;
  }

  // Each origin 'g' is connected to 'g + e_d', with 'd' the translated
  // dimension: the sets are the maximal runs along 'd' of the origins,
  // each extended by one element.
  size_t d = *dimension;
  bool isForward = rhsFirst[d] > lhsFirst[d];
  IndexSet origins =
      getImage(isForward ? lhsToRoot : rhsToRoot, equationIndices);

  // Split the ranges of the origins on a common grid of the other
  // dimensions, so that their segments along 'd' can be compared.
  llvm::SmallVector<llvm::SmallVector<int64_t>, 3> grid(rank);

  for (const MultidimensionalRange &range :
       llvm::make_range(origins.rangesBegin(), origins.rangesEnd())) {
    for (size_t i = 0; i < rank; ++i) {
      if (i != d) {
        grid[i].push_back(range[i].getBegin());
        grid[i].push_back(range[i].getEnd());
      }
    }
  }

  for (llvm::SmallVector<int64_t> &breakpoints : grid) {
    llvm::sort(breakpoints);
    breakpoints.erase(std::unique(breakpoints.begin(), breakpoints.end()),
                      breakpoints.end());
  }

  // Cells of the grid (ranges of the other dimensions, with a placeholder
  // for 'd') and their segments along 'd'.
  using Cell = std::vector<std::pair<int64_t, int64_t>>;
  std::map<Cell, llvm::SmallVector<std::pair<int64_t, int64_t>>> segments;

  for (const MultidimensionalRange &range :
       llvm::make_range(origins.rangesBegin(), origins.rangesEnd())) {
    // Intervals of the grid covered by the range, for each dimension.
    llvm::SmallVector<llvm::SmallVector<std::pair<int64_t, int64_t>>, 3>
        intervals(rank);

    for (size_t i = 0; i < rank; ++i) {
      if (i == d) {
        intervals[i].push_back({0, 0});
        continue;
      }

      const llvm::SmallVector<int64_t> &breakpoints = grid[i];
      auto it = llvm::lower_bound(breakpoints, range[i].getBegin());

      for (; *it != range[i].getEnd(); ++it) {
        intervals[i].push_back({*it, *std::next(it)});
      }
    }

    // Enumerate the cells covered by the range.
    llvm::SmallVector<size_t, 3> positions(rank, 0);

    while (true) {
      Cell cell;

      for (size_t i = 0; i < rank; ++i) {
        cell.push_back(intervals[i][positions[i]]);
      }

      segments[cell].push_back({range[d].getBegin(), range[d].getEnd()});

      size_t i = 0;

      while (i < rank && ++positions[i] == intervals[i].size()) {
        positions[i++] = 0;
      }

      if (i == rank) {
        break;
      }
    }
  }

  for (auto &entry : segments) {
    const Cell &cell = entry.first;
    auto &cellSegments = entry.second;
    llvm::sort(cellSegments);

    auto rePointRun = [&](int64_t begin, int64_t end) {
      // The run is [begin, end] along 'd': the elements after the first
      // slice get the representative in it.
      llvm::SmallVector<Range, 3> ranges;
      llvm::SmallVector<mlir::AffineExpr, 3> expressions;

      for (size_t i = 0; i < rank; ++i) {
        if (i == d) {
          ranges.push_back(Range(begin + 1, end + 1));
          expressions.push_back(mlir::getAffineConstantExpr(begin, context));
        } else {
          ranges.push_back(Range(cell[i].first, cell[i].second));
          expressions.push_back(mlir::getAffineDimExpr(i, context));
        }
      }

      auto childToParent = AccessFunction::build(
          mlir::AffineMap::get(rank, 0, expressions, context));

      rePoint(root, IndexSet(MultidimensionalRange(ranges)), root,
              *childToParent);
    };

    // Segments [begin, end) of origins cover the run [begin, end] of
    // connected elements, so they merge when they share an element.
    int64_t runBegin = cellSegments.front().first;
    int64_t runEnd = cellSegments.front().second;

    for (const auto &segment : llvm::drop_begin(cellSegments)) {
      if (segment.first <= runEnd) {
        runEnd = std::max(runEnd, segment.second);
      } else {
        rePointRun(runBegin, runEnd);
        runBegin = segment.first;
        runEnd = segment.second;
      }
    }

    rePointRun(runBegin, runEnd);
  }

  return true;
}

void ConnectionGraph::sliceUnion(const AccessFunction &lhsToRoot,
                                 const AccessFunction &rhsToRoot,
                                 const IndexSet &equationIndices,
                                 const Edge &edge) {
  size_t rank = equationIndices.rank();
  llvm::SmallBitVector lhsDimensions = getUsedDimensions(lhsToRoot, rank);
  llvm::SmallBitVector rhsDimensions = getUsedDimensions(rhsToRoot, rank);

  // The equations differing only in the dimensions used by neither map
  // connect the same elements. For each value of the dimensions used by both
  // maps, the elements given by all the values of the dimensions used only
  // by the left-hand side map are connected to all the ones given by the
  // dimensions used only by the right-hand side map: fixing the latter
  // (resp. the former) leaves a slice on which the left-hand (resp.
  // right-hand) side map is injective, and the two slices share an equation.
  llvm::SmallBitVector lhsOnly = lhsDimensions;
  lhsOnly.reset(rhsDimensions);
  llvm::SmallBitVector rhsOnly = rhsDimensions;
  rhsOnly.reset(lhsDimensions);

  llvm::SmallBitVector lhsSliceFixed = lhsDimensions;
  lhsSliceFixed.flip();
  llvm::SmallBitVector rhsSliceFixed = rhsDimensions;
  rhsSliceFixed.flip();

  llvm::SmallVector<MultidimensionalRange> slices;

  for (const MultidimensionalRange &range : llvm::make_range(
           equationIndices.rangesBegin(), equationIndices.rangesEnd())) {
    if (lhsOnly.any() || rhsOnly.none()) {
      slices.push_back(fixDimensions(range, lhsSliceFixed));
    }

    if (rhsOnly.any()) {
      slices.push_back(fixDimensions(range, rhsSliceFixed));
    }
  }

  for (const MultidimensionalRange &slice : slices) {
    // A slice equal to the equations would make one of the maps invertible
    // on them, so the recursion ends.
    assert(slice.size() < equationIndices.size());
    uniteEquations(edge, IndexSet(slice));
  }
}

void ConnectionGraph::pointwiseUnion(const IndexSet &equationIndices,
                                     const Edge &edge) {
  // TODO: keep compact once ranges with a step are supported.
  using Element = std::pair<VertexId, Point>;

  // Get the representatives of the elements accessed by the equations, from
  // the pieces as they are before the union.
  auto getRepresentatives = [&](VertexId vertex,
                                const AccessFunction &accessFunction) {
    llvm::SmallVector<Point> accessed;

    for (Point point : equationIndices) {
      accessed.push_back(evaluate(accessFunction, point));
    }

    llvm::sort(accessed);
    accessed.erase(std::unique(accessed.begin(), accessed.end()),
                   accessed.end());

    std::map<Point, Element> result;

    for (const Piece &piece : pieces[vertex]) {
      ++numOfVisitedPieces;

      for (const MultidimensionalRange &range : llvm::make_range(
               piece.domain.rangesBegin(), piece.domain.rangesEnd())) {
        forEachPointIn(accessed, range, [&](const Point &element) {
          result[element] = {piece.root, evaluate(*piece.toRoot, element)};
        });
      }
    }

    return result;
  };

  std::map<Point, Element> lhsRepresentatives =
      getRepresentatives(edge.lhs, *edge.lhsAccess);

  std::map<Point, Element> rhsRepresentatives =
      getRepresentatives(edge.rhs, *edge.rhsAccess);

  // Union-find over the representatives. Each set gets its smallest
  // representative, which belongs to the root vertex with the smallest id.
  std::map<Element, Element> parents;

  auto find = [&](Element element) {
    Element root = element;

    for (auto it = parents.find(root); it != parents.end();
         it = parents.find(root)) {
      root = it->second;
    }

    // Compress the path.
    while (element != root) {
      auto it = parents.find(element);
      element = std::exchange(it->second, root);
    }

    return root;
  };

  for (Point point : equationIndices) {
    Element lhs = find(lhsRepresentatives.at(evaluate(*edge.lhsAccess, point)));
    Element rhs = find(rhsRepresentatives.at(evaluate(*edge.rhsAccess, point)));

    if (lhs != rhs) {
      parents[std::max(lhs, rhs)] = std::min(lhs, rhs);
    }
  }

  // The representatives no longer being such, for each root vertex.
  std::map<VertexId, std::map<Point, Element>> children;

  for (const auto &entry : parents) {
    Element root = find(entry.first);

    if (root != entry.first) {
      children[entry.first.first][entry.first.second] = root;
    }
  }

  for (const auto &entry : children) {
    rePointEach(entry.first, entry.second);
  }
}

void ConnectionGraph::rePoint(VertexId childRoot, const IndexSet &childRoots,
                              VertexId parentRoot,
                              const AccessFunction &childToParent) {
  assert(childRoot != parentRoot ||
         !childRoots.contains(getImage(childToParent, childRoots)));

  // The vertices are copied, as the set is modified while iterating.
  llvm::SmallVector<VertexId> vertexIds(dependents[childRoot].begin(),
                                        dependents[childRoot].end());

  for (VertexId vertex : vertexIds) {
    auto &vertexPieces = pieces[vertex];
    bool stillDependent = false;

    for (size_t i = 0, e = vertexPieces.size(); i < e; ++i) {
      ++numOfVisitedPieces;

      if (vertexPieces[i].root != childRoot) {
        continue;
      }

      IndexSet moved = getPreimage(*vertexPieces[i].toRoot, childRoots,
                                   vertexPieces[i].domain);

      if (moved.empty()) {
        stillDependent = true;
        continue;
      }

      Piece piece;
      piece.domain = std::move(moved);
      piece.root = parentRoot;
      piece.toRoot = vertexPieces[i].toRoot->combine(childToParent);
      piece.isRoot = false;

      vertexPieces[i].domain -= piece.domain;

      if (!vertexPieces[i].domain.empty()) {
        stillDependent = true;
      }

      // The new piece is appended after the ones being visited.
      vertexPieces.push_back(std::move(piece));
    }

    llvm::erase_if(vertexPieces,
                   [](const Piece &piece) { return piece.domain.empty(); });

    if (!stillDependent) {
      dependents[childRoot].remove(vertex);
    }

    dependents[parentRoot].insert(vertex);
    modifiedVertices.insert(vertex);
  }
}

void ConnectionGraph::rePointEach(
    VertexId childRoot,
    const std::map<Point, std::pair<VertexId, Point>> &parents) {
  // The keys of the map are sorted.
  llvm::SmallVector<Point> childRoots;

  for (const auto &entry : parents) {
    childRoots.push_back(entry.first);
  }

  // The vertices are copied, as the set is modified while iterating.
  llvm::SmallVector<VertexId> vertexIds(dependents[childRoot].begin(),
                                        dependents[childRoot].end());

  for (VertexId vertex : vertexIds) {
    auto &vertexPieces = pieces[vertex];
    bool stillDependent = false;

    // The moved elements, for each new representative.
    std::map<std::pair<VertexId, Point>, IndexSet> moved;

    for (Piece &piece : vertexPieces) {
      ++numOfVisitedPieces;

      if (piece.root != childRoot) {
        continue;
      }

      // The elements of the piece whose representatives are among the
      // child ones. They are removed from the piece one range at a time, as
      // building an index set made of many ranges is expensive.
      llvm::SmallVector<MultidimensionalRange> movedElements;
      std::optional<SimpleResults> simpleResults =
          getSimpleResults(*piece.toRoot);

      for (const MultidimensionalRange &range : llvm::make_range(
               piece.domain.rangesBegin(), piece.domain.rangesEnd())) {
        if (!simpleResults) {
          // TODO: keep compact once ranges with a step are supported.
          for (Point element : range) {
            auto it = parents.find(evaluate(*piece.toRoot, element));

            if (it != parents.end()) {
              moved[it->second] += element;
              movedElements.push_back(MultidimensionalRange(element));
            }
          }

          continue;
        }

        IndexSet rangeElements(range);

        forEachPointIn(
            childRoots, getImageBounds(*piece.toRoot, range),
            [&](const Point &childRepresentative) {
              IndexSet elements = getPreimage(
                  *piece.toRoot, IndexSet(childRepresentative), rangeElements);

              if (!elements.empty()) {
                moved[parents.at(childRepresentative)] += elements;

                llvm::append_range(movedElements,
                                   llvm::make_range(elements.rangesBegin(),
                                                    elements.rangesEnd()));
              }
            });
      }

      if (movedElements.empty()) {
        stillDependent = true;
        continue;
      }

      for (const MultidimensionalRange &range : movedElements) {
        piece.domain -= range;
      }

      if (!piece.domain.empty()) {
        stillDependent = true;
      }
    }

    // The vertex is removed from the dependents before adding the new
    // pieces, whose roots may include the child one.
    if (!stillDependent) {
      dependents[childRoot].remove(vertex);
    }

    for (auto &entry : moved) {
      Piece piece;
      piece.domain = std::move(entry.second);
      piece.root = entry.first.first;

      piece.toRoot = getConstantAccess(context, vertices[vertex].indices.rank(),
                                       entry.first.second);

      piece.isRoot = false;
      dependents[piece.root].insert(vertex);
      vertexPieces.push_back(std::move(piece));
    }

    llvm::erase_if(vertexPieces,
                   [](const Piece &piece) { return piece.domain.empty(); });

    modifiedVertices.insert(vertex);
  }
}

void ConnectionGraph::mergePieces(VertexId vertex) {
  auto &vertexPieces = pieces[vertex];

  // The pieces are merged into the first one with the same root, kind and
  // map to the root. The maps are equal if their affine maps are, which are
  // uniqued.
  std::map<std::tuple<VertexId, bool, const void *>, size_t> firstPieces;
  llvm::SmallVector<Piece, 1> mergedPieces;

  for (Piece &piece : vertexPieces) {
    ++numOfVisitedPieces;

    auto key =
        std::make_tuple(piece.root, piece.isRoot,
                        piece.toRoot->getAffineMap().getAsOpaquePointer());

    auto [it, inserted] = firstPieces.try_emplace(key, mergedPieces.size());

    if (inserted) {
      mergedPieces.push_back(std::move(piece));
    } else {
      mergedPieces[it->second].domain += piece.domain;
    }
  }

  vertexPieces = std::move(mergedPieces);
}
} // namespace mlir::modelica::connection
