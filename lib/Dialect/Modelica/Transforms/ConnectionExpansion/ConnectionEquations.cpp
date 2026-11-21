#include "marco/Dialect/Modelica/Transforms/ConnectionExpansion/ConnectionEquations.h"
#include "marco/Dialect/Modelica/Transforms/ConnectionExpansion/AccessFunctionUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include <map>
#include <optional>
#include <tuple>
#include <vector>

using namespace ::marco::modeling;
using namespace ::mlir::modelica::connection;

//===---------------------------------------------------------------------===//
// Potential equations
//===---------------------------------------------------------------------===//

/// Get the indices, among the given ones, on which two affine access
/// functions with the same number of results give the same indices.
static IndexSet getAgreementSet(const AccessFunction &lhs,
                                const AccessFunction &rhs,
                                const IndexSet &indices) {
  IndexSet result;
  std::optional<SimpleResults> lhsResults = getSimpleResults(lhs);
  std::optional<SimpleResults> rhsResults = getSimpleResults(rhs);

  for (const MultidimensionalRange &range :
       llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
    if (agreeOn(lhs, rhs, IndexSet(range))) {
      result += range;
      continue;
    }

    // Restrict the range dimension by dimension, as long as each pair of
    // results constrains a single dimension.
    llvm::SmallVector<int64_t, 3> begins;
    llvm::SmallVector<int64_t, 3> ends;

    for (size_t i = 0, e = range.rank(); i < e; ++i) {
      begins.push_back(range[i].getBegin());
      ends.push_back(range[i].getEnd());
    }

    auto restrict = [&](unsigned dimension, int64_t value) {
      begins[dimension] = std::max(begins[dimension], value);
      ends[dimension] = std::min(ends[dimension], value + 1);
    };

    bool isBox = lhsResults && rhsResults;
    bool isEmpty = false;

    for (size_t i = 0, e = lhs.getNumOfResults(); i < e && isBox; ++i) {
      const SimpleResult &l = (*lhsResults)[i];
      const SimpleResult &r = (*rhsResults)[i];

      if (!l.dimension && !r.dimension) {
        isEmpty |= l.offset != r.offset;
      } else if (!r.dimension) {
        restrict(*l.dimension, r.offset - l.offset);
      } else if (!l.dimension) {
        restrict(*r.dimension, l.offset - r.offset);
      } else if (*l.dimension == *r.dimension) {
        isEmpty |= l.offset != r.offset;
      } else if (range[*l.dimension].size() == 1) {
        restrict(*r.dimension,
                 range[*l.dimension].getBegin() + l.offset - r.offset);
      } else if (range[*r.dimension].size() == 1) {
        restrict(*l.dimension,
                 range[*r.dimension].getBegin() + r.offset - l.offset);
      } else {
        isBox = false;
      }
    }

    if (!isBox) {
      // TODO: avoid the enumeration for the other affine access functions.
      for (Point point : range) {
        if (evaluate(lhs, point) == evaluate(rhs, point)) {
          result += point;
        }
      }

      continue;
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

  return result;
}

/// Vertices of the graph, given their variable and side.
using VertexMap = std::map<std::pair<unsigned, bool>, VertexId>;

/// Get the vertex being the other side of the same variable, if any.
static std::optional<VertexId> getOtherSide(const ConnectionGraph &graph,
                                            const VertexMap &vertices,
                                            VertexId vertex) {
  const Vertex &current = graph.getVertex(vertex);

  if (auto it = vertices.find({current.variable, !current.outside});
      it != vertices.end()) {
    return it->second;
  }

  return std::nullopt;
}

static void addPotentialFamilies(const ConnectionGraph &graph,
                                 const VertexMap &vertices, VertexId vertex,
                                 ConnectionEquations &equations) {
  std::optional<VertexId> otherSide = getOtherSide(graph, vertices, vertex);
  bool isOutside = graph.getVertex(vertex).outside;

  for (const Piece &piece : graph.getPieces(vertex)) {
    if (piece.isRoot) {
      continue;
    }

    IndexSet indices = piece.domain;

    // The inside and outside sides of the same element are the same
    // variable element, which must be neither equated to itself nor
    // equated twice to the same representative: an element is skipped if
    // the other side is its representative, or if it is the outside side and
    // the other side has the same representative.
    if (otherSide) {
      for (const Piece &otherPiece : graph.getPieces(*otherSide)) {
        if (otherPiece.root != piece.root ||
            (!isOutside && !otherPiece.isRoot)) {
          continue;
        }

        IndexSet common = indices.intersect(otherPiece.domain);

        if (!common.empty()) {
          indices -= getAgreementSet(*piece.toRoot, *otherPiece.toRoot, common);
        }
      }
    }

    if (indices.empty()) {
      continue;
    }

    equations.potentials.push_back(
        {std::move(indices), vertex, piece.root, piece.toRoot->clone()});
  }
}

//===---------------------------------------------------------------------===//
// Flow equations
//===---------------------------------------------------------------------===//

namespace {
/// Members of the flow sets of a root vertex coming from the same vertex and
/// accessed in the same way: a term if there are no iterators, a reduction
/// otherwise.
struct Contribution {
  VertexId vertex;

  /// Map from the root indices (followed by the iterators, if any) to the
  /// indices of the vertex.
  std::unique_ptr<AccessFunction> access;

  std::optional<MultidimensionalRange> iterators;

  /// Indices of the representatives of the sets the members belong to.
  IndexSet rootIndices;
};

/// Vertex, access and iterators of a contribution.
using ContributionKey =
    std::tuple<VertexId, const void *, std::vector<int64_t>>;

/// Contributions to the sums of the flow sets of a root vertex.
struct Contributions {
  llvm::SmallVector<Contribution> list;

  /// Position in the list of the contribution with a given key.
  std::map<ContributionKey, size_t> positions;
};
} // namespace

/// Add a contribution, merging it with an existing one accessing the same
/// vertex in the same way.
static void addContribution(Contributions &contributions,
                            Contribution contribution,
                            EquationsStatistics &statistics) {
  ++statistics.numOfVisitedContributions;

  // The access functions are equal if their affine maps are, which are
  // uniqued.
  std::vector<int64_t> iterators;

  if (contribution.iterators) {
    for (size_t i = 0, e = contribution.iterators->rank(); i < e; ++i) {
      iterators.push_back((*contribution.iterators)[i].getBegin());
      iterators.push_back((*contribution.iterators)[i].getEnd());
    }
  }

  ContributionKey key(contribution.vertex,
                      contribution.access->getAffineMap().getAsOpaquePointer(),
                      std::move(iterators));

  auto [it, inserted] =
      contributions.positions.try_emplace(key, contributions.list.size());

  if (!inserted) {
    contributions.list[it->second].rootIndices += contribution.rootIndices;
    return;
  }

  contributions.list.push_back(std::move(contribution));
}

/// Get the contribution of a range of a piece whose elements, given the
/// indices of their representative, are a box spanning the dimensions not
/// used by the map to the root. The map to the root must be simple and not
/// use the same non-degenerate dimension more than once.
static Contribution getRangeContribution(VertexId vertex,
                                         const AccessFunction &toRoot,
                                         const SimpleResults &simpleResults,
                                         const MultidimensionalRange &range) {
  mlir::MLIRContext *context = toRoot.getContext();
  IndexSet rootIndices = getImage(toRoot, IndexSet(range));
  size_t rootRank = simpleResults.size();

  llvm::SmallVector<mlir::AffineExpr, 3> expressions;
  llvm::SmallVector<Range, 3> iterators;

  for (size_t i = 0, e = range.rank(); i < e; ++i) {
    auto it = llvm::find_if(simpleResults, [&](const SimpleResult &result) {
      return result.dimension == i;
    });

    if (it != simpleResults.end()) {
      size_t resultIndex = std::distance(simpleResults.begin(), it);

      expressions.push_back(mlir::getAffineDimExpr(resultIndex, context) -
                            it->offset);
    } else if (range[i].size() == 1) {
      expressions.push_back(
          mlir::getAffineConstantExpr(range[i].getBegin(), context));
    } else {
      expressions.push_back(
          mlir::getAffineDimExpr(rootRank + iterators.size(), context));

      iterators.push_back(range[i]);
    }
  }

  if (iterators.empty()) {
    return {vertex, getInverseOn(toRoot, range), std::nullopt,
            std::move(rootIndices)};
  }

  auto access = AccessFunction::build(mlir::AffineMap::get(
      rootRank + iterators.size(), 0, expressions, context));

  return {vertex, std::move(access), MultidimensionalRange(iterators),
          std::move(rootIndices)};
}

/// Add the contributions of a piece of a flow vertex to the sums of its root.
static void addPieceContributions(VertexId vertex, const Piece &piece,
                                  Contributions &result,
                                  EquationsStatistics &statistics) {
  if (piece.isRoot) {
    addContribution(result,
                    {vertex,
                     getIdentity(piece.toRoot->getContext(),
                                 piece.toRoot->getNumOfResults()),
                     std::nullopt, piece.domain},
                    statistics);

    return;
  }

  std::optional<SimpleResults> simpleResults = getSimpleResults(*piece.toRoot);

  for (const MultidimensionalRange &range :
       llvm::make_range(piece.domain.rangesBegin(), piece.domain.rangesEnd())) {
    if (simpleResults && !hasRepeatedDimensions(*simpleResults, range)) {
      addContribution(
          result,
          getRangeContribution(vertex, *piece.toRoot, *simpleResults, range),
          statistics);

      continue;
    }

    // TODO: keep compact once ranges with a step are supported.
    for (Point point : range) {
      MultidimensionalRange pointRange(point);

      addContribution(result,
                      {vertex, getInverseOn(*piece.toRoot, pointRange),
                       std::nullopt, IndexSet(evaluate(*piece.toRoot, point))},
                      statistics);
    }
  }
}

/// Replace the last dimension of an access function by a constant.
static std::unique_ptr<AccessFunction>
fixLastDimension(const AccessFunction &accessFunction, int64_t value) {
  mlir::AffineMap map = accessFunction.getAffineMap();
  mlir::MLIRContext *context = map.getContext();
  unsigned numOfDims = map.getNumDims() - 1;
  llvm::SmallVector<mlir::AffineExpr, 3> replacements;

  for (unsigned i = 0; i < numOfDims; ++i) {
    replacements.push_back(mlir::getAffineDimExpr(i, context));
  }

  replacements.push_back(mlir::getAffineConstantExpr(value, context));

  return AccessFunction::build(mlir::simplifyAffineMap(
      map.replaceDimsAndSymbols(replacements, {}, numOfDims, 0)));
}

/// Merge the terms of a family into the one-dimensional reductions of the
/// same vertex, when they access the element just before or after the
/// iterated ones. This happens for the representatives of the closures of
/// self-translations, which are the first elements of their runs.
static void mergeTermsIntoReductions(FlowFamily &family) {
  bool changed = true;

  while (changed) {
    changed = false;

    for (FlowReduction &reduction : family.reductions) {
      if (reduction.iterators.rank() != 1) {
        continue;
      }

      // Copied, as the reduction is modified below.
      Range iterators = reduction.iterators[0];

      auto before =
          fixLastDimension(*reduction.access, iterators.getBegin() - 1);
      auto after = fixLastDimension(*reduction.access, iterators.getEnd());

      for (auto *it = family.terms.begin(); it != family.terms.end(); ++it) {
        if (it->vertex != reduction.vertex) {
          continue;
        }

        if (agreeOn(*it->fromRoot, *before, family.rootIndices)) {
          reduction.iterators = MultidimensionalRange(
              Range(iterators.getBegin() - 1, iterators.getEnd()));
        } else if (agreeOn(*it->fromRoot, *after, family.rootIndices)) {
          reduction.iterators = MultidimensionalRange(
              Range(iterators.getBegin(), iterators.getEnd() + 1));
        } else {
          continue;
        }

        family.terms.erase(it);
        changed = true;
        break;
      }

      if (changed) {
        break;
      }
    }
  }
}

/// Add the flow families of a root vertex. The root indices are partitioned
/// into the regions covered by the same contributions, each region becoming
/// a family. The contributions of a single root index, coming from the
/// elements processed point by point, are grouped by it instead: each such
/// root index becomes a family of its own.
static void addFlowFamilies(const ConnectionGraph &graph, VertexId root,
                            llvm::ArrayRef<Contribution> contributions,
                            ConnectionEquations &equations,
                            EquationsStatistics &statistics) {
  struct Region {
    IndexSet rootIndices;
    llvm::SmallVector<size_t> contributions;
  };

  IndexSet rootIndices;

  for (const Piece &piece : graph.getPieces(root)) {
    if (piece.isRoot) {
      rootIndices += piece.domain;
    }
  }

  std::map<Point, llvm::SmallVector<size_t>> pointContributions;
  llvm::SmallVector<size_t> rangeContributions;

  for (const auto &contribution : llvm::enumerate(contributions)) {
    ++statistics.numOfVisitedContributions;
    const IndexSet &contributionIndices = contribution.value().rootIndices;

    if (contributionIndices.size() == 1) {
      pointContributions[*contributionIndices.begin()].push_back(
          contribution.index());
    } else {
      rangeContributions.push_back(contribution.index());
    }
  }

  llvm::SmallVector<Region> regions;
  regions.push_back({std::move(rootIndices), {}});

  for (size_t index : rangeContributions) {
    const Contribution &contribution = contributions[index];
    llvm::SmallVector<Region> newRegions;

    for (Region &region : regions) {
      ++statistics.numOfVisitedContributions;

      // A region covered entirely, as by the contribution of the
      // representatives themselves, is kept as it is: intersecting index sets
      // made of many ranges is expensive.
      if (contribution.rootIndices.contains(region.rootIndices)) {
        region.contributions.push_back(index);
        newRegions.push_back(std::move(region));
        continue;
      }

      IndexSet covered = region.rootIndices.intersect(contribution.rootIndices);

      if (covered.empty()) {
        newRegions.push_back(std::move(region));
        continue;
      }

      IndexSet uncovered = region.rootIndices - covered;
      llvm::SmallVector<size_t> coveredContributions(region.contributions);
      coveredContributions.push_back(index);
      newRegions.push_back({std::move(covered), coveredContributions});

      if (!uncovered.empty()) {
        newRegions.push_back(
            {std::move(uncovered), std::move(region.contributions)});
      }
    }

    regions = std::move(newRegions);
  }

  // The root indices with contributions of their own are taken out of their
  // regions.
  llvm::SmallVector<Region> pointRegions;

  for (auto &entry : pointContributions) {
    const Point &point = entry.first;

    Region *region = llvm::find_if(regions, [&](const Region &candidate) {
      ++statistics.numOfVisitedContributions;
      return candidate.rootIndices.contains(point);
    });

    assert(region != regions.end() && "Contribution to a non-root index");
    region->rootIndices -= point;

    llvm::SmallVector<size_t> pointRegionContributions(region->contributions);
    llvm::append_range(pointRegionContributions, entry.second);
    pointRegions.push_back({IndexSet(point), pointRegionContributions});
  }

  llvm::erase_if(
      regions, [](const Region &region) { return region.rootIndices.empty(); });

  llvm::append_range(regions, std::move(pointRegions));

  for (const Region &region : regions) {
    FlowFamily family;
    family.rootIndices = region.rootIndices;
    family.root = root;

    for (size_t index : region.contributions) {
      const Contribution &contribution = contributions[index];

      if (contribution.iterators) {
        family.reductions.push_back({contribution.vertex,
                                     *contribution.iterators,
                                     contribution.access->clone()});
      } else {
        family.terms.push_back(
            {contribution.vertex, contribution.access->clone()});
      }
    }

    mergeTermsIntoReductions(family);
    equations.flows.push_back(std::move(family));
  }
}

//===---------------------------------------------------------------------===//
// Equations
//===---------------------------------------------------------------------===//

namespace mlir::modelica::connection {
ConnectionEquations computeEquations(const ConnectionGraph &graph,
                                     EquationsStatistics *statistics) {
  EquationsStatistics localStatistics;
  EquationsStatistics &stats = statistics ? *statistics : localStatistics;
  ConnectionEquations equations;
  size_t numOfVertices = graph.getNumVertices();

  VertexMap vertices;

  for (VertexId vertex = 0; vertex < numOfVertices; ++vertex) {
    const Vertex &current = graph.getVertex(vertex);
    vertices[{current.variable, current.outside}] = vertex;
  }

  // The contributions to the sums of the flow sets, for each root vertex.
  std::vector<Contributions> contributions(numOfVertices);

  for (VertexId vertex = 0; vertex < numOfVertices; ++vertex) {
    if (!graph.getVertex(vertex).isFlow) {
      addPotentialFamilies(graph, vertices, vertex, equations);
      continue;
    }

    for (const Piece &piece : graph.getPieces(vertex)) {
      addPieceContributions(vertex, piece, contributions[piece.root], stats);
    }
  }

  for (VertexId root = 0; root < numOfVertices; ++root) {
    if (!contributions[root].list.empty()) {
      addFlowFamilies(graph, root, contributions[root].list, equations, stats);
    }
  }

  return equations;
}
} // namespace mlir::modelica::connection
