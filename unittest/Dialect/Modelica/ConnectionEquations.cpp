#include "marco/Dialect/Modelica/Transforms/ConnectionExpansion/ConnectionEquations.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"
#include <map>
#include <set>

using namespace ::marco::modeling;
using namespace ::mlir::modelica::connection;

namespace {
/// Element of a connection set: a vertex and an index of it.
using Element = std::pair<VertexId, Point>;

using Sets = std::set<std::set<Element>>;

/// Build the one-dimensional access function 'k -> k + offset'.
std::unique_ptr<AccessFunction> shifted(mlir::MLIRContext *context,
                                        int64_t offset) {
  mlir::AffineExpr result = mlir::getAffineDimExpr(0, context) +
                            mlir::getAffineConstantExpr(offset, context);

  return AccessFunction::build(mlir::AffineMap::get(1, 0, result, context));
}

/// Build the one-dimensional access function 'k -> value'.
std::unique_ptr<AccessFunction> constant(mlir::MLIRContext *context,
                                         int64_t value) {
  mlir::AffineExpr result = mlir::getAffineConstantExpr(value, context);
  return AccessFunction::build(mlir::AffineMap::get(1, 0, result, context));
}

/// Build the one-dimensional access function 'k -> factor * k + offset'.
std::unique_ptr<AccessFunction> affine(mlir::MLIRContext *context,
                                       int64_t factor, int64_t offset) {
  mlir::AffineExpr result = mlir::getAffineDimExpr(0, context) * factor +
                            mlir::getAffineConstantExpr(offset, context);

  return AccessFunction::build(mlir::AffineMap::get(1, 0, result, context));
}

/// Index set made of the points [begin, end).
IndexSet indices(int64_t begin, int64_t end) {
  return {MultidimensionalRange(Range(begin, end))};
}

/// Apply an access function to a point.
Point apply(const AccessFunction &accessFunction, const Point &point) {
  IndexSet mapped = accessFunction.map(point);
  EXPECT_EQ(mapped.size(), 1u);
  return *mapped.begin();
}

/// Concatenate the coordinates of two points.
Point concat(const Point &lhs, const Point &rhs) {
  llvm::SmallVector<int64_t, 3> coordinates(lhs.begin(), lhs.end());
  coordinates.append(rhs.begin(), rhs.end());
  return {coordinates};
}

/// Get the connection sets of the vertices with the given kind, from the
/// pieces computed by the graph.
Sets getSets(const ConnectionGraph &graph, bool isFlow) {
  std::map<Element, std::set<Element>> sets;

  for (VertexId vertex = 0; vertex < graph.getNumVertices(); ++vertex) {
    if (graph.getVertex(vertex).isFlow != isFlow) {
      continue;
    }

    for (const Piece &piece : graph.getPieces(vertex)) {
      for (Point index : piece.domain) {
        Element representative(piece.root, apply(*piece.toRoot, index));
        sets[representative].insert(Element(vertex, index));
      }
    }
  }

  Sets result;

  for (const auto &entry : sets) {
    result.insert(entry.second);
  }

  return result;
}

/// Check that the potential families equate each element to an element of
/// its own set, with exactly 'n - 1' equations for a set of 'n' distinct
/// variable elements connecting all of them. The inside and outside sides of
/// the same variable element count as one. Return the number of equations.
size_t checkPotentials(const ConnectionGraph &graph,
                       const ConnectionEquations &equations,
                       const Sets &expected) {
  // Variable element: the variable and an index of it.
  using Key = std::pair<unsigned, Point>;

  auto getKey = [&](const Element &element) {
    return Key(graph.getVertex(element.first).variable, element.second);
  };

  std::map<Key, size_t> setOf;

  for (const auto &set : llvm::enumerate(expected)) {
    for (const Element &element : set.value()) {
      setOf[getKey(element)] = set.index();
    }
  }

  // Union-find over the equations.
  std::map<Key, Key> parents;

  auto find = [&](Key key) {
    while (parents.count(key) && parents[key] != key) {
      key = parents[key];
    }

    return key;
  };

  size_t numOfEquations = 0;

  for (const PotentialFamily &family : equations.potentials) {
    EXPECT_FALSE(graph.getVertex(family.vertex).isFlow);

    for (Point index : family.indices) {
      Key lhs = getKey(Element(family.vertex, index));
      Key rhs = getKey(Element(family.root, apply(*family.toRoot, index)));
      EXPECT_NE(lhs, rhs);
      EXPECT_TRUE(setOf.count(lhs) && setOf.count(rhs));

      if (setOf.count(lhs) && setOf.count(rhs)) {
        EXPECT_EQ(setOf[lhs], setOf[rhs]);
      }

      ++numOfEquations;
      Key lhsRoot = find(lhs);
      Key rhsRoot = find(rhs);
      EXPECT_NE(lhsRoot, rhsRoot) << "Redundant equation";
      parents[lhsRoot] = rhsRoot;
    }
  }

  EXPECT_EQ(numOfEquations, setOf.size() - expected.size());
  return numOfEquations;
}

/// Get the members of the sums of the flow families, checking that each sum
/// has no repeated member.
std::vector<std::set<Element>> getFlowSums(const ConnectionGraph &graph,
                                           const ConnectionEquations &eqs) {
  std::vector<std::set<Element>> result;

  for (const FlowFamily &family : eqs.flows) {
    EXPECT_TRUE(graph.getVertex(family.root).isFlow);

    for (Point rootIndex : family.rootIndices) {
      std::set<Element> members;
      size_t numOfMembers = 0;

      for (const FlowTerm &term : family.terms) {
        members.insert({term.vertex, apply(*term.fromRoot, rootIndex)});
        ++numOfMembers;
      }

      for (const FlowReduction &reduction : family.reductions) {
        for (Point iterator : reduction.iterators) {
          members.insert(
              {reduction.vertex,
               apply(*reduction.access, concat(rootIndex, iterator))});

          ++numOfMembers;
        }
      }

      EXPECT_EQ(members.size(), numOfMembers) << "Repeated member";
      result.push_back(std::move(members));
    }
  }

  return result;
}

/// Check that each flow set gets exactly one sum, whose terms are exactly its
/// members.
void checkFlows(const ConnectionGraph &graph,
                const ConnectionEquations &equations, const Sets &expected) {
  std::vector<std::set<Element>> sums = getFlowSums(graph, equations);
  Sets actual(sums.begin(), sums.end());
  EXPECT_EQ(sums.size(), actual.size()) << "Set summed more than once";
  EXPECT_EQ(actual, expected);
}

/// Compute the representatives and the equations, and check the equations
/// against the sets computed by the graph.
ConnectionEquations computeAndCheck(ConnectionGraph &graph) {
  graph.computeRepresentatives();
  ConnectionEquations equations = computeEquations(graph);
  checkPotentials(graph, equations, getSets(graph, false));
  checkFlows(graph, equations, getSets(graph, true));
  return equations;
}
} // namespace

/// Spec example, zero-based, for both the potential and the flow variables:
///   for k in 0:1 loop connect(r[k].n, r[k + 1].p); end for;
///   connect(r[2].n, g);
///   connect(r[0].p, g);
TEST(ConnectionEquations, ladderEquations) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId rnv = graph.addVertex({0, false, false, indices(0, 3)});
  VertexId rpv = graph.addVertex({1, false, false, indices(0, 3)});
  VertexId gv = graph.addVertex({2, false, false, indices(0, 1)});
  VertexId rni = graph.addVertex({3, false, true, indices(0, 3)});
  VertexId rpi = graph.addVertex({4, false, true, indices(0, 3)});
  VertexId gi = graph.addVertex({5, false, true, indices(0, 1)});

  for (auto [rn, rp, g] :
       {std::tuple(rnv, rpv, gv), std::tuple(rni, rpi, gi)}) {
    graph.addEdge(
        {indices(0, 2), rn, rp, shifted(&context, 0), shifted(&context, 1)});

    graph.addEdge(
        {indices(0, 1), rn, g, constant(&context, 2), constant(&context, 0)});

    graph.addEdge(
        {indices(0, 1), rp, g, constant(&context, 0), constant(&context, 0)});
  }

  graph.computeRepresentatives();
  ConnectionEquations equations = computeEquations(graph);

  auto sets = [](VertexId rn, VertexId rp, VertexId g) -> Sets {
    return {{{rn, Point(0)}, {rp, Point(1)}},
            {{rn, Point(1)}, {rp, Point(2)}},
            {{rn, Point(2)}, {g, Point(0)}, {rp, Point(0)}}};
  };

  EXPECT_EQ(checkPotentials(graph, equations, sets(rnv, rpv, gv)), 4u);
  checkFlows(graph, equations, sets(rni, rpi, gi));

  // One family for the sets of two elements, one for the set of three.
  EXPECT_EQ(equations.flows.size(), 2u);
}

/// The inside and outside sides of the same variable, connected element-wise:
/// no potential equation equates an element to itself.
TEST(ConnectionEquations, insideAndOutsideSameVariable) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId inside = graph.addVertex({7, false, false, indices(0, 3)});
  VertexId outside = graph.addVertex({7, true, false, indices(0, 3)});
  VertexId flowInside = graph.addVertex({8, false, true, indices(0, 3)});
  VertexId flowOutside = graph.addVertex({8, true, true, indices(0, 3)});

  graph.addEdge({indices(0, 3), inside, outside, shifted(&context, 0),
                 shifted(&context, 0)});

  graph.addEdge({indices(0, 3), flowInside, flowOutside, shifted(&context, 0),
                 shifted(&context, 0)});

  graph.computeRepresentatives();
  ConnectionEquations equations = computeEquations(graph);

  EXPECT_TRUE(equations.potentials.empty());
  ASSERT_EQ(equations.flows.size(), 1u);
  EXPECT_EQ(equations.flows[0].rootIndices, indices(0, 3));
  EXPECT_EQ(equations.flows[0].terms.size(), 2u);
  EXPECT_TRUE(equations.flows[0].reductions.empty());
  checkFlows(graph, equations, getSets(graph, true));
}

/// 'w[k]' ~ 'v[k]' (inside) and 'w[k]' ~ 'v[k]' (outside) for k in [0, 3),
/// 'w' being the representative: the two sides of 'v[k]' are one element,
/// equated to 'w[k]' once.
TEST(ConnectionEquations, insideAndOutsideSharedRepresentative) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId w = graph.addVertex({9, false, false, indices(0, 3)});
  VertexId inside = graph.addVertex({7, false, false, indices(0, 3)});
  VertexId outside = graph.addVertex({7, true, false, indices(0, 3)});
  VertexId flowW = graph.addVertex({10, false, true, indices(0, 3)});
  VertexId flowInside = graph.addVertex({8, false, true, indices(0, 3)});
  VertexId flowOutside = graph.addVertex({8, true, true, indices(0, 3)});

  graph.addEdge(
      {indices(0, 3), w, inside, shifted(&context, 0), shifted(&context, 0)});

  graph.addEdge(
      {indices(0, 3), outside, w, shifted(&context, 0), shifted(&context, 0)});

  graph.addEdge({indices(0, 3), flowW, flowInside, shifted(&context, 0),
                 shifted(&context, 0)});

  graph.addEdge({indices(0, 3), flowOutside, flowW, shifted(&context, 0),
                 shifted(&context, 0)});

  ConnectionEquations equations = computeAndCheck(graph);

  size_t numOfEquations = 0;

  for (const PotentialFamily &family : equations.potentials) {
    EXPECT_EQ(family.root, w);
    numOfEquations += family.indices.size();
  }

  EXPECT_EQ(numOfEquations, 3u);

  // The flow sums keep both sides, whose signs are applied at emission.
  ASSERT_EQ(equations.flows.size(), 1u);
  EXPECT_EQ(equations.flows[0].terms.size(), 3u);
}

/// A flow vertex without edges: each element sums to zero alone.
TEST(ConnectionEquations, unconnectedFlow) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId x = graph.addVertex({0, false, true, indices(0, 3)});

  ConnectionEquations equations = computeAndCheck(graph);

  EXPECT_TRUE(equations.potentials.empty());
  ASSERT_EQ(equations.flows.size(), 1u);
  EXPECT_EQ(equations.flows[0].root, x);
  EXPECT_EQ(equations.flows[0].rootIndices, indices(0, 3));
  ASSERT_EQ(equations.flows[0].terms.size(), 1u);
  EXPECT_EQ(equations.flows[0].terms[0].vertex, x);
  EXPECT_TRUE(equations.flows[0].reductions.empty());

  for (int64_t k = 0; k < 3; ++k) {
    EXPECT_EQ(apply(*equations.flows[0].terms[0].fromRoot, Point(k)), Point(k));
  }
}

/// 'x[k]' ~ 'x[k + 1]' for k in [0, 999), with 'x' flow: the members of the
/// only set are summed by a single reduction over the whole chain.
TEST(ConnectionEquations, selfChainReduction) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  int64_t size = 1000;
  VertexId x = graph.addVertex({0, false, true, indices(0, size)});

  graph.addEdge(
      {indices(0, size - 1), x, x, shifted(&context, 0), shifted(&context, 1)});

  graph.computeRepresentatives();
  ConnectionEquations equations = computeEquations(graph);

  EXPECT_TRUE(equations.potentials.empty());
  ASSERT_EQ(equations.flows.size(), 1u);
  const FlowFamily &family = equations.flows[0];
  EXPECT_EQ(family.root, x);
  EXPECT_EQ(family.rootIndices.size(), 1u);
  EXPECT_TRUE(family.terms.empty());
  ASSERT_EQ(family.reductions.size(), 1u);
  EXPECT_EQ(family.reductions[0].vertex, x);

  EXPECT_EQ(family.reductions[0].iterators,
            MultidimensionalRange(Range(0, size)));

  checkFlows(graph, equations, getSets(graph, true));
}

/// 'x[i, j]' ~ 'x[i, j + 1]' for i, j in [0, 4) x [0, 4), with x flow of
/// size 4 x 5: one reduction per row, in a single family.
TEST(ConnectionEquations, selfChain2DReduction) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  IndexSet xIndices(MultidimensionalRange({Range(0, 4), Range(0, 5)}));
  VertexId x = graph.addVertex({0, false, true, xIndices});

  mlir::AffineExpr i = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr j = mlir::getAffineDimExpr(1, &context);

  graph.addEdge(
      {IndexSet(MultidimensionalRange({Range(0, 4), Range(0, 4)})), x, x,
       AccessFunction::build(mlir::AffineMap::get(2, 0, {i, j}, &context)),
       AccessFunction::build(
           mlir::AffineMap::get(2, 0, {i, j + 1}, &context))});

  ConnectionEquations equations = computeAndCheck(graph);

  ASSERT_EQ(equations.flows.size(), 1u);
  EXPECT_EQ(equations.flows[0].rootIndices.size(), 4u);
  EXPECT_TRUE(equations.flows[0].terms.empty());
  ASSERT_EQ(equations.flows[0].reductions.size(), 1u);

  EXPECT_EQ(equations.flows[0].reductions[0].iterators,
            MultidimensionalRange(Range(0, 5)));
}

/// 'a[2k]' ~ 'b[k]' for k in [0, 5), for both the potential and the flow
/// variables: the pieces with a strided map are processed point by point.
TEST(ConnectionEquations, strided) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId av = graph.addVertex({0, false, false, indices(0, 10)});
  VertexId bv = graph.addVertex({1, false, false, indices(0, 5)});
  VertexId ai = graph.addVertex({2, false, true, indices(0, 10)});
  VertexId bi = graph.addVertex({3, false, true, indices(0, 5)});

  graph.addEdge(
      {indices(0, 5), av, bv, affine(&context, 2, 0), shifted(&context, 0)});

  graph.addEdge(
      {indices(0, 5), ai, bi, affine(&context, 2, 0), shifted(&context, 0)});

  computeAndCheck(graph);
}

/// 'x[k]' ~ 'x[2k]' for k in [0, 5), with x flow of size 10.
TEST(ConnectionEquations, selfDoubling) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  graph.addVertex({0, false, true, indices(0, 10)});
  graph.addVertex({1, false, false, indices(0, 10)});

  for (VertexId x : {0u, 1u}) {
    graph.addEdge(
        {indices(0, 5), x, x, shifted(&context, 0), affine(&context, 2, 0)});
  }

  computeAndCheck(graph);
}

/// 'a[k]' ~ 'b[k]' and 'b[k]' ~ 'a[k + 1]' for k in [0, 4), plus 'a[4]' ~
/// 'a[0]': a cycle among two vertices, for both kinds of variables.
TEST(ConnectionEquations, cycle) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  for (bool isFlow : {false, true}) {
    VertexId a =
        graph.addVertex({isFlow ? 2u : 0u, false, isFlow, indices(0, 5)});
    VertexId b =
        graph.addVertex({isFlow ? 3u : 1u, false, isFlow, indices(0, 4)});

    graph.addEdge(
        {indices(0, 4), a, b, shifted(&context, 0), shifted(&context, 0)});

    graph.addEdge(
        {indices(0, 4), b, a, shifted(&context, 0), shifted(&context, 1)});

    graph.addEdge(
        {indices(0, 1), a, a, constant(&context, 4), constant(&context, 0)});
  }

  computeAndCheck(graph);
}

/// 'a[2k]' ~ 'b[k]' for k in [0, n), with flow variables: the pieces with a
/// strided map contribute point by point, with work growing linearly with n.
TEST(ConnectionEquations, stridedScaling) {
  auto getNumOfVisitedContributions = [](int64_t n) {
    mlir::MLIRContext context;
    ConnectionGraph graph(&context);

    VertexId a = graph.addVertex({0, false, true, indices(0, 2 * n)});
    VertexId b = graph.addVertex({1, false, true, indices(0, n)});

    graph.addEdge(
        {indices(0, n), a, b, affine(&context, 2, 0), shifted(&context, 0)});

    graph.computeRepresentatives();
    EquationsStatistics statistics;
    ConnectionEquations equations = computeEquations(graph, &statistics);
    checkFlows(graph, equations, getSets(graph, true));
    return statistics.numOfVisitedContributions;
  };

  size_t small = getNumOfVisitedContributions(100);
  size_t large = getNumOfVisitedContributions(200);
  EXPECT_LE(large, 3 * small) << small << " " << large;
}
