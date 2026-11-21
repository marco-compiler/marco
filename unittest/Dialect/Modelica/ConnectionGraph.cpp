#include "marco/Dialect/Modelica/Transforms/ConnectionExpansion/ConnectionGraph.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"
#include <chrono>
#include <map>
#include <optional>
#include <set>

using namespace ::marco::modeling;
using namespace ::mlir::modelica::connection;

namespace {
/// Element of a connection set: a vertex and an index of it.
using Element = std::pair<VertexId, Point>;

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

/// Index set made of the points [begin, end).
IndexSet indices(int64_t begin, int64_t end) {
  return {MultidimensionalRange(Range(begin, end))};
}

/// Get the representative of an element by evaluating the piece containing
/// it, checking that exactly one piece of the vertex contains it.
Element getRepresentative(const ConnectionGraph &graph, VertexId vertex,
                          const Point &index) {
  std::optional<Element> result;

  for (const Piece &piece : graph.getPieces(vertex)) {
    if (!piece.domain.contains(index)) {
      continue;
    }

    EXPECT_FALSE(result) << "Element covered by multiple pieces";
    IndexSet mapped = piece.toRoot->map(index);
    EXPECT_EQ(mapped.size(), 1u);
    result = Element(piece.root, *mapped.begin());

    if (piece.isRoot) {
      EXPECT_EQ(piece.root, vertex);
      EXPECT_EQ(*mapped.begin(), index);
    }
  }

  if (!result) {
    ADD_FAILURE() << "Element not covered by any piece";
    return Element(vertex, index);
  }

  return *result;
}

/// Build the one-dimensional access function 'k -> factor * k + offset'.
std::unique_ptr<AccessFunction> affine(mlir::MLIRContext *context,
                                       int64_t factor, int64_t offset) {
  mlir::AffineExpr result = mlir::getAffineDimExpr(0, context) * factor +
                            mlir::getAffineConstantExpr(offset, context);

  return AccessFunction::build(mlir::AffineMap::get(1, 0, result, context));
}

/// Build the access function '(d0, ..., dn) -> dimension'.
std::unique_ptr<AccessFunction>
projection(mlir::MLIRContext *context, unsigned numOfDims, unsigned dimension) {
  mlir::AffineExpr result = mlir::getAffineDimExpr(dimension, context);

  return AccessFunction::build(
      mlir::AffineMap::get(numOfDims, 0, result, context));
}

/// Compute the representatives and get the connection sets, checking that
/// the pieces cover each element exactly once.
std::set<std::set<Element>> getSets(ConnectionGraph &graph,
                                    size_t &numOfPieces) {
  graph.computeRepresentatives();

  std::map<Element, std::set<Element>> sets;
  numOfPieces = 0;

  for (VertexId vertex = 0; vertex < graph.getNumVertices(); ++vertex) {
    numOfPieces += graph.getPieces(vertex).size();

    for (Point index : graph.getVertex(vertex).indices) {
      Element representative = getRepresentative(graph, vertex, index);
      sets[representative].insert(Element(vertex, index));
    }
  }

  std::set<std::set<Element>> result;

  for (const auto &entry : sets) {
    result.insert(entry.second);
  }

  return result;
}
} // namespace

/// Spec example, zero-based:
///   for k in 0:1 loop connect(r[k].n, r[k + 1].p); end for;   (A)
///   connect(r[2].n, g);                                        (B)
///   connect(r[0].p, g);                                        (C)
/// The scalar 'g' and the equations not nested in for-equations have the
/// one-point index set {0}.
TEST(ConnectionGraph, ladder) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId rn = graph.addVertex({0, false, false, indices(0, 3)});
  VertexId rp = graph.addVertex({1, false, false, indices(0, 3)});
  VertexId g = graph.addVertex({2, false, false, indices(0, 1)});
  EXPECT_EQ(graph.getNumVertices(), 3u);

  graph.addEdge(
      {indices(0, 2), rn, rp, shifted(&context, 0), shifted(&context, 1)});

  graph.addEdge(
      {indices(0, 1), rn, g, constant(&context, 2), constant(&context, 0)});

  graph.addEdge(
      {indices(0, 1), rp, g, constant(&context, 0), constant(&context, 0)});

  graph.computeRepresentatives();

  std::map<Element, std::set<Element>> sets;
  size_t numOfPieces = 0;

  for (VertexId vertex = 0; vertex < graph.getNumVertices(); ++vertex) {
    numOfPieces += graph.getPieces(vertex).size();

    for (Point index : graph.getVertex(vertex).indices) {
      Element representative = getRepresentative(graph, vertex, index);
      sets[representative].insert(Element(vertex, index));
    }
  }

  std::set<std::set<Element>> expected = {
      {{rn, Point(0)}, {rp, Point(1)}},
      {{rn, Point(1)}, {rp, Point(2)}},
      {{rn, Point(2)}, {g, Point(0)}, {rp, Point(0)}}};

  std::set<std::set<Element>> actual;

  for (const auto &entry : sets) {
    actual.insert(entry.second);
  }

  EXPECT_EQ(actual, expected);
  EXPECT_LE(numOfPieces, 6u);
}

/// 'a[0]' ~ 'g', and 'a[k]' ~ 'b[k]' for k in [0, 10).
TEST(ConnectionGraph, junctionScalarAndArray) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId a = graph.addVertex({0, false, false, indices(0, 10)});
  VertexId b = graph.addVertex({1, false, false, indices(0, 10)});
  VertexId g = graph.addVertex({2, false, false, indices(0, 1)});

  graph.addEdge(
      {indices(0, 1), a, g, constant(&context, 0), constant(&context, 0)});

  graph.addEdge(
      {indices(0, 10), a, b, shifted(&context, 0), shifted(&context, 0)});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);

  std::set<std::set<Element>> expected = {
      {{a, Point(0)}, {b, Point(0)}, {g, Point(0)}}};

  for (int64_t k = 1; k < 10; ++k) {
    expected.insert({{a, Point(k)}, {b, Point(k)}});
  }

  EXPECT_EQ(actual, expected);
  EXPECT_LE(numOfPieces, 6u);
}

/// The same edge 'a[k]' ~ 'b[k]' added twice.
TEST(ConnectionGraph, duplicateEdge) {
  mlir::MLIRContext context;

  ConnectionGraph single(&context);
  VertexId a = single.addVertex({0, false, false, indices(0, 10)});
  VertexId b = single.addVertex({1, false, false, indices(0, 10)});

  single.addEdge(
      {indices(0, 10), a, b, shifted(&context, 0), shifted(&context, 0)});

  ConnectionGraph duplicate(&context);
  duplicate.addVertex({0, false, false, indices(0, 10)});
  duplicate.addVertex({1, false, false, indices(0, 10)});

  for (int i = 0; i < 2; ++i) {
    duplicate.addEdge(
        {indices(0, 10), a, b, shifted(&context, 0), shifted(&context, 0)});
  }

  size_t singlePieces, duplicatePieces;
  std::set<std::set<Element>> singleSets = getSets(single, singlePieces);
  std::set<std::set<Element>> duplicateSets =
      getSets(duplicate, duplicatePieces);

  EXPECT_EQ(duplicateSets, singleSets);
  EXPECT_EQ(singleSets.size(), 10u);

  for (VertexId vertex : {a, b}) {
    llvm::ArrayRef<Piece> singleVertexPieces = single.getPieces(vertex);
    llvm::ArrayRef<Piece> duplicateVertexPieces = duplicate.getPieces(vertex);
    ASSERT_EQ(duplicateVertexPieces.size(), singleVertexPieces.size());

    for (size_t i = 0, e = singleVertexPieces.size(); i < e; ++i) {
      EXPECT_EQ(duplicateVertexPieces[i].domain, singleVertexPieces[i].domain);
      EXPECT_EQ(duplicateVertexPieces[i].root, singleVertexPieces[i].root);
      EXPECT_EQ(*duplicateVertexPieces[i].toRoot,
                *singleVertexPieces[i].toRoot);
      EXPECT_EQ(duplicateVertexPieces[i].isRoot, singleVertexPieces[i].isRoot);
    }
  }
}

/// Equations with indices [0, 3) x [0, 4) connecting 'a[i]' and 'b[i]': the
/// dimension 'j' is unused.
TEST(ConnectionGraph, unusedEquationDimension) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId a = graph.addVertex({0, false, false, indices(0, 3)});
  VertexId b = graph.addVertex({1, false, false, indices(0, 3)});

  graph.addEdge({IndexSet(MultidimensionalRange({Range(0, 3), Range(0, 4)})), a,
                 b, projection(&context, 2, 0), projection(&context, 2, 0)});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);
  std::set<std::set<Element>> expected;

  for (int64_t i = 0; i < 3; ++i) {
    expected.insert({{a, Point(i)}, {b, Point(i)}});
  }

  EXPECT_EQ(actual, expected);
  EXPECT_LE(numOfPieces, 2u);
}

/// 'a[2k]' ~ 'b[k]' for k in [0, 5).
TEST(ConnectionGraph, strided) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId a = graph.addVertex({0, false, false, indices(0, 10)});
  VertexId b = graph.addVertex({1, false, false, indices(0, 5)});

  graph.addEdge(
      {indices(0, 5), a, b, affine(&context, 2, 0), shifted(&context, 0)});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);
  std::set<std::set<Element>> expected;

  for (int64_t k = 0; k < 5; ++k) {
    expected.insert({{a, Point(2 * k)}, {b, Point(k)}});
    expected.insert({{a, Point(2 * k + 1)}});
  }

  EXPECT_EQ(actual, expected);
}

/// 'a[2k]' ~ 'b[2k]' for k in [0, 3): neither side is invertible, so the
/// equations are united one at a time.
TEST(ConnectionGraph, stridedBothSides) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId a = graph.addVertex({0, false, false, indices(0, 6)});
  VertexId b = graph.addVertex({1, false, false, indices(0, 6)});

  graph.addEdge(
      {indices(0, 3), a, b, affine(&context, 2, 0), affine(&context, 2, 0)});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);
  std::set<std::set<Element>> expected;

  for (int64_t k = 0; k < 3; ++k) {
    expected.insert({{a, Point(2 * k)}, {b, Point(2 * k)}});
    expected.insert({{a, Point(2 * k + 1)}});
    expected.insert({{b, Point(2 * k + 1)}});
  }

  EXPECT_EQ(actual, expected);
}

/// The scalars 'g' and 'h' connected by the equations k in [0, 4): both
/// sides are constant, and the unused dimension is dropped.
TEST(ConnectionGraph, constantSides) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId g = graph.addVertex({0, false, false, indices(0, 1)});
  VertexId h = graph.addVertex({1, false, false, indices(0, 1)});

  graph.addEdge(
      {indices(0, 4), g, h, constant(&context, 0), constant(&context, 0)});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);
  std::set<std::set<Element>> expected = {{{g, Point(0)}, {h, Point(0)}}};

  EXPECT_EQ(actual, expected);
  EXPECT_EQ(numOfPieces, 2u);
}

/// 'a[k]' ~ 'b[k]', then 'c[k]' ~ 'a[k + 1]', then 'b[k]' ~ 'z[k]' for k in
/// D = [0, 3) u [5, 8). The root 'a' of 'b' becomes the child of 'z' on D,
/// so the non-root pieces of 'b' and 'c' are re-pointed through the
/// dependents of 'a', once per range of D.
TEST(ConnectionGraph, rePointDependents) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId z = graph.addVertex({0, false, false, indices(0, 10)});
  VertexId a = graph.addVertex({1, false, false, indices(0, 10)});
  VertexId b = graph.addVertex({2, false, false, indices(0, 10)});
  VertexId c = graph.addVertex({3, false, false, indices(0, 9)});

  graph.addEdge(
      {indices(0, 10), a, b, shifted(&context, 0), shifted(&context, 0)});

  graph.addEdge(
      {indices(0, 9), c, a, shifted(&context, 0), shifted(&context, 1)});

  IndexSet d(
      {MultidimensionalRange(Range(0, 3)), MultidimensionalRange(Range(5, 8))});

  graph.addEdge({d, b, z, shifted(&context, 0), shifted(&context, 0)});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);
  std::set<std::set<Element>> expected;

  for (int64_t k = 0; k < 10; ++k) {
    std::set<Element> set = {{a, Point(k)}, {b, Point(k)}};

    if (k >= 1) {
      set.insert({c, Point(k - 1)});
    }

    if (d.contains(Point(k))) {
      set.insert({z, Point(k)});
    } else {
      expected.insert({{z, Point(k)}});
    }

    expected.insert(set);
  }

  EXPECT_EQ(actual, expected);

  // 'z': root. 'a': root outside D, 'z[k]' on D. 'b': 'a[k]' outside D,
  // 'z[k]' on D. 'c': 'a[k + 1]' and 'z[k + 1]'.
  EXPECT_EQ(numOfPieces, 7u);
}

/// 'a[k]' ~ 'g', 'b[k]' ~ 'h', then 'a[k]' ~ 'b[k]' for k in [0, 100): the
/// maps of the last edge to the roots are both constant, and the edge must
/// not be processed one equation at a time.
TEST(ConnectionGraph, constantMapsToRoots) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId a = graph.addVertex({0, false, false, indices(0, 100)});
  VertexId b = graph.addVertex({1, false, false, indices(0, 100)});
  VertexId g = graph.addVertex({2, false, false, indices(0, 1)});
  VertexId h = graph.addVertex({3, false, false, indices(0, 1)});

  graph.addEdge(
      {indices(0, 100), a, g, shifted(&context, 0), constant(&context, 0)});

  graph.addEdge(
      {indices(0, 100), b, h, shifted(&context, 0), constant(&context, 0)});

  graph.addEdge(
      {indices(0, 100), a, b, shifted(&context, 0), shifted(&context, 0)});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);
  std::set<Element> set = {{g, Point(0)}, {h, Point(0)}};

  for (int64_t k = 0; k < 100; ++k) {
    set.insert({a, Point(k)});
    set.insert({b, Point(k)});
  }

  std::set<std::set<Element>> expected = {set};
  EXPECT_EQ(actual, expected);
  EXPECT_LE(numOfPieces, 4u);
  EXPECT_LE(graph.getNumOfExaminedPairs(), 12u)
      << graph.getNumOfExaminedPairs();
}

/// 'a[i]' ~ 'b[j]' for i, j in [0, 10) x [0, 10): each side uses one of the
/// dimensions, and the edge must not be processed one equation at a time.
TEST(ConnectionGraph, allToAll) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId a = graph.addVertex({0, false, false, indices(0, 10)});
  VertexId b = graph.addVertex({1, false, false, indices(0, 10)});

  graph.addEdge({IndexSet(MultidimensionalRange({Range(0, 10), Range(0, 10)})),
                 a, b, projection(&context, 2, 0), projection(&context, 2, 1)});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);
  std::set<Element> set;

  for (int64_t k = 0; k < 10; ++k) {
    set.insert({a, Point(k)});
    set.insert({b, Point(k)});
  }

  std::set<std::set<Element>> expected = {set};
  EXPECT_EQ(actual, expected);
  EXPECT_LE(numOfPieces, 4u);
  EXPECT_LE(graph.getNumOfExaminedPairs(), 12u)
      << graph.getNumOfExaminedPairs();
}

/// 'x[k]' ~ 'x[k + 1]' for k in [0, 999999): the closure of the translation
/// must be computed without per-element work.
TEST(ConnectionGraph, selfChain) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  int64_t size = 1000000;
  VertexId x = graph.addVertex({0, false, false, indices(0, size)});

  graph.addEdge(
      {indices(0, size - 1), x, x, shifted(&context, 0), shifted(&context, 1)});

  auto start = std::chrono::steady_clock::now();
  graph.computeRepresentatives();
  auto end = std::chrono::steady_clock::now();

#ifdef NDEBUG
  // The assertions of the index sets enumerate their points in debug builds:
  // there, the compactness is checked only by the counts below.
  EXPECT_LT(std::chrono::duration<double>(end - start).count(), 1.0);
#else
  (void)start;
  (void)end;
#endif

  EXPECT_LE(graph.getNumOfExaminedPairs(), 4u) << graph.getNumOfExaminedPairs();

  // The elements are too many to be enumerated: the pieces are checked to
  // cover the vertex and to map all the elements to the same representative.
  llvm::ArrayRef<Piece> pieces = graph.getPieces(x);
  EXPECT_LE(pieces.size(), 2u);

  IndexSet covered;
  size_t numOfElements = 0;
  IndexSet representatives;

  for (const Piece &piece : pieces) {
    EXPECT_EQ(piece.root, x);
    covered += piece.domain;
    numOfElements += piece.domain.size();
    representatives += piece.toRoot->map(piece.domain);
  }

  EXPECT_EQ(numOfElements, static_cast<size_t>(size));
  EXPECT_EQ(covered, indices(0, size));
  EXPECT_EQ(representatives.size(), 1u);

  for (int64_t k : {int64_t(0), int64_t(1), size / 2, size - 1}) {
    Element representative = getRepresentative(graph, x, Point(k));
    EXPECT_EQ(representative.first, x);
    EXPECT_TRUE(representatives.contains(representative.second));
  }
}

/// 'x[i, j]' ~ 'x[i, j + 1]' for i, j in [0, 4) x [0, 4), with x of size
/// 4 x 5: the sets are the rows.
TEST(ConnectionGraph, selfChain2D) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId x = graph.addVertex(
      {0, false, false,
       IndexSet(MultidimensionalRange({Range(0, 4), Range(0, 5)}))});

  mlir::AffineExpr i = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr j = mlir::getAffineDimExpr(1, &context);

  graph.addEdge(
      {IndexSet(MultidimensionalRange({Range(0, 4), Range(0, 4)})), x, x,
       AccessFunction::build(mlir::AffineMap::get(2, 0, {i, j}, &context)),
       AccessFunction::build(
           mlir::AffineMap::get(2, 0, {i, j + 1}, &context))});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);
  std::set<std::set<Element>> expected;

  for (int64_t row = 0; row < 4; ++row) {
    std::set<Element> set;

    for (int64_t column = 0; column < 5; ++column) {
      set.insert({x, Point({row, column})});
    }

    expected.insert(set);
  }

  EXPECT_EQ(actual, expected);
  EXPECT_LE(numOfPieces, 2u);
}

/// 'x[k]' ~ 'x[k + 2]' for k in [0, 8), with x of size 10: the sets are the
/// even and the odd elements.
TEST(ConnectionGraph, selfChainStride2) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId x = graph.addVertex({0, false, false, indices(0, 10)});

  graph.addEdge(
      {indices(0, 8), x, x, shifted(&context, 0), shifted(&context, 2)});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);
  std::set<Element> even, odd;

  for (int64_t k = 0; k < 10; ++k) {
    (k % 2 == 0 ? even : odd).insert({x, Point(k)});
  }

  std::set<std::set<Element>> expected = {even, odd};
  EXPECT_EQ(actual, expected);
}

/// 'x[k]' ~ 'x[k + 1]' for k in [0, 4), plus 'x[4]' ~ 'x[0]': the last edge
/// connects elements already in the same set.
TEST(ConnectionGraph, cycleNoRedundancy) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId x = graph.addVertex({0, false, false, indices(0, 5)});

  graph.addEdge(
      {indices(0, 4), x, x, shifted(&context, 0), shifted(&context, 1)});

  graph.addEdge(
      {indices(0, 1), x, x, constant(&context, 4), constant(&context, 0)});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);
  std::set<Element> set;

  for (int64_t k = 0; k < 5; ++k) {
    set.insert({x, Point(k)});
  }

  std::set<std::set<Element>> expected = {set};
  EXPECT_EQ(actual, expected);
  EXPECT_LE(numOfPieces, 2u);
}

/// 'x[k + 1]' ~ 'x[k]' for k in [0, 3) u [5, 8), with x of size 10: backward
/// translation over two separate runs.
TEST(ConnectionGraph, selfChainBackwardRuns) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId x = graph.addVertex({0, false, false, indices(0, 10)});

  graph.addEdge({IndexSet({MultidimensionalRange(Range(0, 3)),
                           MultidimensionalRange(Range(5, 8))}),
                 x, x, shifted(&context, 1), shifted(&context, 0)});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);

  auto run = [&](int64_t begin, int64_t end) {
    std::set<Element> set;

    for (int64_t k = begin; k < end; ++k) {
      set.insert({x, Point(k)});
    }

    return set;
  };

  std::set<std::set<Element>> expected = {run(0, 4), run(4, 5), run(5, 9),
                                          run(9, 10)};

  EXPECT_EQ(actual, expected);
  EXPECT_LE(numOfPieces, 3u);
}

/// 'x[i, j]' ~ 'x[i, j + 1]' for (i, j) in [0, 2) x [0, 2) u [0, 1) x [2, 3),
/// with x of size 2 x 4: the origins have different ranges along i.
TEST(ConnectionGraph, selfChain2DUnalignedRanges) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId x = graph.addVertex(
      {0, false, false,
       IndexSet(MultidimensionalRange({Range(0, 2), Range(0, 4)}))});

  mlir::AffineExpr i = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr j = mlir::getAffineDimExpr(1, &context);

  graph.addEdge(
      {IndexSet({MultidimensionalRange({Range(0, 2), Range(0, 2)}),
                 MultidimensionalRange({Range(0, 1), Range(2, 3)})}),
       x, x,
       AccessFunction::build(mlir::AffineMap::get(2, 0, {i, j}, &context)),
       AccessFunction::build(
           mlir::AffineMap::get(2, 0, {i, j + 1}, &context))});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);

  std::set<std::set<Element>> expected = {
      {{x, Point({0, 0})},
       {x, Point({0, 1})},
       {x, Point({0, 2})},
       {x, Point({0, 3})}},
      {{x, Point({1, 0})}, {x, Point({1, 1})}, {x, Point({1, 2})}},
      {{x, Point({1, 3})}}};

  EXPECT_EQ(actual, expected);
}

/// 'x[i, j]' ~ 'x[j, i]' for i, j in [0, 3) x [0, 3): the maps agree on the
/// first equation but are not a translation.
TEST(ConnectionGraph, selfTranspose) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId x = graph.addVertex(
      {0, false, false,
       IndexSet(MultidimensionalRange({Range(0, 3), Range(0, 3)}))});

  mlir::AffineExpr i = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr j = mlir::getAffineDimExpr(1, &context);

  graph.addEdge(
      {IndexSet(MultidimensionalRange({Range(0, 3), Range(0, 3)})), x, x,
       AccessFunction::build(mlir::AffineMap::get(2, 0, {i, j}, &context)),
       AccessFunction::build(mlir::AffineMap::get(2, 0, {j, i}, &context))});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);
  std::set<std::set<Element>> expected;

  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t column = row; column < 3; ++column) {
      expected.insert({{x, Point({row, column})}, {x, Point({column, row})}});
    }
  }

  EXPECT_EQ(actual, expected);
}

/// 'x[k]' ~ 'x[2k]' for k in [0, 5), with x of size 10: the maps agree on
/// the first equation but are not a translation.
TEST(ConnectionGraph, selfDoubling) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId x = graph.addVertex({0, false, false, indices(0, 10)});

  graph.addEdge(
      {indices(0, 5), x, x, shifted(&context, 0), affine(&context, 2, 0)});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);

  auto set = [&](std::initializer_list<int64_t> elements) {
    std::set<Element> result;

    for (int64_t k : elements) {
      result.insert({x, Point(k)});
    }

    return result;
  };

  std::set<std::set<Element>> expected = {
      set({0}), set({1, 2, 4, 8}), set({3, 6}), set({5}), set({7}), set({9})};

  EXPECT_EQ(actual, expected);
}

/// 'y[0]' ~ 'x[2]', then 'x[k]' ~ 'x[2k]' for k in [0, 5), then 'w[0]' ~
/// 'x[1]', with x of size 10: the representative of y[0] is moved by the
/// self-relation within x, and then once more by the last edge.
TEST(ConnectionGraph, pointwiseSelfRelationDependents) {
  mlir::MLIRContext context;
  ConnectionGraph graph(&context);

  VertexId w = graph.addVertex({0, false, false, indices(0, 1)});
  VertexId x = graph.addVertex({1, false, false, indices(0, 10)});
  VertexId y = graph.addVertex({2, false, false, indices(0, 1)});

  graph.addEdge(
      {indices(0, 1), y, x, constant(&context, 0), constant(&context, 2)});

  graph.addEdge(
      {indices(0, 5), x, x, shifted(&context, 0), affine(&context, 2, 0)});

  graph.addEdge(
      {indices(0, 1), w, x, constant(&context, 0), constant(&context, 1)});

  size_t numOfPieces;
  std::set<std::set<Element>> actual = getSets(graph, numOfPieces);

  auto set = [&](std::initializer_list<int64_t> elements) {
    std::set<Element> result;

    for (int64_t k : elements) {
      result.insert({x, Point(k)});
    }

    return result;
  };

  std::set<Element> first = set({1, 2, 4, 8});
  first.insert({w, Point(0)});
  first.insert({y, Point(0)});

  std::set<std::set<Element>> expected = {
      first, set({0}), set({3, 6}), set({5}), set({7}), set({9})};

  EXPECT_EQ(actual, expected);
}

/// 'a[2k]' ~ 'b[3k]' and 'x[k]' ~ 'x[2k]' for k in [0, n): neither side can
/// be inverted, so the equations are united one at a time, with work growing
/// linearly with n.
TEST(ConnectionGraph, pointwiseScaling) {
  auto getNumOfVisitedPieces = [](int64_t n) {
    mlir::MLIRContext context;
    ConnectionGraph graph(&context);

    VertexId a = graph.addVertex({0, false, false, indices(0, 2 * n)});
    VertexId b = graph.addVertex({1, false, false, indices(0, 3 * n)});
    VertexId x = graph.addVertex({2, false, false, indices(0, 2 * n)});

    graph.addEdge(
        {indices(0, n), a, b, affine(&context, 2, 0), affine(&context, 3, 0)});

    graph.addEdge(
        {indices(0, n), x, x, shifted(&context, 0), affine(&context, 2, 0)});

    graph.computeRepresentatives();
    return graph.getNumOfVisitedPieces();
  };

  size_t small = getNumOfVisitedPieces(100);
  size_t large = getNumOfVisitedPieces(200);
  EXPECT_LE(large, 3 * small) << small << " " << large;
}
