#ifndef MARCO_DIALECT_MODELICA_TRANSFORMS_CONNECTIONEXPANSION_CONNECTIONGRAPH_H
#define MARCO_DIALECT_MODELICA_TRANSFORMS_CONNECTIONEXPANSION_CONNECTIONGRAPH_H

#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/IndexSet.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace mlir::modelica::connection {
using VertexId = unsigned;

/// Primitive variable of the connectors referenced by the connect-equations,
/// together with the side of the connector it belongs to.
struct Vertex {
  /// Opaque id of the primitive variable.
  unsigned variable;

  /// Whether the variable belongs to an outside connector.
  bool outside;

  /// Whether the variable is a flow one.
  bool isFlow;

  /// Indices of the elements of the vertex: the dimensions of the enclosing
  /// arrays of instances followed by the dimensions of the variable.
  marco::modeling::IndexSet indices;
};

/// Family of connect-equations between the elements of two vertices.
struct Edge {
  /// Indices of the equations of the family.
  marco::modeling::IndexSet equationIndices;

  VertexId lhs, rhs;

  /// Access functions mapping the equation indices to the indices of the
  /// left-hand and right-hand side vertices.
  std::unique_ptr<marco::modeling::AccessFunction> lhsAccess, rhsAccess;
};

/// Elements of a vertex sharing the way their representative is computed.
/// A root piece contains only representatives (toRoot is the identity and
/// root is the vertex itself); a non-root piece contains none.
struct Piece {
  marco::modeling::IndexSet domain;

  /// Vertex containing the representatives of the elements of the piece.
  VertexId root;

  /// Access function mapping the elements of the piece to the indices of
  /// their representatives.
  std::unique_ptr<marco::modeling::AccessFunction> toRoot;

  bool isRoot;
};

/// Graph of the connections among the elements of the connectors, computing
/// for every element the representative of its connection set.
class ConnectionGraph {
public:
  explicit ConnectionGraph(mlir::MLIRContext *context);

  /// Add a vertex. Initially, each of its elements is its own
  /// representative.
  VertexId addVertex(Vertex vertex);

  /// Add an edge. Its access functions must be affine. The dimensions of the
  /// equation indices not used by any of the access functions are dropped.
  void addEdge(Edge edge);

  size_t getNumVertices() const;

  const Vertex &getVertex(VertexId id) const;

  /// Compute the representatives of the connection sets, by processing the
  /// edges in insertion order.
  void computeRepresentatives();

  /// Get the pieces of a vertex. Their domains are disjoint and cover the
  /// indices of the vertex.
  llvm::ArrayRef<Piece> getPieces(VertexId id) const;

  /// Get the number of pairs of pieces with connected elements examined by
  /// the last computation of the representatives. It doesn't depend on the
  /// number of elements for offset, constant and permutation accesses.
  size_t getNumOfExaminedPairs() const;

  /// Get the number of pieces visited by the last computation of the
  /// representatives: each pair of pieces searched for connected elements,
  /// and each piece searched for elements to re-point or to merge, counts
  /// as one visit. It grows linearly with the number of elements processed
  /// point by point.
  size_t getNumOfVisitedPieces() const;

private:
  /// Unite the connection sets of the elements connected by an edge.
  void unite(const Edge &edge);

  /// Unite the connection sets of the elements connected by the given
  /// equations of an edge, one pair of pieces at a time. The processed
  /// equations are not searched again, as their elements stay connected.
  void uniteEquations(const Edge &edge,
                      marco::modeling::IndexSet equationIndices);

  /// Process the first pair of pieces of the vertices of an edge whose
  /// elements are connected by the given equations of the edge but don't
  /// have the same representatives yet. Return the processed equations, or
  /// std::nullopt if there is no such pair.
  std::optional<marco::modeling::IndexSet>
  uniteFirstPair(const Edge &edge,
                 const marco::modeling::IndexSet &equationIndices);

  /// Unite the sets of the elements connected by an edge whose sides already
  /// share the root vertex, but not the representatives, in case one of the
  /// sides has the same representative for all the equations: the
  /// representatives of the other side get it. Return false if neither side
  /// has a single representative.
  bool constantSelfUnion(VertexId root,
                         const marco::modeling::AccessFunction &lhsToRoot,
                         const marco::modeling::AccessFunction &rhsToRoot,
                         const marco::modeling::IndexSet &equationIndices);

  /// Unite the sets of the elements connected by an edge whose sides already
  /// share the root vertex, but not the representatives, in case the
  /// right-hand side map is the left-hand side one followed by a translation
  /// by one along a single dimension: each maximal run of connected elements
  /// along that dimension gets the representative in its first slice.
  /// Return false for the other maps.
  bool selfUnion(VertexId root,
                 const marco::modeling::AccessFunction &lhsToRoot,
                 const marco::modeling::AccessFunction &rhsToRoot,
                 const marco::modeling::IndexSet &equationIndices);

  /// Unite the sets of the elements connected by the given equations of an
  /// edge, whose maps to the roots are of the form 'd + offset' or 'offset'
  /// but can't be inverted. Each range of equations is restricted to slices
  /// on which one of the maps can be inverted and that connect the same
  /// elements.
  void sliceUnion(const marco::modeling::AccessFunction &lhsToRoot,
                  const marco::modeling::AccessFunction &rhsToRoot,
                  const marco::modeling::IndexSet &equationIndices,
                  const Edge &edge);

  /// Unite the sets of the elements connected by the given equations of an
  /// edge, one equation at a time, as neither side can be inverted. The
  /// representatives of the connected elements are computed once, and
  /// re-pointed together.
  void pointwiseUnion(const marco::modeling::IndexSet &equationIndices,
                      const Edge &edge);

  /// Make the representatives with the given indices of the child root
  /// vertex no longer representatives: the elements having them as
  /// representatives get the ones given by 'childToParent' instead. The
  /// child and parent root vertices may coincide, provided that
  /// 'childToParent' maps to indices not among the given ones.
  void rePoint(VertexId childRoot, const marco::modeling::IndexSet &childRoots,
               VertexId parentRoot,
               const marco::modeling::AccessFunction &childToParent);

  /// Make the representatives of the child root vertex given by the keys of
  /// the map no longer representatives: the elements having one of them as
  /// representative get the element of a root vertex it is mapped to, which
  /// must be a representative.
  void rePointEach(
      VertexId childRoot,
      const std::map<marco::modeling::Point,
                     std::pair<VertexId, marco::modeling::Point>> &parents);

  /// Merge the pieces of a vertex computing the representatives in the same
  /// way.
  void mergePieces(VertexId vertex);

private:
  mlir::MLIRContext *context;
  llvm::SmallVector<Vertex> vertices;
  std::vector<Edge> edges;
  std::vector<llvm::SmallVector<Piece, 1>> pieces;

  /// For each root vertex, the vertices having pieces pointing to it.
  std::vector<llvm::SetVector<VertexId>> dependents;

  /// Vertices whose pieces have been modified since their last merge.
  llvm::SetVector<VertexId> modifiedVertices;

  size_t numOfExaminedPairs = 0;
  size_t numOfVisitedPieces = 0;
};
} // namespace mlir::modelica::connection

#endif // MARCO_DIALECT_MODELICA_TRANSFORMS_CONNECTIONEXPANSION_CONNECTIONGRAPH_H
