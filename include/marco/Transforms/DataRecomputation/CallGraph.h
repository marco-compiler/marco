#ifndef INCLUDE_MARCO_DIALECT_TRANSFORMS_DATARECOMPUTATION_CALLGRAPH_H
#define INCLUDE_MARCO_DIALECT_TRANSFORMS_DATARECOMPUTATION_CALLGRAPH_H

#include <llvm/ADT/DirectedGraph.h>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"


// Forward declarations

class DRCallGraphNode;

template <class NodeTy>
class DRCallGraphEdge  : public llvm::DGEdge<NodeTy, DRCallGraphEdge<NodeTy>>{
public:
  using Base = llvm::DGEdge<NodeTy, DRCallGraphEdge<NodeTy>>;

  // The operation that created this edge
  mlir::Operation *callOp = nullptr;

  DRCallGraphEdge() = delete;
  DRCallGraphEdge(DRCallGraphNode &target, mlir::Operation *op = nullptr)
    : Base(target), callOp(op) { }

  bool isEqualTo(const DRCallGraphEdge &E) const {
    return &Base::getTargetNode() == &E.getTargetNode() && callOp == E.callOp;
  }
};

class DRCallGraphNode {
public:
  explicit DRCallGraphNode(mlir::func::FuncOp funcOp)
    : function{funcOp} {}

  using EdgeListTy = llvm::SmallVector<DRCallGraphEdge<DRCallGraphNode>, 4>;
  using iterator = EdgeListTy::iterator;
  using const_iterator = EdgeListTy::const_iterator;

  iterator begin() { return outEdges.begin(); }
  iterator end() { return outEdges.end(); }

  const_iterator begin() const { return outEdges.begin(); }
  const_iterator end() const { return outEdges.end(); }

  void addEdge(const DRCallGraphEdge<DRCallGraphNode> &edge) { outEdges.push_back(edge); }

  bool hasEdgeTo(const DRCallGraphNode &node) const {
    for ( const auto &edge : outEdges ) {
      if ( &edge.getTargetNode() == &node) return true;
    }
    return false;
  }


private:
  mlir::func::FuncOp function;
  EdgeListTy outEdges;

  };




class DRCallGraph : public llvm::DirectedGraph<DRCallGraphNode, DRCallGraphEdge>
{
  using Base = llvm::DirectedGraph<DRCallGraphNode, DRCallGraphEdge>;
  using Base::addNode;
  using Base::connect;
};



#endif /* end of include guard:                                                \
        INCLUDE_MARCO_DIALECT_TRANSFORMS_DATARECOMPUTATION_CALLGRAPH_H*/
