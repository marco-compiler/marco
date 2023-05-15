#ifndef MARCO_CYCLESSYMBOLICSOLVER_H
#define MARCO_CYCLESSYMBOLICSOLVER_H

#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DirectedGraph.h"

#include <algorithm>

namespace marco::codegen {
  class CyclesSymbolicSolver
  {
  private:
    mlir::OpBuilder& builder;

  public:
    CyclesSymbolicSolver(mlir::OpBuilder& builder);

    bool solve(Model<MatchedEquation>& model);

  };

  class OperandIterator {
    private:

  };

  class OperationEdge {

  };

  class ValueNode {
    private:
      mlir::Value value;
      ValueNode* father;
      std::vector<ValueNode*> children;

    public:
      ValueNode(
          mlir::Value value,
          ValueNode* father
      );

      mlir::Value getValue();
      std::vector<ValueNode*>& getChildren();
      void addChild(ValueNode*);
      ValueNode* getFather();
  };

  class OperationNode : llvm::DGNode<OperationNode, OperationEdge> {
  private:
    mlir::Operation* operation;
    OperationNode* next;
    OperationNode* prev;
    OperationNode* father;
    OperationNode* child;
    size_t childNumber;
    size_t numberOfChildren;
  public:
    OperationNode(
      mlir::Operation* operation,
      OperationNode* next,
      OperationNode* prev,
      OperationNode* father,
      OperationNode* child,
      size_t childNumber,
      size_t numberOfChildren
    );
    mlir::Operation* getOperation();
    void setNext(OperationNode* next);
    void setChild(OperationNode* child);
    OperationNode* getChild();
    OperationNode* getNext();
  };

  class EquationGraph {
    private:
      marco::codegen::MatchedEquation* equation;
      OperationNode* entryNode;
    public:
      explicit EquationGraph(MatchedEquation* equation);
      OperationNode* getEntryNode();
      void print();
      void erase();
  };

  class EquationValueGraph {
    private:
      marco::codegen::MatchedEquation* equation;
      ValueNode entryNode = ValueNode(nullptr, nullptr);

    public:
      explicit EquationValueGraph(MatchedEquation* equation);
      ValueNode* getEntryNode();
      void erase();
      void print();
      void walk(void (*func)(ValueNode*));
  };
}


namespace llvm
{
  template<>
  struct GraphTraits<const marco::codegen::EquationGraph>
  {
    using Graph = const marco::codegen::EquationGraph;
    using GraphPtr = Graph*;

    using NodeRef = typename mlir::Operation*;
    // Need an iterator that dereferences to a NodeRef
    using ChildIteratorType = typename marco::codegen::OperationNode*;

    static NodeRef getEntryNode(const GraphPtr& graph) {
      //return graph->getEntryNode().getOperation();
    }

    static ChildIteratorType child_begin(NodeRef node) {

    }

    static ChildIteratorType child_end(NodeRef node) {

    }
  };
}

#endif//MARCO_CYCLESSYMBOLICSOLVER_H
