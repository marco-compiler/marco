#ifndef MARCO_CYCLESSYMBOLICSOLVER_H
#define MARCO_CYCLESSYMBOLICSOLVER_H

#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DirectedGraph.h"

#include <algorithm>
#include <ginac/ginac.h>
#include <ginac/flags.h>

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
      ValueNode* getChild(size_t);
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

  class SymbolicVisitor
      : public GiNaC::visitor,
        public GiNaC::add::visitor,
        public GiNaC::mul::visitor,
        public GiNaC::power::visitor,
        public GiNaC::function::visitor,
        public GiNaC::relational::visitor,
        public GiNaC::numeric::visitor
        //public GiNaC::basic::visitor
  {
      void visit(const GiNaC::add & x) override {
        std::cerr << "Add\n" << x.nops() << '\n' << std::flush;
      }

      void visit(const GiNaC::mul & x) override {
        std::cerr << "Mul\n" << x << '\n' << std::flush;
      }

      void visit(const GiNaC::power & x) override {
        std::cerr << "Power\n" << x << '\n' << std::flush;
      }

      void visit(const GiNaC::function & x) override {
        if (x.get_name() == "sin") {
          std::cerr << "Function\n" << x << '\n' << std::flush;
        }
      }

      void visit(const GiNaC::relational & x) override {
        if (x.info(GiNaC::info_flags::relation_equal)) {
          std::cerr << "Equals\n" << '\n' << std::flush;
        }
      }

      void visit(const GiNaC::numeric & x) override {
        std::cerr << "Numeric\n" << x << '\n' << std::flush;
      }

//      void visit(const GiNaC::basic & x) {
//        std::cerr << "Basic\n" << x << '\n' << std::flush;
//      }
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
