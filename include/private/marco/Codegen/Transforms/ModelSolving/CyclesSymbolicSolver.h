#ifndef MARCO_CYCLESSYMBOLICSOLVER_H
#define MARCO_CYCLESSYMBOLICSOLVER_H

#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

#include "marco/Codegen/Utils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <ginac/flags.h>
#include <ginac/ginac.h>

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
        public GiNaC::numeric::visitor,
        public GiNaC::symbol::visitor
  {
      private:
      mlir::OpBuilder& builder;
      mlir::Location loc;
      // Map from expression to Value, to be able to get the Value of a subexpression while traversing the expression.
      llvm::DenseMap<unsigned int, mlir::Value> expressionHashToValueMap;
      std::map<std::string, mlir::Value> symbolNameToValueMap;
      std::map<mlir::Value, mlir::Value> variableLoadingMapping;
      mlir::Value timeValue;

      public:
      explicit SymbolicVisitor(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          std::map<std::string, mlir::Value>& symbolValueMap
      ) : builder(builder), loc(loc), symbolNameToValueMap(symbolValueMap)
      {

      }

      void visit(const GiNaC::add & x) override {
        std::cerr << "Add\n" << x.nops() << '\n' << std::flush;
        // todo: loop to add also the third argument

        mlir::Value lhs = expressionHashToValueMap[x.op(0).gethash()];
        mlir::Value rhs = expressionHashToValueMap[x.op(1).gethash()];

        lhs.dump();
        rhs.dump();

        mlir::Type type = getMostGenericType(lhs.getType(), rhs.getType());

        mlir::Value value = builder.create<mlir::modelica::AddOp>(loc, type, lhs, rhs);
        expressionHashToValueMap[x.gethash()] = value;
      }

      void visit(const GiNaC::mul & x) override {
        std::cerr << "Mul\n" << x << '\n' << std::flush;

        mlir::Value lhs = expressionHashToValueMap[x.op(0).gethash()];
        mlir::Value rhs = expressionHashToValueMap[x.op(1).gethash()];

        mlir::Type type = getMostGenericType(lhs.getType(), rhs.getType());

        mlir::Value value = builder.create<mlir::modelica::MulOp>(loc, type, lhs, rhs);
        expressionHashToValueMap[x.gethash()] = value;
      }

      void visit(const GiNaC::power & x) override {
        std::cerr << "Power\n" << x << '\n' << std::flush;

        mlir::Value lhs = expressionHashToValueMap[x.op(0).gethash()];
        mlir::Value rhs = expressionHashToValueMap[x.op(1).gethash()];

        mlir::Type type = getMostGenericType(lhs.getType(), rhs.getType());

        mlir::Value value = builder.create<mlir::modelica::PowOp>(loc, type, lhs, rhs);
        expressionHashToValueMap[x.gethash()] = value;
      }

      void visit(const GiNaC::function & x) override {
        if (x.get_name() == "sin") {
          //todo: add time to expvalue mapping
          std::cerr << "Function\n" << x << '\n' << std::flush;

          mlir::Value lhs = expressionHashToValueMap[x.op(0).gethash()];

          mlir::Type type = lhs.getType();

          mlir::Value value = builder.create<mlir::modelica::SinOp>(loc, type, lhs);
          expressionHashToValueMap[x.gethash()] = value;
        }
      }

      void visit(const GiNaC::relational & x) override {
        if (x.info(GiNaC::info_flags::relation_equal)) {
          std::cerr << "Equals\n" << '\n' << std::flush;

          mlir::Value lhs = expressionHashToValueMap[x.op(0).gethash()];
          mlir::Value rhs = expressionHashToValueMap[x.op(1).gethash()];

          lhs = builder.create<mlir::modelica::EquationSideOp>(loc, lhs);
          rhs = builder.create<mlir::modelica::EquationSideOp>(loc, rhs);

          builder.create<mlir::modelica::EquationSidesOp>(loc, lhs, rhs);
        }
      }

      void visit(const GiNaC::numeric & x) override {
        std::cerr << "Numeric\n" << x << '\n' << std::flush;

        mlir::Attribute attribute;

        if (x.is_cinteger()) {
          attribute = mlir::modelica::IntegerAttr::get(builder.getContext(), x.to_int());
        } else if (x.is_real()) {
          attribute = mlir::modelica::RealAttr::get(builder.getContext(), x.to_double());
        } else {
          llvm_unreachable("Unknown variable type, aborting.");
        }

        mlir::Value value = builder.create<mlir::modelica::ConstantOp>(loc, attribute);
        expressionHashToValueMap[x.gethash()] = value;

      }

      void visit(const GiNaC::symbol & x) override {
        std::cerr << "Numeric\n" << x << '\n' << std::flush;

        mlir::Value value = expressionHashToValueMap[x.gethash()];
        if (value == nullptr) {
          if (x.get_name() == "time") {
            value = builder.create<mlir::modelica::TimeOp>(loc);
          } else {
            std::cerr << x.get_name() << std::flush;
            value = symbolNameToValueMap[x.get_name()];
            value.dump();
            // todo: create appropriate LoadOp / SubscriptionOp for arrays
            value = builder.create<mlir::modelica::LoadOp>(loc, value);
          }

          expressionHashToValueMap[x.gethash()] = value;
        }
      }
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
