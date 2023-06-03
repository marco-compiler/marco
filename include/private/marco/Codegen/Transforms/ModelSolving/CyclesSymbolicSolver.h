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

    // The equations which originally had cycles but have been partially or fully solved.
    std::vector<MatchedEquation*> solvedEquations_;

    // The newly created equations which has no cycles anymore.
    Equations<MatchedEquation> newEquations_;

    // The cycles that can't be solved by substitution.
    std::vector<MatchedEquation*> unsolvedCycles_;

  public:
    explicit CyclesSymbolicSolver(mlir::OpBuilder& builder);

    bool solve(const std::set<MatchedEquation*>& equationSet);

    [[nodiscard]] Equations<MatchedEquation> getSolution() const;

    [[nodiscard]] bool hasUnsolvedCycles() const;

    [[nodiscard]] Equations<MatchedEquation> getUnsolvedEquations() const;

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
      MatchedEquation* equation;
      // Map from expression to Value, to be able to get the Value of a subexpression while traversing the expression.
      llvm::DenseMap<unsigned int, mlir::Value> expressionHashToValueMap;
      std::map<std::string, mlir::Type> nameToTypeMap;
      std::map<std::string, std::vector<GiNaC::ex>> nameToIndicesMap;

      public:
      explicit SymbolicVisitor(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          MatchedEquation* equation,
          std::map<std::string, mlir::Type>& nameToTypeMap,
          std::map<std::string, std::vector<GiNaC::ex>>& nameToIndicesMap
      );

      void visit(const GiNaC::add & x) override;
      void visit(const GiNaC::mul & x) override;
      void visit(const GiNaC::power & x) override;
      void visit(const GiNaC::function & x) override;
      void visit(const GiNaC::relational & x) override;
      void visit(const GiNaC::numeric & x) override;
      void visit(const GiNaC::symbol & x) override;
  };
}

#endif//MARCO_CYCLESSYMBOLICSOLVER_H
