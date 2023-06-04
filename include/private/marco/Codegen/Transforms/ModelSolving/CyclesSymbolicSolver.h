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
  struct SymbolInfo {
    GiNaC::symbol symbol;
    std::string variableName;
    mlir::Type variableType;
    std::vector<GiNaC::ex> indices;
    MatchedEquation* matchedEquation;
  };

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

  class SymbolicToModelicaEquationVisitor
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
      MatchedEquation* matchedEquation;
      // Map from expression to Value, to be able to get the Value of a subexpression while traversing the expression.
      llvm::DenseMap<unsigned int, mlir::Value> expressionHashToValueMap;
      std::map<std::string, SymbolInfo> symbolNameToInfoMap;

      public:
      explicit SymbolicToModelicaEquationVisitor(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          MatchedEquation* matchedEquation,
          std::map<std::string, SymbolInfo>& symbolNameToInfoMap
      );

      void visit(const GiNaC::add & x) override;
      void visit(const GiNaC::mul & x) override;
      void visit(const GiNaC::power & x) override;
      void visit(const GiNaC::function & x) override;
      void visit(const GiNaC::relational & x) override;
      void visit(const GiNaC::numeric & x) override;
      void visit(const GiNaC::symbol & x) override;
  };

  class ModelicaToSymbolicEquationVisitor
  {
      private:
      MatchedEquation* matchedEquation;
      std::map<std::string, SymbolInfo>& symbolNameToInfoMap;
      llvm::DenseMap<mlir::Value, GiNaC::ex> valueToExpressionMap;
      GiNaC::ex& solution;

      public:
      ModelicaToSymbolicEquationVisitor(
          MatchedEquation* matchedEquation,
          std::map<std::string, SymbolInfo>& symbolNameToInfoMap,
          GiNaC::ex& solution);

      void visit(mlir::modelica::VariableGetOp);
      void visit(mlir::modelica::SubscriptionOp);
      void visit(mlir::modelica::LoadOp);
      void visit(mlir::modelica::ConstantOp);
      void visit(mlir::modelica::TimeOp);
      void visit(mlir::modelica::NegateOp);
      void visit(mlir::modelica::AddOp);
      void visit(mlir::modelica::SubOp);
      void visit(mlir::modelica::MulOp);
      void visit(mlir::modelica::DivOp);
      void visit(mlir::modelica::PowOp);
      void visit(mlir::modelica::SinOp);
      void visit(mlir::modelica::EquationSideOp);
      void visit(mlir::modelica::EquationSidesOp);
  };
}

#endif//MARCO_CYCLESSYMBOLICSOLVER_H
