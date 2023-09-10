#ifndef MARCO_CYCLESSYMBOLICSOLVER_H
#define MARCO_CYCLESSYMBOLICSOLVER_H

#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Modeling/Cycles.h"


#include "marco/Codegen/Utils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <ginac/flags.h>
#include <ginac/ginac.h>

namespace marco::codegen {
  struct MatchedEquationSubscription
  {
    MatchedEquationSubscription(MatchedEquation* equation, llvm::ArrayRef<modeling::IndexSet> subscriptionIndices)
        : equation(std::move(equation))
    {
      for (const auto& indices : subscriptionIndices) {
        this->solvedIndices += indices;
      }
    }

    // The equation which originally had cycles.
    // The pointer refers to the original equation, which presented (and still presents) cycles.
    MatchedEquation* equation;

    // The indices of the equations for which the cycles have been solved
    modeling::IndexSet solvedIndices;
  };

  /// An equation which originally presented cycles but now, for some indices, does not anymore.
  struct SolvedEquation
  {
    SolvedEquation(const Equation* equation, llvm::ArrayRef<modeling::IndexSet> solvedIndices)
        : equation(std::move(equation))
    {
      for (const auto& indices : solvedIndices) {
        this->solvedIndices += indices;
      }
    }

    // The equation which originally had cycles.
    // The pointer refers to the original equation, which presented (and still presents) cycles.
    const Equation* equation;

    // The indices of the equations for which the cycles have been solved
    modeling::IndexSet solvedIndices;
  };

  struct SymbolInfo {
    GiNaC::symbol symbol;
    std::string variableName;
    mlir::Type variableType;
    std::vector<GiNaC::ex> indices;
    MatchedEquation* matchedEquation;
    modeling::IndexSet subscriptionIndices;
  };

  class CyclesSymbolicSolver
  {
  private:
    mlir::OpBuilder& builder;

    // The equations which originally had cycles but have been partially or fully solved.
    std::vector<SolvedEquation> solvedEquations_;

    // The newly created equations which has no cycles anymore.
    Equations<MatchedEquation> newEquations_;

    // The cycles that can't be solved by substitution.
    std::vector<MatchedEquation*> unsolvedCycles_;

  public:
    void addSolvedEquation(
        std::vector<SolvedEquation>& solvedEquations,
        Equation* const equation,
        modeling::IndexSet indices)
    {
        auto it = llvm::find_if(solvedEquations, [&](SolvedEquation& solvedEquation) {
          return solvedEquation.equation == equation;
        });

        if (it != solvedEquations.end()) {
          modeling::IndexSet& solvedIndices = it->solvedIndices;
          solvedIndices += indices;
        } else {
          solvedEquations.push_back(SolvedEquation(equation, indices));
        }
    }

    bool hasSolvedEquation(Equation* const equation, modeling::IndexSet indices) const
    {
        auto it = llvm::find_if(solvedEquations_, [&](const SolvedEquation& solvedEquation) {
          return solvedEquation.equation == equation && solvedEquation.solvedIndices.contains(indices);
        });

        return it != solvedEquations_.end();
    }

    explicit CyclesSymbolicSolver(mlir::OpBuilder& builder);

    bool solve(const std::vector<MatchedEquationSubscription>& equations);

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
      MatchedEquationSubscription matchedEquation;
      // Map from expression to Value, to be able to get the Value of a subexpression while traversing the expression.
      std::map<GiNaC::ex, mlir::Value, GiNaC::ex_is_less> expressionHashToValueMap;
      std::map<std::string, SymbolInfo> symbolNameToInfoMap;
      std::vector<std::pair<llvm::APInt, llvm::APInt>> matchedEquationForEquationRanges;

      public:
      explicit SymbolicToModelicaEquationVisitor(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          MatchedEquationSubscription matchedEquation,
          std::map<std::string, SymbolInfo> symbolNameToInfoMap
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
      mlir::modelica::MatchedEquationInstanceOp* equationInstance;
      std::map<std::string, SymbolInfo>& symbolNameToInfoMap;
      llvm::DenseMap<mlir::Value, GiNaC::ex> valueToExpressionMap;
      GiNaC::ex& solution;
      modeling::IndexSet subscriptionIndices;
      size_t numberOfForLoops;

      public:
      ModelicaToSymbolicEquationVisitor(
          mlir::modelica::MatchedEquationInstanceOp* equationInstance,
          std::map<std::string, SymbolInfo>& symbolNameToInfoMap,
          GiNaC::ex& solution, modeling::IndexSet subscriptionIndices);

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
