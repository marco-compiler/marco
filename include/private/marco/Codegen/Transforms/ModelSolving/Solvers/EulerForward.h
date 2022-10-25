#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SOLVERS_EULERFORWARD_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SOLVERS_EULERFORWARD_H

#include "marco/Codegen/Transforms/ModelSolving/Solvers/ModelSolver.h"

namespace marco::codegen
{
  class EulerForwardSolver : public ModelSolver
  {
    public:
      static constexpr llvm::StringLiteral calcICFunctionName = "calcIC";
      static constexpr llvm::StringLiteral updateNonStateVariablesFunctionName = "updateNonStateVariables";
      static constexpr llvm::StringLiteral updateStateVariablesFunctionName = "updateStateVariables";
      static constexpr llvm::StringLiteral incrementTimeFunctionName = "incrementTime";

      struct ConversionInfo
      {
        std::set<std::unique_ptr<Equation>> explicitEquations;
        std::map<ScheduledEquation*, Equation*> explicitEquationsMap;
        std::set<ScheduledEquation*> implicitEquations;
        std::set<ScheduledEquation*> cyclicEquations;
      };

      EulerForwardSolver(
          mlir::LLVMTypeConverter& typeConverter,
          VariableFilter& variablesFilter,
          double startTime,
          double endTime,
          double timeStep);

      mlir::LogicalResult solveICModel(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model) override;

      mlir::LogicalResult solveMainModel(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model) override;

    private:
      /// Create the function that computes the initial conditions.
      mlir::LogicalResult createCalcICFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          const ConversionInfo& conversionInfo) const;

      /// Create the functions that calculates the values that the non-state
      /// variables will have in the next iteration.
      mlir::LogicalResult createUpdateNonStateVariablesFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          const ConversionInfo& conversionInfo) const;

      /// Create the functions that calculates the values that the state
      /// variables will have in the next iteration.
      mlir::LogicalResult createUpdateStateVariablesFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model) const;

      /// Create the function to be used to increment the time.
      mlir::LogicalResult createIncrementTimeFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model) const;

      mlir::func::FuncOp createEquationFunction(
          mlir::OpBuilder& builder,
          const ScheduledEquation& equation,
          llvm::StringRef equationFunctionName,
          mlir::func::FuncOp templateFunction,
          mlir::TypeRange varsTypes) const;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SOLVERS_EULERFORWARD_H
