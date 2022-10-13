#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_EQUATIONIMPL_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_EQUATIONIMPL_H

#include "marco/Codegen/Transforms/ModelSolving/Equation.h"
#include "marco/Codegen/Transforms/ModelSolving/Path.h"
#include "marco/Modeling/AccessFunction.h"

namespace marco::codegen
{
  class BaseEquation : public Equation
  {
    public:
      BaseEquation(mlir::modelica::EquationInterface equation, Variables variables);

      mlir::modelica::EquationInterface getOperation() const override;

      Variables getVariables() const override;

      void setVariables(Variables variables) override;

      mlir::Value getValueAtPath(const EquationPath& path) const override;

      void traversePath(
          const EquationPath& path,
          std::function<bool(mlir::Value)> traverseFn) const override;

      mlir::LogicalResult explicitate(
          mlir::OpBuilder& builder,
          const ::marco::modeling::IndexSet& equationIndices,
          const EquationPath& path) override;

      std::unique_ptr<Equation> cloneIRAndExplicitate(
          mlir::OpBuilder& builder,
          const ::marco::modeling::IndexSet& equationIndices,
          const EquationPath& path) const override;

      mlir::LogicalResult replaceInto(
          mlir::OpBuilder& builder,
          const ::marco::modeling::IndexSet& equationIndices,
          Equation& destination,
          const ::marco::modeling::AccessFunction& destinationAccessFunction,
          const EquationPath& destinationPath) const override;

      virtual mlir::func::FuncOp createTemplateFunction(
          llvm::ThreadPool& threadPool,
          mlir::OpBuilder& builder,
          llvm::StringRef functionName,
          ::marco::modeling::scheduling::Direction iterationDirection,
          std::vector<unsigned int>& usedVariables) const override;

    protected:
      mlir::modelica::EquationSidesOp getTerminator() const;

      mlir::LogicalResult explicitate(
          mlir::OpBuilder& builder,
          size_t argumentIndex,
          EquationPath::EquationSide side);

      mlir::LogicalResult groupLeftHandSide(
          mlir::OpBuilder& builder,
          const ::marco::modeling::IndexSet& equationIndices,
          const Access& access);

      std::pair<unsigned int, mlir::Value> getMultiplyingFactor(
          mlir::OpBuilder& builder,
          const ::marco::modeling::IndexSet& equationIndices,
          mlir::Value value,
          mlir::Value variable,
          const ::marco::modeling::IndexSet& variableIndices) const;

      virtual mlir::LogicalResult mapInductionVariables(
          mlir::OpBuilder& builder,
          mlir::BlockAndValueMapping& mapping,
          Equation& destination,
          const ::marco::modeling::AccessFunction& transformation) const = 0;

      void createIterationLoops(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ValueRange beginIndices,
          mlir::ValueRange endIndices,
          mlir::ValueRange steps,
          marco::modeling::scheduling::Direction iterationDirection,
          std::function<void(mlir::OpBuilder&, mlir::ValueRange)> bodyBuilder) const;

      virtual mlir::LogicalResult createTemplateFunctionBody(
          mlir::OpBuilder& builder,
          mlir::BlockAndValueMapping& mapping,
          mlir::ValueRange beginIndexes,
          mlir::ValueRange endIndexes,
          mlir::ValueRange steps,
          ::marco::modeling::scheduling::Direction iterationDirection) const = 0;

    private:
      mlir::Operation* equationOp;
      Variables variables;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_EQUATIONIMPL_H
