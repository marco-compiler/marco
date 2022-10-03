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
          mlir::OpBuilder& builder,
          llvm::StringRef functionName,
          mlir::ValueRange vars,
          ::marco::modeling::scheduling::Direction iterationDirection) const override;

      mlir::LogicalResult getCoefficients(
          mlir::OpBuilder& builder,
          std::vector<mlir::Value>& coefficients,
          mlir::Value& constantTerm,
          const modeling::IndexSet& equationIndices) const override;

      mlir::LogicalResult getSideCoefficients(
          mlir::OpBuilder& builder,
          std::vector<mlir::Value>& coefficients,
          mlir::Value& constantTerm,
          std::vector<mlir::Value> values,
          EquationPath::EquationSide side,
          const modeling::IndexSet& equationIndices) const override;

      mlir::LogicalResult convertAndCollectSide(
          mlir::OpBuilder& builder,
          std::vector<mlir::Value>& output,
          EquationPath::EquationSide side) const override;

      void replaceSides(
          mlir::OpBuilder& builder,
          mlir::Value lhs,
          mlir::Value rhs) const override;

      size_t getSizeUntilVariable(
          size_t index) const override;

      size_t getFlatAccessIndex(
          const Access& access,
          const ::marco::modeling::IndexSet& equationIndices,
          const ::marco::modeling::IndexSet& variableIndices) const override;

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
