#ifndef MARCO_CODEGEN_PASSES_MODEL_EQUATIONIMPL_H
#define MARCO_CODEGEN_PASSES_MODEL_EQUATIONIMPL_H

#include "marco/codegen/passes/model/Equation.h"
#include "marco/codegen/passes/model/Path.h"

namespace marco::codegen
{
  namespace impl
  {
    mlir::LogicalResult explicitate(
        mlir::OpBuilder& builder,
        modelica::EquationOp equationOp,
        const EquationPath& path);

    mlir::LogicalResult explicitate(
        mlir::OpBuilder& builder,
        modelica::EquationOp equationOp,
        size_t argumentIndex,
        EquationPath::EquationSide side);
  }

  class BaseEquation : public Equation
  {
    public:
      BaseEquation(modelica::EquationOp equation, Variables variables);

      modelica::EquationOp getOperation() const override;

      const Variables& getVariables() const override;

      void setVariables(Variables variables) override;

      virtual mlir::FuncOp createTemplateFunction(
          mlir::OpBuilder& builder,
          llvm::StringRef functionName,
          mlir::ValueRange vars,
          const EquationPath& path) const override;

    protected:
      virtual std::vector<mlir::Value> createTemplateFunctionLoops(
          mlir::OpBuilder& builder,
          mlir::ValueRange lowerBounds,
          mlir::ValueRange upperBounds,
          mlir::ValueRange steps) const;

      virtual void mapIterationVars(mlir::BlockAndValueMapping& mapping, mlir::ValueRange iterationVars) const;

    private:
      mlir::Operation* equationOp;
      Variables variables;
  };
}

#endif // MARCO_CODEGEN_PASSES_MODEL_EQUATIONIMPL_H
