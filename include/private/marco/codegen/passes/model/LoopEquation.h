#ifndef MARCO_CODEGEN_PASSES_MODEL_LOOPEQUATION_H
#define MARCO_CODEGEN_PASSES_MODEL_LOOPEQUATION_H

#include "marco/codegen/passes/model/EquationImpl.h"

namespace marco::codegen
{
  /// Loop Equation.
  /// An equation that present explicit or implicit induction variables.
  class LoopEquation : public BaseEquation
  {
    private:
      using DimensionAccess = ::marco::modeling::DimensionAccess;

    public:
      LoopEquation(modelica::EquationOp equation, Variables variables);

      std::unique_ptr<Equation> clone() const override;

      size_t getNumOfIterationVars() const override;
      long getRangeBegin(size_t inductionVarIndex) const override;
      long getRangeEnd(size_t inductionVarIndex) const override;

      std::vector<Access> getAccesses() const override;

      DimensionAccess resolveDimensionAccess(std::pair<mlir::Value, long> access) const override;

      mlir::FuncOp createTemplateFunction(
          mlir::OpBuilder& builder,
          llvm::StringRef functionName,
          mlir::ValueRange vars,
          const EquationPath& path) const override;

    protected:
      std::vector<mlir::Value> createTemplateFunctionLoops(
          mlir::OpBuilder& builder,
          mlir::ValueRange lowerBounds,
          mlir::ValueRange upperBounds,
          mlir::ValueRange steps) const override;

      void mapIterationVars(mlir::BlockAndValueMapping& mapping, mlir::ValueRange variables) const override;

    private:
      size_t getNumberOfExplicitLoops() const;

      modelica::ForEquationOp getExplicitLoop(size_t index) const;

      size_t getNumberOfImplicitLoops() const;

      long getImplicitLoopStart(size_t index) const;
      long getImplicitLoopEnd(size_t index) const;
  };
}

#endif // MARCO_CODEGEN_PASSES_MODEL_LOOPEQUATION_H
