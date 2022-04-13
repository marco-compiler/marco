#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_LOOPEQUATION_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_LOOPEQUATION_H

#include "marco/Codegen/Transforms/ModelSolving/EquationImpl.h"

namespace marco::codegen
{
  /// Loop Equation.
  /// An equation that present explicit or implicit induction variables.
  class LoopEquation : public BaseEquation
  {
    private:
      using DimensionAccess = ::marco::modeling::DimensionAccess;

    public:
      LoopEquation(mlir::modelica::EquationOp equation, Variables variables);

      std::unique_ptr<Equation> clone() const override;

      mlir::modelica::EquationOp cloneIR() const override;

      void eraseIR() override;

      void dumpIR(llvm::raw_ostream& os) const override;

      size_t getNumOfIterationVars() const override;

      modeling::MultidimensionalRange getIterationRanges() const override;

      std::vector<Access> getAccesses() const override;

      DimensionAccess resolveDimensionAccess(std::pair<mlir::Value, long> access) const override;

      std::vector<mlir::Value> getInductionVariables() const override;

    protected:
      mlir::LogicalResult mapInductionVariables(
          mlir::OpBuilder& builder,
          mlir::BlockAndValueMapping& mapping,
          Equation& destination,
          const ::marco::modeling::AccessFunction& transformation) const override;

      mlir::LogicalResult createTemplateFunctionBody(
          mlir::OpBuilder& builder,
          mlir::BlockAndValueMapping& mapping,
          mlir::ValueRange beginIndexes,
          mlir::ValueRange endIndexes,
          mlir::ValueRange steps,
          ::marco::modeling::scheduling::Direction iterationDirection) const override;

    private:
      size_t getNumberOfExplicitLoops() const;

      std::vector<mlir::modelica::ForEquationOp> getExplicitLoops() const;

      mlir::modelica::ForEquationOp getExplicitLoop(size_t index) const;

      size_t getNumberOfImplicitLoops() const;

      std::vector<modeling::Range> getImplicitLoops() const;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_LOOPEQUATION_H
