#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SCALAREQUATION_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SCALAREQUATION_H

#include "marco/Codegen/Transforms/ModelSolving/EquationImpl.h"

namespace marco::codegen
{
  /// Scalar Equation with Scalar Assignments.
  /// An equation that does not present induction variables, neither
  /// explicit or implicit.
  class ScalarEquation : public BaseEquation
  {
    private:
      using DimensionAccess = ::marco::modeling::DimensionAccess;

    public:
      ScalarEquation(mlir::modelica::EquationInterface equation, Variables variables);

      std::unique_ptr<Equation> clone() const override;

      mlir::modelica::EquationInterface cloneIR() const override;

      void eraseIR() override;

      void dumpIR(llvm::raw_ostream& os) const override;

      size_t getNumOfIterationVars() const override;

      modeling::IndexSet getIterationRanges() const override;

      std::vector<Access> getAccesses() const override;

      DimensionAccess resolveDimensionAccess(std::pair<mlir::Value, long> access) const override;

      Access getAccessAtPath(const EquationPath& path) const override;

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
          llvm::StringMap<mlir::Value>& variablesMap,
          ::marco::modeling::scheduling::Direction iterationDirection) const override;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SCALAREQUATION_H
