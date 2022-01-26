#ifndef MARCO_CODEGEN_PASSES_MODEL_SCALAREQUATION_H
#define MARCO_CODEGEN_PASSES_MODEL_SCALAREQUATION_H

#include "marco/codegen/passes/model/EquationImpl.h"

namespace marco::codegen
{
  /// Scalar Equation with Scalar Assignments.
  /// An equation that does not present induction variables, neither
  /// explicit or implicit.
  class ScalarEquation : public Equation::Impl
  {
    private:
      using DimensionAccess = ::marco::modeling::DimensionAccess;

    public:
      ScalarEquation(modelica::EquationOp equation, Variables variables);

      std::unique_ptr<Equation::Impl> clone() const override;

      std::unique_ptr<Impl> cloneIR() const override;
      void eraseIR() override;

      size_t getNumOfIterationVars() const override;

      long getRangeBegin(size_t inductionVarIndex) const override;
      long getRangeEnd(size_t inductionVarIndex) const override;

      std::vector<Access> getAccesses() const override;

      DimensionAccess resolveDimensionAccess(std::pair<mlir::Value, long> access) const override;

      /*
      void getWrites(llvm::SmallVectorImpl<ScalarEquation::Access>& accesses) const override;
      void getReads(llvm::SmallVectorImpl<ScalarEquation::Access>& accesses) const override;
       */
  };
}

#endif // MARCO_CODEGEN_PASSES_MODEL_SCALAREQUATION_H
