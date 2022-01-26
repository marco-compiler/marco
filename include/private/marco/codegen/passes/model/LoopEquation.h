#ifndef MARCO_CODEGEN_PASSES_MODEL_LOOPEQUATION_H
#define MARCO_CODEGEN_PASSES_MODEL_LOOPEQUATION_H

#include "marco/codegen/passes/model/EquationImpl.h"

namespace marco::codegen
{
  /// Loop Equation.
  /// An equation that present explicit or implicit induction variables.
  class LoopEquation : public Equation::Impl
  {
    private:
      using DimensionAccess = ::marco::modeling::DimensionAccess;

    public:
      LoopEquation(modelica::EquationOp equation, Variables variables);

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

    private:
      size_t getNumberOfExplicitLoops() const;

      modelica::ForEquationOp getExplicitLoop(size_t index) const;

      size_t getNumberOfImplicitLoops() const;

      long getImplicitLoopStart(size_t index) const;
      long getImplicitLoopEnd(size_t index) const;
  };
}

#endif // MARCO_CODEGEN_PASSES_MODEL_LOOPEQUATION_H
