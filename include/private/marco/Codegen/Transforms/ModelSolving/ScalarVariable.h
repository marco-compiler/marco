#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SCALARVARIABLE_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SCALARVARIABLE_H

#include "marco/Codegen/Transforms/ModelSolving/VariableImpl.h"

namespace marco::codegen
{
  /// Variable implementation for scalar values.
  /// The arrays declaration are kept untouched within the IR, but they
  /// are masked by this class as arrays with just one element.
  class ScalarVariable : public BaseVariable
  {
    public:
      ScalarVariable(mlir::Value value);

      std::unique_ptr<Variable> clone() const override;

      size_t getRank() const override;

      long getDimensionSize(size_t index) const override;

      modeling::IndexSet getIndices() const override;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SCALARVARIABLE_H
