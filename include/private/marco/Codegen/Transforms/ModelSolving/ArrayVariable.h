#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_ARRAYVARIABLE_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_ARRAYVARIABLE_H

#include "marco/Codegen/Transforms/ModelSolving/VariableImpl.h"

namespace marco::codegen
{
  /// Variable implementation for array values.
  /// The class just acts as a forwarder.
  class ArrayVariable : public BaseVariable
  {
    public:
      ArrayVariable(mlir::modelica::VariableOp variableOp);

      std::unique_ptr<Variable> clone() const override;

      size_t getRank() const override;

      long getDimensionSize(size_t index) const override;

      modeling::IndexSet getIndices() const override;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_ARRAYVARIABLE_H
