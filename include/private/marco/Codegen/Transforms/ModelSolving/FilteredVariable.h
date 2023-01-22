#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_FILTEREDVARIABLE_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_FILTEREDVARIABLE_H

#include "marco/Codegen/Transforms/ModelSolving/VariableImpl.h"

namespace marco::codegen
{
  /// Variable implementation for array values.
  /// The class just acts as a forwarder.
  class FilteredVariable : public Variable
  {
    public:
      using Id = Variable::Id;

      FilteredVariable(std::unique_ptr<Variable> variable, modeling::IndexSet indices);

      FilteredVariable(const FilteredVariable& other);

      ~FilteredVariable();

      FilteredVariable& operator=(const FilteredVariable& other);
      FilteredVariable& operator=(FilteredVariable&& other);

      friend void swap(FilteredVariable& first, FilteredVariable& second);

      std::unique_ptr<Variable> clone() const override;

      /// @name Forwarded methods
      /// {

      Id getId() const override;

      size_t getRank() const override;

      long getDimensionSize(size_t index) const override;

      mlir::Value getValue() const override;

      mlir::modelica::MemberCreateOp getDefiningOp() const override;

      bool isParameter() const override;

      /// }
      /// @name Modified methods
      /// {

      modeling::IndexSet getIndices() const override;

      /// }

    private:
      std::unique_ptr<Variable> variable;
      modeling::IndexSet indices;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_FILTEREDVARIABLE_H
