#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_DERIVATIVESMAP_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_DERIVATIVESMAP_H

#include "marco/Modeling/IndexSet.h"
#include "mlir/IR/Value.h"
#include <map>

namespace marco::codegen
{
  class DerivedVariable
  {
    public:
      DerivedVariable(unsigned int argNumber, modeling::MultidimensionalRange indices);

      unsigned int getArgNumber() const;

      const modeling::MultidimensionalRange& getIndices() const;

      bool operator<(const DerivedVariable& other) const;

    private:
      unsigned int argNumber;
      modeling::MultidimensionalRange indices;
  };

  class Derivative
  {
    public:
      Derivative(unsigned int argNumber, std::vector<long> offsets);

      unsigned int getArgNumber() const;

      llvm::ArrayRef<long> getOffsets() const;

    private:
      unsigned int argNumber;
      std::vector<long> offsets;
  };

  namespace impl
  {
    struct DerivedVariablesComparator : public std::binary_function<DerivedVariable, DerivedVariable, bool>
    {
      bool operator()(const DerivedVariable& lhs, const DerivedVariable& rhs) const;
    };
  }

  class DerivativesMap
  {
    public:
      void addDerivedIndices(unsigned int variable, modeling::MultidimensionalRange indices);

      modeling::IndexSet getDerivedIndices(unsigned int variable) const;

      void setDerivative(
          unsigned int variable,
          modeling::MultidimensionalRange variableIndices,
          unsigned int derivative,
          modeling::MultidimensionalRange derivativeIndices);

      bool hasDerivative(unsigned int variable) const;

      std::vector<std::pair<const modeling::MultidimensionalRange*, const Derivative*>> getDerivative(unsigned int variable) const;

      const Derivative& getDerivative(unsigned int variable, const modeling::MultidimensionalRange& indices) const;

      bool isDerivative(unsigned int variable) const;

      const DerivedVariable* getDerivedVariable(unsigned int derivative) const;

    private:
      std::map<unsigned int, modeling::IndexSet> derivedIndices;
      std::map<DerivedVariable, Derivative, impl::DerivedVariablesComparator> derivatives;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_DERIVATIVESMAP_H
