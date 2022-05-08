#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_DERIVATIVESMAP_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_DERIVATIVESMAP_H

#include "marco/Modeling/IndexSet.h"
#include "mlir/IR/Value.h"
#include <map>

namespace marco::codegen
{
  class DerivativesMap
  {
    public:
      bool hasDerivative(unsigned int variable) const;

      unsigned int getDerivative(unsigned int variable) const;

      void setDerivative(unsigned int variable, unsigned int derivative);

      const modeling::IndexSet& getDerivedIndices(unsigned int variable) const;

      void setDerivedIndices(unsigned int variable, modeling::IndexSet indices);

      bool isDerivative(unsigned int variable) const;

      unsigned int getDerivedVariable(unsigned int derivative) const;

    private:
      std::map<unsigned int, unsigned int> derivatives;
      std::map<unsigned int, unsigned int> inverseDerivatives;
      std::map<unsigned int, modeling::IndexSet> derivedIndices;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_DERIVATIVESMAP_H
