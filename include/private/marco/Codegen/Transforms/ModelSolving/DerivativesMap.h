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
      void setDerivative(unsigned int variable, unsigned int derivative);

      bool hasDerivative(unsigned int variable) const;

      unsigned int getDerivative(unsigned int variable) const;

      bool isDerivative(unsigned int variable) const;

      unsigned int getDerivedVariable(unsigned int derivative) const;

    private:
      std::map<unsigned int, unsigned int> derivatives;
      std::map<unsigned int, unsigned int> inverseDerivatives;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_DERIVATIVESMAP_H
