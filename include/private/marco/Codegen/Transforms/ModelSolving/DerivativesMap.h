#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_DERIVATIVESMAP_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_DERIVATIVESMAP_H

#include "marco/Modeling/IndexSet.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringMap.h"

namespace marco::codegen
{
  class DerivativesMap
  {
    public:
      bool hasDerivative(llvm::StringRef variable) const;

      llvm::StringRef getDerivative(llvm::StringRef variable) const;

      void setDerivative(llvm::StringRef variable, llvm::StringRef derivative);

      const modeling::IndexSet& getDerivedIndices(llvm::StringRef variable) const;

      void setDerivedIndices(llvm::StringRef variable, modeling::IndexSet indices);

      bool isDerivative(llvm::StringRef variable) const;

      llvm::StringRef getDerivedVariable(llvm::StringRef derivative) const;

    private:
      llvm::StringMap<llvm::StringRef> derivatives;
      llvm::StringMap<llvm::StringRef> inverseDerivatives;
      llvm::StringMap<modeling::IndexSet> derivedIndices;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_DERIVATIVESMAP_H
