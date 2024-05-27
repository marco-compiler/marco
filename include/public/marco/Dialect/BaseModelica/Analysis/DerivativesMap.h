#ifndef MARCO_DIALECT_BASEMODELICA_ANALYSIS_DERIVATIVESMAP_H
#define MARCO_DIALECT_BASEMODELICA_ANALYSIS_DERIVATIVESMAP_H

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::bmodelica
{
  class DerivativesMap
  {
    public:
      DerivativesMap(ModelOp modelOp);

      /// Compute the derivatives map.
      void initialize();

      llvm::DenseSet<mlir::SymbolRefAttr> getDerivedVariables() const;

      /// Get the derivative variable of a given state variable.
      std::optional<mlir::SymbolRefAttr> getDerivative(
          mlir::SymbolRefAttr variable) const;

      /// Set the derivative variable for a state one.
      void setDerivative(
          mlir::SymbolRefAttr variable, mlir::SymbolRefAttr derivative);

      std::optional<std::reference_wrapper<const mlir::modeling::IndexSet>>
      getDerivedIndices(mlir::SymbolRefAttr variable) const;

      void setDerivedIndices(
          mlir::SymbolRefAttr variable, mlir::modeling::IndexSet indices);

      void addDerivedIndices(
          mlir::SymbolRefAttr variable, mlir::modeling::IndexSet indices);

      /// Get the state variable of a given derivative variable.
      std::optional<mlir::SymbolRefAttr>
      getDerivedVariable(mlir::SymbolRefAttr derivative) const;

    private:
      ModelOp modelOp;
      bool initialized;

      llvm::DenseMap<
          mlir::SymbolRefAttr,
          mlir::SymbolRefAttr> derivatives;

      llvm::DenseMap<
          mlir::SymbolRefAttr,
          mlir::modeling::IndexSet> derivedIndices;

      llvm::DenseMap<
          mlir::SymbolRefAttr,
          mlir::SymbolRefAttr> inverseDerivatives;
  };
}

#endif // MARCO_DIALECT_BASEMODELICA_ANALYSIS_DERIVATIVESMAP_H
