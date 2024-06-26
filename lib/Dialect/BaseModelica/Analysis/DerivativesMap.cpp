#include "marco/Dialect/BaseModelica/Analysis/DerivativesMap.h"

using namespace ::mlir::bmodelica;

namespace mlir::bmodelica
{
  DerivativesMap::DerivativesMap(ModelOp modelOp)
      : modelOp(modelOp),
        initialized(false)
  {
  }

  void DerivativesMap::initialize()
  {
    for (VarDerivativeAttr attr :
         modelOp.getDerivativesMap().getAsRange<VarDerivativeAttr>()) {
      setDerivative(attr.getVariable(), attr.getDerivative());

      if (auto indices = attr.getDerivedIndices()) {
        setDerivedIndices(attr.getVariable(), indices.getValue());
      } else {
        setDerivedIndices(attr.getVariable(), {});
      }
    }

    initialized = true;
  }

  llvm::DenseSet<mlir::SymbolRefAttr>
  DerivativesMap::getDerivedVariables() const
  {
    llvm::DenseSet<mlir::SymbolRefAttr> result;

    for (auto& entry : derivatives) {
      result.insert(entry.getFirst());
    }

    return result;
  }

  std::optional<mlir::SymbolRefAttr>
  DerivativesMap::getDerivative(mlir::SymbolRefAttr variable) const
  {
    assert(initialized && "Derivatives map not initialized");
    auto it = derivatives.find(variable);

    if (it == derivatives.end()) {
      return std::nullopt;
    }

    return it->getSecond();
  }

  /// Set the derivative variable for a state one.
  void DerivativesMap::setDerivative(
      mlir::SymbolRefAttr variable, mlir::SymbolRefAttr derivative)
  {
    derivatives[variable] = derivative;
    inverseDerivatives[derivative] = variable;
  }

  std::optional<std::reference_wrapper<const mlir::modeling::IndexSet>>
  DerivativesMap::getDerivedIndices(mlir::SymbolRefAttr variable) const
  {
    assert(initialized && "Derivatives map not initialized");
    auto it = derivedIndices.find(variable);
    assert(!getDerivative(variable) || it != derivedIndices.end());

    if (it == derivedIndices.end()) {
      return std::nullopt;
    }

    return std::reference_wrapper(it->getSecond());
  }

  void DerivativesMap::setDerivedIndices(
      mlir::SymbolRefAttr variable, mlir::modeling::IndexSet indices)
  {
    derivedIndices[variable] = std::move(indices);
  }

  void DerivativesMap::addDerivedIndices(
      mlir::SymbolRefAttr variable, mlir::modeling::IndexSet indices)
  {
    derivedIndices[variable] += indices;
  }

  std::optional<mlir::SymbolRefAttr>
  DerivativesMap::getDerivedVariable(mlir::SymbolRefAttr derivative) const
  {
    assert(initialized && "Derivatives map not initialized");
    auto it = inverseDerivatives.find(derivative);

    if (it == inverseDerivatives.end()) {
      return std::nullopt;
    }

    return it->getSecond();
  }
}
