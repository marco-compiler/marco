#include "marco/Codegen/Transforms/ModelSolving/DerivativesMap.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "llvm/ADT/STLExtras.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  bool DerivativesMap::hasDerivative(llvm::StringRef variable) const
  {
    return derivatives.find(variable) != derivatives.end();
  }

  llvm::StringRef DerivativesMap::getDerivative(llvm::StringRef variable) const
  {
    auto it = derivatives.find(variable);
    assert(it != derivatives.end());
    return it->second;
  }

  void DerivativesMap::setDerivative(llvm::StringRef variable, llvm::StringRef derivative)
  {
    derivatives[variable] = derivative;
    inverseDerivatives[derivative] = variable;
  }

  bool DerivativesMap::isDerivative(llvm::StringRef variable) const
  {
    return inverseDerivatives.find(variable) != inverseDerivatives.end();
  }

  llvm::StringRef DerivativesMap::getDerivedVariable(llvm::StringRef derivative) const
  {
    auto it = inverseDerivatives.find(derivative);
    assert(it != inverseDerivatives.end());
    return it->second;
  }

  const IndexSet& DerivativesMap::getDerivedIndices(llvm::StringRef variable) const
  {
    auto it = derivedIndices.find(variable);
    assert(it != derivedIndices.end());
    return it->second;
  }

  void DerivativesMap::setDerivedIndices(llvm::StringRef variable, modeling::IndexSet indices)
  {
    derivedIndices[variable] = std::move(indices);
  }
}
