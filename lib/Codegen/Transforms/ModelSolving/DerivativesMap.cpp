#include "marco/Codegen/Transforms/ModelSolving/DerivativesMap.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "llvm/ADT/STLExtras.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  bool DerivativesMap::hasDerivative(unsigned int variable) const
  {
    return derivatives.find(variable) != derivatives.end();
  }

  unsigned int DerivativesMap::getDerivative(unsigned int variable) const
  {
    auto it = derivatives.find(variable);
    assert(it != derivatives.end());
    return it->second;
  }

  void DerivativesMap::setDerivative(unsigned int variable, unsigned int derivative)
  {
    derivatives[variable] = derivative;
    inverseDerivatives[derivative] = variable;
  }

  bool DerivativesMap::isDerivative(unsigned int variable) const
  {
    return inverseDerivatives.find(variable) != inverseDerivatives.end();
  }

  unsigned int DerivativesMap::getDerivedVariable(unsigned int derivative) const
  {
    auto it = inverseDerivatives.find(derivative);
    assert(it != inverseDerivatives.end());
    return it->second;
  }

  const IndexSet& DerivativesMap::getDerivedIndices(unsigned int variable) const
  {
    auto it = derivedIndices.find(variable);
    assert(it != derivedIndices.end());
    return it->second;
  }

  void DerivativesMap::setDerivedIndices(unsigned int variable, modeling::IndexSet indices)
  {
    derivedIndices[variable] = std::move(indices);
  }
}
