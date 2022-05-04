#include "marco/Codegen/Transforms/ModelSolving/DerivativesMap.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "llvm/ADT/STLExtras.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  DerivedVariable::DerivedVariable(unsigned int argNumber, MultidimensionalRange indices)
    : argNumber(argNumber),
      indices(std::move(indices))
  {
  }

  unsigned int DerivedVariable::getArgNumber() const
  {
    return argNumber;
  }

  const MultidimensionalRange& DerivedVariable::getIndices() const
  {
    return indices;
  }

  bool DerivedVariable::operator<(const DerivedVariable& other) const
  {
    if (getArgNumber() < other.getArgNumber()) {
      return true;
    }

    return getIndices() < other.getIndices();
  }

  Derivative::Derivative(unsigned int argNumber, std::vector<long> offsets)
      : argNumber(argNumber), offsets(std::move(offsets))
  {
  }

  unsigned int Derivative::getArgNumber() const
  {
    return argNumber;
  }

  llvm::ArrayRef<long> Derivative::getOffsets() const
  {
    return offsets;
  }

  namespace impl
  {
    bool DerivedVariablesComparator::operator()(const DerivedVariable& lhs, const DerivedVariable& rhs) const
    {
      if (lhs.getArgNumber() != rhs.getArgNumber()) {
        return lhs.getArgNumber() < rhs.getArgNumber();
      }

      if (lhs.getIndices().rank() != rhs.getIndices().rank()) {
        return lhs.getIndices().rank() < rhs.getIndices().rank();
      }

      return lhs.getIndices() < rhs.getIndices();
    }
  }

  void DerivativesMap::addDerivedIndices(unsigned int variable, MultidimensionalRange indices)
  {
    derivedIndices[variable] += indices;
  }

  IndexSet DerivativesMap::getDerivedIndices(unsigned int variable) const
  {
    auto it = derivedIndices.find(variable);

    if (it != derivedIndices.end()) {
      return it->second;
    }

    return IndexSet();
  }

  void DerivativesMap::setDerivative(
      unsigned int variable,
      MultidimensionalRange variableIndices,
      unsigned int derivative,
      MultidimensionalRange derivativeIndices)
  {
    assert(variableIndices.rank() == derivativeIndices.rank());
    std::vector<long> offsets;

    for (size_t i = 0; i < variableIndices.rank(); ++i) {
      assert(variableIndices[i].size() == derivativeIndices[i].size());
      offsets.push_back(derivativeIndices[i].getBegin() - variableIndices[i].getBegin());
    }

    derivatives.try_emplace(
        DerivedVariable(variable, std::move(variableIndices)),
        Derivative(derivative, std::move(offsets)));
  }

  bool DerivativesMap::hasDerivative(unsigned int variable) const
  {
    return llvm::any_of(derivatives, [&](const auto& entry) {
      return entry.first.getArgNumber() == variable;
    });
  }

  std::vector<std::pair<const MultidimensionalRange*, const Derivative*>> DerivativesMap::getDerivative(unsigned int variable) const
  {
    std::vector<std::pair<const MultidimensionalRange*, const Derivative*>> result;

    for (const auto& entry : derivatives) {
      if (entry.first.getArgNumber() == variable) {
        result.emplace_back(&entry.first.getIndices(), &entry.second);
      }
    }

    return result;
  }

  const Derivative& DerivativesMap::getDerivative(unsigned int variable, const MultidimensionalRange& indices) const
  {
    auto it = llvm::find_if(derivatives, [&](const auto& entry) {
      return entry.first.getArgNumber() == variable && entry.first.getIndices() == indices;
    });

    assert(it != derivatives.end());
    return it->second;
  }

  bool DerivativesMap::isDerivative(unsigned int variable) const
  {
    return llvm::any_of(derivatives, [&](const auto& entry) {
      return entry.second.getArgNumber() == variable;
    });
  }

  const DerivedVariable* DerivativesMap::getDerivedVariable(unsigned int derivative) const
  {
    auto it = llvm::find_if(derivatives, [&](const auto& entry) {
      return entry.second.getArgNumber() == derivative;
    });

    if (it == derivatives.end()) {
      return nullptr;
    }

    return &it->first;
  }
}
