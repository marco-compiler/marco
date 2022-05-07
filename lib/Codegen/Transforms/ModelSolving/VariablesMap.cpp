#include "marco/Codegen/Transforms/ModelSolving/VariablesMap.h"

using namespace ::marco::codegen;

namespace marco::codegen
{
  SplitVariable::SplitVariable(modeling::MultidimensionalRange indices, unsigned int argNumber)
      : indices(std::move(indices)), argNumber(argNumber)
  {
  }

  const modeling::MultidimensionalRange& SplitVariable::getIndices() const
  {
    return indices;
  }

  unsigned int SplitVariable::getArgNumber() const
  {
    return argNumber;
  }

  void VariablesMap::add(llvm::StringRef name, SplitVariable variable)
  {
    assert(llvm::none_of(variables[name], [&](const auto& splitVariable) {
      return splitVariable.getIndices().overlaps(variable.getIndices());
    }));

    variables[name].push_back(std::move(variable));

    llvm::sort(variables[name], [](const auto& first, const auto& second) {
      return first.getIndices() < second.getIndices();
    });
  }

  const std::vector<SplitVariable>& VariablesMap::operator[](llvm::StringRef name) const
  {
    auto it = variables.find(name);
    assert(it != variables.end());
    return it->second;
  }
}
