#include "marco/Modeling/LocalMatchingSolutionsMCIM.h"
#include "marco/Modeling/LocalMatchingSolutionsVAF.h"

namespace marco::modeling::internal
{
  LocalMatchingSolutions::ImplInterface::~ImplInterface() = default;

  LocalMatchingSolutions::LocalMatchingSolutions(
      llvm::ArrayRef<AccessFunction> accessFunctions,
      MultidimensionalRange equationRanges,
      MultidimensionalRange variableRanges)
      : impl(std::make_unique<VAFSolutions>(std::move(accessFunctions), std::move(equationRanges), std::move(variableRanges)))
  {
  }

  LocalMatchingSolutions::LocalMatchingSolutions(const MCIM& obj)
      : impl(std::make_unique<MCIMSolutions>(obj))
  {
  }

  LocalMatchingSolutions::~LocalMatchingSolutions() = default;

  MCIM& LocalMatchingSolutions::operator[](size_t index)
  {
    return (*impl)[index];
  }

  size_t LocalMatchingSolutions::size() const
  {
    return impl->size();
  }

  LocalMatchingSolutions::iterator LocalMatchingSolutions::begin()
  {
    return iterator(*this, 0);
  }

  LocalMatchingSolutions::iterator LocalMatchingSolutions::end()
  {
    return iterator(*this, size());
  }

  LocalMatchingSolutions solveLocalMatchingProblem(
      const MultidimensionalRange& equationRanges,
      const MultidimensionalRange& variableRanges,
      llvm::ArrayRef<AccessFunction> accessFunctions)
  {
    return LocalMatchingSolutions(std::move(accessFunctions), equationRanges, variableRanges);
  }

  LocalMatchingSolutions solveLocalMatchingProblem(const MCIM& obj)
  {
    return LocalMatchingSolutions(obj);
  }
}
