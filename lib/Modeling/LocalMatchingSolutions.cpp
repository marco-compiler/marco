#include "marco/Modeling/LocalMatchingSolutionsMCIM.h"
#include "marco/Modeling/LocalMatchingSolutionsVAF.h"

namespace marco::modeling::internal
{
  LocalMatchingSolutions::ImplInterface::~ImplInterface() = default;

  LocalMatchingSolutions::LocalMatchingSolutions(
      llvm::ArrayRef<AccessFunction> accessFunctions,
      IndexSet equationIndices,
      IndexSet variableIndices)
      : impl(std::make_unique<VAFSolutions>(
          std::move(accessFunctions),
          std::move(equationIndices),
          std::move(variableIndices)))
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
      const IndexSet& equationIndices,
      const IndexSet& variableIndices,
      llvm::ArrayRef<AccessFunction> accessFunctions)
  {
    return LocalMatchingSolutions(
        std::move(accessFunctions), equationIndices, variableIndices);
  }

  LocalMatchingSolutions solveLocalMatchingProblem(const MCIM& obj)
  {
    return LocalMatchingSolutions(obj);
  }
}
