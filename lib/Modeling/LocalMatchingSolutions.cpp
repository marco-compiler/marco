#include "marco/Modeling/LocalMatchingSolutionsMCIM.h"
#include "marco/Modeling/LocalMatchingSolutionsVAF.h"

namespace marco::modeling::internal {
LocalMatchingSolutions::ImplInterface::~ImplInterface() = default;

LocalMatchingSolutions::LocalMatchingSolutions(
    llvm::ArrayRef<std::unique_ptr<AccessFunction>> accessFunctions,
    IndexSet equationIndices, IndexSet variableIndices)
    : impl(std::make_unique<VAFSolutions>(accessFunctions,
                                          std::move(equationIndices),
                                          std::move(variableIndices))) {}

LocalMatchingSolutions::LocalMatchingSolutions(const MCIM &obj)
    : impl(std::make_unique<MCIMSolutions>(obj)) {}

LocalMatchingSolutions::~LocalMatchingSolutions() = default;

MCIM &LocalMatchingSolutions::operator[](size_t index) {
  return (*impl)[index];
}

size_t LocalMatchingSolutions::size() const { return impl->size(); }

LocalMatchingSolutions::iterator LocalMatchingSolutions::begin() {
  return iterator(*this, 0);
}

LocalMatchingSolutions::iterator LocalMatchingSolutions::end() {
  return iterator(*this, size());
}

bool isValidLocalMatchingSolution(const MCIM &matrix) {
  IndexSet equations = matrix.getEquationRanges();
  IndexSet variables = matrix.getVariableRanges();

  // Check that each row has at most one 1.
  for (Point equation : equations) {
    size_t amount =
        std::count_if(variables.begin(), variables.end(), [&](Point variable) {
          return matrix.get(equation, variable);
        });

    if (amount > 1) {
      return false;
    }
  }

  // Check that each column has at most one 1.
  for (Point variable : variables) {
    size_t amount =
        std::count_if(equations.begin(), equations.end(), [&](Point equation) {
          return matrix.get(equation, variable);
        });

    if (amount > 1) {
      return false;
    }
  }

  return true;
}

LocalMatchingSolutions solveLocalMatchingProblem(
    const IndexSet &equationIndices, const IndexSet &variableIndices,
    llvm::ArrayRef<std::unique_ptr<AccessFunction>> accessFunctions) {
  return LocalMatchingSolutions(accessFunctions, equationIndices,
                                variableIndices);
}

LocalMatchingSolutions solveLocalMatchingProblem(const MCIM &obj) {
  return LocalMatchingSolutions(obj);
}
} // namespace marco::modeling::internal
