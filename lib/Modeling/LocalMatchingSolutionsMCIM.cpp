#include "marco/Modeling/LocalMatchingSolutionsMCIM.h"

using namespace ::marco::modeling;
using namespace ::marco::modeling::internal;

namespace marco::modeling::internal {
MCIMSolutions::MCIMSolutions(const MCIM &obj) { compute(obj); }

MCIM &MCIMSolutions::operator[](size_t index) { return solutions[index]; }

size_t MCIMSolutions::size() const { return solutions.size(); }

void MCIMSolutions::compute(const MCIM &obj) {
  for (auto &solution : obj.splitGroups()) {
    assert(isValidLocalMatchingSolution(solution) &&
           "Invalid MCIM local matching solution");

    solutions.push_back(std::move(solution));
  }

  assert(allIndicesCovered(obj) &&
         "The computed local matching solutions discard some indices");
}

bool MCIMSolutions::allIndicesCovered(const MCIM &obj) const {
  for (auto coordinates :
       llvm::make_range(obj.indicesBegin(), obj.indicesEnd())) {
    Point equation = coordinates.first;
    Point variable = coordinates.second;

    if (obj.get(equation, variable)) {
      if (!llvm::any_of(solutions, [&](const MCIM &solution) {
            return solution.get(equation, variable);
          })) {
        return false;
      }
    }
  }

  return true;
}
} // namespace marco::modeling::internal
