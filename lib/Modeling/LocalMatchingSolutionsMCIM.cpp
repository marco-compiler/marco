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
}
} // namespace marco::modeling::internal
