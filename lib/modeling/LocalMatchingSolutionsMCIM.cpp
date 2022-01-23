#include <marco/modeling/LocalMatchingSolutionsMCIM.h>

namespace marco::modeling::internal
{
  MCIMSolutions::MCIMSolutions(const MCIM& obj)
  {
    compute(obj);
  }

  MCIM& MCIMSolutions::operator[](size_t index)
  {
    return solutions[index];
  }

  size_t MCIMSolutions::size() const
  {
    return solutions.size();
  }

  void MCIMSolutions::compute(const MCIM& obj)
  {
    for (const auto& solution: obj.splitGroups()) {
      solutions.push_back(std::move(solution));
    }
  }
}
