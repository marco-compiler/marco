#include "MatchingCommon.h"

namespace marco::modeling::matching::test
{
  Variable::Variable(llvm::StringRef name, llvm::ArrayRef<long> dimensions)
      : name(name.str()), dimensions(dimensions.begin(), dimensions.end())
  {
    if (this->dimensions.empty())
      this->dimensions.emplace_back(1);
  }

  llvm::StringRef Variable::getName() const
  {
    return name;
  }

  size_t Variable::getRank() const
  {
    return dimensions.size();
  }

  long Variable::getDimensionSize(size_t index) const
  {
    return dimensions[index];
  }

  Equation::Equation(llvm::StringRef name) : name(name.str())
  {
  }

  llvm::StringRef Equation::getName() const
  {
    return name;
  }

  size_t Equation::getNumOfIterationVars() const
  {
    return ranges.size();
  }

  long Equation::getRangeBegin(size_t index) const
  {
    return ranges[index].getBegin();
  }

  long Equation::getRangeEnd(size_t index) const
  {
    return ranges[index].getEnd();
  }

  void Equation::addIterationRange(Range range)
  {
    ranges.push_back(range);
  }

  std::vector<Access<Variable>> Equation::getVariableAccesses() const
  {
    std::vector<Access<Variable>> result;

    for (const auto& access : accesses)
      result.push_back(access);

    return result;
  }

  void Equation::addVariableAccess(Access<Variable> access)
  {
    accesses.push_back(access);
  }
}
