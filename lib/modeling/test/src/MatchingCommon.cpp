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

  long Equation::getRangeStart(size_t index) const
  {
    return ranges[index].getBegin();
  }

  long Equation::getRangeEnd(size_t index) const
  {
    return ranges[index].getEnd();
  }

  void Equation::addIterationRange(internal::Range range)
  {
    ranges.push_back(range);
  }

  llvm::iterator_range<Equation::AccessIt> Equation::getVariableAccesses() const
  {
    return llvm::iterator_range<Equation::AccessIt>(accesses.begin(), accesses.end());
  }

  void Equation::addVariableAccess(Access<Variable> access)
  {
    accesses.push_back(access);
  }
}
