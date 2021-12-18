#include "MatchingCommon.h"

namespace marco::modeling::matching::test
{
  Variable::Variable(llvm::StringRef name, llvm::ArrayRef<long> dimensions)
      : name(name.str()), dimensions(dimensions.begin(), dimensions.end())
  {
    if (this->dimensions.empty())
      this->dimensions.emplace_back(1);
  }

  Variable::Id Variable::getId() const
  {
    return name;
  }

  unsigned int Variable::getRank() const
  {
    return dimensions.size();
  }

  long Variable::getDimensionSize(size_t index) const
  {
    return dimensions[index];
  }

  llvm::StringRef Variable::getName() const
  {
    return name;
  }

  Equation::Equation(llvm::StringRef name) : name(name.str())
  {
  }

  Equation::Id Equation::getId() const
  {
    return name;
  }

  unsigned int Equation::getNumOfIterationVars() const
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

  void Equation::getVariableAccesses(llvm::SmallVectorImpl<Access<Variable>>& v) const
  {
    v.insert(v.begin(), this->accesses.begin(), this->accesses.end());
  }

  void Equation::addVariableAccess(Access<Variable> access)
  {
    accesses.push_back(access);
  }

  llvm::StringRef Equation::getName() const
  {
    return name;
  }
}
