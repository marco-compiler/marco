#include "SCCCommon.h"

namespace marco::modeling::scc::test
{
  Variable::Variable(llvm::StringRef name)
      : name(name.str())
  {
  }

  llvm::StringRef Variable::getName() const
  {
    return name;
  }

  Equation::Equation(llvm::StringRef name, Access<Variable> write, llvm::ArrayRef<Access<Variable>> reads)
      : name(name.str()), write(std::move(write)), reads(reads.begin(), reads.end())
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

  const Access<Variable>& Equation::getWrite() const
  {
    return write;
  }

  std::vector<Access<Variable>> Equation::getReads() const
  {
    std::vector<Access<Variable>> result;

    for (const auto& read : reads)
      result.push_back(read);

    return result;
  }
}
