#include "SCCCommon.h"

namespace marco::modeling::scc::test
{
  Variable::Variable(llvm::StringRef name)
          : name(name.str())
  {
  }

  Variable::Id Variable::getId() const
  {
    return name;
  }

  Equation::Equation(llvm::StringRef name, Access<Variable> write, llvm::ArrayRef<Access<Variable>> reads)
          : name(name.str()), write(std::move(write)), reads(reads.begin(), reads.end())
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

  void Equation::addIterationRange(Range range)
  {
    ranges.push_back(range);
  }

  const Access<Variable>& Equation::getWrite() const
  {
    return write;
  }

  void Equation::getReads(llvm::SmallVectorImpl<Access<Variable>>& v) const
  {
    v.append(reads);
  }
}
