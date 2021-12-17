#ifndef MARCO_SCHEDULINGCOMMON_H
#define MARCO_SCHEDULINGCOMMON_H

#include <llvm/ADT/StringRef.h>
#include <marco/modeling/VVarDependencyGraph.h>

namespace marco::modeling::scc::test
{
  class Variable
  {
    public:
    using Id = std::string;

    Variable(llvm::StringRef name)
            : name(name.str())
    {
    }

    Id getId() const
    {
      return name;
    }

    private:
    std::string name;
  };

  class Equation
  {
    public:
    using Id = std::string;
    using Range = internal::Range;

    Equation(llvm::StringRef name, Access<Variable> write, llvm::ArrayRef<Access<Variable>> reads)
            : name(name.str()), write(std::move(write)), reads(reads.begin(), reads.end())
    {
    }

    Id getId() const
    {
      return name;
    }

    unsigned int getNumOfIterationVars() const
    {
      return ranges.size();
    }

    long getRangeStart(size_t index) const
    {
      return ranges[index].getBegin();
    }

    long getRangeEnd(size_t index) const
    {
      return ranges[index].getEnd();
    }

    void addIterationRange(Range range)
    {
      ranges.push_back(range);
    }

    const Access<Variable>& getWrite() const
    {
      return write;
    }

    void getReads(llvm::SmallVectorImpl<Access<Variable>>& v) const
    {
      v.append(reads);
    }

    private:
    std::string name;
    llvm::SmallVector<Range, 3> ranges;
    Access<Variable> write;
    llvm::SmallVector<Access<Variable>, 3> reads;
  };
}

#endif //MARCO_SCHEDULINGCOMMON_H
