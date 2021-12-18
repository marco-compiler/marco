#ifndef MARCO_MODELING_TEST_SCCCOMMON_H
#define MARCO_MODELING_TEST_SCCCOMMON_H

#include <llvm/ADT/StringRef.h>
#include <marco/modeling/VVarDependencyGraph.h>

namespace marco::modeling::scc::test
{
  class Variable
  {
    public:
      using Id = std::string;

      Variable(llvm::StringRef name);

      Id getId() const;

    private:
      std::string name;
  };

  class Equation
  {
    public:
      using Id = std::string;
      using Range = internal::Range;

      Equation(llvm::StringRef name, Access<Variable> write, llvm::ArrayRef<Access<Variable>> reads);

      Id getId() const;

      unsigned int getNumOfIterationVars() const;

      long getRangeStart(size_t index) const;

      long getRangeEnd(size_t index) const;

      void addIterationRange(Range range);

      const Access<Variable>& getWrite() const;

      void getReads(llvm::SmallVectorImpl<Access<Variable>>& v) const;

    private:
      std::string name;
      llvm::SmallVector<Range, 3> ranges;
      Access<Variable> write;
      llvm::SmallVector<Access<Variable>, 3> reads;
  };
}

#endif // MARCO_MODELING_TEST_SCCCOMMON_H
