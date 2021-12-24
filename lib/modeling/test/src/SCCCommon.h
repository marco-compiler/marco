#ifndef MARCO_MODELING_TEST_SCCCOMMON_H
#define MARCO_MODELING_TEST_SCCCOMMON_H

#include <llvm/ADT/StringRef.h>
#include <marco/modeling/VVarDependencyGraph.h>

namespace marco::modeling::scc
{
  namespace test
  {
    class Variable
    {
      public:
        Variable(llvm::StringRef name);

        llvm::StringRef getName() const;

      private:
        std::string name;
    };
  }

  template<>
  struct VariableTraits<test::Variable>
  {
    using Id = std::string;

    static Id getId(const test::Variable* variable)
    {
      return variable->getName().str();
    }
  };

  namespace test
  {
    class Equation
    {
      public:
        using Id = std::string;
        using Range = internal::Range;

        Equation(llvm::StringRef name, Access<Variable> write, llvm::ArrayRef<Access<Variable>> reads);

        llvm::StringRef getName() const;

        size_t getNumOfIterationVars() const;

        long getRangeBegin(size_t index) const;

        long getRangeEnd(size_t index) const;

        void addIterationRange(Range range);

        const Access<Variable>& getWrite() const;

        std::vector<Access<Variable>> getReads() const;

      private:
        std::string name;
        llvm::SmallVector<Range, 3> ranges;
        Access<Variable> write;
        llvm::SmallVector<Access<Variable>, 3> reads;
    };
  }

  template<>
  struct EquationTraits<test::Equation>
  {
    using Id = std::string;

    static Id getId(const test::Equation* equation)
    {
      return equation->getName().str();
    }

    static size_t getNumOfIterationVars(const test::Equation* equation)
    {
      return equation->getNumOfIterationVars();
    }

    static long getRangeBegin(const test::Equation* equation, size_t inductionVarIndex)
    {
      return equation->getRangeBegin(inductionVarIndex);
    }

    static long getRangeEnd(const test::Equation* equation, size_t inductionVarIndex)
    {
      return equation->getRangeEnd(inductionVarIndex);
    }

    using VariableType = test::Variable;

    static Access<VariableType> getWrite(const test::Equation* equation)
    {
      return equation->getWrite();
    }

    static std::vector<Access<VariableType>> getReads(const test::Equation* equation)
    {
      return equation->getReads();
    }
  };
}

#endif // MARCO_MODELING_TEST_SCCCOMMON_H
