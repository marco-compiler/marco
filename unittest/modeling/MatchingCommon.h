#ifndef MARCO_MODELING_TEST_MATCHINGCOMMON_H
#define MARCO_MODELING_TEST_MATCHINGCOMMON_H

#include <llvm/ADT/iterator_range.h>
#include <llvm/ADT/StringRef.h>
#include <marco/modeling/Matching.h>

namespace marco::modeling::matching
{
  namespace test
  {
    class Variable
    {
      public:
        Variable(llvm::StringRef name, llvm::ArrayRef<long> dimensions = llvm::None);

        llvm::StringRef getName() const;

        size_t getRank() const;

        long getDimensionSize(size_t index) const;

      private:
        std::string name;
        llvm::SmallVector<long, 3> dimensions;
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

    static size_t getRank(const test::Variable* variable)
    {
      return variable->getRank();
    }

    static size_t getDimensionSize(const test::Variable* variable, size_t index)
    {
      return variable->getDimensionSize(index);
    }
  };

  namespace test
  {
    class Equation
    {
      public:
        using VariableType = Variable;

        Equation(llvm::StringRef name);

        llvm::StringRef getName() const;

        size_t getNumOfIterationVars() const;

        long getRangeBegin(size_t index) const;

        long getRangeEnd(size_t index) const;

        void addIterationRange(internal::Range range);

        std::vector<Access<Variable>> getVariableAccesses() const;

        void addVariableAccess(Access<Variable> access);

      private:
        std::string name;
        llvm::SmallVector<internal::Range, 3> ranges;
        llvm::SmallVector<Access<Variable>, 3> accesses;
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

    static std::vector<Access<VariableType>> getAccesses(const test::Equation* equation)
    {
      return equation->getVariableAccesses();
    }
  };
}

#endif  // MARCO_MODELING_TEST_MATCHINGCOMMON_H
