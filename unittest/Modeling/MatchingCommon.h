#ifndef MARCO_MODELING_TEST_MATCHINGCOMMON_H
#define MARCO_MODELING_TEST_MATCHINGCOMMON_H

#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/StringRef.h"
#include "marco/Modeling/Matching.h"

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

        IndexSet getIndices() const;

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

    static IndexSet getIndices(const test::Variable* variable)
    {
      return variable->getIndices();
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

        void addIterationRange(Range range);

        std::vector<Access<Variable>> getVariableAccesses() const;

        void addVariableAccess(Access<Variable> access);

      private:
        std::string name;
        llvm::SmallVector<Range, 3> ranges;
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

    static IndexSet getIterationRanges(const test::Equation* equation)
    {
      std::vector<Range> ranges;

      for (size_t i = 0, e = getNumOfIterationVars(equation); i < e; ++i) {
        ranges.emplace_back(equation->getRangeBegin(i), equation->getRangeEnd(i));
      }

      return IndexSet(MultidimensionalRange(std::move(ranges)));
    }

    using VariableType = test::Variable;

    static std::vector<Access<VariableType>> getAccesses(const test::Equation* equation)
    {
      return equation->getVariableAccesses();
    }
  };
}

#endif  // MARCO_MODELING_TEST_MATCHINGCOMMON_H
