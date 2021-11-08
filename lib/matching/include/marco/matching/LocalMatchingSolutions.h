#ifndef MARCO_MATCHING_LOCALMATCHINGSOLUTIONS_H
#define MARCO_MATCHING_LOCALMATCHINGSOLUTIONS_H

#include "AccessFunction.h"
#include "IncidenceMatrix.h"
#include "Range.h"

namespace marco::matching
{
  namespace detail
  {
    class LocalMatchingSolutions;

    template<typename Container, typename ValueType>
    class LocalMatchingSolutionsIterator
    {
      public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = ValueType;
      using difference_type = std::ptrdiff_t;
      using pointer = ValueType*;
      using reference = ValueType&;

      LocalMatchingSolutionsIterator(Container& container, size_t index)
              : container(&container),
                index(std::move(index))
      {
      }

      operator bool() const
      {
        return index != container->size();
      }

      bool operator==(const LocalMatchingSolutionsIterator& it) const
      {
        return index == it.index && container == it.container;
      }

      bool operator!=(const LocalMatchingSolutionsIterator& it) const
      {
        return index != it.index || container != it.container;
      }

      LocalMatchingSolutionsIterator& operator++()
      {
        index = std::min(index + 1, container->size());
        return *this;
      }

      LocalMatchingSolutionsIterator operator++(int)
      {
        auto temp = *this;
        index = std::min(index + 1, container->size());
        return temp;
      }

      value_type& operator*()
      {
        return (*container)[index];
      }

      private:
      Container* container;
      size_t index;
    };

    class LocalMatchingSolutions
    {
      public:
      using iterator = LocalMatchingSolutionsIterator<
              LocalMatchingSolutions, IncidenceMatrix>;

      LocalMatchingSolutions(
              llvm::ArrayRef<AccessFunction> accessFunctions,
              MultidimensionalRange equationRanges,
              MultidimensionalRange variableRanges);

      IncidenceMatrix& operator[](size_t index);

      size_t size() const;

      iterator begin();
      iterator end();

      private:
      void fetchNext();

      void getInductionVariablesUsage(
              llvm::SmallVectorImpl<size_t>& usages,
              const AccessFunction& accessFunction) const;

      llvm::SmallVector<AccessFunction, 3> accessFunctions;
      MultidimensionalRange equationRanges;
      MultidimensionalRange variableRanges;

      // Total number of possible match matrices
      size_t solutionsAmount;

      // List of the computed match matrices
      llvm::SmallVector<IncidenceMatrix, 3> matrices;

      size_t currentAccessFunction = 0;
      size_t groupSize;
      llvm::SmallVector<Range, 3> reorderedRanges;
      std::unique_ptr<MultidimensionalRange> range;
      llvm::SmallVector<size_t, 3> ordering;
      std::unique_ptr<MultidimensionalRange::iterator> rangeIt;
      std::unique_ptr<MultidimensionalRange::iterator> rangeEnd;
    };

    LocalMatchingSolutions solveLocalMatchingProblem(
            const MultidimensionalRange& equationRanges,
            const MultidimensionalRange& variableRanges,
            llvm::ArrayRef<AccessFunction> accessFunctions);
  }
}

#endif	// MARCO_MATCHING_LOCALMATCHINGSOLUTIONS_H
