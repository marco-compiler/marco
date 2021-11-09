#ifndef MARCO_MATCHING_LOCALMATCHINGSOLUTIONS_H
#define MARCO_MATCHING_LOCALMATCHINGSOLUTIONS_H

#include <memory>

#include "AccessFunction.h"
#include "IncidenceMatrix.h"
#include "Range.h"

namespace marco::matching::detail
{
  class LocalMatchingSolutions
  {
    public:
    class ImplInterface;

    template<typename Container, typename ValueType>
    class Iterator
    {
      public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = ValueType;
      using difference_type = std::ptrdiff_t;
      using pointer = ValueType*;
      using reference = ValueType&;

      Iterator(Container& container, size_t index)
              : container(&container),
                index(std::move(index))
      {
      }

      bool operator==(const Iterator& it) const
      {
        return index == it.index && container == it.container;
      }

      bool operator!=(const Iterator& it) const
      {
        return index != it.index || container != it.container;
      }

      Iterator& operator++()
      {
        index = std::min(index + 1, container->size());
        return *this;
      }

      Iterator operator++(int)
      {
        auto temp = *this;
        index = std::min(index + 1, container->size());
        return temp;
      }

      reference operator*()
      {
        return (*container)[index];
      }

      private:
      Container* container;
      size_t index;
    };

    using iterator = Iterator<LocalMatchingSolutions, IncidenceMatrix>;

    LocalMatchingSolutions(
            llvm::ArrayRef<AccessFunction> accessFunctions,
            MultidimensionalRange equationRanges,
            MultidimensionalRange variableRanges);

    explicit LocalMatchingSolutions(const IncidenceMatrix& matrix);

    ~LocalMatchingSolutions();

    IncidenceMatrix& operator[](size_t index);

    size_t size() const;

    iterator begin();
    iterator end();

    private:
    std::unique_ptr<ImplInterface> impl;
  };

  LocalMatchingSolutions solveLocalMatchingProblem(
          const MultidimensionalRange& equationRanges,
          const MultidimensionalRange& variableRanges,
          llvm::ArrayRef<AccessFunction> accessFunctions);

  LocalMatchingSolutions solveLocalMatchingProblem(const IncidenceMatrix& matrix);
}

#endif	// MARCO_MATCHING_LOCALMATCHINGSOLUTIONS_H
