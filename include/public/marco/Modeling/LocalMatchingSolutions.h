#ifndef MARCO_MODELING_LOCALMATCHINGSOLUTIONS_H
#define MARCO_MODELING_LOCALMATCHINGSOLUTIONS_H

#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/MCIM.h"
#include "marco/Modeling/MultidimensionalRange.h"
#include <memory>

namespace marco::modeling::internal
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

      using iterator = Iterator<LocalMatchingSolutions, MCIM>;

      LocalMatchingSolutions(
          llvm::ArrayRef<AccessFunction> accessFunctions,
          IndexSet equationIndices,
          IndexSet variableIndices);

      explicit LocalMatchingSolutions(const MCIM& mcim);

      ~LocalMatchingSolutions();

      MCIM& operator[](size_t index);

      size_t size() const;

      iterator begin();

      iterator end();

    private:
      std::unique_ptr<ImplInterface> impl;
  };

  LocalMatchingSolutions solveLocalMatchingProblem(
      const IndexSet& equationRanges,
      const IndexSet& variableRanges,
      llvm::ArrayRef<AccessFunction> accessFunctions);

  LocalMatchingSolutions solveLocalMatchingProblem(const MCIM& matrix);
}

#endif  // MARCO_MODELING_LOCALMATCHINGSOLUTIONS_H
