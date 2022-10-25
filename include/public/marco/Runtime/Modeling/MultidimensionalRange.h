#ifndef MARCO_RUNTIME_MODELING_MULTIDIMENSIONALRANGE_H
#define MARCO_RUNTIME_MODELING_MULTIDIMENSIONALRANGE_H

#include "marco/Runtime/Modeling/Range.h"
#include <functional>
#include <vector>

namespace marco::runtime
{
  using MultidimensionalRange = std::vector<Range>;

  class MultidimensionalRangeIterator
  {
    public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = size_t*;
      using difference_type = std::ptrdiff_t;
      using pointer = const int64_t**;
      using reference = const int64_t*&;

      static MultidimensionalRangeIterator begin(const MultidimensionalRange& range);
      static MultidimensionalRangeIterator end(const MultidimensionalRange& range);

      bool operator==(const MultidimensionalRangeIterator& it) const;

      bool operator!=(const MultidimensionalRangeIterator& it) const;

      MultidimensionalRangeIterator& operator++();

      MultidimensionalRangeIterator operator++(int);

      const int64_t* operator*() const;

    private:
      MultidimensionalRangeIterator(
        const MultidimensionalRange& range,
        std::function<RangeIterator(const Range&)> initFunction);

      void fetchNext();

    private:
      std::vector<RangeIterator> beginIterators;
      std::vector<RangeIterator> currentIterators;
      std::vector<RangeIterator> endIterators;
      std::vector<int64_t> indices;
  };
}

#endif // MARCO_RUNTIME_MODELING_MULTIDIMENSIONALRANGE_H
