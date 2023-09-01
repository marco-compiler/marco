#include "marco/Runtime/Modeling/MultidimensionalRange.h"
#include <cassert>

namespace marco::runtime
{
  MultidimensionalRangeIterator::MultidimensionalRangeIterator(const MultidimensionalRange& ranges, std::function<RangeIterator(const Range&)> initFunction)
  {
    for (const Range& range : ranges) {
      beginIterators.push_back(RangeIterator::begin(range));
      auto it = initFunction(range);
      currentIterators.push_back(it);
      endIterators.push_back(RangeIterator::end(range));
      indices.push_back(*it);
    }

    assert(ranges.size() == beginIterators.size());
    assert(ranges.size() == currentIterators.size());
    assert(ranges.size() == endIterators.size());
    assert(ranges.size() == indices.size());
  }

  MultidimensionalRangeIterator MultidimensionalRangeIterator::begin(const MultidimensionalRange& ranges)
  {
    return MultidimensionalRangeIterator(ranges, [](const Range& range) {
      return RangeIterator::begin(range);
    });
  }

  MultidimensionalRangeIterator MultidimensionalRangeIterator::end(const MultidimensionalRange& ranges)
  {
    return MultidimensionalRangeIterator(ranges, [](const Range& range) {
      return RangeIterator::end(range);
    });
  }

  bool MultidimensionalRangeIterator::operator==(const MultidimensionalRangeIterator& it) const
  {
    return currentIterators == it.currentIterators;
  }

  bool MultidimensionalRangeIterator::operator!=(const MultidimensionalRangeIterator& it) const
  {
    return currentIterators != it.currentIterators;
  }

  MultidimensionalRangeIterator& MultidimensionalRangeIterator::operator++()
  {
    fetchNext();
    return *this;
  }

  MultidimensionalRangeIterator MultidimensionalRangeIterator::operator++(int)
  {
    MultidimensionalRangeIterator temp = *this;
    fetchNext();
    return temp;
  }

  const int64_t* MultidimensionalRangeIterator::operator*() const
  {
    return indices.data();
  }

  void MultidimensionalRangeIterator::fetchNext()
  {
    size_t size = indices.size();

    auto findIndex = [&]() -> std::pair<bool, size_t> {
      for (size_t i = 0, e = size; i < e; ++i) {
        size_t pos = e - i - 1;

        if (++currentIterators[pos] != endIterators[pos]) {
          return std::make_pair(true, pos);
        }
      }

      return std::make_pair(false, 0);
    };

    std::pair<bool, size_t> index = findIndex();

    if (index.first) {
      size_t pos = index.second;

      indices[pos] = *currentIterators[pos];

      for (size_t i = pos + 1; i < size; ++i) {
        currentIterators[i] = beginIterators[i];
        indices[i] = *currentIterators[i];
      }
    }
  }
}
