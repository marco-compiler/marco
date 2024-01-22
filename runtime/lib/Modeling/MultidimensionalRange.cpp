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

  uint64_t getFlatSize(const MultidimensionalRange& ranges)
  {
    uint64_t result = 1;

    for (const Range& range : ranges) {
      result *= range.end - range.begin;
    }

    return result;
  }

  uint64_t getFlatIndex(
      const std::vector<int64_t>& indices,
      const MultidimensionalRange& ranges)
  {
    assert(indices[0] >= ranges[0].begin);
    uint64_t offset = indices[0] - ranges[0].begin;

    for (size_t i = 1, e = ranges.size(); i < e; ++i) {
      assert(ranges[i].end > ranges[i].begin);
      offset = offset * (ranges[i].end - ranges[i].begin) +
          (indices[i] - ranges[i].begin);
    }

    return offset;
  }

  void getIndicesFromFlatIndex(
      uint64_t flatIndex,
      std::vector<int64_t>& result,
      const MultidimensionalRange& ranges)
  {
    result.resize(ranges.size());
    uint64_t size = 1;

    for (size_t i = 1, e = ranges.size(); i < e; ++i) {
      assert(ranges[i].end > ranges[i].begin);
      size *= ranges[i].end - ranges[i].begin;
    }

    for (size_t i = 1, e = ranges.size(); i < e; ++i) {
      result[i - 1] =
          static_cast<int64_t>(flatIndex / size) + ranges[i - 1].begin;

      flatIndex %= size;
      assert(ranges[i].end > ranges[i].begin);
      size /= ranges[i].end - ranges[i].begin;
    }

    result[ranges.size() - 1] =
        static_cast<int64_t>(flatIndex) + ranges.back().begin;

    assert(size == 1);

    assert(([&]() -> bool {
             for (size_t i = 0, e = result.size(); i < e; ++i) {
               if (result[i] < ranges[i].begin ||
                   result[i] >= ranges[i].end) {
                 return false;
               }
             }

             return true;
           }()) && "Wrong index unflattening result");
  }
}
