#include "marco/Modeling/MultidimensionalRange.h"

namespace marco::modeling
{
  MultidimensionalRange::MultidimensionalRange(llvm::ArrayRef<Range> ranges)
      : ranges(ranges.begin(), ranges.end())
  {
    assert(!ranges.empty());
  }

  MultidimensionalRange::MultidimensionalRange(Point point)
  {
    for (const auto& index : point) {
      ranges.emplace_back(index, index + 1);
    }
  }

  bool MultidimensionalRange::operator==(const Point& other) const
  {
    if (rank() != other.rank())
      return false;

    for (size_t i = 0, e = rank(); i < e; ++i) {
      if (ranges[i] != other[i]) {
        return false;
      }
    }

    return true;
  }

  bool MultidimensionalRange::operator==(const MultidimensionalRange& other) const
  {
    if (rank() != other.rank()) {
      return false;
    }

    for (size_t i = 0, e = rank(); i < e; ++i) {
      if (ranges[i] != other.ranges[i]) {
        return false;
      }
    }

    return true;
  }

  bool MultidimensionalRange::operator!=(const Point& other) const
  {
    if (rank() != other.rank())
      return true;

    for (size_t i = 0, e = rank(); i < e; ++i) {
      if (ranges[i] != other[i]) {
        return true;
      }
    }

    return false;
  }

  bool MultidimensionalRange::operator!=(const MultidimensionalRange& other) const
  {
    if (rank() != other.rank()) {
      return true;
    }

    for (const auto&[lhs, rhs]: llvm::zip(ranges, other.ranges)) {
      if (lhs != rhs) {
        return true;
      }
    }

    return false;
  }

  bool MultidimensionalRange::operator<(const MultidimensionalRange& other) const
  {
    assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");

    for (size_t i = 0, e = rank(); i < e; ++i) {
      if (ranges[i] < other.ranges[i]) {
        return true;
      }

      if (ranges[i] > other.ranges[i]) {
        return false;
      }
    }

    return false;
  }

  bool MultidimensionalRange::operator>(const MultidimensionalRange& other) const
  {
    assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");

    for (size_t i = 0, e = rank(); i < e; ++i) {
      if (ranges[i] > other.ranges[i]) {
        return true;
      }

      if (ranges[i] < other.ranges[i]) {
        return false;
      }
    }

    return false;
  }

  Range& MultidimensionalRange::operator[](size_t index)
  {
    assert(index < ranges.size());
    return ranges[index];
  }

  const Range& MultidimensionalRange::operator[](size_t index) const
  {
    assert(index < ranges.size());
    return ranges[index];
  }

  unsigned int MultidimensionalRange::rank() const
  {
    return ranges.size();
  }

  void MultidimensionalRange::getSizes(llvm::SmallVectorImpl<size_t>& sizes) const
  {
    for (size_t i = 0, e = rank(); i < e; ++i) {
      sizes.push_back(ranges[i].size());
    }
  }

  unsigned int MultidimensionalRange::flatSize() const
  {
    unsigned int result = 1;

    for (unsigned int i = 0, r = rank(); i < r; ++i) {
      result *= (*this)[i].getEnd() - (*this)[i].getBegin();
    }

    return result;
  }

  bool MultidimensionalRange::contains(const Point& other) const
  {
    for (const auto&[position, range]: llvm::zip(other, ranges)) {
      if (!range.contains(position)) {
        return false;
      }
    }

    return true;
  }

  bool MultidimensionalRange::contains(const MultidimensionalRange& other) const
  {
    assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");

    for (size_t i = 0, e = rank(); i < e; ++i) {
      if (!ranges[i].contains(other.ranges[i])) {
        return false;
      }
    }

    return true;
  }

  bool MultidimensionalRange::overlaps(const MultidimensionalRange& other) const
  {
    assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");

    for (const auto&[x, y]: llvm::zip(ranges, other.ranges)) {
      if (!x.overlaps(y)) {
        return false;
      }
    }

    return true;
  }

  MultidimensionalRange MultidimensionalRange::intersect(const MultidimensionalRange& other) const
  {
    assert(overlaps(other));
    llvm::SmallVector<Range, 3> intersectionRanges;

    for (const auto&[x, y]: llvm::zip(ranges, other.ranges)) {
      intersectionRanges.push_back(x.intersect(y));
    }

    return MultidimensionalRange(std::move(intersectionRanges));
  }

  std::pair<bool, size_t> MultidimensionalRange::canBeMerged(const MultidimensionalRange& other) const
  {
    assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");

    bool found = false;
    size_t dimension = 0;

    for (size_t i = 0, e = rank(); i < e; ++i) {
      if (const Range& first = ranges[i], second = other.ranges[i]; first != second) {
        if (first.canBeMerged(other.ranges[i])) {
          if (found) {
            return std::make_pair(false, 0);
          }

          found = true;
          dimension = i;
        } else {
          return std::make_pair(false, 0);
        }
      }
    }

    return std::make_pair(found, dimension);
  }

  MultidimensionalRange MultidimensionalRange::merge(const MultidimensionalRange& other, size_t dimension) const
  {
    assert(rank() == other.rank());
    llvm::SmallVector<Range, 3> mergedRanges;

    for (size_t i = 0, e = rank(); i < e; ++i) {
      if (i == dimension) {
        Range merged = ranges[i].merge(other.ranges[i]);
        mergedRanges.push_back(std::move(merged));
      } else {
        assert(ranges[i] == other.ranges[i]);
        mergedRanges.push_back(ranges[i]);
      }
    }

    return MultidimensionalRange(mergedRanges);
  }

  std::vector<MultidimensionalRange> MultidimensionalRange::subtract(const MultidimensionalRange& other) const
  {
    assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");
    std::vector<MultidimensionalRange> results;

    if (!overlaps(other)) {
      results.push_back(*this);
    } else {
      llvm::SmallVector<Range, 3> resultRanges(ranges.begin(), ranges.end());

      for (size_t i = 0, e = rank(); i < e; ++i) {
        const auto& x = ranges[i];
        const auto& y = other.ranges[i];
        assert(x.overlaps(y));

        for (const auto& subRange: x.subtract(y)) {
          resultRanges[i] = std::move(subRange);
          results.emplace_back(resultRanges);
        }

        resultRanges[i] = x.intersect(y);
      }
    }

    return results;
  }

  MultidimensionalRange::const_iterator MultidimensionalRange::begin() const
  {
    return const_iterator(
        ranges, [](const Range& range) {
          return range.begin();
        });
  }

  MultidimensionalRange::const_iterator MultidimensionalRange::end() const
  {
    return const_iterator(
        ranges, [](const Range& range) {
          return range.end();
        });
  }

  MultidimensionalRange MultidimensionalRange::slice(size_t dimensions) const
  {
    assert(dimensions <= rank());
    llvm::SmallVector<Range, 3> slicedRanges;

    for (size_t i = 0; i < dimensions; ++i) {
      slicedRanges.push_back(ranges[i]);
    }

    return MultidimensionalRange(std::move(slicedRanges));
  }

  std::ostream& operator<<(std::ostream& stream, const MultidimensionalRange& obj)
  {
    stream << "[";

    for (size_t i = 0, e = obj.rank(); i < e; ++i) {
      if (i != 0) {
        stream << ",";
      }

      stream << obj[i];
    }

    stream << "]";
    return stream;
  }

  MultidimensionalRange::Iterator::Iterator(
      llvm::ArrayRef<Range> ranges,
      std::function<Range::const_iterator(const Range&)> initFunction)
  {
    for (const auto& range: ranges) {
      beginIterators.push_back(range.begin());
      auto it = initFunction(range);
      currentIterators.push_back(it);
      endIterators.push_back(range.end());
      indices.push_back(*it);
    }

    assert(ranges.size() == beginIterators.size());
    assert(ranges.size() == currentIterators.size());
    assert(ranges.size() == endIterators.size());
    assert(ranges.size() == indices.size());
  }

  bool MultidimensionalRange::Iterator::operator==(const Iterator& it) const
  {
    return currentIterators == it.currentIterators;
  }

  bool MultidimensionalRange::Iterator::operator!=(const Iterator& it) const
  {
    return currentIterators != it.currentIterators;
  }

  MultidimensionalRange::Iterator& MultidimensionalRange::Iterator::operator++()
  {
    fetchNext();
    return *this;
  }

  MultidimensionalRange::Iterator MultidimensionalRange::Iterator::operator++(int)
  {
    auto temp = *this;
    fetchNext();
    return temp;
  }

  MultidimensionalRange::Iterator::value_type MultidimensionalRange::Iterator::operator*() const
  {
    return Point(indices);
  }

  void MultidimensionalRange::Iterator::fetchNext()
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
