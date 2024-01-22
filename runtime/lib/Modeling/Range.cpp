#include "marco/Runtime/Modeling/Range.h"
#include <algorithm>
#include <cassert>

namespace marco::runtime
{
  Range::Range()
      : begin(0), end(1)
  {
  }

  Range::Range(int64_t begin, int64_t end)
      : begin(begin),
        end(end)
  {
  }

  bool Range::operator<(const Range& other) const
  {
    if (begin == other.begin) {
      return end < other.end;
    }

    return begin < other.begin;
  }

  RangeIterator::RangeIterator(int64_t begin, int64_t end) : current_(begin), end_(end)
  {
    assert(begin <= end);
  }

  RangeIterator RangeIterator::begin(const Range& range)
  {
    return {range.begin, range.end};
  }

  RangeIterator RangeIterator::end(const Range& range)
  {
    return {range.end, range.end};
  }

  bool RangeIterator::operator==(const RangeIterator& it) const
  {
    return current_ == it.current_ && end_ == it.end_;
  }

  bool RangeIterator::operator!=(const RangeIterator& it) const
  {
    return current_ != it.current_ || end_ != it.end_;
  }

  RangeIterator& RangeIterator::operator++()
  {
    current_ = std::min(current_ + 1, end_);
    return *this;
  }

  RangeIterator RangeIterator::operator++(int)
  {
    RangeIterator temp = *this;
    current_ = std::min(current_ + 1, end_);
    return temp;
  }

  int64_t RangeIterator::operator*() const
  {
    return current_;
  }
}
