#include "marco/Runtime/Modeling/Range.h"
#include <cassert>

namespace marco::runtime
{
  RangeIterator::RangeIterator(int64_t begin, int64_t end) : current_(begin), end_(end)
  {
    assert(begin <= end);
  }

  RangeIterator RangeIterator::begin(const Range& range)
  {
    return RangeIterator(range.begin, range.end);
  }

  RangeIterator RangeIterator::end(const Range& range)
  {
    return RangeIterator(range.end, range.end);
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

  int64_t RangeIterator::operator*()
  {
    return current_;
  }
}
