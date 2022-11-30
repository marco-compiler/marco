#ifndef MARCO_RUNTIME_MODELING_RANGE_H
#define MARCO_RUNTIME_MODELING_RANGE_H

#include <cstdint>
#include <iterator>

namespace marco::runtime
{
  /// Mono-dimensional range in the form [begin, end).
  struct Range
  {
    int64_t begin;
    int64_t end;

    bool operator<(const Range& other) const;
  };

  class RangeIterator
  {
    public:
      using iterator_category = std::input_iterator_tag;
      using value_type = int64_t;
      using difference_type = std::ptrdiff_t;
      using pointer = int64_t*;
      using reference = int64_t&;

      static RangeIterator begin(const Range& range);
      static RangeIterator end(const Range& range);

      bool operator==(const RangeIterator& it) const;

      bool operator!=(const RangeIterator& it) const;

      RangeIterator& operator++();

      RangeIterator operator++(int);

      value_type operator*();

    private:
      RangeIterator(int64_t begin, int64_t end);

    private:
      int64_t current_;
      int64_t end_;
  };
}

#endif // MARCO_RUNTIME_MODELING_RANGE_H
