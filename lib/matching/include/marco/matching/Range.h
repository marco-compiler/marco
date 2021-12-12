#ifndef MARCO_MATCHING_RANGE_H
#define MARCO_MATCHING_RANGE_H

#include "Point.h"

namespace marco::matching
{
  namespace detail
  {
    template<typename ValueType>
    class RangeIterator
    {
      public:
      using iterator_category = std::input_iterator_tag;
      using value_type = ValueType;
      using difference_type = std::ptrdiff_t;
      using pointer = ValueType*;
      using reference = ValueType&;

      RangeIterator(ValueType begin, ValueType end) : current(begin), end(end)
      {
        assert(begin <= end);
      }

      bool operator==(const RangeIterator& it) const
      {
        return current == it.current && end == it.end;
      }

      bool operator!=(const RangeIterator& it) const
      {
        return current != it.current || end != it.end;
      }

      RangeIterator& operator++()
      {
        current = std::min(current + 1, end);
        return *this;
      }

      RangeIterator operator++(int)
      {
        auto temp = *this;
        current = std::min(current + 1, end);
        return temp;
      }

      value_type operator*()
      {
        return current;
      }

      private:
      ValueType current;
      ValueType end;
    };
  }

	/**
	 * 1-D half-open range [a,b).
	 */
	class Range
	{
		public:
		using data_type = Point::data_type;
		using const_iterator = detail::RangeIterator<data_type>;

		Range(data_type begin, data_type end);

		bool operator==(const Range& other) const;
		bool operator!=(const Range& other) const;

    bool operator<(const Range& other) const;
    bool operator>(const Range& other) const;

    // TODO remove
    data_type getBegin() const;

    // TODO remove
    data_type getEnd() const;

		size_t size() const;

		bool contains(data_type value) const;
    bool contains(const Range& other) const;

		bool overlaps(const Range& other) const;

    Range intersect(const Range& other) const;

    /**
     * Check whether the range can be merged with another one.
     * Two ranges can be merged if they overlap or if they are contiguous.
     *
     * @param other  range to be merged
     * @return whether the ranges can be merged
     */
    bool canBeMerged(const Range& other) const;

    /**
     * Create a range that is the resulting of merging this one with
     * another one that can be merged.
     *
     * @param other  ranged to be merged
     * @return merged range
     */
    Range merge(const Range& other) const;

    /**
     * Subtract a range from the current one.
     * Multiple results are created if the removed range is fully contained
     * and does not touch the borders.
     *
     * @param other  ranged to be removed
     * @return ranges resulting from the subtraction
     */
    std::vector<Range> subtract(const Range& other) const;

		const_iterator begin() const;
		const_iterator end() const;

		private:
		data_type _begin;
		data_type _end;
	};

	std::ostream& operator<<(std::ostream& stream, const Range& obj);
}

#endif	// MARCO_MATCHING_RANGE_H
