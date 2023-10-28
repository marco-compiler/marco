#ifndef MARCO_MODELING_RANGE_H
#define MARCO_MODELING_RANGE_H

#include "marco/Modeling/Point.h"
#include "llvm/ADT/Hashing.h"

namespace llvm
{
  class raw_ostream;
}

namespace marco::modeling
{
  /// 1-D half-open range [a,b).
  class Range
  {
    public:
      using data_type = Point::data_type;

    private:
      class Iterator
      {
        public:
          using iterator_category = std::input_iterator_tag;
          using value_type = data_type;
          using difference_type = std::ptrdiff_t;
          using pointer = data_type*;
          using reference = data_type&;

          Iterator(data_type begin, data_type end);

          bool operator==(const Iterator& other) const;
          bool operator!=(const Iterator& other) const;

          Iterator& operator++();

          Iterator operator++(int);

          value_type operator*();

        private:
          data_type current;
          data_type end;
      };

    public:
      using const_iterator = Iterator;

      Range(data_type begin, data_type end);

      friend llvm::hash_code hash_value(const Range& value);

      bool operator==(data_type other) const;

      bool operator==(const Range& other) const;

      bool operator!=(data_type other) const;

      bool operator!=(const Range& other) const;

      bool operator<(const Range& other) const;

      bool operator>(const Range& other) const;

      data_type getBegin() const;

      data_type getEnd() const;

      size_t size() const;

      /// Check if the range contains a point.
      bool contains(data_type value) const;

      /// Check if the range contains all the points of another range.
      bool contains(const Range& other) const;

      /// Check if the range has some points in common with another one.
      bool overlaps(const Range& other) const;

      /// Get the intersection with another range.
      /// The ranges must overlap.
      Range intersect(const Range& other) const;

      /// Check whether the range can be merged with another one.
      /// Two ranges can be merged if they overlap or if they are contiguous.
      bool canBeMerged(const Range& other) const;

      /// Create a range that is the resulting of merging this one with
      /// another one that can be merged.
      Range merge(const Range& other) const;

      /// Subtract a range from the current one.
      /// Multiple results are created if the removed range is fully contained
      /// and does not touch the borders.
      std::vector<Range> subtract(const Range& other) const;

      //// @name Iterators for the contained points
      /// {

      const_iterator begin() const;

      const_iterator end() const;

      /// }

    private:
      data_type begin_;
      data_type end_;
  };

  llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Range& obj);
}

#endif  // MARCO_MODELING_RANGE_H
