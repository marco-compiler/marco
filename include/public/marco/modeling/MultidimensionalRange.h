#ifndef MARCO_MODELING_MULTIDIMENSIONALRANGE_H
#define MARCO_MODELING_MULTIDIMENSIONALRANGE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "marco/modeling/Range.h"

namespace marco::modeling
{
  namespace impl
  {
    template<typename ValueType>
    class MultidimensionalRangeIterator
    {
      public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Point;
        using difference_type = std::ptrdiff_t;
        using pointer = Point*;
        using reference = Point&;

        MultidimensionalRangeIterator(
            llvm::ArrayRef<Range> ranges,
            std::function<RangeIterator<ValueType>(const Range&)> initFunction)
        {
          for (const auto& range: ranges) {
            beginIterators.push_back(range.begin());
            auto it = initFunction(range);
            currentIterators.push_back(it);
            endIterators.push_back(range.end());
            indexes.push_back(*it);
          }

          assert(ranges.size() == beginIterators.size());
          assert(ranges.size() == currentIterators.size());
          assert(ranges.size() == endIterators.size());
          assert(ranges.size() == indexes.size());
        }

        bool operator==(const MultidimensionalRangeIterator& it) const
        {
          return currentIterators == it.currentIterators;
        }

        bool operator!=(const MultidimensionalRangeIterator& it) const
        {
          return currentIterators != it.currentIterators;
        }

        MultidimensionalRangeIterator& operator++()
        {
          fetchNext();
          return *this;
        }

        MultidimensionalRangeIterator operator++(int)
        {
          auto temp = *this;
          fetchNext();
          return temp;
        }

        value_type operator*() const
        {
          return Point(indexes);
        }

      private:
        void fetchNext()
        {
          size_t size = indexes.size();

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

            indexes[pos] = *currentIterators[pos];

            for (size_t i = pos + 1; i < size; ++i) {
              currentIterators[i] = beginIterators[i];
              indexes[i] = *currentIterators[i];
            }
          }
        }

        llvm::SmallVector<RangeIterator<ValueType>, 3> beginIterators;
        llvm::SmallVector<RangeIterator<ValueType>, 3> currentIterators;
        llvm::SmallVector<RangeIterator<ValueType>, 3> endIterators;
        llvm::SmallVector<ValueType, 3> indexes;
    };
  }

  /// n-D range. Each dimension is half-open as the 1-D range.
  class MultidimensionalRange
  {
    private:
      using Container = llvm::SmallVector<Range, 2>;

    public:
      using data_type = Range::data_type;
      using const_iterator = impl::MultidimensionalRangeIterator<data_type>;

      MultidimensionalRange(llvm::ArrayRef<Range> ranges);

      // TODO test
      MultidimensionalRange(Point point);

      bool operator==(const Point& other) const;

      bool operator==(const MultidimensionalRange& other) const;

      bool operator!=(const Point& other) const;

      bool operator!=(const MultidimensionalRange& other) const;

      bool operator<(const MultidimensionalRange& other) const;

      bool operator>(const MultidimensionalRange& other) const;

      Range& operator[](size_t index);

      const Range& operator[](size_t index) const;
    
      llvm::ArrayRef<Range> getRanges() const;

      unsigned int rank() const;

      void getSizes(llvm::SmallVectorImpl<size_t>& sizes) const;

      unsigned int flatSize() const;

      bool contains(const Point& other) const;

      bool contains(const MultidimensionalRange& other) const;

      bool overlaps(const MultidimensionalRange& other) const;

      MultidimensionalRange intersect(const MultidimensionalRange& other) const;

      /// Check if two multidimensional ranges can be merged.
      ///
      /// @return a pair whose first element is whether the merge is possible
      /// and the second is the dimension to be merged
      std::pair<bool, size_t> canBeMerged(const MultidimensionalRange& other) const;

      MultidimensionalRange merge(const MultidimensionalRange& other, size_t dimension) const;

      std::vector<MultidimensionalRange> subtract(const MultidimensionalRange& other) const;

      const_iterator begin() const;

      const_iterator end() const;

    private:
      Container ranges;
  };

  std::ostream& operator<<(std::ostream& stream, const MultidimensionalRange& obj);
}

#endif // MARCO_MODELING_MULTIDIMENSIONALRANGE_H
