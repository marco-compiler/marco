#ifndef MARCO_MODELING_INDEXSET_H
#define MARCO_MODELING_INDEXSET_H

#include "marco/Modeling/MultidimensionalRange.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>
#include <list>
#include <set>
#include <variant>

namespace marco::modeling
{
   /// Multidimensional Compressed Index Set (MCIS).
   /// It replaces the multidimensional vectors in order to achieve O(1) scaling.
  class IndexSet
  {
    public:
      // this iterator class is to iterate MultidimensionalRanges
      class RangeIterator
      {
        public:
          using iterator_category = std::input_iterator_tag;
          using value_type = MultidimensionalRange;
          using difference_type = std::ptrdiff_t;
          using pointer = const MultidimensionalRange*;
          using reference = const MultidimensionalRange&;

          class Impl;

          RangeIterator(const RangeIterator& other);

          RangeIterator(RangeIterator&& other);

          ~RangeIterator();

          RangeIterator& operator=(const RangeIterator& other);

          friend void swap(RangeIterator& first, RangeIterator& second);

          static RangeIterator begin(const IndexSet& indexSet);

          static RangeIterator end(const IndexSet& indexSet);

          bool operator==(const RangeIterator& it) const;

          bool operator!=(const RangeIterator& it) const;

          RangeIterator& operator++();

          RangeIterator operator++(int);

          reference operator*() const;

        private:
          RangeIterator(std::unique_ptr<Impl> impl);

          std::unique_ptr<Impl> impl;
      };

      // this iterator class is to iterate indexes (points)
      class PointIterator
      {
        public:
          using value_type = Point;
          using reference = const Point&;

          class Impl;

          PointIterator(const PointIterator& other);

          PointIterator(PointIterator&& other);

          ~PointIterator();

          PointIterator& operator=(const PointIterator& other);

          friend void swap(PointIterator& first, PointIterator& second);

          static PointIterator begin(const IndexSet& indexSet);

          static PointIterator end(const IndexSet& indexSet);

          bool operator==(const PointIterator& it) const;

          bool operator!=(const PointIterator& it) const;

          PointIterator& operator++();

          PointIterator operator++(int);

          value_type operator*() const;

        private:
          PointIterator(std::unique_ptr<Impl> impl);

          std::unique_ptr<Impl> impl;
      };

      using const_range_iterator = RangeIterator;
      using const_point_iterator = PointIterator;
      class Impl;

      IndexSet();

      IndexSet(llvm::ArrayRef<Point> points);

      IndexSet(llvm::ArrayRef<MultidimensionalRange> ranges);

      IndexSet(const IndexSet& other);

      IndexSet(IndexSet&& other);

      ~IndexSet();

      IndexSet& operator=(const IndexSet& other);

      friend void swap(IndexSet& first, IndexSet& second);

      friend std::ostream& operator<<(std::ostream& os, const IndexSet& obj);

      bool operator==(const Point& rhs) const;

      bool operator==(const MultidimensionalRange& rhs) const;

      bool operator==(const IndexSet& rhs) const;

      bool operator!=(const Point& rhs) const;

      bool operator!=(const MultidimensionalRange& rhs) const;

      bool operator!=(const IndexSet& rhs) const;

      IndexSet& operator+=(const Point& rhs);

      IndexSet& operator+=(const MultidimensionalRange& rhs);

      IndexSet& operator+=(const IndexSet& rhs);

      IndexSet operator+(const Point& rhs) const;

      IndexSet operator+(const MultidimensionalRange& rhs) const;

      IndexSet operator+(const IndexSet& rhs) const;

      IndexSet& operator-=(const MultidimensionalRange& rhs);

      IndexSet& operator-=(const IndexSet& rhs);

      IndexSet operator-(const MultidimensionalRange& rhs) const;

      IndexSet operator-(const IndexSet& rhs) const;

      bool empty() const;

      size_t rank() const;

      size_t flatSize() const;

      void clear();

      const_point_iterator begin() const;

      const_point_iterator end() const;

      // todo: if called on something that is not a lvalue, it produces a LLVM ERROR: out of memory
      llvm::iterator_range<const_range_iterator> getRanges() const;

      bool contains(const Point& other) const;

      bool contains(const MultidimensionalRange& other) const;

      bool contains(const IndexSet& other) const;

      bool overlaps(const MultidimensionalRange& other) const;

      bool overlaps(const IndexSet& other) const;

      IndexSet intersect(const MultidimensionalRange& other) const;

      IndexSet intersect(const IndexSet& other) const;

      IndexSet complement(const MultidimensionalRange& other) const;

    private:
      IndexSet(std::unique_ptr<Impl> impl);

      std::unique_ptr<Impl> impl;
  };
}

#endif  // MARCO_MODELING_INDEXSET_H
