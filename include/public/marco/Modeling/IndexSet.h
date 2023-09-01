#ifndef MARCO_MODELING_INDEXSET_H
#define MARCO_MODELING_INDEXSET_H

#include "marco/Modeling/MultidimensionalRange.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>

namespace marco::modeling
{
  class IndexSet
  {
    public:
      class PointIterator
      {
        public:
          using iterator_category = std::input_iterator_tag;
          using value_type = Point;
          using difference_type = std::ptrdiff_t;
          using pointer = const Point*;
          using reference = const Point&;

          class Impl;

          PointIterator(std::unique_ptr<Impl> impl);

          PointIterator(const PointIterator& other);

          PointIterator(PointIterator&& other);

          ~PointIterator();

          PointIterator& operator=(const PointIterator& other);

          friend void swap(PointIterator& first, PointIterator& second);

          bool operator==(const PointIterator& it) const;

          bool operator!=(const PointIterator& it) const;

          PointIterator& operator++();

          PointIterator operator++(int);

          value_type operator*() const;

        private:
          std::unique_ptr<Impl> impl;
      };

      class RangeIterator
      {
        public:
          using iterator_category = std::input_iterator_tag;
          using value_type = MultidimensionalRange;
          using difference_type = std::ptrdiff_t;
          using pointer = const MultidimensionalRange*;
          using reference = const MultidimensionalRange&;

          class Impl;

          RangeIterator(std::unique_ptr<Impl> impl);

          RangeIterator(const RangeIterator& other);

          RangeIterator(RangeIterator&& other);

          ~RangeIterator();

          RangeIterator& operator=(const RangeIterator& other);

          friend void swap(RangeIterator& first, RangeIterator& second);

          bool operator==(const RangeIterator& it) const;

          bool operator!=(const RangeIterator& it) const;

          RangeIterator& operator++();

          RangeIterator operator++(int);

          reference operator*() const;

        private:
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

      IndexSet& operator=(IndexSet&& other);

      friend void swap(IndexSet& first, IndexSet& second);

      friend std::ostream& operator<<(std::ostream& os, const IndexSet& obj);

      friend llvm::hash_code hash_value(const IndexSet& value);

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

      IndexSet& operator-=(const Point& rhs);

      IndexSet& operator-=(const MultidimensionalRange& rhs);

      IndexSet& operator-=(const IndexSet& rhs);

      IndexSet operator-(const Point& rhs) const;

      IndexSet operator-(const MultidimensionalRange& rhs) const;

      IndexSet operator-(const IndexSet& rhs) const;

      bool empty() const;

      size_t rank() const;

      size_t flatSize() const;

      void clear();

      const_point_iterator begin() const;

      const_point_iterator end() const;

      const_range_iterator rangesBegin() const;

      const_range_iterator rangesEnd() const;

      bool contains(const Point& other) const;

      bool contains(const MultidimensionalRange& other) const;

      bool contains(const IndexSet& other) const;

      bool overlaps(const MultidimensionalRange& other) const;

      bool overlaps(const IndexSet& other) const;

      IndexSet intersect(const MultidimensionalRange& other) const;

      IndexSet intersect(const IndexSet& other) const;

      IndexSet complement(const MultidimensionalRange& other) const;

      IndexSet takeFirstDimensions(size_t n) const;

      IndexSet takeLastDimensions(size_t n) const;

      IndexSet takeDimensions(const llvm::SmallBitVector& dimensions) const;

      IndexSet dropFirstDimensions(size_t n) const;

      IndexSet dropLastDimensions(size_t n) const;

      IndexSet append(const IndexSet& other) const;

      IndexSet getCanonicalRepresentation() const;

    private:
      IndexSet(std::unique_ptr<Impl> impl);

    private:
      std::unique_ptr<Impl> impl;
  };
}

#endif  // MARCO_MODELING_INDEXSET_H
