#ifndef MARCO_MODELING_INDEXSET_H
#define MARCO_MODELING_INDEXSET_H

#include "llvm/ADT/ArrayRef.h"
#include "marco/Modeling/MultidimensionalRange.h"
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
      class Iterator
      {
        public:
          using iterator_category = std::input_iterator_tag;
          using value_type = MultidimensionalRange;
          using difference_type = std::ptrdiff_t;
          using pointer = const MultidimensionalRange*;
          using reference = const MultidimensionalRange&;

          class Impl;

          Iterator(const Iterator& other);

          Iterator(Iterator&& other);

          ~Iterator();

          Iterator& operator=(const Iterator& other);

          friend void swap(Iterator& first, Iterator& second);

          static Iterator begin(const IndexSet& indexSet);

          static Iterator end(const IndexSet& indexSet);

          bool operator==(const Iterator& it) const;

          bool operator!=(const Iterator& it) const;

          Iterator& operator++();

          Iterator operator++(int);

          reference operator*() const;

        private:
          Iterator(std::unique_ptr<Impl> impl);

          std::unique_ptr<Impl> impl;
      };

      using const_iterator = Iterator;
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

      const_iterator begin() const;

      const_iterator end() const;

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
