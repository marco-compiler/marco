#ifndef MARCO_MODELING_INDEXSET_H
#define MARCO_MODELING_INDEXSET_H

#include "llvm/ADT/ArrayRef.h"
#include "marco/modeling/MultidimensionalRange.h"
#include <list>

namespace marco::modeling
{
   /// Multidimensional Compressed Index Set (MCIS).
   /// It replaces the multidimensional vectors in order to achieve O(1) scaling.
  class IndexSet
  {
    private:
      using Container = std::list<MultidimensionalRange>;

    public:
      class Iterator
      {
        using RangeIterator = MultidimensionalRange::const_iterator;
      public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Point;
        using difference_type = std::ptrdiff_t;
        using pointer = Point*;
        using reference = Point&;

        Iterator(const IndexSet &container);
        Iterator(const IndexSet &container, bool end);

        bool operator==(const Iterator& it) const;
        bool operator!=(const Iterator& it) const;
        
        Iterator& operator++();
        Iterator operator++(int);
        value_type operator*() const;

      private:
        void fetchNext();

        const IndexSet* container;
        llvm::ArrayRef<MultidimensionalRange>::const_iterator rangeIt;
        MultidimensionalRange::const_iterator it;
        bool end;
      };
      friend class Iterator;
      using const_iterator = Iterator;
      using const_range_iterator = Container::const_iterator;

      IndexSet();

      // IndexSet(const Point &point);
      IndexSet(llvm::ArrayRef<Point> points);

      //IndexSet(const MultidimensionalRange &range);
      IndexSet(llvm::ArrayRef<MultidimensionalRange> ranges);

      friend std::ostream& operator<<(std::ostream& stream, const IndexSet& obj);

      bool operator==(const Point& rhs) const;

      bool operator==(const MultidimensionalRange& rhs) const;

      bool operator==(const IndexSet& rhs) const;

      bool operator!=(const Point& rhs) const;

      bool operator!=(const MultidimensionalRange& rhs) const;

      bool operator!=(const IndexSet& rhs) const;

      const MultidimensionalRange& operator[](size_t index) const;

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

      size_t size() const;

      size_t rank() const;

      llvm::ArrayRef<MultidimensionalRange> getRanges()	const;

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

      IndexSet complement(const IndexSet& other) const;

      MultidimensionalRange minContainingRange() const;

    private:
      void sort();

      void merge();

      llvm::SmallVector<MultidimensionalRange,3> ranges;
  };
}

#endif  // MARCO_MODELING_INDEXSET_H
