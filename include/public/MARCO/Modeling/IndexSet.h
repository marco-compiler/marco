#ifndef MARCO_MODELING_INDEXSET_H
#define MARCO_MODELING_INDEXSET_H

#include "llvm/ADT/ArrayRef.h"
#include "marco/Modeling/MultidimensionalRange.h"
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
      using const_iterator = Container::const_iterator;

      IndexSet();

      IndexSet(llvm::ArrayRef<Point> points);

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
      void sort();

      void merge();

      std::list<MultidimensionalRange> ranges;
  };
}

#endif  // MARCO_MODELING_INDEXSET_H
