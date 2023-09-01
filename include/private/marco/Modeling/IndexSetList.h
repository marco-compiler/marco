#ifndef MARCO_MODELING_INDEXSETLIST_H
#define MARCO_MODELING_INDEXSETLIST_H

#include "marco/Modeling/IndexSetImpl.h"
#include <list>

namespace marco::modeling::impl
{
  class ListIndexSet : public IndexSet::Impl
  {
    private:
      class PointIterator;
      class RangeIterator;

      using Container = std::list<MultidimensionalRange>;

    public:
      using const_point_iterator = IndexSet::const_point_iterator;
      using const_range_iterator = IndexSet::const_range_iterator;

      ListIndexSet();

      ListIndexSet(llvm::ArrayRef<Point> points);

      ListIndexSet(llvm::ArrayRef<MultidimensionalRange> ranges);

      ListIndexSet(const ListIndexSet& other);

      ListIndexSet(ListIndexSet&& other);

      ~ListIndexSet();

      ListIndexSet& operator=(const ListIndexSet& other) = delete;

      ListIndexSet& operator=(ListIndexSet&& other) = delete;

      /// @name LLVM-style RTTI methods
      /// {

      static bool classof(const IndexSet::Impl* obj)
      {
        return obj->getKind() == List;
      }

      /// }

      friend std::ostream& operator<<(
          std::ostream& stream, const ListIndexSet& obj);

      std::unique_ptr<Impl> clone() const override;

      friend llvm::hash_code hash_value(const ListIndexSet& value);

      bool operator==(const Point& rhs) const override;

      bool operator==(const MultidimensionalRange& rhs) const override;

      bool operator==(const IndexSet::Impl& rhs) const override;

      bool operator==(const ListIndexSet& rhs) const;

      bool operator!=(const Point& rhs) const override;

      bool operator!=(const MultidimensionalRange& rhs) const override;

      bool operator!=(const IndexSet::Impl& rhs) const override;

      bool operator!=(const ListIndexSet& rhs) const;

      IndexSet::Impl& operator+=(const Point& rhs) override;

      IndexSet::Impl& operator+=(const MultidimensionalRange& rhs) override;

      IndexSet::Impl& operator+=(const IndexSet::Impl& rhs) override;

      IndexSet::Impl& operator+=(const ListIndexSet& rhs);

      IndexSet::Impl& operator-=(const Point& rhs) override;

      IndexSet::Impl& operator-=(const MultidimensionalRange& rhs) override;

      IndexSet::Impl& operator-=(const IndexSet::Impl& rhs) override;

      IndexSet::Impl& operator-=(const ListIndexSet& rhs);

      bool empty() const override;

      size_t rank() const override;

      size_t flatSize() const override;

      void clear() override;

      const_point_iterator begin() const override;

      const_point_iterator end() const override;

      const_range_iterator rangesBegin() const override;

      const_range_iterator rangesEnd() const override;

      bool contains(const Point& other) const override;

      bool contains(const MultidimensionalRange& other) const override;

      bool contains(const IndexSet::Impl& other) const override;

      bool contains(const ListIndexSet& other) const;

      bool overlaps(const MultidimensionalRange& other) const override;

      bool overlaps(const IndexSet::Impl& other) const override;

      bool overlaps(const ListIndexSet& other) const;

      IndexSet intersect(const MultidimensionalRange& other) const override;

      IndexSet intersect(const IndexSet::Impl& other) const override;

      IndexSet intersect(const ListIndexSet& other) const;

      IndexSet complement(const MultidimensionalRange& other) const override;

      std::unique_ptr<IndexSet::Impl> takeFirstDimensions(size_t n) const override;

      std::unique_ptr<IndexSet::Impl> takeLastDimensions(size_t n) const override;

      std::unique_ptr<IndexSet::Impl> takeDimensions(
          const llvm::SmallBitVector& dimensions) const override;

      std::unique_ptr<IndexSet::Impl> dropFirstDimensions(size_t n) const override;

      std::unique_ptr<IndexSet::Impl> dropLastDimensions(size_t n) const override;

      std::unique_ptr<IndexSet::Impl> append(const IndexSet& other) const override;

      std::unique_ptr<IndexSet::Impl> getCanonicalRepresentation() const override;

    private:
      void split();

      bool shouldSplitRange(
          const MultidimensionalRange& range,
          const MultidimensionalRange& grid,
          size_t dimension) const;

      std::list<MultidimensionalRange> splitRange(
          const MultidimensionalRange& range,
          const MultidimensionalRange& grid,
          size_t dimension) const;

      void sort();

      void removeDuplicates();

      void merge();

    private:
      std::list<MultidimensionalRange> ranges;
      bool initialized;
      size_t allowedRank;
  };
}

#endif // MARCO_MODELING_INDEXSETLIST_H
