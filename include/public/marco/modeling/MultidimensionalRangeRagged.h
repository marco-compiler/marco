#pragma once
#include "marco/modeling/MultidimensionalRange.h"
#include "marco/modeling/RangeRagged.h"
#include "marco/modeling/IndexSet.h"
#include "marco/utils/Shape.h"

#include <vector>

namespace marco::modeling
{

  class MultidimensionalRangeRagged
  {
    public:

    class MultidimensionalRangeRaggedIterator
    {
      public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Point;
        using difference_type = std::ptrdiff_t;
        using pointer = Point*;
        using reference = Point&;

        MultidimensionalRangeRaggedIterator(const MultidimensionalRangeRagged &container);
        MultidimensionalRangeRaggedIterator(const MultidimensionalRangeRagged &container, bool end);

        bool operator==(const MultidimensionalRangeRaggedIterator& it) const;

        bool operator!=(const MultidimensionalRangeRaggedIterator& it) const;
        MultidimensionalRangeRaggedIterator& operator++();

        MultidimensionalRangeRaggedIterator operator++(int);
        value_type operator*() const;

      private:
        void fetchNext();
        void updatePoint();
        void calculateRaggedStart();

        const MultidimensionalRangeRagged *container;

        llvm::SmallVector<size_t, 3> indexes;
        llvm::SmallVector<Point::data_type, 3> point;
        size_t raggedStart;
        bool end = false;
    };
  
    using const_iterator = MultidimensionalRangeRaggedIterator;

    MultidimensionalRangeRagged(const MultidimensionalRange& range);

    MultidimensionalRangeRagged(std::initializer_list<RangeRagged> list)
    {
      for (auto i : list) ranges.push_back({i});
    }

    MultidimensionalRangeRagged(llvm::ArrayRef<RangeRagged> ranges);

    [[nodiscard]] bool isRagged() const;

    unsigned int rank() const;
    [[nodiscard]] size_t dimensions() const { return ranges.size(); }

    RangeRagged& operator[](size_t index);
    const RangeRagged& operator[](size_t index) const;

    [[nodiscard]] const_iterator begin() const;
    [[nodiscard]] const_iterator end() const;

    [[nodiscard]] bool operator==(const Point& other) const;
    [[nodiscard]] bool operator==(const MultidimensionalRangeRagged& other) const;
    [[nodiscard]] bool operator!=(const Point& other) const;
    [[nodiscard]] bool operator!=(const MultidimensionalRangeRagged& other) const;
    [[nodiscard]] bool operator>(const MultidimensionalRangeRagged& other) const;
    [[nodiscard]] bool operator<(const MultidimensionalRangeRagged& other) const;

    [[nodiscard]] unsigned int flatSize() const;

    [[nodiscard]] bool contains(Point point) const;
    [[nodiscard]] bool contains(const MultidimensionalRangeRagged& other) const;

    bool overlaps(const MultidimensionalRangeRagged& other) const;

    MultidimensionalRangeRagged intersect(const MultidimensionalRangeRagged& other) const;

    /// Check if two multidimensional ranges can be merged.
    ///
    /// @return a pair whose first element is whether the merge is possible
    /// and the second is the dimension to be merged
    [[nodiscard]] std::pair<bool, size_t> canBeMerged(const MultidimensionalRangeRagged& other) const;

    MultidimensionalRangeRagged merge(const MultidimensionalRangeRagged& other, size_t dimension) const;

    std::vector<MultidimensionalRangeRagged> subtract(const MultidimensionalRangeRagged& other) const;

    [[nodiscard]] llvm::SmallVector<MultidimensionalRange, 3> toMultidimensionalRanges() const;
    [[nodiscard]] MultidimensionalRange toMultidimensionalRange() const;
    [[nodiscard]] MultidimensionalRange toMinContainingMultidimensionalRange() const;


    [[nodiscard]] bool empty() const { return flatSize() == 0; }

    private:
    llvm::SmallVector<RangeRagged, 2> ranges;

    [[nodiscard]] friend bool areDisjoint(
        const MultidimensionalRangeRagged& left, const MultidimensionalRangeRagged& right);
    [[nodiscard]] friend MultidimensionalRangeRagged intersection(
        const MultidimensionalRangeRagged& left, const MultidimensionalRangeRagged& right);
  };

  [[nodiscard]] bool areDisjoint(const RangeRagged& left, const RangeRagged& right);

  extern std::string toString(const MultidimensionalRangeRagged& value);

  extern MultidimensionalRangeRagged getMultidimensionalRangeRaggedFromShape(const Shape& shape);
  extern IndexSet getIndexSetFromRaggedRange(const MultidimensionalRangeRagged& range);
  extern IndexSet getIndexSetFromShape(const Shape& shape);

}// namespace marco::modeling
