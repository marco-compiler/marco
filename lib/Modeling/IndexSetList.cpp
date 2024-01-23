#include "marco/Modeling/IndexSetList.h"
#include "llvm/Support/raw_ostream.h"
#include <queue>

using namespace ::marco::modeling;
using namespace ::marco::modeling::impl;

//===---------------------------------------------------------------------===//
// List-IndexSet: point iterator
//===---------------------------------------------------------------------===//

namespace marco::modeling::impl
{
  class ListIndexSet::PointIterator : public IndexSet::PointIterator::Impl
  {
    public:
      using iterator_category =
          IndexSet::PointIterator::Impl::iterator_category;

      using value_type = IndexSet::PointIterator::Impl::value_type;
      using difference_type = IndexSet::PointIterator::Impl::difference_type;
      using pointer = IndexSet::PointIterator::Impl::pointer;
      using reference = IndexSet::PointIterator::Impl::reference;

      /// @name LLVM-style RTTI methods
      /// {

      static bool classof(const IndexSet::PointIterator::Impl* obj)
      {
        return obj->getKind() == List;
      }

      /// }

      std::unique_ptr<IndexSet::PointIterator::Impl> clone() const override;

      /// @name Construction methods
      /// {

      static IndexSet::PointIterator begin(const ListIndexSet& indexSet);

      static IndexSet::PointIterator end(const ListIndexSet& indexSet);

      /// }

      bool operator==(
          const IndexSet::PointIterator::Impl& other) const override;

      bool operator!=(
          const IndexSet::PointIterator::Impl& other) const override;

      IndexSet::PointIterator::Impl& operator++() override;

      IndexSet::PointIterator operator++(int) override;

      value_type operator*() const override;

    private:
      PointIterator(
          IndexSet::RangeIterator currentRangeIt,
          IndexSet::RangeIterator endRangeIt,
          std::optional<MultidimensionalRange::const_iterator> currentPointIt,
          std::optional<MultidimensionalRange::const_iterator> endPointIt);

      bool shouldProceed() const;

      void fetchNext();

      void advance();

    private:
      IndexSet::RangeIterator currentRangeIt;
      IndexSet::RangeIterator endRangeIt;
      std::optional<MultidimensionalRange::const_iterator> currentPointIt;
      std::optional<MultidimensionalRange::const_iterator> endPointIt;
  };

  ListIndexSet::PointIterator::PointIterator(
      IndexSet::RangeIterator currentRangeIt,
      IndexSet::RangeIterator endRangeIt,
      std::optional<MultidimensionalRange::const_iterator> currentPointIt,
      std::optional<MultidimensionalRange::const_iterator> endPointIt)
      : IndexSet::PointIterator::Impl(List),
        currentRangeIt(std::move(currentRangeIt)),
        endRangeIt(std::move(endRangeIt)),
        currentPointIt(std::move(currentPointIt)),
        endPointIt(std::move(endPointIt))
  {
    fetchNext();
  }

  std::unique_ptr<IndexSet::PointIterator::Impl>
  ListIndexSet::PointIterator::clone() const
  {
    return std::make_unique<ListIndexSet::PointIterator>(*this);
  }

  IndexSet::PointIterator ListIndexSet::PointIterator::begin(
      const ListIndexSet& indexSet)
  {
    auto currentRangeIt = indexSet.rangesBegin();
    auto endRangeIt = indexSet.rangesEnd();

    if (currentRangeIt == endRangeIt) {
      // There are no ranges. The current range iterator is already
      // past-the-end, and thus we must avoid dereferencing it.

      ListIndexSet::PointIterator it(
          currentRangeIt, endRangeIt, std::nullopt, std::nullopt);

      return { std::make_unique<ListIndexSet::PointIterator>(std::move(it)) };
    }

    auto currentPointIt = (*currentRangeIt).begin();
    auto endPointIt = (*currentRangeIt).end();

    ListIndexSet::PointIterator it(
        currentRangeIt, endRangeIt, currentPointIt, endPointIt);

    return { std::make_unique<ListIndexSet::PointIterator>(std::move(it)) };
  }

  IndexSet::PointIterator ListIndexSet::PointIterator::end(
      const ListIndexSet& indexSet)
  {
    ListIndexSet::PointIterator it(
        indexSet.rangesEnd(), indexSet.rangesEnd(), std::nullopt, std::nullopt);

    return { std::make_unique<ListIndexSet::PointIterator>(std::move(it)) };
  }

  bool ListIndexSet::PointIterator::operator==(
      const IndexSet::PointIterator::Impl& other) const
  {
    if (auto* it = other.dyn_cast<ListIndexSet::PointIterator>()) {
      return currentRangeIt == it->currentRangeIt &&
          currentPointIt == it->currentPointIt;
    }

    return false;
  }

  bool ListIndexSet::PointIterator::operator!=(
      const IndexSet::PointIterator::Impl& other) const
  {
    if (auto* it = other.dyn_cast<ListIndexSet::PointIterator>()) {
      return currentRangeIt != it->currentRangeIt ||
          currentPointIt != it->currentPointIt;
    }

    return true;
  }

  IndexSet::PointIterator::Impl& ListIndexSet::PointIterator::operator++()
  {
    advance();
    return *this;
  }

  IndexSet::PointIterator ListIndexSet::PointIterator::operator++(int)
  {
    IndexSet::PointIterator result(clone());
    ++(*this);
    return result;
  }

  ListIndexSet::PointIterator::value_type
  ListIndexSet::PointIterator::operator*() const
  {
    return **currentPointIt;
  }

  bool ListIndexSet::PointIterator::shouldProceed() const
  {
    if (currentRangeIt == endRangeIt) {
      return false;
    }

    return currentPointIt == endPointIt;
  }

  void ListIndexSet::PointIterator::fetchNext()
  {
    while (shouldProceed()) {
      bool advanceToNextRange = currentPointIt == endPointIt;

      if (advanceToNextRange) {
        ++currentRangeIt;

        if (currentRangeIt == endRangeIt) {
          currentPointIt = std::nullopt;
          endPointIt = std::nullopt;
        } else {
          currentPointIt = (*currentRangeIt).begin();
          endPointIt = (*currentRangeIt).end();
        }
      } else {
        ++(*currentPointIt);
      }
    }
  }

  void ListIndexSet::PointIterator::advance()
  {
    ++(*currentPointIt);
    fetchNext();
  }
}

//===---------------------------------------------------------------------===//
// List-IndexSet: range iterator
//===---------------------------------------------------------------------===//

namespace marco::modeling::impl
{
  class ListIndexSet::RangeIterator : public IndexSet::RangeIterator::Impl
  {
    public:
      using iterator_category =
          IndexSet::RangeIterator::Impl::iterator_category;

      using value_type = IndexSet::RangeIterator::Impl::value_type;
      using difference_type = IndexSet::RangeIterator::Impl::difference_type;
      using pointer = IndexSet::RangeIterator::Impl::pointer;
      using reference = IndexSet::RangeIterator::Impl::reference;

    private:
      using ListIterator = decltype(ListIndexSet::ranges)::const_iterator;

    public:
      /// @name LLVM-style RTTI methods
      /// {

      static bool classof(const IndexSet::RangeIterator::Impl* obj)
      {
        return obj->getKind() == List;
      }

      /// }

      std::unique_ptr<IndexSet::RangeIterator::Impl> clone() const override;

      /// @name Construction methods
      /// {

      static IndexSet::RangeIterator begin(const ListIndexSet& indexSet);

      static IndexSet::RangeIterator end(const ListIndexSet& indexSet);

      /// }

      bool operator==(
          const IndexSet::RangeIterator::Impl& other) const override;

      bool operator!=(
          const IndexSet::RangeIterator::Impl& other) const override;

      IndexSet::RangeIterator::Impl& operator++() override;

      IndexSet::RangeIterator operator++(int) override;

      reference operator*() const override;

    private:
      explicit RangeIterator(ListIterator rangeIt);

    private:
      ListIterator rangeIt;
  };


  ListIndexSet::RangeIterator::RangeIterator(
      ListIndexSet::RangeIterator::ListIterator rangeIt)
      : IndexSet::RangeIterator::Impl(List),
        rangeIt(rangeIt)
  {
  }

  std::unique_ptr<IndexSet::RangeIterator::Impl>
  ListIndexSet::RangeIterator::clone() const
  {
    return std::make_unique<ListIndexSet::RangeIterator>(*this);
  }

  IndexSet::RangeIterator ListIndexSet::RangeIterator::begin(
      const ListIndexSet& indexSet)
  {
    ListIndexSet::RangeIterator it(indexSet.ranges.begin());

    return { std::make_unique<ListIndexSet::RangeIterator>(std::move(it)) };
  }

  IndexSet::RangeIterator ListIndexSet::RangeIterator::end(
      const ListIndexSet& indexSet)
  {
    ListIndexSet::RangeIterator it(indexSet.ranges.end());

    return { std::make_unique<ListIndexSet::RangeIterator>(std::move(it)) };
  }

  bool ListIndexSet::RangeIterator::operator==(
      const IndexSet::RangeIterator::Impl& other) const
  {
    if (auto* it = other.dyn_cast<ListIndexSet::RangeIterator>()) {
      return rangeIt == it->rangeIt;
    }

    return false;
  }

  bool ListIndexSet::RangeIterator::operator!=(
      const IndexSet::RangeIterator::Impl& other) const
  {
    if (auto* it = other.dyn_cast<ListIndexSet::RangeIterator>()) {
      return rangeIt != it->rangeIt;
    }

    return true;
  }

  IndexSet::RangeIterator::Impl& ListIndexSet::RangeIterator::operator++()
  {
    ++rangeIt;
    return *this;
  }

  IndexSet::RangeIterator ListIndexSet::RangeIterator::operator++(int)
  {
    IndexSet::RangeIterator result(clone());
    ++(*this);
    return result;
  }

  ListIndexSet::RangeIterator::reference
  ListIndexSet::RangeIterator::operator*() const
  {
    return *rangeIt;
  }
}

//===----------------------------------------------------------------------===//
// List-IndexSet
//===----------------------------------------------------------------------===//

namespace marco::modeling::impl
{
  ListIndexSet::ListIndexSet()
      : IndexSet::Impl(List),
        initialized(false),
        allowedRank(0)
  {
  }

  ListIndexSet::ListIndexSet(llvm::ArrayRef<Point> points)
      : ListIndexSet()
  {
    for (const Point& point : points) {
      this->ranges.push_back(MultidimensionalRange(point));
    }

    if (!this->ranges.empty()) {
      this->initialized = true;
      this->allowedRank = this->ranges.front().rank();

      split();
      sort();
      merge();
    }
  }

  ListIndexSet::ListIndexSet(llvm::ArrayRef<MultidimensionalRange> ranges)
      : ListIndexSet()
  {
    for (const MultidimensionalRange& range : ranges) {
      this->ranges.push_back(range);
    }

    if (!this->ranges.empty()) {
      this->initialized = true;
      this->allowedRank = this->ranges.front().rank();

      split();
      sort();
      removeDuplicates();
      merge();
    }
  }

  ListIndexSet::ListIndexSet(const ListIndexSet& other)
      : IndexSet::Impl(List),
        ranges(other.ranges),
        initialized(other.initialized),
        allowedRank(other.allowedRank)
  {
  }

  ListIndexSet::ListIndexSet(ListIndexSet&& other) = default;

  ListIndexSet::~ListIndexSet() = default;

  std::unique_ptr<IndexSet::Impl> ListIndexSet::clone() const
  {
    return std::make_unique<ListIndexSet>(*this);
  }

  llvm::hash_code hash_value(const ListIndexSet& value)
  {
    return llvm::hash_combine_range(value.ranges.begin(), value.ranges.end());
  }

  llvm::raw_ostream& ListIndexSet::dump(llvm::raw_ostream& os) const
  {
    os << "{";

    bool separator = false;

    for (const MultidimensionalRange& range : ranges) {
      if (separator) {
        os << ", ";
      }

      separator = true;
      os << range;
    }

    return os << "}";
  }

  bool ListIndexSet::operator==(const Point& rhs) const
  {
    if (ranges.size() != 1) {
      return false;
    }

    return ranges.front() == rhs;
  }

  bool ListIndexSet::operator==(const MultidimensionalRange& rhs) const
  {
    if (ranges.size() != 1) {
      return false;
    }

    return ranges.front() == rhs;
  }

  bool ListIndexSet::operator==(const IndexSet::Impl& rhs) const
  {
    if (auto* rhsCasted = rhs.dyn_cast<ListIndexSet>()) {
      return *this == *rhsCasted;
    }

    if (rank() != rhs.rank()) {
      return false;
    }

    return contains(rhs) && rhs.contains(*this);
  }

  bool ListIndexSet::operator==(const ListIndexSet& rhs) const
  {
    if (rank() != rhs.rank()) {
      return false;
    }

    return contains(rhs) && rhs.contains(*this);
  }

  bool ListIndexSet::operator!=(const Point& rhs) const
  {
    return !(*this == rhs);
  }

  bool ListIndexSet::operator!=(const MultidimensionalRange& rhs) const
  {
    return !(*this == rhs);
  }

  bool ListIndexSet::operator!=(const IndexSet::Impl& rhs) const
  {
    return !(*this == rhs);
  }

  bool ListIndexSet::operator!=(const ListIndexSet& rhs) const
  {
    return !(*this == rhs);
  }

  IndexSet::Impl& ListIndexSet::operator+=(const Point& rhs)
  {
    llvm::SmallVector<Range, 3> elementRanges;

    for (const Point::data_type& index : rhs) {
      elementRanges.emplace_back(index, index + 1);
    }

    return *this += MultidimensionalRange(elementRanges);
  }

  IndexSet::Impl& ListIndexSet::operator+=(const MultidimensionalRange& rhs)
  {
    if (!initialized) {
      allowedRank = rhs.rank();
      initialized = true;
    }

    assert(rhs.rank() == allowedRank && "Incompatible rank");

    llvm::SmallVector<MultidimensionalRange> nonOverlappingRanges;
    nonOverlappingRanges.push_back(rhs);

    for (const MultidimensionalRange& range : ranges) {
      llvm::SmallVector<MultidimensionalRange> newCandidates;

      for (const MultidimensionalRange& candidate : nonOverlappingRanges) {
        for (MultidimensionalRange& diff : candidate.subtract(range)) {
          newCandidates.push_back(std::move(diff));
        }
      }

      nonOverlappingRanges = std::move(newCandidates);
    }

    for (MultidimensionalRange& range : nonOverlappingRanges) {
      assert(llvm::none_of(ranges, [&](const MultidimensionalRange& r) {
               return r.overlaps(range);
             }) && "New range must not overlap the existing ones");

      auto it = llvm::find_if(
          ranges, [&range](const MultidimensionalRange& r) {
            return r > range;
          });

      ranges.insert(it, std::move(range));
    }

    split();
    sort();
    merge();

    return *this;
  }

  IndexSet::Impl& ListIndexSet::operator+=(const IndexSet::Impl& rhs)
  {
    if (auto* rhsCasted = rhs.dyn_cast<ListIndexSet>()) {
      return *this += *rhsCasted;
    }

    for (const MultidimensionalRange& range :
         llvm::make_range(rhs.rangesBegin(), rhs.rangesEnd())) {
      *this += range;
    }

    return *this;
  }

  IndexSet::Impl& ListIndexSet::operator+=(const ListIndexSet& rhs)
  {
    auto beginIt = rhs.ranges.rbegin();
    auto endIt = rhs.ranges.rend();

    for (auto it = beginIt; it != endIt; ++it) {
      const MultidimensionalRange& range = *it;
      *this += range;
    }

    return *this;
  }

  IndexSet::Impl& ListIndexSet::operator-=(const Point& rhs)
  {
    llvm::SmallVector<Range, 3> elementRanges;

    for (const Point::data_type& index : rhs) {
      elementRanges.emplace_back(index, index + 1);
    }

    return *this -= MultidimensionalRange(elementRanges);
  }

  IndexSet::Impl& ListIndexSet::operator-=(const MultidimensionalRange& rhs)
  {
    if (!initialized) {
      allowedRank = rhs.rank();
      initialized = true;
    }

    assert(rhs.rank() == allowedRank && "Incompatible rank");

    if (ranges.empty()) {
      return *this;
    }

    [[maybe_unused]] auto hasCompatibleRank =
        [&](const MultidimensionalRange& range) {
          if (ranges.empty()) {
            return true;
          }

          return ranges.front().rank() == range.rank();
        };

    assert(hasCompatibleRank(rhs) && "Incompatible ranges");

    llvm::SmallVector<MultidimensionalRange> newRanges;

    for (const MultidimensionalRange& range : ranges) {
      for (MultidimensionalRange& diff: range.subtract(rhs)) {
        newRanges.push_back(std::move(diff));
      }
    }

    ranges.clear();
    llvm::sort(newRanges);
    ranges.insert(ranges.begin(), newRanges.begin(), newRanges.end());

    split();
    sort();
    merge();

    return *this;
  }

  IndexSet::Impl& ListIndexSet::operator-=(const IndexSet::Impl& rhs)
  {
    if (auto* rhsCasted = rhs.dyn_cast<ListIndexSet>()) {
      return *this -= *rhsCasted;
    }

    for (const MultidimensionalRange& range :
         llvm::make_range(rhs.rangesBegin(), rhs.rangesEnd())) {
      *this -= range;
    }

    return *this;
  }

  IndexSet::Impl& ListIndexSet::operator-=(const ListIndexSet& rhs)
  {
    for (const MultidimensionalRange& range : rhs.ranges) {
      *this -= range;
    }

    return *this;
  }

  bool ListIndexSet::empty() const
  {
    return ranges.empty();
  }

  size_t ListIndexSet::rank() const
  {
    return allowedRank;
  }

  size_t ListIndexSet::flatSize() const
  {
    size_t result = 0;

    for (const MultidimensionalRange& range : ranges) {
      result += range.flatSize();
    }

    return result;
  }

  void ListIndexSet::clear()
  {
    ranges.clear();
  }

  ListIndexSet::const_point_iterator ListIndexSet::begin() const
  {
    return ListIndexSet::PointIterator::begin(*this);
  }

  ListIndexSet::const_point_iterator ListIndexSet::end() const
  {
    return ListIndexSet::PointIterator::end(*this);
  }

  ListIndexSet::const_range_iterator ListIndexSet::rangesBegin() const
  {
    return ListIndexSet::RangeIterator::begin(*this);
  }

  ListIndexSet::const_range_iterator ListIndexSet::rangesEnd() const
  {
    return ListIndexSet::RangeIterator::end(*this);
  }

  bool ListIndexSet::contains(const Point& other) const
  {
    return llvm::any_of(ranges, [&](const MultidimensionalRange& range) {
      return range.contains(other);
    });
  }

  bool ListIndexSet::contains(const MultidimensionalRange& other) const
  {
    if (empty()) {
      return false;
    }

    std::queue<MultidimensionalRange> nonOverlappingRanges;
    nonOverlappingRanges.push(other);

    bool shouldContinue = true;

    while (!nonOverlappingRanges.empty() && shouldContinue) {
      const MultidimensionalRange& current = nonOverlappingRanges.front();

      for (const MultidimensionalRange& range : ranges) {
        if (range.contains(current)) {
          nonOverlappingRanges.pop();
          break;
        } else if (range.overlaps(current)) {
          for (MultidimensionalRange& diff : current.subtract(range)) {
            assert(other.contains(diff));
            nonOverlappingRanges.push(std::move(diff));
          }

          nonOverlappingRanges.pop();
          break;
        } else if (range > current) {
          shouldContinue = false;
          break;
        }
      }
    }

    return nonOverlappingRanges.empty();
  }

  bool ListIndexSet::contains(const IndexSet::Impl& other) const
  {
    if (auto* otherCasted = other.dyn_cast<ListIndexSet>()) {
      return contains(*otherCasted);
    }

    return llvm::all_of(
        llvm::make_range(other.rangesBegin(), other.rangesEnd()),
        [&](const MultidimensionalRange& range) {
          return contains(range);
        });
  }

  bool ListIndexSet::contains(const ListIndexSet& other) const
  {
    return llvm::all_of(other.ranges, [&](const MultidimensionalRange& range) {
      return contains(range);
    });
  }

  bool ListIndexSet::overlaps(const MultidimensionalRange& other) const
  {
    for (const MultidimensionalRange& range : ranges) {
      if (range.overlaps(other)) {
        return true;
      }

      if (range > other) {
        return false;
      }
    }

    return false;
  }

  bool ListIndexSet::overlaps(const IndexSet::Impl& other) const
  {
    if (auto* otherCasted = other.dyn_cast<ListIndexSet>()) {
      return overlaps(*otherCasted);
    }

    return llvm::any_of(ranges, [&](const MultidimensionalRange& range) {
      return other.overlaps(range);
    });
  }

  bool ListIndexSet::overlaps(const ListIndexSet& other) const
  {
    return llvm::any_of(ranges, [&](const MultidimensionalRange& range) {
      return other.overlaps(range);
    });
  }

  IndexSet ListIndexSet::intersect(const MultidimensionalRange& other) const
  {
    IndexSet result;

    for (const MultidimensionalRange& range : ranges) {
      if (range.overlaps(other)) {
        result += range.intersect(other);
      }
    }

    return result;
  }

  IndexSet ListIndexSet::intersect(const IndexSet::Impl& other) const
  {
    if (auto* otherCasted = other.dyn_cast<ListIndexSet>()) {
      return intersect(*otherCasted);
    }

    IndexSet result;

    for (const MultidimensionalRange& range1 : ranges) {
      for (const MultidimensionalRange& range2 :
           llvm::make_range(other.rangesBegin(), other.rangesEnd())) {
        if (range1.overlaps(range2)) {
          result += range1.intersect(range2);
        }
      }
    }

    return result;
  }

  IndexSet ListIndexSet::intersect(const ListIndexSet& other) const
  {
    IndexSet result;

    for (const MultidimensionalRange& range1 : ranges) {
      for (const MultidimensionalRange& range2 : other.ranges) {
        if (range1.overlaps(range2)) {
          result += range1.intersect(range2);
        }
      }
    }

    return result;
  }

  IndexSet ListIndexSet::complement(const MultidimensionalRange& other) const
  {
    if (ranges.empty()) {
      return {other};
    }

    llvm::SmallVector<MultidimensionalRange> result;

    llvm::SmallVector<MultidimensionalRange> current;
    current.push_back(other);

    for (const MultidimensionalRange& range : ranges) {
      llvm::SmallVector<MultidimensionalRange> next;

      for (const MultidimensionalRange& curr : current) {
        for (MultidimensionalRange& diff: curr.subtract(range)) {
          if (overlaps(diff)) {
            next.push_back(std::move(diff));
          } else {
            result.push_back(std::move(diff));
          }
        }
      }

      current = next;
    }

    return { result };
  }

  std::unique_ptr<IndexSet::Impl>
  ListIndexSet::takeFirstDimensions(size_t n) const
  {
    assert(n < rank());
    llvm::SmallVector<MultidimensionalRange> result;

    for (const MultidimensionalRange& range :
         llvm::make_range(rangesBegin(), rangesEnd())) {
      result.push_back(range.takeFirstDimensions(n));
    }

    return std::make_unique<ListIndexSet>(std::move(result));
  }

  std::unique_ptr<IndexSet::Impl>
  ListIndexSet::takeLastDimensions(size_t n) const
  {
    assert(n < rank());
    llvm::SmallVector<MultidimensionalRange> result;

    for (const MultidimensionalRange& range :
         llvm::make_range(rangesBegin(), rangesEnd())) {
      result.push_back(range.takeLastDimensions(n));
    }

    return std::make_unique<ListIndexSet>(std::move(result));
  }

  std::unique_ptr<IndexSet::Impl>
  ListIndexSet::takeDimensions(const llvm::SmallBitVector& dimensions) const
  {
    assert(dimensions.size() == rank());
    llvm::SmallVector<MultidimensionalRange> result;

    for (const MultidimensionalRange& range : ranges) {
      result.push_back(range.takeDimensions(dimensions));
    }

    return std::make_unique<ListIndexSet>(std::move(result));
  }

  std::unique_ptr<IndexSet::Impl>
  ListIndexSet::dropFirstDimensions(size_t n) const
  {
    assert(n < rank());
    llvm::SmallVector<MultidimensionalRange> result;

    for (const MultidimensionalRange& range :
         llvm::make_range(rangesBegin(), rangesEnd())) {
      result.push_back(range.dropFirstDimensions(n));
    }

    return std::make_unique<ListIndexSet>(std::move(result));
  }

  std::unique_ptr<IndexSet::Impl>
  ListIndexSet::dropLastDimensions(size_t n) const
  {
    assert(n < rank());
    llvm::SmallVector<MultidimensionalRange> result;

    for (const MultidimensionalRange& range :
         llvm::make_range(rangesBegin(), rangesEnd())) {
      result.push_back(range.dropLastDimensions(n));
    }

    return std::make_unique<ListIndexSet>(std::move(result));
  }

  std::unique_ptr<IndexSet::Impl>
  ListIndexSet::append(const IndexSet& other) const
  {
    auto result = std::make_unique<ListIndexSet>();

    if (empty()) {
      for (const MultidimensionalRange& range :
           llvm::make_range(other.rangesBegin(), other.rangesEnd())) {
        *result += range;
      }
    } else if (other.empty()) {
      for (const MultidimensionalRange& range :
           llvm::make_range(rangesBegin(), rangesEnd())) {
        *result += range;
      }
    } else {
      for (const MultidimensionalRange& range :
           llvm::make_range(rangesBegin(), rangesEnd())) {
        for (const MultidimensionalRange& otherRange :
             llvm::make_range(other.rangesBegin(), other.rangesEnd())) {
          *result += range.append(otherRange);
        }
      }
    }

    return std::move(result);
  }

  std::unique_ptr<IndexSet::Impl>
  ListIndexSet::getCanonicalRepresentation() const
  {
    return clone();
  }

  void ListIndexSet::split()
  {
    if (ranges.empty()) {
      return;
    }

    assert(allowedRank > 0);

    for (size_t dimension = 0; dimension < allowedRank - 1; ++dimension) {
      for (auto it1 = ranges.begin(); it1 != ranges.end(); ++it1) {
        auto it2 = ranges.begin();

        while (it2 != ranges.end()) {
          if (it1 == it2) {
            ++it2;
            continue;
          }

          const MultidimensionalRange& range = *it2;
          const MultidimensionalRange& grid = *it1;

          if (shouldSplitRange(range, grid, dimension)) {
            std::list<MultidimensionalRange> splitRanges = splitRange(
                range, grid, dimension);

            ranges.splice(it2, splitRanges);
            it2 = ranges.erase(it2);
          } else {
            ++it2;
          }
        }
      }
    }
  }

  bool ListIndexSet::shouldSplitRange(
      const MultidimensionalRange& range,
      const MultidimensionalRange& grid,
      size_t dimension) const
  {
    assert(range.rank() == grid.rank());
    assert(dimension < range.rank());

    return grid[dimension].overlaps(range[dimension]);
  }

  std::list<MultidimensionalRange> ListIndexSet::splitRange(
      const MultidimensionalRange& range,
      const MultidimensionalRange& grid,
      size_t dimension) const
  {
    assert(range[dimension].overlaps(grid[dimension]));

    std::list<MultidimensionalRange> result;

    llvm::SmallVector<Range> newDimensionRanges;
    newDimensionRanges.push_back(range[dimension].intersect(grid[dimension]));

    for (Range& diff : range[dimension].subtract(grid[dimension])) {
      newDimensionRanges.push_back(std::move(diff));
    }

    llvm::SmallVector<Range> newRanges;

    for (Range& newDimensionRange : newDimensionRanges) {
      newRanges.clear();

      for (size_t i = 0; i < range.rank(); ++i) {
        if (i == dimension) {
          newRanges.push_back(std::move(newDimensionRange));
        } else {
          newRanges.push_back(range[i]);
        }
      }

      result.push_back(MultidimensionalRange(newRanges));
    }

    return result;
  }

  void ListIndexSet::sort()
  {
    ranges.sort([](const MultidimensionalRange& first,
                   const MultidimensionalRange& second) {
      return first < second;
    });
  }

  void ListIndexSet::removeDuplicates()
  {
    assert(llvm::is_sorted(
        ranges, [](const MultidimensionalRange& first,
                   const MultidimensionalRange& second) {
          return first < second;
        }));

    auto it = ranges.begin();

    if (it == ranges.end()) {
      return;
    }

    while (it != ranges.end()) {
      auto next = std::next(it);

      while (next != ranges.end() && *next == *it) {
        ranges.erase(next);
        next = std::next(it);
      }

      ++it;
    }
  }

  void ListIndexSet::merge()
  {
    using It = decltype(ranges)::iterator;

    auto findCandidates = [&](It begin, It end) -> std::tuple<It, It, size_t> {
      for (auto it1 = begin; it1 != end; ++it1) {
        for (auto it2 = std::next(it1); it2 != end; ++it2) {
          if (auto mergePossibility = it1->canBeMerged(*it2);
              mergePossibility.first) {
            return std::make_tuple(it1, it2, mergePossibility.second);
          }
        }
      }

      return std::make_tuple(end, end, 0);
    };

    auto candidates = findCandidates(ranges.begin(), ranges.end());

    while (std::get<0>(candidates) != ranges.end() &&
           std::get<1>(candidates) != ranges.end()) {
      auto& first = std::get<0>(candidates);
      auto& second = std::get<1>(candidates);
      size_t dimension = std::get<2>(candidates);

      *first = first->merge(*second, dimension);
      ranges.erase(second);
      candidates = findCandidates(ranges.begin(), ranges.end());
    }
  }
}
