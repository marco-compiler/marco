#include "llvm/ADT/SmallVector.h"
#include "marco/modeling/IndexSet.h"

namespace marco::modeling
{
  IndexSet::IndexSet() = default;
  // IndexSet::IndexSet(const Point &point)
  // {
  //   this->operator+=(point);
  // }
  IndexSet::IndexSet(llvm::ArrayRef<Point> points)
  {
    for (const auto& point: points) {
      this->operator+=(point);
    }
  }

  // IndexSet::IndexSet(const MultidimensionalRange& range)
  // {
  //   this->operator+=(range);
  // }

  IndexSet::IndexSet(llvm::ArrayRef<MultidimensionalRange> ranges)
  {
    for (const auto& range: ranges) {
      this->operator+=(range);
    }
  }

  bool IndexSet::operator==(const Point& rhs) const
  {
    if (ranges.size() != 1) {
      return false;
    }

    return ranges.front() == rhs;
  }

  bool IndexSet::operator==(const MultidimensionalRange& rhs) const
  {
    if (ranges.size() != 1) {
      return false;
    }

    return ranges.front() == rhs;
  }

  bool IndexSet::operator==(const IndexSet& rhs) const
  {
    return contains(rhs) && rhs.contains(*this);
  }

  bool IndexSet::operator!=(const Point& rhs) const
  {
    if (ranges.size() != 1) {
      return true;
    }

    return ranges.front() != rhs;
  }

  bool IndexSet::operator!=(const MultidimensionalRange& rhs) const
  {
    if (ranges.size() != 1) {
      return true;
    }

    return ranges.front() != rhs;
  }

  bool IndexSet::operator!=(const IndexSet& rhs) const
  {
    return !contains(rhs) || !rhs.contains(*this);
  }

  const MultidimensionalRange& IndexSet::operator[](size_t index) const
  {
    assert(index < ranges.size());
    return *(std::next(ranges.begin(), index));
  }

  IndexSet& IndexSet::operator+=(const Point& rhs)
  {
    llvm::SmallVector<Range, 3> elementRanges;

    for (const auto& index: rhs) {
      elementRanges.emplace_back(index, index + 1);
    }

    return this->operator+=(MultidimensionalRange(std::move(elementRanges)));
  }

  IndexSet& IndexSet::operator+=(const MultidimensionalRange& rhs)
  {
    [[maybe_unused]] auto hasCompatibleRank = [&](const MultidimensionalRange& range) {
      if (ranges.empty()) {
        return true;
      }

      return ranges.front().rank() == range.rank();
    };

    assert(hasCompatibleRank(rhs) && "Incompatible ranges");

    std::vector<MultidimensionalRange> nonOverlappingRanges;
    nonOverlappingRanges.push_back(std::move(rhs));

    for (const auto& range: ranges) {
      std::vector<MultidimensionalRange> newCandidates;

      for (const auto& candidate: nonOverlappingRanges) {
        for (const auto& subRange: candidate.subtract(range)) {
          newCandidates.push_back(std::move(subRange));
        }
      }

      nonOverlappingRanges = std::move(newCandidates);
    }

    for (const auto& range: nonOverlappingRanges) {
      assert(llvm::none_of(ranges, [&](const MultidimensionalRange& r) {
        return r.overlaps(range);
      }) && "New range must not overlap the existing ones");

      auto it = std::find_if(ranges.begin(), ranges.end(), [&range](const MultidimensionalRange& r) {
        return r > range;
      });

      ranges.insert(it, std::move(range));
    }

    // Insertion has already been done in-order, so we can avoid sorting the ranges
    merge();

    return *this;
  }

  IndexSet& IndexSet::operator+=(const IndexSet& rhs)
  {
    [[maybe_unused]] auto hasCompatibleRank = [&](const IndexSet& mcis) {
      if (ranges.empty() || mcis.ranges.empty()) {
        return true;
      }

      return ranges.front().rank() == mcis.ranges.front().rank();
    };

    assert(hasCompatibleRank(rhs) && "Incompatible ranges");

    for (const auto& range: rhs.ranges) {
      this->operator+=(range);
    }

    return *this;
  }

  IndexSet IndexSet::operator+(const Point& rhs) const
  {
    IndexSet result(*this);
    result += rhs;
    return result;
  }

  IndexSet IndexSet::operator+(const MultidimensionalRange& rhs) const
  {
    IndexSet result(*this);
    result += rhs;
    return result;
  }

  IndexSet IndexSet::operator+(const IndexSet& rhs) const
  {
    IndexSet result(*this);
    result += rhs;
    return result;
  }

  IndexSet& IndexSet::operator-=(const MultidimensionalRange& rhs)
  {
    if (ranges.empty()) {
      return *this;
    }

    [[maybe_unused]] auto hasCompatibleRank = [&](const MultidimensionalRange& range) {
      if (ranges.empty()) {
        return true;
      }

      return ranges.front().rank() == range.rank();
    };

    assert(hasCompatibleRank(rhs) && "Incompatible ranges");

    llvm::SmallVector<MultidimensionalRange, 3> newRanges;

    for (const auto& range: ranges) {
      for (const auto& diff: range.subtract(rhs)) {
        newRanges.push_back(std::move(diff));
      }
    }

    ranges.clear();
    llvm::sort(newRanges);
    ranges.insert(ranges.begin(), newRanges.begin(), newRanges.end());

    merge();
    return *this;
  }

  IndexSet& IndexSet::operator-=(const IndexSet& rhs)
  {
    for (const auto& range: rhs.ranges) {
      this->operator-=(range);
    }

    return *this;
  }

  IndexSet IndexSet::operator-(const MultidimensionalRange& rhs) const
  {
    IndexSet result(*this);
    result -= rhs;
    return result;
  }

  IndexSet IndexSet::operator-(const IndexSet& rhs) const
  {
    IndexSet result(*this);
    result -= rhs;
    return result;
  }

  bool IndexSet::empty() const
  {
    return ranges.empty();
  }

  size_t IndexSet::size() const
  {
    size_t result = 0;

    for (const auto& range: ranges) {
      result += range.flatSize();
    }

    return result;
  }

  size_t IndexSet::rank() const
  {
    for(const auto& range: ranges)
      return range.rank();
    
    assert(false && "requested rank of empty IndexSet");
    return 0;
  }

  llvm::ArrayRef<MultidimensionalRange> IndexSet::getRanges() const
  {
    return ranges;
  }

  IndexSet::const_iterator IndexSet::begin() const
  {
    return Iterator(*this);
  }

  IndexSet::const_iterator IndexSet::end() const
  {
    return Iterator(*this,true);
  }

  bool IndexSet::contains(const Point& other) const
  {
    for (const auto& range: ranges) {
      if (range.contains(other)) {
        return true;
      }
    }

    return false;
  }

  bool IndexSet::contains(const MultidimensionalRange& other) const
  {
    for (const auto& range: ranges) {
      if (range.contains(other)) {
        return true;
      }

      if (range > other) {
        return false;
      }
      
      if (range.overlaps(other)) {
        for(auto remainder: other.subtract(range))
        {
          if(!contains(remainder))
            return false;
        }
        return true;
      }
    }

    return false;
  }

  bool IndexSet::contains(const IndexSet& other) const
  {
    llvm::SmallVector<MultidimensionalRange, 3> current;

    for (const auto& range: other.ranges) {
      if (!contains(range)) {
        return false;
      }
    }

    return true;
  }

  bool IndexSet::overlaps(const MultidimensionalRange& other) const
  {
    for (const auto& range: ranges) {
      if (range.overlaps(other)) {
        return true;
      }

      if (range > other) {
        return false;
      }
    }

    return false;
  }

  bool IndexSet::overlaps(const IndexSet& other) const
  {
    for (const auto& range: ranges) {
      if (other.overlaps(range)) {
        return true;
      }
    }

    return false;
  }

  IndexSet IndexSet::intersect(const MultidimensionalRange& other) const
  {
    IndexSet result;

    for (const auto& range : ranges) {
      if (range.overlaps(other)) {
        result += range.intersect(other);
      }
    }

    return result;
  }

  IndexSet IndexSet::intersect(const IndexSet& other) const
  {
    IndexSet result;

    for (const auto& range1: ranges) {
      for (const auto& range2: other.ranges) {
        if (range1.overlaps(range2)) {
          result += range1.intersect(range2);
        }
      }
    }

    return result;
  }

  IndexSet IndexSet::complement(const MultidimensionalRange& other) const
  {
    if (ranges.empty()) {
      return IndexSet(other);
    }

    llvm::SmallVector<MultidimensionalRange, 3> result;

    llvm::SmallVector<MultidimensionalRange, 3> current;
    current.push_back(other);

    for (const auto& range: ranges) {
      llvm::SmallVector<MultidimensionalRange, 3> next;

      for (const auto& curr: current) {
        for (const auto& diff: curr.subtract(range)) {
          if (overlaps(diff)) {
            next.push_back(std::move(diff));
          } else {
            result.push_back(std::move(diff));
          }
        }
      }

      current = next;
    }

    return IndexSet(result);
  }

  IndexSet IndexSet::complement(const IndexSet& other) const
  {
    if (ranges.empty()) {
      return other;
    }

    llvm::SmallVector<MultidimensionalRange, 3> result;

    llvm::SmallVector<MultidimensionalRange, 3> current;
    for(auto r: other)
      current.push_back(r);

    for (const auto& range: ranges) {
      llvm::SmallVector<MultidimensionalRange, 3> next;

      for (const auto& curr: current) {
        for (const auto& diff: curr.subtract(range)) {
          if (overlaps(diff)) {
            next.push_back(std::move(diff));
          } else {
            result.push_back(std::move(diff));
          }
        }
      }

      current = next;
    }

    return IndexSet(result);
  }

  void IndexSet::sort()
  {
    llvm::sort(ranges,[](const MultidimensionalRange& first, const MultidimensionalRange& second) {
      return first < second;
    });
  }

  void IndexSet::merge()
  {
    using It = decltype(ranges)::iterator;

    auto findCandidates = [&](It begin, It end) -> std::tuple<It, It, size_t> {
      for (It it1 = begin; it1 != end; ++it1) {
        for (It it2 = std::next(it1); it2 != end; ++it2) {
          if (auto mergePossibility = it1->canBeMerged(*it2); mergePossibility.first) {
            return std::make_tuple(it1, it2, mergePossibility.second);
          }
        }
      }

      return std::make_tuple(end, end, 0);
    };

    auto candidates = findCandidates(ranges.begin(), ranges.end());

    while (std::get<0>(candidates) != ranges.end() && std::get<1>(candidates) != ranges.end()) {
      auto& first = std::get<0>(candidates);
      auto& second = std::get<1>(candidates);
      size_t dimension = std::get<2>(candidates);

      *first = first->merge(*second, dimension);
      ranges.erase(second);
      candidates = findCandidates(ranges.begin(), ranges.end());
    }
  }

  MultidimensionalRange IndexSet::minContainingRange() const
  {
    assert(!ranges.empty());

    if(ranges.size()==1)
      return ranges.front();

    llvm::SmallVector<Range,2> containingRanges;

    auto getMinContaining = [](const Range& a, const Range& b){
      return Range(std::min(a.getBegin(),b.getBegin()),std::max(a.getEnd(),b.getEnd()));
    };

    for(size_t i=0UL; i<rank(); i++)
    {
      containingRanges.push_back(ranges.front()[i]);

      for(auto it=std::next(ranges.begin()); it!=ranges.end(); ++it)
      {
        containingRanges[i] = getMinContaining(containingRanges[i],(*it)[i]);
      }
    }

    return MultidimensionalRange(containingRanges);
  }


  std::ostream& operator<<(std::ostream& stream, const IndexSet& obj)
  {
    stream << "{";

    bool positionSeparator = false;
    for (const auto& range: obj.ranges) {
      if (positionSeparator) {
        stream << "; ";
      }
      positionSeparator = false;
      for (auto indexes: range) {
        if (positionSeparator) {
          stream << ", ";
        }

        positionSeparator = true;
        stream << indexes;
      }
    }

    stream << "}";
    return stream;
  }



  // Iterator

  IndexSet::Iterator::Iterator(const IndexSet &p_container)
    : container(&p_container),rangeIt(container->ranges.begin()),it(rangeIt->begin()),end(false)
  {}

  IndexSet::Iterator::Iterator(const IndexSet &container, bool end)
    : container(&container),rangeIt(container.ranges.begin()),it(rangeIt->begin()),end(end)
  {}

  bool IndexSet::Iterator::operator==(const IndexSet::Iterator& other) const
  {
    assert(container==other.container);
    return end==other.end && (end || (it==other.it && rangeIt==other.rangeIt));
  }

  bool IndexSet::Iterator::operator!=(const IndexSet::Iterator& it) const
  {
    return !(*this==it);
  }

  IndexSet::Iterator& IndexSet::Iterator::operator++()
  {
    fetchNext();
    return *this;
  }

  IndexSet::Iterator IndexSet::Iterator::operator++(int)
  {
    auto temp = *this;
    fetchNext();
    return temp;
  }

  IndexSet::Iterator::value_type IndexSet::Iterator::operator*() const
  {
    return *it;
  }

  void IndexSet::Iterator::fetchNext()
  {
    if(end)
      return;

    if(++it==rangeIt->end())
    {
      if(++rangeIt == container->ranges.end())
      {
        
        end=true;
      }
      else
      {
        it=rangeIt->begin();
      }
    }
  }
  
}
