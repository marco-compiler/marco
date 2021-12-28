#include <llvm/ADT/SmallVector.h>
#include <marco/modeling/MCIS.h>

namespace marco::modeling::internal
{
  MCIS::MCIS(llvm::ArrayRef<MultidimensionalRange> ranges)
      : ranges(ranges.begin(), ranges.end())
  {
    for (const auto& range: ranges) {
      this->operator+=(range);
    }
  }

  bool MCIS::operator==(const MCIS& rhs) const
  {
    return contains(rhs) && rhs.contains(*this);
  }

  bool MCIS::operator!=(const MCIS& rhs) const
  {
    return !contains(rhs) || !rhs.contains(*this);
  }

  const MultidimensionalRange& MCIS::operator[](size_t index) const
  {
    assert(index < ranges.size());
    return *(std::next(ranges.begin(), index));
  }

  MCIS& MCIS::operator+=(const Point& rhs)
  {
    llvm::SmallVector<Range, 3> elementRanges;

    for (const auto& index: rhs) {
      elementRanges.emplace_back(index, index + 1);
    }

    return this->operator+=(MultidimensionalRange(std::move(elementRanges)));
  }

  MCIS& MCIS::operator+=(const MultidimensionalRange& rhs)
  {
    auto hasCompatibleRank = [&](const MultidimensionalRange& range) {
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

  MCIS& MCIS::operator+=(const MCIS& rhs)
  {
    auto hasCompatibleRank = [&](const MCIS& mcis) {
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

  MCIS MCIS::operator+(const Point& rhs) const
  {
    MCIS result(*this);
    result += rhs;
    return result;
  }

  MCIS MCIS::operator+(const MultidimensionalRange& rhs) const
  {
    MCIS result(*this);
    result += rhs;
    return result;
  }

  MCIS MCIS::operator+(const MCIS& rhs) const
  {
    MCIS result(*this);
    result += rhs;
    return result;
  }

  MCIS& MCIS::operator-=(const MultidimensionalRange& rhs)
  {
    if (ranges.empty()) {
      return *this;
    }

    auto hasCompatibleRank = [&](const MultidimensionalRange& range) {
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

  MCIS& MCIS::operator-=(const MCIS& rhs)
  {
    for (const auto& range: rhs.ranges) {
      this->operator-=(range);
    }

    return *this;
  }

  MCIS MCIS::operator-(const MultidimensionalRange& rhs) const
  {
    MCIS result(*this);
    result -= rhs;
    return result;
  }

  MCIS MCIS::operator-(const MCIS& rhs) const
  {
    MCIS result(*this);
    result -= rhs;
    return result;
  }

  bool MCIS::empty() const
  {
    return ranges.empty();
  }

  size_t MCIS::size() const
  {
    size_t result = 0;

    for (const auto& range: ranges) {
      result += range.flatSize();
    }

    return result;
  }

  MCIS::const_iterator MCIS::begin() const
  {
    return ranges.begin();
  }

  MCIS::const_iterator MCIS::end() const
  {
    return ranges.end();
  }

  bool MCIS::contains(const Point& other) const
  {
    for (const auto& range: ranges) {
      if (range.contains(other)) {
        return true;
      }
    }

    return false;
  }

  bool MCIS::contains(const MultidimensionalRange& other) const
  {
    for (const auto& range: ranges) {
      if (range.contains(other)) {
        return true;
      }

      if (range > other) {
        return false;
      }
    }

    return false;
  }

  bool MCIS::contains(const MCIS& other) const
  {
    llvm::SmallVector<MultidimensionalRange, 3> current;

    for (const auto& range: other.ranges) {
      if (!contains(range)) {
        return false;
      }
    }

    return true;
  }

  bool MCIS::overlaps(const MultidimensionalRange& other) const
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

  bool MCIS::overlaps(const MCIS& other) const
  {
    for (const auto& range: ranges) {
      if (other.overlaps(range)) {
        return true;
      }
    }

    return false;
  }

  MCIS MCIS::intersect(const MultidimensionalRange& other) const
  {
    MCIS result;

    for (const auto& range : ranges) {
      if (range.overlaps(other)) {
        result += range.intersect(other);
      }
    }

    return result;
  }

  MCIS MCIS::intersect(const MCIS& other) const
  {
    MCIS result;

    for (const auto& range1: ranges) {
      for (const auto& range2: other.ranges) {
        if (range1.overlaps(range2)) {
          result += range1.intersect(range2);
        }
      }
    }

    return result;
  }

  MCIS MCIS::complement(const MultidimensionalRange& other) const
  {
    if (ranges.empty()) {
      return MCIS(other);
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

    return MCIS(result);
  }

  void MCIS::sort()
  {
    ranges.sort([](const MultidimensionalRange& first, const MultidimensionalRange& second) {
      return first < second;
    });
  }

  void MCIS::merge()
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

  std::ostream& operator<<(std::ostream& stream, const MCIS& obj)
  {
    stream << "{";

    for (const auto& range: obj.ranges) {
      bool positionSeparator = false;

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
}
