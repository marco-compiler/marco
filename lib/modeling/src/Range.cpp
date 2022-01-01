#include <marco/modeling/Range.h>

namespace marco::modeling::internal
{
  Range::Range(Range::data_type begin, Range::data_type end)
      : _begin(begin), _end(end)
  {
    assert(begin < end && "Range is not well-formed");
  }

  bool Range::operator==(Range::data_type other) const
  {
    return getBegin() == other && getEnd() == other + 1;
  }

  bool Range::operator==(const Range& other) const
  {
    return getBegin() == other.getBegin() && getEnd() == other.getEnd();
  }

  bool Range::operator!=(Range::data_type other) const
  {
    return getBegin() != other && getEnd() != other + 1;
  }

  bool Range::operator!=(const Range& other) const
  {
    return getBegin() != other.getBegin() || getEnd() != other.getEnd();
  }

  bool Range::operator<(const Range& other) const
  {
    if (getBegin() == other.getBegin()) {
      return getEnd() < other.getEnd();
    }

    return getBegin() < other.getBegin();
  }

  bool Range::operator>(const Range& other) const
  {
    if (getBegin() == other.getBegin()) {
      return getEnd() > other.getEnd();
    }

    return getBegin() > other.getBegin();
  }

  Range::data_type Range::getBegin() const
  {
    return _begin;
  }

  Range::data_type Range::getEnd() const
  {
    return _end;
  }

  size_t Range::size() const
  {
    return getEnd() - getBegin();
  }

  bool Range::contains(Range::data_type value) const
  {
    return value >= getBegin() && value < getEnd();
  }

  bool Range::contains(const Range& other) const
  {
    return getBegin() <= other.getBegin() && getEnd() >= other.getEnd();
  }

  bool Range::overlaps(const Range& other) const
  {
    return (getBegin() <= other.getEnd() - 1) && (getEnd() - 1 >= other.getBegin());
  }

  Range Range::intersect(const Range& other) const
  {
    assert(overlaps(other));

    if (contains(other)) {
      return other;
    }

    if (other.contains(*this)) {
      return *this;
    }

    if (getBegin() <= other.getBegin()) {
      return Range(other.getBegin(), getEnd());
    }

    return Range(getBegin(), other.getEnd());
  }

  bool Range::canBeMerged(const Range& other) const
  {
    return getBegin() == other.getEnd() || getEnd() == other.getBegin() || overlaps(other);
  }

  Range Range::merge(const Range& other) const
  {
    assert(canBeMerged(other));

    if (overlaps(other)) {
      return Range(std::min(getBegin(), other.getBegin()), std::max(getEnd(), other.getEnd()));
    }

    if (getBegin() == other.getEnd()) {
      return Range(other.getBegin(), getEnd());
    }

    return Range(getBegin(), other.getEnd());
  }

  std::vector<Range> Range::subtract(const Range& other) const
  {
    std::vector<Range> results;

    if (!overlaps(other)) {
      results.push_back(*this);
    } else if (contains(other)) {
      if (getBegin() != other.getBegin()) {
        results.emplace_back(getBegin(), other.getBegin());
      }

      if (getEnd() != other.getEnd()) {
        results.emplace_back(other.getEnd(), getEnd());
      }
    } else if (!other.contains(*this)) {
      if (getBegin() <= other.getBegin()) {
        results.emplace_back(getBegin(), other.getBegin());
      } else {
        results.emplace_back(other.getEnd(), getEnd());
      }
    }

    return results;
  }

  Range::const_iterator Range::begin() const
  {
    return const_iterator(getBegin(), getEnd());
  }

  Range::const_iterator Range::end() const
  {
    return const_iterator(getEnd(), getEnd());
  }

  std::ostream& operator<<(std::ostream& stream, const Range& obj)
  {
    return stream << "[" << obj.getBegin() << "," << obj.getEnd() << ")";
  }
}
