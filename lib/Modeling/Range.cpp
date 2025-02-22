#include "marco/Modeling/Range.h"
#include "llvm/Support/raw_ostream.h"

namespace marco::modeling {
Range::Range(Range::data_type begin, Range::data_type end)
    : begin_(begin), end_(end) {
  assert(begin < end && "Range is not well-formed");
}

llvm::hash_code hash_value(const Range &value) {
  return llvm::hash_combine(value.begin_, value.end_);
}

bool Range::operator==(Range::data_type other) const {
  return getBegin() == other && getEnd() == other + 1;
}

bool Range::operator==(const Range &other) const {
  return getBegin() == other.getBegin() && getEnd() == other.getEnd();
}

bool Range::operator!=(Range::data_type other) const {
  return getBegin() != other && getEnd() != other + 1;
}

bool Range::operator!=(const Range &other) const {
  return getBegin() != other.getBegin() || getEnd() != other.getEnd();
}

bool Range::operator<(const Range &other) const {
  if (getBegin() == other.getBegin()) {
    return getEnd() < other.getEnd();
  }

  return getBegin() < other.getBegin();
}

bool Range::operator>(const Range &other) const {
  if (getBegin() == other.getBegin()) {
    return getEnd() > other.getEnd();
  }

  return getBegin() > other.getBegin();
}

Range::data_type Range::getBegin() const { return begin_; }

Range::data_type Range::getEnd() const { return end_; }

size_t Range::size() const { return getEnd() - getBegin(); }

int Range::compare(const Range &other) const {
  if (getBegin() == other.getBegin()) {
    if (getEnd() == other.getEnd()) {
      return 0;
    }

    if (getEnd() < other.getEnd()) {
      return -1;
    }

    return 1;
  }

  if (getBegin() < other.getBegin()) {
    return -1;
  }

  return 1;
}

bool Range::contains(Range::data_type value) const {
  return value >= getBegin() && value < getEnd();
}

bool Range::contains(const Range &other) const {
  return getBegin() <= other.getBegin() && getEnd() >= other.getEnd();
}

bool Range::overlaps(const Range &other) const {
  return (getBegin() <= other.getEnd() - 1) &&
         (getEnd() - 1 >= other.getBegin());
}

Range Range::intersect(const Range &other) const {
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

bool Range::canBeMerged(const Range &other) const {
  return getBegin() == other.getEnd() || getEnd() == other.getBegin() ||
         overlaps(other);
}

Range Range::merge(const Range &other) const {
  assert(canBeMerged(other));

  if (overlaps(other)) {
    Point::data_type begin = std::min(getBegin(), other.getBegin());
    Point::data_type end = std::max(getEnd(), other.getEnd());
    return Range(begin, end);
  }

  if (getBegin() == other.getEnd()) {
    return Range(other.getBegin(), getEnd());
  }

  return Range(getBegin(), other.getEnd());
}

std::vector<Range> Range::subtract(const Range &other) const {
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

Range::const_iterator Range::begin() const {
  return const_iterator(getBegin(), getEnd());
}

Range::const_iterator Range::end() const {
  return const_iterator(getEnd(), getEnd());
}

Range::Iterator::Iterator(Point::data_type begin, Point::data_type end)
    : current(begin), end(end) {
  assert(begin <= end);
}

bool Range::Iterator::operator==(const Range::Iterator &other) const {
  return current == other.current && end == other.end;
}

bool Range::Iterator::operator!=(const Range::Iterator &other) const {
  return current != other.current || end != other.end;
}

Range::Iterator &Range::Iterator::operator++() {
  current = std::min(current + 1, end);
  return *this;
}

Range::Iterator Range::Iterator::operator++(int) {
  auto temp = *this;
  current = std::min(current + 1, end);
  return temp;
}

Range::Iterator::value_type Range::Iterator::operator*() { return current; }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Range &obj) {
  return os << "[" << obj.getBegin() << "," << obj.getEnd() << ")";
}
} // namespace marco::modeling
