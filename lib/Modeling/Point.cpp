#include "marco/Modeling/Point.h"
#include "llvm/Support/raw_ostream.h"

namespace marco::modeling {
Point::Point() : Point(0) {}

Point::Point(Point::data_type value) { values.push_back(std::move(value)); }

Point::Point(std::initializer_list<Point::data_type> values)
    : values(std::move(values)) {}

Point::Point(llvm::ArrayRef<Point::data_type> values)
    : values(values.begin(), values.end()) {}

llvm::hash_code hash_value(const Point &value) {
  return llvm::hash_combine_range(value.values.begin(), value.values.end());
}

bool Point::operator==(const Point &other) const {
  if (values.size() != other.values.size()) {
    return false;
  }

  for (size_t i = 0, e = rank(); i < e; ++i) {
    if (values[i] != other.values[i]) {
      return false;
    }
  }

  return true;
}

bool Point::operator!=(const Point &other) const { return !(*this == other); }

Point::data_type Point::operator[](size_t index) const {
  assert(index < values.size());
  return values[index];
}

size_t Point::rank() const { return values.size(); }

Point::const_iterator Point::begin() const { return values.begin(); }

Point::const_iterator Point::end() const { return values.end(); }

Point Point::append(const Point &other) const {
  llvm::SmallVector<data_type> newValues{llvm::ArrayRef<data_type>(values)};
  newValues.append(other.values);
  return Point(newValues);
}

Point Point::takeFront(size_t n) const {
  return Point(llvm::ArrayRef<data_type>(values).take_front(n));
}

Point Point::takeBack(size_t n) const {
  return Point(llvm::ArrayRef<data_type>(values).take_back(n));
}

Point::operator llvm::ArrayRef<data_type>() const { return values; }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Point &obj) {
  os << "(";

  for (size_t i = 0, e = obj.rank(); i < e; ++i) {
    if (i != 0) {
      os << ",";
    }

    os << obj[i];
  }

  return os << ")";
}
} // namespace marco::modeling
