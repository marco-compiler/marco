#ifndef MARCO_MODELING_POINT_H
#define MARCO_MODELING_POINT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include <initializer_list>

namespace llvm {
class raw_ostream;
}

namespace marco::modeling {
/// n-D point.
class Point {
public:
  using data_type = int64_t;

private:
  using Container = llvm::SmallVector<data_type, 3>;

public:
  using const_iterator = Container::const_iterator;

  Point();

  Point(data_type value);

  Point(std::initializer_list<data_type> values);

  Point(llvm::ArrayRef<data_type> values);

  friend llvm::hash_code hash_value(const Point &value);

  bool operator==(const Point &other) const;

  bool operator!=(const Point &other) const;

  data_type operator[](size_t index) const;

  /// Get the number of dimensions.
  size_t rank() const;

  const_iterator begin() const;

  const_iterator end() const;

  Point append(const Point &other) const;

  Point takeFront(size_t n) const;

  Point takeBack(size_t n) const;

  operator llvm::ArrayRef<data_type>() const;

private:
  Container values;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Point &obj);
} // namespace marco::modeling

#endif // MARCO_MODELING_POINT_H
