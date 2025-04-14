#ifndef MARCO_MODELING_POINT_H
#define MARCO_MODELING_POINT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMapInfo.h"
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
  using Container = llvm::SmallVector<data_type>;

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

  bool operator<(const Point &other) const;

  /// Get the number of dimensions.
  size_t rank() const;

  const_iterator begin() const;

  const_iterator end() const;

  int compare(const Point &other) const;

  Point append(const Point &other) const;

  Point takeFront(size_t n) const;

  Point takeBack(size_t n) const;

  Point slice(const llvm::BitVector &filter) const;

  operator llvm::ArrayRef<data_type>() const;

private:
  Container values;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Point &obj);
} // namespace marco::modeling

namespace llvm {
template <>
struct DenseMapInfo<marco::modeling::Point> {
  using Key = marco::modeling::Point;

  static inline Key getEmptyKey() {
    using ArrayRefDenseInfo =
        llvm::DenseMapInfo<llvm::ArrayRef<Key::data_type>>;

    return {ArrayRefDenseInfo::getEmptyKey()};
  }

  static inline Key getTombstoneKey() {
    using ArrayRefDenseInfo =
        llvm::DenseMapInfo<llvm::ArrayRef<Key::data_type>>;

    return {ArrayRefDenseInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const Key &val) { return hash_value(val); }

  static bool isEqual(const Key &lhs, const Key &rhs) { return lhs == rhs; }
};
} // namespace llvm

#endif // MARCO_MODELING_POINT_H
