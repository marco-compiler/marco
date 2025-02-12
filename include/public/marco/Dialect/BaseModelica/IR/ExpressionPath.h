#ifndef MARCO_DIALECT_BASEMODELICA_IR_EXPRESSIONPATH_H
#define MARCO_DIALECT_BASEMODELICA_IR_EXPRESSIONPATH_H

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::bmodelica {
class ExpressionPath {
public:
  class Guard {
  public:
    Guard(ExpressionPath &path);

    ~Guard();

  private:
    ExpressionPath *path;
    size_t size;
  };

private:
  using Container = llvm::SmallVector<uint64_t, 3>;

public:
  using const_iterator = Container::const_iterator;

  ExpressionPath(llvm::ArrayRef<uint64_t> path);

  friend llvm::hash_code hash_value(const ExpressionPath &value);

  bool operator==(const ExpressionPath &other) const;
  bool operator!=(const ExpressionPath &other) const;

  uint64_t operator[](size_t index) const;
  size_t size() const;

  const_iterator begin() const;
  const_iterator end() const;

  void append(uint64_t index);

  ExpressionPath operator+(uint64_t index) const;

  ExpressionPath &operator+=(uint64_t index);

private:
  Container path;
};
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_IR_EXPRESSIONPATH_H
