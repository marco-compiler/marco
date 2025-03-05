#include "marco/Dialect/BaseModelica/IR/ExpressionPath.h"
#include "llvm/ADT/ArrayRef.h"

using namespace ::mlir::bmodelica;

namespace mlir::bmodelica {
ExpressionPath::Guard::Guard(ExpressionPath &path)
    : path(&path), size(path.size()) {}

ExpressionPath::Guard::~Guard() {
  if (path->size() > size) {
    path->path.erase(std::next(path->path.begin(), size), path->path.end());
  }
}

ExpressionPath::ExpressionPath(llvm::ArrayRef<uint64_t> path)
    : path(path.begin(), path.end()) {}

llvm::hash_code hash_value(const ExpressionPath &value) {
  return llvm::hash_value(llvm::ArrayRef(value.path));
}

bool ExpressionPath::operator==(const ExpressionPath &other) const {
  return path == other.path;
}

bool ExpressionPath::operator!=(const ExpressionPath &other) const {
  return path != other.path;
}

bool ExpressionPath::operator<(const ExpressionPath &other) const {
  for (auto [lhs, rhs] : llvm::zip(path, other.path)) {
    if (lhs < rhs) {
      return true;
    } else if (lhs > rhs) {
      return false;
    }
  }

  if (path.size() < other.path.size()) {
    return true;
  }

  return false;
}

uint64_t ExpressionPath::operator[](size_t index) const {
  assert(index < path.size());
  return path[index];
}

size_t ExpressionPath::size() const { return path.size(); }

ExpressionPath::const_iterator ExpressionPath::begin() const {
  return path.begin();
}

ExpressionPath::const_iterator ExpressionPath::end() const {
  return path.end();
}

void ExpressionPath::append(uint64_t index) { path.push_back(index); }

ExpressionPath ExpressionPath::operator+(uint64_t index) const {
  ExpressionPath result(*this);
  result.append(index);
  return result;
}

ExpressionPath &ExpressionPath::operator+=(uint64_t index) {
  append(index);
  return *this;
}
} // namespace mlir::bmodelica
