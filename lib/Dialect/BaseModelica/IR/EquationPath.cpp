#include "marco/Dialect/BaseModelica/IR/EquationPath.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::mlir::bmodelica;

namespace mlir::bmodelica {
EquationPath::Guard::Guard(EquationPath &path)
    : ExpressionPath::Guard(path.expressionPath) {}

EquationPath::EquationPath(EquationSide equationSide,
                           llvm::ArrayRef<uint64_t> path)
    : EquationPath(equationSide, ExpressionPath(path)) {}

EquationPath::EquationPath(EquationSide equationSide,
                           ExpressionPath expressionPath)
    : equationSide(equationSide), expressionPath(std::move(expressionPath)) {}

llvm::hash_code hash_value(const EquationPath &value) {
  return llvm::hash_combine(value.equationSide, value.expressionPath);
}

bool EquationPath::operator==(const EquationPath &other) const {
  return equationSide == other.equationSide &&
         expressionPath == other.expressionPath;
}

bool EquationPath::operator!=(const EquationPath &other) const {
  return equationSide != other.equationSide ||
         expressionPath != other.expressionPath;
}

bool EquationPath::operator<(const EquationPath &other) const {
  if (equationSide == LEFT && other.equationSide == RIGHT) {
    return true;
  }

  if (equationSide == RIGHT && other.equationSide == LEFT) {
    return false;
  }

  return expressionPath < other.expressionPath;
}

EquationPath::EquationSide EquationPath::getEquationSide() const {
  return equationSide;
}

uint64_t EquationPath::operator[](size_t index) const {
  return expressionPath[index];
}

EquationPath::const_iterator EquationPath::begin() const {
  return expressionPath.begin();
}

EquationPath::const_iterator EquationPath::end() const {
  return expressionPath.end();
}

EquationPath EquationPath::operator+(uint64_t index) const {
  return EquationPath(equationSide, expressionPath + index);
}

EquationPath &EquationPath::operator+=(uint64_t index) {
  expressionPath += index;
  return *this;
}

size_t EquationPath::size() const { return expressionPath.size(); }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const EquationPath &obj) {
  os << "[";

  if (obj.getEquationSide() == EquationPath::LEFT) {
    os << "L";
  } else {
    os << "R";
  }

  for (uint64_t index : obj) {
    os << ", " << index;
  }

  os << "]";
  return os;
}
} // namespace mlir::bmodelica
