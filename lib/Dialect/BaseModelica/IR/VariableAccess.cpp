#include "marco/Dialect/BaseModelica/IR/VariableAccess.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

using namespace ::mlir::bmodelica;

namespace mlir::bmodelica {
VariableAccess::VariableAccess(EquationPath path, mlir::SymbolRefAttr variable,
                               std::unique_ptr<AccessFunction> accessFunction)
    : path(std::move(path)), variable(variable),
      accessFunction(std::move(accessFunction)) {}

VariableAccess::VariableAccess(const VariableAccess &other)
    : path(other.path), variable(other.variable),
      accessFunction(other.accessFunction->clone()) {}

VariableAccess::~VariableAccess() = default;

VariableAccess &VariableAccess::operator=(const VariableAccess &other) {
  VariableAccess result(other);
  swap(*this, result);
  return *this;
}

VariableAccess &VariableAccess::operator=(VariableAccess &&other) = default;

bool VariableAccess::operator<(const VariableAccess &other) const {
  return path < other.path;
}

void swap(VariableAccess &first, VariableAccess &second) {
  using std::swap;
  swap(first.path, second.path);
  swap(first.variable, second.variable);
  swap(first.accessFunction, second.accessFunction);
}

const EquationPath &VariableAccess::getPath() const { return path; }

mlir::SymbolRefAttr VariableAccess::getVariable() const { return variable; }

const AccessFunction &VariableAccess::getAccessFunction() const {
  return *accessFunction;
}
} // namespace mlir::bmodelica
