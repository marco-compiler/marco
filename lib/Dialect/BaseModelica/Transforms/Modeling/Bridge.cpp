#include "marco/Dialect/BaseModelica/Transforms/Modeling/Bridge.h"

using namespace ::mlir::bmodelica::bridge;

namespace mlir::bmodelica::bridge {
struct StorageImpl : Storage {};

std::unique_ptr<Storage> Storage::create() {
  return std::make_unique<StorageImpl>();
}

bool Storage::hasVariable(VariableBridge::Id id) const {
  return variablesMap.contains(id);
}

VariableBridge &Storage::getVariable(VariableBridge::Id id) const {
  assert(hasVariable(id));
  return *variablesMap.lookup(id);
}

llvm::ArrayRef<std::unique_ptr<VariableBridge>> Storage::getVariables() const {
  return variableBridges;
}

bool Storage::hasEquation(EquationBridge::Id id) const {
  return equationsMap.contains(id);
}

EquationBridge &Storage::getEquation(EquationBridge::Id id) const {
  assert(hasEquation(id));
  return *equationsMap.lookup(id);
}

llvm::ArrayRef<std::unique_ptr<EquationBridge>> Storage::getEquations() const {
  return equationBridges;
}

void Storage::clear() {
  clearVariables();
  clearEquations();
  clearSCCs();
}

void Storage::clearVariables() {
  variableBridges.clear();
  variablesMap.clear();
}

void Storage::clearEquations() {
  equationBridges.clear();
  equationsMap.clear();
}

void Storage::clearSCCs() { sccBridges.clear(); }
} // namespace mlir::bmodelica::bridge
