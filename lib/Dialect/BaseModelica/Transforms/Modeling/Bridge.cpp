#include "marco/Dialect/BaseModelica/Transforms/Modeling/Bridge.h"

using namespace ::mlir::bmodelica::bridge;

namespace mlir::bmodelica::bridge {
struct StorageImpl : Storage {};

std::unique_ptr<Storage> Storage::create() {
  return std::make_unique<StorageImpl>();
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
