#include "marco/Dialect/BaseModelica/Transforms/Modeling/Bridge.h"

using namespace ::mlir::bmodelica::bridge;

namespace mlir::bmodelica::bridge {
struct StorageImpl : Storage {};

std::unique_ptr<Storage> Storage::create() {
  return std::make_unique<StorageImpl>();
}
} // namespace mlir::bmodelica::bridge
