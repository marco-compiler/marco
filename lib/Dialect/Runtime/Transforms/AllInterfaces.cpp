#include "marco/Dialect/Runtime/Transforms/AllInterfaces.h"
#include "marco/Dialect/Runtime/Transforms/BufferizableOpInterfaceImpl.h"

namespace mlir::runtime
{
  void registerAllDialectInterfaceImplementations(
      mlir::DialectRegistry& registry)
  {
    registerBufferizableOpInterfaceExternalModels(registry);
  }
}
