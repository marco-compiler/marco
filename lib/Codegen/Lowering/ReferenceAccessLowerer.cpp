#include "marco/Codegen/Lowering/ReferenceAccessLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  ReferenceAccessLowerer::ReferenceAccessLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  Results ReferenceAccessLowerer::lower(
      const ast::ReferenceAccess& referenceAccess)
  {
    return lookupVariable(referenceAccess.getName());
  }
}
