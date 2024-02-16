#include "marco/Codegen/Lowering/WhenEquationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  WhenEquationLowerer::WhenEquationLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void WhenEquationLowerer::lower(const ast::WhenEquation& equation)
  {
    llvm_unreachable("When equation is not implemented yet");
  }
}
